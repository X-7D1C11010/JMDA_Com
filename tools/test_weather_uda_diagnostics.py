"""Regression coverage for learned weights and observational diagnostics."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from weather_uda_diagnostics import RunDiagnostics, install_weather_weights
from historical_uda_metrics import HistoricalEfficiencyLogger, classification_metrics


class TinyPairedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sharedNet = nn.Sequential(
            nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.irNet = nn.Sequential(
            nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(4, 2)

    def forward(self, images):
        return self.fc(torch.cat((self.sharedNet(images[:, :3]),
                                  self.irNet(images[:, 3:])), dim=1))


class TinyPairedDataset(Dataset):
    def __init__(self):
        self.images = torch.arange(384, dtype=torch.float32).reshape(4, 6, 4, 4) / 384
        self.label_map = {1: 0, 5: 1}
        self.samples = [(f"VIS/{i}.png", i % 2) for i in range(4)]
        self.paired_samples = [(path, f"IR/{i}.png", label)
                               for i, (path, label) in enumerate(self.samples)]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.images[index], self.samples[index][1]


class WeatherDiagnosticsTests(unittest.TestCase):
    def args(self):
        from argparse import Namespace
        return Namespace(target="dark", seed=42, batch_size=2)

    def transform(self):
        return transforms.Compose([transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def test_equal_initial_weights_reproduce_concat_and_are_trainable(self):
        model = TinyPairedModel().eval()
        images = torch.randn(4, 6, 4, 4)
        expected = model(images).detach()
        rng = torch.get_rng_state()
        gate = install_weather_weights(model, "deepcoral", "dark")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
        loss = F.cross_entropy(model(images), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertIsNotNone(gate.logit.grad)
        self.assertNotEqual(float(gate.logit.grad), 0)
        with torch.no_grad():
            gate.logit.fill_(100)
        vis, ir = gate.weights()
        self.assertAlmostEqual(float(vis + ir), 1)
        self.assertAlmostEqual(float(vis), 0.95)
        self.assertAlmostEqual(float(ir), 0.05)

    def test_source_evaluation_preserves_training_state_and_exports_all_samples(self):
        dataset = TinyPairedDataset()
        model = TinyPairedModel().train()
        install_weather_weights(model, "deepcoral", "dark")
        state = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state()
        with tempfile.TemporaryDirectory() as tmp:
            audit = RunDiagnostics(
                model, "deepcoral", self.args(), dataset, dataset,
                self.transform(), lambda m, x: m(x), tmp,
            )
            with redirect_stdout(io.StringIO()):
                audit.evaluate_source(1)
            rows = [json.loads(line) for line in
                    (Path(tmp) / "diagnostics/predictions_source_train.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), len(dataset))
        self.assertEqual({row["sample_id"] for row in rows}, {p for p, _ in dataset.samples})
        self.assertTrue(all(row["partition"] == "source_train" for row in rows))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_diagnostic_gradients_preserve_the_optimizer_gradient(self):
        dataset = TinyPairedDataset()
        model, reference = TinyPairedModel(), TinyPairedModel()
        install_weather_weights(model, "deepcoral", "dark")
        install_weather_weights(reference, "deepcoral", "dark")
        reference.load_state_dict(model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            audit = RunDiagnostics(
                model, "deepcoral", self.args(), dataset, dataset,
                self.transform(), lambda m, x: m(x), tmp,
            )
            audit.set_domain("mixed", 2)
            outputs = model(dataset.images)
            cls = F.cross_entropy(outputs[:2], torch.tensor([0, 1]))
            transfer = outputs[2:].square().mean()
            audit.training_step(cls, transfer, 0.2 * transfer, 2, 2, True, 1)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            (cls + 0.2 * transfer).backward()
            expected = reference(dataset.images)
            (F.cross_entropy(expected[:2], torch.tensor([0, 1])) +
             0.2 * expected[2:].square().mean()).backward()
            for actual, expected_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected_parameter.grad)

    def test_completeness_audit_rejects_a_missing_gradient_file(self):
        dataset = TinyPairedDataset()
        model = TinyPairedModel()
        install_weather_weights(model, "deepcoral", "dark")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            audit = RunDiagnostics(
                model, "deepcoral", self.args(), dataset, dataset,
                self.transform(), lambda m, x: m(x), tmp,
                target_dataset=dataset, optimizer=optimizer,
            )
            audit.set_domain("mixed", 2)
            outputs = model(dataset.images)
            cls = F.cross_entropy(outputs[:2], torch.tensor([0, 1]))
            transfer = outputs[2:].square().mean()
            audit.training_step(cls, transfer, 0.2 * transfer, 2, 2, True, 1)
            (cls + 0.2 * transfer).backward()
            optimizer.step()
            with redirect_stdout(io.StringIO()):
                audit.report_training(1)
                audit.evaluate_source(1)
                audit.evaluate_target_train(1)
                audit.begin_evaluation("target_val", 1)
                model.eval()
                with torch.no_grad():
                    predictions = model(dataset.images).argmax(1).tolist()
                metrics = classification_metrics([0, 1, 0, 1], predictions, 2)
                audit.finish_evaluation(metrics, dataset)
                summary = audit.summary(1)
            self.assertTrue(summary["diagnostics_complete"])
            report_path = Path(tmp) / "diagnostics/completeness.json"
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["status"], "PASS")
            (Path(tmp) / "diagnostics/backbone_gradients.jsonl").unlink()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "missing backbone"):
                audit.validate_outputs(1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["status"], "FAIL")

    def test_a_learnable_gate_must_belong_to_the_optimizer(self):
        dataset = TinyPairedDataset()
        model = TinyPairedModel()
        install_weather_weights(model, "deepcoral", "dark")
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "missing from the optimizer"):
                RunDiagnostics(
                    model, "deepcoral", self.args(), dataset, dataset,
                    self.transform(), lambda m, x: m(x), tmp,
                    target_dataset=dataset, optimizer=optimizer,
                )

    def test_target_prediction_export_ignores_training_labels(self):
        class LabelMustBeIgnored:
            def __int__(self):
                raise AssertionError("A target training label was inspected")

        class TargetDataset(TinyPairedDataset):
            def __getitem__(self, index):
                images, _ = super().__getitem__(index)
                return images, LabelMustBeIgnored()

        source, target = TinyPairedDataset(), TargetDataset()
        model = TinyPairedModel().train()
        install_weather_weights(model, "deepcoral", "dark")
        with tempfile.TemporaryDirectory() as tmp:
            audit = RunDiagnostics(
                model, "deepcoral", self.args(), source, source,
                self.transform(), lambda m, x: m(x), tmp, target_dataset=target,
            )
            audit.evaluate_target_train(1)
            rows = [json.loads(line) for line in
                    (Path(tmp) / "diagnostics/predictions_target_train.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), len(target))
        self.assertTrue(all(row["labels_used"] is False for row in rows))
        self.assertTrue(all("truth_compact" not in row for row in rows))
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_missing_complexity_metric_fails_the_completeness_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "diagnostics"
            directory.mkdir()
            report_path = directory / "completeness.json"
            report_path.write_text('{"status": "PASS", "errors": []}', encoding="utf-8")
            logger = HistoricalEfficiencyLogger(tmp, "Tiny-MM", device="cpu")
            logger.epoch_start(1)
            logger.epoch_end(1)
            logger.record_diagnostics({"diagnostics_dir": str(directory), "diagnostics_complete": True})
            logger.inference_metrics = {
                "total_params": 1, "trainable_params": 1, "complexity_status": "complete",
                "latency_mean_ms": 1, "throughput_samples_s": 1,
            }
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "inference_GFLOPs"):
                logger.finalize()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["complexity_status"], "FAIL")
            self.assertFalse((Path(tmp) / "efficiency_runs.csv").exists())


if __name__ == "__main__":
    unittest.main()
