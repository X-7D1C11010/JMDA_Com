"""Protocol, objective-gradient and multimodal tests; no real-data training."""

import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from vds2raw.adaptation import (
    METHODS, AdaptationDataset, AdaptationModel, PairedEpochSampler,
    cdan_entropy_loss, prepare_statistics, reverse, optimizer_groups,
)
from vds2raw.dataset import CLASS_NAMES, PARTITIONS
from vds2raw import train_adaptation
from vds2raw.summarize_adaptation import summarize


def fixture(root):
    schema = {"train_partitions": ["source_train", "target_train"], "feature_dim": 13,
              "ais_feature_names": ["heading_sin", "heading_cos", "length_z", "width_z"]+[f"nav{i}" for i in range(9)]}
    (root/"feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    for partition in PARTITIONS:
        for label, name in enumerate(CLASS_NAMES):
            directory = root/"samples"/partition/name
            directory.mkdir(parents=True)
            for index in range(2 if partition.endswith("train") else 1):
                value = 10+label*5+index if partition.endswith("train") else 5000
                ais = np.asarray([0.6, 0.8, label+index, 2*label+index]+[float(i == label) for i in range(9)], dtype=np.float32)
                mask = np.ones(13, dtype=np.float32)
                if label == 2:
                    mask[2], ais[2] = 0, 0
                np.savez(directory/f"{index}.npz", rgb=np.full((3, 96, 96), value, np.uint16),
                         nir=np.full((1, 96, 96), value+2, np.uint16), ais=ais, ais_mask=mask,
                         label=np.asarray(label, dtype=np.int64))


class DatasetAndSamplerTests(unittest.TestCase):
    def test_target_loader_never_reads_npz_label_or_exports_training_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            images, ais = prepare_statistics(root, root/"statistics.json")
            target = AdaptationDataset(root, "target_train", images, ais)
            path = target.paths[0]
            with np.load(path) as data:
                stripped = {key: data[key] for key in data.files if key != "label"}
            np.savez(path, **stripped)
            self.assertNotIn("label", target[0])
            self.assertEqual({label for _, label in target.samples}, {-1})
    def test_complete_13_features_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            images, ais = prepare_statistics(root, root/"statistics.json")
            self.assertEqual(images.fit_samples, 12)
            self.assertEqual(ais["fit_samples"], 12)
            self.assertLess(images.rgb_mean[0], 100)
            self.assertEqual(ais["valid_counts"][2], 8)
            ds = AdaptationDataset(root, "source_train", images, ais)
            sample = ds[-1]
            self.assertEqual(sample["ais"].shape, (13,))
            self.assertEqual(sample["ais_mask"].shape, (13,))
            self.assertEqual(float(sample["ais"][2]), 0)
            self.assertEqual(float(sample["ais"][6]), 1)
            self.assertEqual(sample["nir"].shape, (1, 96, 96))
            with self.assertRaises(ValueError):
                AdaptationDataset(root, "target_val", images, ais, augment=True)

    def test_flips_change_heading_and_keep_registered_native_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            images, ais = prepare_statistics(root, root/"statistics.json")
            ds = AdaptationDataset(root, "source_train", images, ais, augment=True)
            with patch("vds2raw.adaptation.torch.rand", side_effect=[torch.tensor(0.), torch.tensor(1.), torch.tensor(0.5)]):
                sample = ds[0]
            self.assertAlmostEqual(float(sample["ais"][0]), -0.6, places=6)
            self.assertAlmostEqual(float(sample["ais"][1]), 0.8, places=6)
            self.assertEqual(sample["rgb"].shape, (3, 96, 96))

    def test_sampler_covers_all_and_repeats_reproducibly(self):
        for size in (94, 116):
            sampler = PairedEpochSampler(range(size), 16, 8, 42)
            batches = list(sampler)
            self.assertEqual(len(batches), 8)
            self.assertTrue(all(len(b) == 16 for b in batches))
            self.assertEqual(set(sum(batches, [])), set(range(size)))
            self.assertEqual(list(sampler), batches)
            sampler.epoch += 1
            self.assertNotEqual(list(sampler), batches)


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_five_ignore_target_truth_and_backpropagate_all_modalities(self):
        args = SimpleNamespace(label_smoothing=0.05,
                               steps_per_epoch=8, adaptation_warmup_epochs=0,
                               adaptation_ramp_epochs=1, trade_off=0.2)
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(42)
                model = AdaptationModel(method, pretrained=False).eval()
                batch = {"rgb": torch.randn(2, 3, 32, 32), "nir": torch.randn(2, 1, 32, 32),
                         "ais": torch.randn(2, 13), "ais_mask": torch.ones(2, 13), "label": torch.tensor([0, 1])}
                target = {**batch, "label": torch.tensor([1, 2])}
                loss, detail = model.objective(batch, target, 10, 100, args)
                changed = {**target, "label": object()}
                changed_loss, changed_detail = model.objective(batch, changed, 10, 100, args)
                torch.testing.assert_close(loss, changed_loss)
                torch.testing.assert_close(detail["classification"], changed_detail["classification"])
                torch.testing.assert_close(detail["classification"], detail["source_ce"])
                self.assertNotIn("target_ce", detail)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                for parameter in (model.rgb_encoder.conv1.weight, model.nir_encoder.conv1.weight,
                                  model.ais_encoder[0].weight, next(model.head.parameters()), model.modality_logits):
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(float(parameter.grad.norm()), 0)
                self.assertAlmostEqual(float(model.modality_weights().sum().detach()), 1, places=6)

    def test_reference_learning_rate_groups_and_forward_modes(self):
        for method in METHODS:
            model = AdaptationModel(method, pretrained=False).eval()
            groups = optimizer_groups(model, 0.001)
            self.assertEqual([g["lr"] for g in groups],
                             [0.001, 0.01] if method in {"deepcoral", "cdan"} else [0.0001, 0.001])
            ids = [id(p) for group in groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            if method == "cdan":
                self.assertIsInstance(model.bottleneck, nn.Linear)
                self.assertEqual(model.discriminator[0].out_features, 1024)
            if method == "mdd":
                self.assertIsInstance(model.head, nn.Sequential)
                self.assertIsInstance(model.bottleneck[-1], nn.Dropout)

    def test_actual_released_cdan_function_values_and_gradients(self):
        path = Path(__file__).resolve().parents[1]/"CDAN/pytorch/loss.py"
        spec = importlib.util.spec_from_file_location("vds_reference_cdan_loss", path)
        released = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(released)
        torch.manual_seed(17)
        features = torch.randn(8, 5, requires_grad=True)
        logits = torch.randn(8, 3, requires_grad=True)
        discriminator = nn.Linear(15, 1)
        coefficient = 0.3

        class ReleasedAdversary(nn.Module):
            def forward(self, inputs):
                return discriminator(reverse(inputs, coefficient)).sigmoid()

        actual, _ = cdan_entropy_loss(features, logits, discriminator, 4, coefficient)
        probabilities = logits.softmax(1)
        # The old released function hardcodes .cuda() only for domain labels.
        with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
            expected = released.CDAN([features, probabilities], ReleasedAdversary(),
                                     released.Entropy(probabilities), coefficient)
        torch.testing.assert_close(actual, expected)
        ag = torch.autograd.grad(actual, (features, logits), retain_graph=True)
        eg = torch.autograd.grad(expected, (features, logits))
        for a, b in zip(ag, eg):
            torch.testing.assert_close(a, b)

    def test_gradient_reversal_direction(self):
        inputs = torch.tensor([1., -2.], requires_grad=True)
        (reverse(inputs, 0.4).square().sum()).backward()
        torch.testing.assert_close(inputs.grad, -0.4*2*inputs.detach())

    def test_cdan_entropy_weighting_matches_released_formula_and_gradient(self):
        torch.manual_seed(3)
        features = torch.randn(8, 5, requires_grad=True)
        logits = torch.randn(8, 3, requires_grad=True)
        discriminator = nn.Linear(15, 1)
        loss, _ = cdan_entropy_loss(features, logits, discriminator, 4, 0.3)
        probabilities = logits.softmax(1)
        joint = torch.bmm(probabilities.detach().unsqueeze(2), features.unsqueeze(1)).flatten(1)
        scores = discriminator(reverse(joint, 0.3)).flatten().sigmoid()
        labels = torch.tensor([1.]*4+[0.]*4)
        entropy = -(probabilities*(probabilities+1e-5).log()).sum(1)
        w = 1+torch.exp(-reverse(entropy, 0.3))
        weights = torch.cat((w[:4]/w[:4].sum().detach(), w[4:]/w[4:].sum().detach()))
        reference = (weights*nn.functional.binary_cross_entropy(scores, labels, reduction="none")).sum()/weights.sum().detach()
        torch.testing.assert_close(loss, reference)
        actual = torch.autograd.grad(loss, (features, logits), retain_graph=True)
        expected = torch.autograd.grad(reference, (features, logits))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)


class TinyModel(AdaptationModel):
    def __init__(self, method, *_args):
        nn.Module.__init__(self)
        self.method = method
        self.rgb_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
        self.nir_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 8))
        self.ais_encoder = nn.Sequential(nn.Linear(26, 8), nn.ReLU())
        self.modality_logits = nn.Parameter(torch.zeros(3))
        self.bottleneck = nn.Identity()
        self.head, self.discriminator = nn.Linear(24, 3), nn.Linear(24, 1)


class TrainingProtocolTests(unittest.TestCase):
    def test_test_is_called_once_after_val_selection_and_selected_state_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            output = root/"output"
            args = train_adaptation.parser().parse_args([
                "--method", "dann", "--data-root", str(root), "--save-dir", str(output),
                "--epochs", "2", "--batch-size", "2", "--max-train-batches", "1",
                "--adaptation-warmup-epochs", "0", "--num-workers", "0", "--cpu-threads", "2",
                "--no-pretrained", "--skip-audit", "--device", "cpu"])
            order, states = [], {}
            evaluator = train_adaptation.evaluate

            def observed(model, dataset, device, epoch, directory, **kwargs):
                order.append((dataset.partition, epoch))
                if dataset.partition == "target_test":
                    self.assertEqual(sum(p == "target_val" for p, _ in order), 2)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value.cpu(), states[epoch][name])
                result = evaluator(model, dataset, device, epoch, directory, **kwargs)
                if dataset.partition == "target_val":
                    states[epoch] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                return result

            with patch.object(train_adaptation, "AdaptationModel", TinyModel), \
                 patch.object(train_adaptation, "evaluate", side_effect=observed), \
                 patch.object(train_adaptation, "build_run_provenance", return_value={}):
                result = train_adaptation.run(args)
            self.assertEqual(sum(p == "target_test" for p, _ in order), 1)
            self.assertEqual(order[-1][0], "target_test")
            self.assertFalse(result["test_selection_bias"])
            self.assertEqual(result["protocol"], "UDA")
            self.assertFalse(result["target_training_labels_used"])
            self.assertEqual(result["selection_partition"], "target_val")
            self.assertEqual(result["test_evaluations_for_selection"], 0)
            report = json.loads((output/"dann_seed42/completeness.json").read_text())
            self.assertEqual(report["status"], "PASS")
            rows = summarize(output)
            self.assertEqual(rows[0]["test_samples"], 3)
            self.assertTrue((output/"summary/runs.csv").is_file())
            prediction_rows = [json.loads(s) for s in (output/"dann_seed42/predictions.jsonl").read_text().splitlines()]
            for row in prediction_rows:
                if row["partition"] == "target_train":
                    self.assertNotIn("y_true", row)
                    self.assertNotIn("correct", row)
                    self.assertFalse(row["labels_used"])
            # A PASS must not survive deletion of a persisted diagnostic.
            (output/"dann_seed42/gradients.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                train_adaptation.validate_diagnostics(output/"dann_seed42", (), 2, result["selected_epoch"])


if __name__ == "__main__":
    unittest.main()
