"""Weather shared-core, exact-pair, label-space and result-selection regressions."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from vds2raw.adaptation import AdaptationModel, resolve_training_settings
from weather import train_vds_core as weather
from weather.summarize_vds_core import summarize


def fixture(root):
    """Noncontiguous labels, a source-only class, and separate exact pairs."""
    lists = root/"lists"
    lists.mkdir()
    rng = np.random.default_rng(42)
    for partition, filename, labels in (("source", "sunny_train.txt", (0, 2, 5, 7, 9)),
                                        ("target", "backlight_train.txt", (0, 2, 5, 7)),
                                        ("val", "backlight_val.txt", (0, 2, 5, 7))):
        lines, pairs = [], []
        for label in labels:
            for index in range(2 if partition != "val" else 1):
                paths = []
                values = rng.integers(0, 256, (270, 270, 3), dtype=np.uint8)
                for modality in ("可见光", "红外"):
                    path = root/"images"/partition/modality/f"{label}_{index}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image = Image.fromarray(values) if modality == "可见光" else Image.fromarray(values[:, :, 0])
                    image.save(path)
                    paths.append(str(path))
                lines.append(f"{paths[0]} {label}\n")
                pairs.append("\t".join(paths)+"\n")
        (lists/filename).write_text("".join(lines), encoding="utf-8")
        if partition == "target":
            (lists/"backlight_train_pairs.tsv").write_text("".join(pairs), encoding="utf-8")
    return lists


class TinyWeather(AdaptationModel):
    def __init__(self, method, *_args, num_classes=3, **_kwargs):
        nn.Module.__init__(self)
        self.method, self.use_ais, self.num_classes = method, False, num_classes
        self.feature_names = ("VIS", "IR")
        self.input_keys = ("rgb", "nir")
        self.rgb_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
        self.nir_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
        self.modality_logits = nn.Parameter(torch.zeros(2))
        self.bottleneck = nn.Identity()
        self.head, self.discriminator = nn.Linear(16, num_classes), nn.Linear(16, 1)


class WeatherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_timing_v2_has_separate_weather_defaults_and_keeps_explicit_cli_settings(self):
        for method, lr, trade in (("dann", .00015, .20), ("deepcoral", .0001, .10)):
            args = weather.parser().parse_args(["--method", method, "--target", "dark", "--data-dir", ".",
                                                "--weather-profile", "timing_v2"])
            weather.resolve_weather_settings(args)
            self.assertEqual((args.epochs, args.lr, args.trade_off, args.domain_forward), (40, lr, trade, "joint"))
            self.assertEqual((args.adaptation_ramp_epochs, args.lr_scheduler), (5, "cosine"))
            self.assertEqual((args.early_stopping_min_epochs, args.early_stopping_patience), (20, 15))
            explicit = weather.parser().parse_args(["--method", method, "--target", "dark", "--data-dir", ".",
                        "--weather-profile", "timing_v2", "--epochs", "2", "--lr", ".001", "--domain-forward", "reference"])
            weather.resolve_weather_settings(explicit)
            self.assertEqual((explicit.epochs, explicit.lr, explicit.domain_forward), (2, .001, "reference"))

    def test_joint_forward_shares_bn_update_without_target_labels_and_logs_moments(self):
        for method in ("dann", "deepcoral"):
            args = weather.parser().parse_args(["--method", method, "--target", "backlight", "--data-dir", ".",
                                                "--weather-profile", "timing_v2"])
            weather.resolve_weather_settings(args)
            args.steps_per_epoch = 2
            model = AdaptationModel(method, pretrained=False, num_classes=4, use_ais=False, ir_channels=3)
            source = {"rgb": torch.randn(4, 3, 64, 64), "nir": torch.randn(4, 3, 64, 64), "label": torch.arange(4)}
            target = {"rgb": torch.randn(4, 3, 64, 64), "nir": torch.randn(4, 3, 64, 64)}
            before = int(model.rgb_encoder.bn1.num_batches_tracked)
            loss, details = model.objective(source, target, 3, 80, args)
            self.assertEqual(int(model.rgb_encoder.bn1.num_batches_tracked)-before, 1)
            self.assertEqual(details["target_logits"].shape, (4, 4))
            self.assertEqual(set(details["domain_feature_moments"]), {"before_bottleneck", "after_bottleneck"})
            loss.backward()
            self.assertGreater(float(model.rgb_encoder.conv1.weight.grad.norm()), 0.)
            self.assertGreater(float(model.nir_encoder.conv1.weight.grad.norm()), 0.)

    def test_same_class_map_filters_source_and_target_has_no_labels(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JMDA_DATA_ROOT": ""}):
            lists = fixture(Path(temporary))
            datasets, audit = weather.prepare_datasets(lists, "backlight")
            source, target, _, target_eval, validation = datasets
            self.assertEqual(audit["compact_label_map"], {0: 0, 2: 1, 5: 2, 7: 3})
            self.assertEqual(audit["source_samples_excluded_outside_target_support"], 2)
            self.assertEqual(len(source), 8)
            self.assertEqual({source[i]["label"] for i in range(len(source))}, {0, 1, 2, 3})
            self.assertEqual(source[0]["rgb"].shape, (3, 224, 224))
            self.assertEqual(target[0]["nir"].shape, (3, 224, 224))
            self.assertNotIn("label", target[0])
            self.assertNotIn("label", target_eval[0])
            self.assertEqual(len(validation), 4)
            # Once metadata checks complete, the target batch loader works
            # even with the labelled target metadata physically removed.
            (lists/"backlight_train.txt").unlink()
            self.assertNotIn("label", target[0])

    def test_corrupt_target_pair_cohort_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JMDA_DATA_ROOT": ""}):
            lists = fixture(Path(temporary))
            path = lists/"backlight_train_pairs.tsv"
            path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[:-1])+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cohort"):
                weather.prepare_datasets(lists, "backlight")

    def test_shared_real_model_dynamic_classes_and_both_encoders_receive_da_gradients(self):
        for method in ("dann", "deepcoral"):
            model = AdaptationModel(method, pretrained=False, num_classes=4, use_ais=False, ir_channels=3)
            args = weather.parser().parse_args(["--method", method, "--target", "backlight", "--data-dir", "."])
            resolve_training_settings(args)
            args.steps_per_epoch, args.adaptation_ramp_epochs = 2, 1
            self.assertFalse(hasattr(model, "ais_encoder"))
            source = {"rgb": torch.randn(4, 3, 64, 64), "nir": torch.randn(4, 3, 64, 64), "label": torch.arange(4)}
            target = {"rgb": torch.randn(4, 3, 64, 64), "nir": torch.randn(4, 3, 64, 64)}
            loss, details = model.objective(source, target, 1, 200, args)
            self.assertEqual(details["target_logits"].shape, (4, 4))
            gradients = torch.autograd.grad(details["weighted_transfer"],
                                           (model.rgb_encoder.conv1.weight, model.nir_encoder.conv1.weight), retain_graph=True)
            self.assertTrue(all(float(g.norm()) > 0 for g in gradients))
            loss.backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())
            self.assertAlmostEqual(float(model.modality_weights().sum()), 1., places=6)

    def test_each_epoch_diagnostics_and_peak_epoch_metrics_are_persisted(self):
        for method in ("dann", "deepcoral"):
            with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JMDA_DATA_ROOT": ""}):
                root = Path(temporary)
                lists = fixture(root)
                output = root/"output"
                args = weather.parser().parse_args([
                    "--method", method, "--target", "backlight", "--data-dir", str(lists),
                    "--save-dir", str(output), "--epochs", "2", "--batch-size", "4",
                    "--num-workers", "0", "--cpu-threads", "2", "--no-pretrained", "--device", "cpu"])
                with patch.object(weather, "AdaptationModel", TinyWeather), \
                     patch.object(weather, "build_run_provenance", return_value={}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = weather.run(args)
                    rows = summarize(output)
                run_dir = output/f"{method}_backlight_seed42"
                history = [json.loads(line) for line in (run_dir/"evaluation.jsonl").read_text().splitlines()]
                val = [row for row in history if row["partition"] == "target_val"]
                best = max(val, key=lambda row: (row["metrics"]["accuracy"], row["metrics"]["f1_macro"]))
                self.assertEqual(result["selected_epoch"], best["epoch"])
                self.assertEqual(result["target_val"], best["metrics"])
                self.assertEqual(result["optimizer_updates"], 4)
                self.assertEqual(rows[0]["ACC_percent"], 100*best["metrics"]["accuracy"])
                self.assertEqual(json.loads((run_dir/"completeness.json").read_text())["status"], "PASS")
                self.assertFalse(list(run_dir.glob("*.pth")))
                self.assertFalse(any(row["partition"] == "target_test" for row in history))
                for line in (run_dir/"predictions.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    if row["partition"] == "target_train":
                        self.assertNotIn("y_true", row)
                        self.assertFalse(row["labels_used"])


if __name__ == "__main__":
    unittest.main()
