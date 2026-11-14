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

from vds2raw.adaptation import AdaptationModel, resolve_training_settings, METHODS
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


def fast_five_method_model(method, *_args, **kwargs):
    model = AdaptationModel(method, pretrained=False, num_classes=kwargs["num_classes"], use_ais=False,
                            ir_channels=3, feature_normalization=kwargs["feature_normalization"],
                            auxiliary_source_weight=kwargs["auxiliary_source_weight"])
    # Preserve the actual bottleneck, heads and DA objective; reduce only image
    # encoder cost in this synthetic persistence/protocol check.
    model.rgb_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512), nn.ReLU())
    model.nir_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512), nn.ReLU())
    return model


class WeatherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_balanced_v4_all_five_have_explicit_method_settings_and_encoder_only_bn(self):
        for method in METHODS:
            args = weather.parser().parse_args(["--method", method, "--target", "dark", "--data-dir", ".",
                                                "--weather-profile", "balanced_v4"])
            weather.resolve_weather_settings(args)
            self.assertEqual(args.feature_normalization, "l2")
            self.assertEqual((args.auxiliary_source_weight, args.gradient_clip_norm), (.20, 5.))
            self.assertEqual(args.freeze_bn_scope, "encoders")
            self.assertTrue(args.source_class_balance)
            self.assertEqual(args.validation_control_start_epoch, args.adaptation_warmup_epochs+args.adaptation_ramp_epochs)
            if method == "deepcoral":
                self.assertEqual((args.lr, args.trade_off, args.freeze_bn_after_epochs), (.0001, .05, -1))
        model = AdaptationModel("dann", pretrained=False, use_ais=False, ir_channels=3).train()
        weather.apply_batchnorm_policy(model, 4, 3, "encoders")
        self.assertFalse(model.rgb_encoder.bn1.training)
        self.assertFalse(model.nir_encoder.bn1.training)
        self.assertTrue(model.bottleneck[1].training)
        self.assertTrue(model.discriminator[2].training)

    def test_prefix_screen_preserves_full_training_budget_lr_and_grl(self):
        histories = []
        for limit in (0, 2):
            with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JMDA_DATA_ROOT": ""}):
                root = Path(temporary)
                lists = fixture(root)
                args = weather.parser().parse_args(["--method", "dann", "--target", "backlight", "--data-dir", str(lists),
                    "--save-dir", str(root/"output"), "--epochs", "4", "--stop-after-epochs", str(limit),
                    "--batch-size", "4", "--num-workers", "0", "--cpu-threads", "2", "--device", "cpu", "--no-pretrained"])
                with patch.object(weather, "AdaptationModel", TinyWeather), contextlib.redirect_stdout(io.StringIO()):
                    result = weather.run(args)
                updates = [json.loads(s) for s in (root/"output/dann_backlight_seed42/updates.jsonl").read_text().splitlines()]
                histories.append(updates[:4])
                self.assertEqual(result["planned_optimizer_updates"], 8)
                self.assertEqual(result["completed_epochs"], limit or 4)
                self.assertEqual(result["screening_only"], bool(limit))
        self.assertEqual(histories[0], histories[1])

    def test_eval_input_cache_is_exact_label_free_and_does_not_consume_training_rng(self):
        import random
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JMDA_DATA_ROOT": ""}):
            lists = fixture(Path(temporary))
            sets, _ = weather.prepare_datasets(lists, "backlight", False)
            for dataset in sets[2:]:
                before = [dataset[i] for i in range(len(dataset))]
                torch_state, python_state = torch.get_rng_state(), random.getstate()
                report = dataset.cache_evaluation_inputs()
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                self.assertEqual(random.getstate(), python_state)
                self.assertFalse(report["model_features_cached"])
                for index, expected in enumerate(before):
                    actual = dataset[index]
                    self.assertEqual(set(actual), set(expected))
                    for key in expected:
                        if torch.is_tensor(expected[key]):
                            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                        else:
                            self.assertEqual(actual[key], expected[key])
            stochastic, _ = weather.prepare_datasets(lists, "backlight", True)
            with self.assertRaisesRegex(ValueError, "stochastic"):
                stochastic[0].cache_evaluation_inputs()

    def test_eval_cache_preserves_training_updates_and_validation_scores(self):
        records = []
        for cache in (False, True):
            with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JMDA_DATA_ROOT": ""}):
                root = Path(temporary)
                lists = fixture(root)
                argv = ["--method", "dann", "--target", "backlight", "--data-dir", str(lists),
                        "--save-dir", str(root/"output"), "--epochs", "2", "--batch-size", "4",
                        "--num-workers", "0", "--cpu-threads", "2", "--device", "cpu", "--no-pretrained"]
                if cache:
                    argv.append("--cache-evaluation")
                args = weather.parser().parse_args(argv)
                with patch.object(weather, "AdaptationModel", TinyWeather), contextlib.redirect_stdout(io.StringIO()):
                    result = weather.run(args)
                directory = root/"output/dann_backlight_seed42"
                updates = [json.loads(s) for s in (directory/"updates.jsonl").read_text().splitlines()]
                records.append((result["target_val"], result["selected_epoch"], updates))
        self.assertEqual(records[0], records[1])

    def test_all_five_auxiliary_classification_uses_only_source_and_keeps_da_gradients(self):
        for method in METHODS:
            args = weather.parser().parse_args(["--method", method, "--target", "dark", "--data-dir", ".",
                                                "--weather-profile", "balanced_v4"])
            weather.resolve_weather_settings(args)
            args.steps_per_epoch = 2
            model = AdaptationModel(method, pretrained=False, num_classes=4, use_ais=False, ir_channels=3,
                                    feature_normalization="l2", auxiliary_source_weight=.2).eval()
            source = {"rgb": torch.randn(4, 3, 64, 64), "nir": torch.randn(4, 3, 64, 64), "label": torch.arange(4)}
            target_count = 4 if method == "mdd" else 3
            target = {"rgb": torch.randn(target_count, 3, 64, 64), "nir": torch.randn(target_count, 3, 64, 64)}
            loss, details = model.objective(source, target, 100, 200, args)
            expected = torch.stack([nn.functional.cross_entropy(logits[:4], source["label"], label_smoothing=.05)
                                    for logits in model.last_branch_logits]).mean()
            torch.testing.assert_close(details["auxiliary_source_ce"], expected)
            torch.testing.assert_close(details["classification"], details["source_ce"]+.2*expected)
            changed_loss, _ = model.objective(source, {**target, "label": object()}, 100, 200, args)
            torch.testing.assert_close(loss, changed_loss)
            gradients = torch.autograd.grad(details["weighted_transfer"],
                         (model.rgb_encoder.conv1.weight, model.nir_encoder.conv1.weight), retain_graph=True)
            self.assertTrue(all(float(g.norm()) > 0 for g in gradients))
            loss.backward()
            for head in model.auxiliary_heads:
                self.assertGreater(float(head.weight.grad.norm()), 0.)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if method == "mdd":
                with self.assertRaisesRegex(ValueError, "equal source/target"):
                    model.objective(source, {"rgb": target["rgb"][:3], "nir": target["nir"][:3]}, 100, 200, args)

    def test_balanced_v4_all_five_persist_branch_diagnostics_and_no_target_training_labels(self):
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"JMDA_DATA_ROOT": ""}):
                root = Path(temporary)
                lists = fixture(root)
                args = weather.parser().parse_args([
                    "--method", method, "--target", "backlight", "--data-dir", str(lists),
                    "--weather-profile", "balanced_v4", "--save-dir", str(root/"output"),
                    "--epochs", "2", "--batch-size", "4", "--num-workers", "0", "--cpu-threads", "2",
                    "--device", "cpu", "--no-pretrained", "--max-train-batches", "1",
                    "--adaptation-warmup-epochs", "0", "--adaptation-ramp-epochs", "1"])
                with patch.object(weather, "AdaptationModel", fast_five_method_model), contextlib.redirect_stdout(io.StringIO()):
                    result = weather.run(args)
                directory = root/"output"/f"{method}_backlight_seed42"
                self.assertTrue(result["smoke"])
                self.assertTrue(result["modality_diagnostics"]["inference_independent_of_label_and_path"])
                evaluations = [json.loads(line) for line in (directory/"evaluation.jsonl").read_text().splitlines()]
                for row in evaluations:
                    self.assertEqual(set(row["auxiliary_branches"]), {"VIS", "IR"})
                    for branch in row["auxiliary_branches"].values():
                        self.assertEqual("metrics" in branch, row["partition"] != "target_train")
                updates = [json.loads(line) for line in (directory/"updates.jsonl").read_text().splitlines()]
                self.assertTrue(all(v["auxiliary_source_weight"] == .2 and v["gradient_norm_before_clipping"] is not None for v in updates))
                self.assertEqual(json.loads((directory/"completeness.json").read_text())["status"], "PASS")

    def test_bn_stable_changes_only_running_stat_policy_relative_to_baseline(self):
        parsed = []
        for profile in ("baseline", "bn_stable"):
            args = weather.parser().parse_args(["--method", "dann", "--target", "backlight", "--data-dir", ".", "--weather-profile", profile])
            weather.resolve_weather_settings(args)
            parsed.append(args)
        for key in ("epochs", "lr", "trade_off", "lr_scheduler", "domain_forward", "adaptation_ramp_epochs", "early_stopping_patience"):
            self.assertEqual(getattr(parsed[0], key), getattr(parsed[1], key))
        model = AdaptationModel("dann", pretrained=False, use_ais=False, ir_channels=3).train()
        self.assertFalse(weather.apply_batchnorm_policy(model, 3, 3)["BN_running_stats_frozen"])
        policy = weather.apply_batchnorm_policy(model, 4, 3)
        self.assertTrue(policy["BN_running_stats_frozen"])
        self.assertFalse(model.rgb_encoder.bn1.training)
        self.assertFalse(model.bottleneck[1].training)
        self.assertTrue(model.rgb_encoder.bn1.weight.requires_grad)
        self.assertTrue(model.discriminator[2].training)

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
