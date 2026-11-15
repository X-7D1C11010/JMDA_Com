"""Server-plan, preset-fidelity and fail-before-dispatch checks; no training."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_backlight_rainy_suite as suite
from weather.rainy_presets import arguments_for, EVIDENCE
from weather.train_vds_core import parser, resolve_weather_settings

ROOT = Path(__file__).resolve().parents[1]


class WeatherServerSuiteChecks(unittest.TestCase):
    def test_default_plan_repairs_only_missing_backlight_seeds_and_runs_rain_five_seeds(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)/"project with spaces"; output = Path(name)/"output"
            plan = suite.build_plan(root, output)
            self.assertEqual(plan["requested"], 35)
            self.assertEqual(plan["required_weights"], ["convnext_small", "resnet18", "unip_tiny"])
            for target, seeds, count in (("backlight", {42, 43}, 10), ("rainy", set(range(42, 47)), 25)):
                jobs = [j for j in plan["jobs"] if j["target"] == target]
                self.assertEqual(len(jobs), count)
                self.assertEqual({j["seed"] for j in jobs}, seeds)
                self.assertEqual({j["method"] for j in jobs}, set(suite.METHODS))
                for job in jobs:
                    args = parser().parse_args(job["argv"][4:]); resolve_weather_settings(args)
                    self.assertEqual((args.method, args.target, args.seed), (job["method"], target, job["seed"]))
                    self.assertEqual(args.stop_after_epochs, 0)
                    self.assertEqual(args.max_train_batches, 0)
                    self.assertTrue(str(args.weights).startswith(str(root)))
                    for key in ("data_dir", "weights", "prediction_teacher_weights", "ir_weights"):
                        value = getattr(args, key)
                        if value:
                            self.assertNotIn("balanced_v4_seed42", str(value))
                            self.assertNotIn("weather_enhanced_screen", str(value))
                    if target == "rainy":
                        self.assertTrue(args.prediction_teacher_weights.startswith(str(root)))
                        if args.ir_weights:
                            self.assertTrue(args.ir_weights.startswith(str(root)))
                    if target == "rainy":
                        self.assertEqual(args.prediction_distillation_weight, 4.)
                        self.assertEqual(args.prediction_distillation_start_epoch, 2)
                        self.assertEqual(args.epochs, 40 if args.method == "cdan" else 20)
                    else:
                        self.assertEqual(args.epochs, 60)
                        self.assertEqual(args.encoder_training_scope, "frozen")
            self.assertFalse(output.exists())

    def test_scope_does_not_repeat_other_weather_or_VDS(self):
        for scope, count in (("backlight", 10), ("rainy", 25)):
            plan = suite.build_plan(Path("project"), Path("output"), scope)
            self.assertEqual(plan["requested"], count)
            self.assertEqual({j["target"] for j in plan["jobs"]}, {scope})
            self.assertFalse(plan["VDS_repeated"])
        with self.assertRaises(ValueError):
            suite.build_plan(Path("project"), Path("output"), backlight_seeds=[42, 42])

    def test_exported_rain_recipes_match_actual_successful_local_runs(self):
        checked = 0
        runtime_keys = {"data_dir", "save_dir", "weights", "prediction_teacher_weights", "ir_weights", "seed"}
        parser_keys = {a.dest for a in parser()._actions}
        for method, evidence in EVIDENCE.items():
            config = ROOT/Path(evidence["local_result"]).parent/"run_config.json"
            if not config.exists():
                continue
            expected = json.loads(config.read_text(encoding="utf-8"))["arguments"]
            values = arguments_for(method, ROOT, ROOT/"lists", ROOT/"new output", 46)
            from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
            restored = parser().parse_args(recorded_arguments_to_cli(values)); resolve_weather_settings(restored)
            for key in parser_keys & expected.keys() - runtime_keys:
                self.assertEqual(getattr(restored, key), expected[key], (method, key))
            self.assertEqual(restored.seed, 46)
            checked += 1
        if not checked:
            self.skipTest("Local experimental records are not shipped to the server")
        self.assertEqual(checked, 5)

    def test_missing_weight_fails_before_any_dataset_or_training_subprocess(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)/"output"
            argv = ["queue", "--project-root", str(ROOT), "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(suite, "require_weight", side_effect=FileNotFoundError("missing weight")), \
                    patch.object(suite, "dependency_preflight"), patch.object(suite.subprocess, "run") as dispatch:
                with self.assertRaisesRegex(FileNotFoundError, "missing weight"):
                    suite.main()
                dispatch.assert_not_called()
                self.assertFalse(output.exists())

    def make_results(self, root):
        plan = suite.build_plan(root, root)
        for job in plan["jobs"]:
            path = Path(job["result_file"]); path.parent.mkdir(parents=True, exist_ok=True)
            accuracy = .80 + (job["seed"]-42)*.01
            result = {"method": job["method"], "target": job["target"], "seed": job["seed"],
                      "protocol": "UDA", "target_training_labels_used": False, "screening_only": False,
                      "smoke": False, "completed_epochs": job["epochs"], "selected_epoch": 3,
                      "target_val": {k: accuracy for k in suite.METRICS.values()},
                      "complexity": {"total_params": 1000}, "enhancement": {"prediction_distillation_weight": 4.,
                      "prediction_distillation": {"inference_includes_teacher": False}}}
            suite.write_json(path, result)
            suite.write_json(path.parent/"completeness.json", {"status": "PASS"})
        return plan

    def test_summary_counts_missing_runs_and_uses_population_std_for_all_four_metrics(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); plan = self.make_results(root)
            self.assertEqual(suite.summarize(root, plan), (35, []))
            with (root/"summary/summary.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 10)
            for row in rows:
                n = 5 if row["target"] == "rainy" else 2
                self.assertEqual(int(row["num_seeds"]), n)
                for key in suite.METRICS:
                    self.assertAlmostEqual(float(row[key+"_mean"]), 82. if n == 5 else 80.5)
                    self.assertAlmostEqual(float(row[key+"_std"]), 2**.5 if n == 5 else .5)
            Path(plan["jobs"][-1]["result_file"]).unlink()
            self.assertEqual(suite.summarize(root, plan), (34, [plan["jobs"][-1]["tag"]]))

    def test_summary_rejects_target_supervision_prefix_training_and_teacher_inference(self):
        for change in ({"target_training_labels_used": True}, {"screening_only": True},
                       {"completed_epochs": 1}, {"enhancement": {"prediction_distillation_weight": 4.,
                       "prediction_distillation": {"inference_includes_teacher": True}}}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as name:
                root = Path(name); plan = self.make_results(root)
                path = Path(next(j for j in plan["jobs"] if j["target"] == "rainy")["result_file"])
                r = json.loads(path.read_text()); r.update(change); suite.write_json(path, r)
                with self.assertRaises(ValueError):
                    suite.summarize(root, plan)


if __name__ == "__main__":
    unittest.main()
