import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.run_vds_backlight_suite import build_plan, summarize, write_json, COMPLEXITY


def fixture(root):
    plan = build_plan(root, root)
    write_json(root/"run_plan.json", plan)
    paths = []
    for experiment in plan["experiments"]:
        vds = experiment["dataset"] == "VDS2Raw"
        directory = root/("vds" if vds else "weather")/experiment["method"]/str(experiment["seed"])
        directory.mkdir(parents=True)
        def metrics(accuracy):
            return {"accuracy": accuracy, "precision_macro": accuracy, "recall_macro": accuracy, "f1_macro": accuracy}
        result = {"method": experiment["method"], "seed": experiment["seed"], "protocol": "UDA",
                  "target_training_labels_used": False, "smoke": False, "selected_epoch": 12,
                  "completed_epochs": 66 if vds else 60,
                  "target_val": metrics(.2 if vds else .8), "target_test": metrics(.9) if vds else None,
                  "complexity": {key: 1. for key in COMPLEXITY}}
        if vds:
            result.update(ais_protocol="heading_size", optimization_profile=experiment["profile"],
                          selection_partition="target_val", test_selection_bias=False,
                          test_evaluations=1, test_evaluations_for_selection=0)
        else:
            result.update(target="backlight", screening_only=False, image_backbone="convnext_small", encoder_training_scope="frozen")
            write_json(directory/"completeness.json", {"status": "PASS"})
        path = directory/"result.json"
        write_json(path, result)
        paths.append(path)
    return plan, paths


class VDSBacklightSuiteTests(unittest.TestCase):
    def test_plan_requests_exactly_35_experiments_with_latest_profiles_and_all_five_seeds(self):
        plan = build_plan(Path("project"), Path("output"))
        experiments = plan["experiments"]
        self.assertEqual(plan["requested_experiments"], 35)
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual({e["seed"] for e in experiments}, set(range(42, 47)))
        self.assertEqual({e["target"] for e in experiments}, {"backlight", "non_day3"})
        self.assertEqual({(e["method"], e["profile"]) for e in experiments if e["dataset"] == "VDS2Raw"},
                         {("mcc", "mcc_v3"), ("cdan", "cdan_safe_v4")})
        self.assertEqual(len({(e["dataset"], e["target"], e["method"], e["seed"]) for e in experiments}), 35)
        for job, seed in zip(plan["jobs"][1:], range(42, 47)):
            self.assertEqual(job["argv"][job["argv"].index("--seed")+1], str(seed))

    def test_individual_scopes_do_not_repeat_the_other_dataset(self):
        for scope, count, dataset in (("vds", 10, "VDS2Raw"), ("backlight", 25, "weather")):
            plan = build_plan(Path("project"), Path("output"), scope)
            self.assertEqual(len(plan["experiments"]), count)
            self.assertEqual({e["dataset"] for e in plan["experiments"]}, {dataset})

    def test_summary_keeps_test_metrics_separate_from_weather_validation_and_counts_all_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture(root)
            count, missing = summarize(root)
            self.assertEqual((count, missing), (35, []))
            with (root/"summary/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(int(row["num_seeds"]) == 5 for row in rows))
            for row in rows:
                vds = row["dataset"] == "VDS2Raw"
                self.assertEqual(row["metric_partition"], "target_test" if vds else "target_val")
                self.assertAlmostEqual(float(row["ACC_percent_mean"]), 90. if vds else 80.)

    def test_partial_results_are_not_reported_as_a_completed_35_run_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, paths = fixture(root)
            paths[-1].unlink()
            count, missing = summarize(root)
            self.assertEqual((count, len(missing)), (34, 1))
            self.assertFalse(json.loads((root/"summary/completeness.json").read_text())["all_results_present"])

    def test_summary_rejects_wrong_supervision_stale_ais_profile_and_prefix_training(self):
        for change in ({"target_training_labels_used": True}, {"ais_protocol": "full13"},
                       {"optimization_profile": "optimized"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); _, paths = fixture(root)
                path = paths[0]; result = json.loads(path.read_text()); result.update(change); write_json(path, result)
                with self.assertRaises(ValueError): summarize(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, paths = fixture(root)
            path = paths[-1]; result = json.loads(path.read_text()); result["screening_only"] = True; write_json(path, result)
            with self.assertRaises(ValueError): summarize(root)

    def test_summary_rejects_duplicate_seed_instead_of_counting_it_as_an_extra_replicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, paths = fixture(root)
            duplicate = root/"vds/duplicate"; duplicate.mkdir()
            (duplicate/"result.json").write_bytes(paths[0].read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
