import json
from pathlib import Path
import tempfile
import unittest

from tools.summarize_jmda_label_guided_formal import (
    EFFICIENCY,
    METRICS,
    summarize,
    write_outputs,
)


class FormalLabelGuidedSummaryTests(unittest.TestCase):
    targets = ["backlight", "rainy", "foggy", "dark"]

    def _summary(self):
        runs = []
        for seed in range(42, 47):
            run = {
                "seed": seed,
                "gpu_name": "test-gpu",
                "precision": "float32",
                "inference_data_loading_included": False,
                "flops_convention": "1 MAC = 2 FLOPs",
            }
            for key, _ in EFFICIENCY:
                run[key] = float(seed)
            runs.append(run)
        return {
            "protocol": "LABEL_GUIDED_DA_NO_TARGET_CE",
            "num_iterations": 5,
            "metrics": {
                key: {
                    "values": [0.7, 0.71, 0.72, 0.73, 0.74],
                    "mean": 0.72,
                    "std": 0.0141421356,
                }
                for key, _ in METRICS
            },
            "efficiency": {
                "enabled": True,
                "per_iteration": runs,
                "aggregate": {},
            },
        }

    def test_complete_suite_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            output = root / "out"
            logs.mkdir()
            for target in self.targets:
                path = logs / f"{target}_label_guided_tensor_ot.log"
                path.write_text(
                    "JMDA_RUN_SUMMARY_JSON: "
                    + json.dumps(self._summary())
                    + "\n",
                    encoding="utf-8",
                )
            rows, full, errors = summarize(logs, self.targets, 5)
            self.assertEqual(errors, [])
            self.assertEqual(len(rows), 4)
            paths = write_outputs(output, rows, full)
            for path in paths:
                self.assertTrue(path.is_file())
            markdown = paths[2].read_text(encoding="utf-8")
            self.assertIn("ACC", markdown)
            self.assertIn("GFLOPs", markdown)
            self.assertIn("Peak train GPU", markdown)

    def test_missing_weather_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, errors = summarize(Path(directory), self.targets, 5)
            self.assertEqual(rows, [])
            self.assertEqual(len(errors), 4)


if __name__ == "__main__":
    unittest.main()
