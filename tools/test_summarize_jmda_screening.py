import json
from pathlib import Path
import tempfile
import unittest

from tools.summarize_jmda_screening import parse_log


class ScreeningSummaryTests(unittest.TestCase):
    def test_parser_checks_zero_target_classification_loss(self):
        summary = {
            "protocol": "ORACLE_PAIRED_UDA_DIAGNOSTIC",
            "uda_pairing_mode": "target_ground_truth_oracle",
            "target_ground_truth_used_for_pairing": True,
            "target_training_item_labels_exposed": False,
            "num_iterations": 1,
            "metrics": {
                "accuracy": {"mean": 0.5},
                "precision_macro": {"mean": 0.4},
                "recall_macro": {"mean": 0.3},
                "f1_macro": {"mean": 0.35},
            },
        }
        epoch_template = (
            "Epoch [{epoch}/30] | Loss: 1.0 "
            "(Cls: 1.0 [src/tgt/mid=1.0/0.0000/0.0000; "
            "labeled-target-observations=0], TAL: 0.0, Adv: 0.0) | "
            "Source Train Acc: 0.9000 | Val Acc: {accuracy:.4f} | "
            "Val P/R/F1 (Macro): 0.4000/0.3000/0.3500\n"
        )
        text = (
            "JMDA_SCREEN_EXPERIMENT=oracle_tensor_only\n"
            "JMDA_SCREEN_TARGET=backlight\n"
            + epoch_template.format(epoch=10, accuracy=0.6)
            + epoch_template.format(epoch=11, accuracy=0.55)
            + epoch_template.format(epoch=30, accuracy=0.5)
            + "JMDA_RUN_SUMMARY_JSON: "
            + json.dumps(summary)
            + "\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.log"
            path.write_text(text, encoding="utf-8")
            parsed = parse_log(path)

        self.assertEqual(parsed["experiment"], "oracle_tensor_only")
        self.assertAlmostEqual(parsed["warmup_acc"], 0.6)
        self.assertAlmostEqual(parsed["transition_acc"], 0.55)
        self.assertAlmostEqual(parsed["final_acc"], 0.5)
        self.assertEqual(parsed["target_ce_max"], 0.0)
        self.assertEqual(parsed["target_observations_max"], 0)
        self.assertTrue(parsed["protocol_safe"])


if __name__ == "__main__":
    unittest.main()
