import json
import unittest

from tools.audit_jmda_label_guided_log import audit_text


class LabelGuidedLogAuditTests(unittest.TestCase):
    def _summary(self):
        return {
            "protocol": "LABEL_GUIDED_DA_NO_TARGET_CE",
            "adaptation_protocol": "LABEL_GUIDED_DA",
            "classification_protocol": "UDA",
            "label_space_mode": "oracle_shared",
            "uda_pairing_mode": "target_ground_truth_label_guided",
            "target_ground_truth_used_for_pairing": True,
            "target_training_item_labels_exposed": False,
            "label_guided_alignment": True,
            "class_conditional_ot": True,
            "balance_svd_classes": True,
            "transport_mode": "sinkhorn",
            "num_iterations": 1,
            "metrics": {
                "accuracy": {"mean": 0.75},
                "precision_macro": {"mean": 0.78},
                "recall_macro": {"mean": 0.79},
                "f1_macro": {"mean": 0.74},
            },
        }

    def _log(self, target_loss="0.0000", balanced="True"):
        return (
            "Epoch [11/30] | SVD update #1: pairs=113, sweeps=3, "
            "converged=True, rel_change=1e-4, effective_ranks=(40, 30), "
            f"class_paired=True, class_balanced={balanced}, "
            "target_labels_source=target-ground-truth-label-guided, "
            "basis_overlap_src/tgt=0.8/0.8\n"
            "Epoch [21/30] | Loss: 1.0 (Cls: 0.9 "
            f"[src/tgt/mid=0.8/{target_loss}/0.1; "
            "labeled-target-observations=0], TAL: 0.1, Adv: 0.1, "
            "D: 0.1, OT-Reg: 0.0) | OT scale: 0.20 | "
            "OT inner objective: 0.1 | Marginal residual row/col/max: "
            "1e-6/1e-6/1e-6\n"
            "JMDA_RUN_SUMMARY_JSON: " + json.dumps(self._summary()) + "\n"
        )

    def test_valid_log_passes(self):
        summary, errors = audit_text(self._log(), 1)
        self.assertEqual(errors, [])
        self.assertEqual(summary["protocol"], "LABEL_GUIDED_DA_NO_TARGET_CE")

    def test_target_ce_is_rejected(self):
        _, errors = audit_text(self._log(target_loss="0.2"), 1)
        self.assertTrue(any("target classification CE" in item for item in errors))

    def test_unbalanced_svd_is_rejected(self):
        _, errors = audit_text(self._log(balanced="False"), 1)
        self.assertTrue(any("not class-balanced" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
