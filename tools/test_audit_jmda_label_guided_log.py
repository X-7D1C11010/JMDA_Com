import json
import unittest

from tools.audit_jmda_label_guided_log import audit_text


class LabelGuidedLogAuditTests(unittest.TestCase):
    def _summary(self):
        summary = {
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
            "report_strategy": "best",
            "metrics": {
                "accuracy": {"mean": 0.75},
                "precision_macro": {"mean": 0.78},
                "recall_macro": {"mean": 0.79},
                "f1_macro": {"mean": 0.74},
            },
        }
        run = {
            "training_total_params": 100,
            "training_trainable_params": 50,
            "inference_total_params": 80,
            "inference_gflops": 1.2,
            "total_training_time_s": 10.0,
            "train_epoch_mean_s": 1.0,
            "inference_latency_mean_ms": 2.0,
            "inference_latency_p50_ms": 1.9,
            "inference_latency_p95_ms": 2.2,
            "inference_throughput_samples_s": 500.0,
            "peak_train_allocated_gb": 2.0,
            "peak_train_reserved_gb": 3.0,
            "peak_inference_allocated_gb": 1.0,
            "peak_inference_reserved_gb": 1.5,
            "inference_data_loading_included": False,
            "flops_convention": "1 MAC = 2 FLOPs",
        }
        summary["efficiency"] = {
            "enabled": True,
            "per_iteration": [run],
            "aggregate": {
                key: {"values": [value], "mean": value, "std": 0.0}
                for key, value in run.items()
                if isinstance(value, (int, float)) and value > 0
            },
            "summary_json": "/tmp/summary.json",
            "runs_csv": "/tmp/runs.csv",
        }
        return summary

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

    def test_valid_efficiency_evidence_passes(self):
        _, errors = audit_text(self._log(), 1, require_efficiency=True)
        self.assertEqual(errors, [])

    def test_expected_best_report_strategy_passes(self):
        _, errors = audit_text(
            self._log(),
            1,
            expected_report_strategy="best",
        )
        self.assertEqual(errors, [])

    def test_unexpected_report_strategy_is_rejected(self):
        summary = self._summary()
        summary["report_strategy"] = "last"
        text = self._log().replace(
            json.dumps(self._summary()),
            json.dumps(summary),
        )
        _, errors = audit_text(
            text,
            1,
            expected_report_strategy="best",
        )
        self.assertTrue(any("report_strategy" in item for item in errors))

    def test_missing_efficiency_evidence_is_rejected(self):
        summary = self._summary()
        summary.pop("efficiency")
        text = self._log().replace(
            json.dumps(self._summary()),
            json.dumps(summary),
        )
        _, errors = audit_text(text, 1, require_efficiency=True)
        self.assertTrue(any("efficiency summary" in item for item in errors))

    def test_target_ce_is_rejected(self):
        _, errors = audit_text(self._log(target_loss="0.2"), 1)
        self.assertTrue(any("target classification CE" in item for item in errors))

    def test_unbalanced_svd_is_rejected(self):
        _, errors = audit_text(self._log(balanced="False"), 1)
        self.assertTrue(any("not class-balanced" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
