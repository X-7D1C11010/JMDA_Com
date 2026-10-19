import json
import unittest

from tools.audit_jmda_uda_log import audit_text
from tools.collect_run_result import apply_protocol_metric_policy, parse_metrics


def _iteration(index, total, final_accuracy=0.30, min_cosine=0.80):
    return "\n".join(
        [
            f"迭代 {index}/{total}",
            f"随机种子: {41 + index}",
            "TARGET LABEL SUPPORT ORACLE; NOT STRICT UDA. classifier classes=4.",
            "Epoch [1/4] | UDA source-only warm-up: cross-domain SVD/Tensor/OT updates are disabled.",
            "Epoch [1/4] | Loss: 1.0 (Cls: 1.0 [src/tgt/mid=1.0/0.0/0.0; labeled-target-observations=0], TAL: 0.0) | Source Train Acc: 0.6000 | Val Acc: 0.2000 | Val P/R/F1 (Macro): 0.1000/0.2000/0.1300",
            "Epoch [2/4] | UDA source-only warm-up: cross-domain SVD/Tensor/OT updates are disabled.",
            "Epoch [2/4] | Loss: 0.8 (Cls: 0.8 [src/tgt/mid=0.8/0.0/0.0; labeled-target-observations=0], TAL: 0.0) | Source Train Acc: 0.8000 | Val Acc: 0.2500 | Val P/R/F1 (Macro): 0.2000/0.2500/0.2200",
            (
                "Epoch [3/4] | SVD update #1: pairs=(10, 8), sweeps=2, "
                "effective_ranks=(16, 16), stabilized_null_dims=(0, 0), "
                "rank_thresholds=(1e-5, 1e-5), basis_overlap_src/tgt=0.9/0.9, "
                f"basis_min_cos_src/tgt={min_cosine}/0.7, "
                "basis_nonorth_src/tgt=1e-6/1e-6, transport_nonorth=1e-6, "
                "shared_transport_ranks=(16, 16), "
                "shared_transport_rank_thresholds=(1e-6, 1e-6), "
                "classifier_basis=source"
            ),
            "Epoch [3/4] | Loss: 0.7 (Cls: 0.7 [src/tgt/mid=0.7/0.0/0.0; labeled-target-observations=0], TAL: -0.1) | Source Train Acc: 0.7800 | Val Acc: 0.2600 | Val P/R/F1 (Macro): 0.2400/0.2600/0.2500",
            "Epoch [4/4] | SVD update #2: pairs=(10, 8), sweeps=2, effective_ranks=(2, 2)",
            f"Epoch [4/4] | Loss: 0.6 (Cls: 0.6 [src/tgt/mid=0.6/0.0/0.0; labeled-target-observations=0], TAL: -0.1) | Source Train Acc: 0.8200 | Val Acc: {final_accuracy:.4f} | Val P/R/F1 (Macro): 0.3100/0.3200/0.3150",
            f"Report strategy: last, window=10, reported_acc={final_accuracy:.4f}; target-label best checkpoint is not selected or reported in UDA.",
            "Saved final checkpoint: checkpoint.pth",
        ]
    )


def _log(iterations=2, final_accuracy=0.30, min_cosine=0.80):
    summary = {
        "protocol": "ORACLE_SHARED_UDA",
        "adaptation_protocol": "UDA",
        "label_space_mode": "oracle_shared",
        "report_strategy": "last",
        "projection_dim": 16,
        "uda_source_warmup_epochs": 2,
        "code_fingerprint": "0123456789abcdef",
        "num_iterations": iterations,
        "seeds": list(range(42, 42 + iterations)),
        "oracle_target_label_support_used": True,
        "oracle_shared_labels": [1, 2, 3, 4],
        "label_map_original_to_index": {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
        },
        "metrics": {
            "accuracy": {
                "values": [final_accuracy] * iterations,
                "mean": final_accuracy,
                "std": 0.0,
            },
            "precision_macro": {
                "values": [0.31] * iterations,
                "mean": 0.31,
                "std": 0.0,
            },
            "recall_macro": {
                "values": [0.32] * iterations,
                "mean": 0.32,
                "std": 0.0,
            },
            "f1_macro": {
                "values": [0.315] * iterations,
                "mean": 0.315,
                "std": 0.0,
            },
        },
    }
    parts = [
        "训练代码指纹: 0123456789abcdef",
        "Tensor projection update: projection_dim=16",
    ]
    parts.extend(
        _iteration(i, iterations, final_accuracy, min_cosine)
        for i in range(1, iterations + 1)
    )
    parts.append("JMDA_RUN_SUMMARY_JSON: " + json.dumps(summary))
    return "\n".join(parts)


class JmdaLogAuditTests(unittest.TestCase):
    def audit(
        self,
        text,
        *,
        expected_num_iterations=None,
        expected_code_fingerprint="0123456789abcdef",
    ):
        if expected_num_iterations is None:
            expected_num_iterations = len(
                [line for line in text.splitlines() if line.startswith("迭代 ")]
            )
        return audit_text(
            text,
            expected_protocol="ORACLE_SHARED_UDA",
            expected_projection_dim=16,
            expected_warmup_epochs=2,
            expected_total_epochs=4,
            expected_code_fingerprint=expected_code_fingerprint,
            expected_num_iterations=expected_num_iterations,
            expected_base_seed=42,
        )

    def test_correct_multiseed_log_passes_protocol_audit(self):
        result = self.audit(_log(iterations=2))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["protocol_errors"], [])
        self.assertEqual(len(result["iterations"]), 2)
        self.assertEqual(
            result["iterations"][0]["first_effective_ranks"], [16, 16]
        )
        self.assertEqual(
            result["iterations"][0]["first_shared_transport_ranks"],
            [16, 16],
        )

    def test_label_use_and_wrong_projection_fail(self):
        corrupted = _log(iterations=1).replace(
            "projection_dim=16", "projection_dim=64"
        ).replace(
            "labeled-target-observations=0",
            "labeled-target-observations=1",
            1,
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("projection_dim mismatch" in error for error in result["protocol_errors"])
        )
        self.assertTrue(
            any("labeled target" in error for error in result["protocol_errors"])
        )

    def test_low_accuracy_and_low_transition_cosine_are_warnings_only(self):
        result = self.audit(
            _log(iterations=1, final_accuracy=0.10, min_cosine=0.05)
        )
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["protocol_errors"], [])
        self.assertGreaterEqual(len(result["performance_warnings"]), 2)

    def test_rank_deficient_first_transition_is_a_performance_warning(self):
        text = _log(iterations=1).replace(
            "effective_ranks=(16, 16), stabilized_null_dims=(0, 0)",
            "effective_ranks=(12, 16), stabilized_null_dims=(4, 0)",
        ).replace(
            "shared_transport_ranks=(16, 16)",
            "shared_transport_ranks=(15, 16)",
        )
        result = self.audit(text)
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["protocol_errors"], [])
        self.assertTrue(
            any("numerical-null" in warning for warning in result["performance_warnings"])
        )
        self.assertTrue(
            any("rank-deficient" in warning for warning in result["performance_warnings"])
        )

    def test_appended_audit_output_does_not_change_collected_metrics(self):
        training_log = _log(iterations=1, final_accuracy=0.30)
        audited_log = training_log + "\n" + "\n".join(
            [
                "JMDA log audit status: PASS",
                'JMDA_LOG_AUDIT_JSON: {"status": "PASS", '
                '"iterations": [{"final_target_accuracy": 0.01}]}',
            ]
        )
        self.assertEqual(parse_metrics(training_log), parse_metrics(audited_log))

    def test_uda_result_record_cannot_publish_target_best_accuracy(self):
        parsed = parse_metrics(
            "Val Acc: 0.2000\nVal Acc: 0.3000\n"
            "Diagnostic peak target val accuracy: 0.9000\n"
            "reported_acc=0.3000\n"
        )
        self.assertEqual(parsed["best_acc"], 0.9)
        governed = apply_protocol_metric_policy(parsed, "oracle_shared_uda")
        self.assertIsNone(governed["best_acc"])

    def test_reported_accuracy_must_equal_final_epoch(self):
        corrupted = _log(iterations=1).replace(
            "reported_acc=0.3000", "reported_acc=0.9000"
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("does not equal final-epoch" in error for error in result["protocol_errors"])
        )

    def test_summary_accuracy_cannot_override_final_epoch(self):
        corrupted = _log(iterations=2).replace(
            '"values": [0.3, 0.3]', '"values": [0.99, 0.99]'
        ).replace('"mean": 0.3', '"mean": 0.99')
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "does not match that iteration's final/reported accuracy" in error
                for error in result["protocol_errors"]
            )
        )

    def test_iteration_headers_and_seeds_must_be_unique_and_consistent(self):
        corrupted = _log(iterations=2).replace("迭代 2/2", "迭代 1/2").replace(
            "随机种子: 43", "随机种子: 42"
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("header indices" in error for error in result["protocol_errors"])
        )
        self.assertTrue(
            any("not unique" in error for error in result["protocol_errors"])
        )

    def test_external_iteration_count_and_code_fingerprint_are_enforced(self):
        result = self.audit(_log(iterations=1), expected_num_iterations=2)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("iteration count mismatch" in error for error in result["protocol_errors"])
        )

        forged = _log(iterations=1).replace(
            "0123456789abcdef", "ffffffffffffffff"
        )
        result = self.audit(forged)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("fingerprint mismatch" in error for error in result["protocol_errors"])
        )

    def test_macro_summary_cannot_override_final_epoch(self):
        corrupted = _log(iterations=1).replace(
            '"values": [0.315]', '"values": [0.99]'
        ).replace('"mean": 0.315', '"mean": 0.99')
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("summary f1_macro value" in error for error in result["protocol_errors"])
        )


if __name__ == "__main__":
    unittest.main()
