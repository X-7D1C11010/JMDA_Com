import json
import unittest

from tools.audit_jmda_uda_log import (
    _expected_training_batches_without_singleton,
    audit_text,
)
from tools.collect_run_result import apply_protocol_metric_policy, parse_metrics


def _iteration(index, total, final_accuracy=0.30, min_cosine=0.80):
    def pseudo(epoch, change_rate):
        return (
            f"Epoch [{epoch}/4] | UDA pseudo-class pairing: "
            "target_label_source=source-prototype (no target ground truth), "
            f"pseudo_feature_space={'native-pre-svd' if epoch == 1 else 'aligned-svd'}, "
            "class_counts=[4, 2, 1, 1], coverage_repairs=1, "
            "assigned_confidence=0.7000, repaired-vs-nearest agreement=0.8750, "
            "modality_source_bal_acc=[0.8, 0.6], modality_weights=[0.7, 0.3], "
            "modality_reliability_estimator=leave-one-out-source-prototypes, "
            f"change_rate={change_rate}, target_samples=8, "
            "balanced_epoch_samples=16, training_batches=1"
        )

    return "\n".join(
        [
            f"迭代 {index}/{total}",
            f"随机种子: {41 + index}",
            "TARGET LABEL SUPPORT ORACLE; NOT STRICT UDA. classifier classes=4.",
            "Target training item label exposure: include_label=False; "
            "UDA must be False.",
            pseudo(1, "n/a"),
            (
                "Epoch [1/4] | SVD update #1: pairs=80, sweeps=2, "
                "effective_ranks=(64, 64), stabilized_null_dims=(0, 0), "
                "rank_thresholds=(1e-5, 1e-5), "
                "classifier_basis=shared, class_paired=True, class_balanced=True, "
                "target_labels_source=pseudo"
            ),
            "Epoch [1/4] | Loss: 1.0 (Cls: 1.0 [src/tgt/mid=1.0/0.0/0.0; labeled-target-observations=0], TAL: -0.1) | OT scale: 0.00 | Source Train Acc: 0.6000 | Val Acc: 0.2000 | Val P/R/F1 (Macro): 0.1000/0.2000/0.1300 | LR feature/other/discriminator: 5e-6/2e-4/2e-4",
            pseudo(2, "0.2500"),
            (
                "Epoch [2/4] | SVD update #2: pairs=80, sweeps=2, "
                "effective_ranks=(64, 64), stabilized_null_dims=(0, 0), "
                "rank_thresholds=(1e-5, 1e-5), basis_overlap_src/tgt=0.9/0.9, "
                f"basis_min_cos_src/tgt={min_cosine}/0.7, "
                "basis_nonorth_src/tgt=1e-6/1e-6, transport_nonorth=1e-6, "
                "basis_domain_gap=0.1, "
                "shared_transport_ranks=(64, 64), "
                "shared_transport_rank_thresholds=(1e-6, 1e-6), "
                "classifier_basis=shared, basis_transport=legacy_supervised, "
                "class_paired=True, class_balanced=True, "
                "target_labels_source=pseudo"
            ),
            "Epoch [2/4] | Loss: 0.8 (Cls: 0.8 [src/tgt/mid=0.8/0.0/0.0; labeled-target-observations=0], TAL: -0.1) | OT scale: 0.00 | Source Train Acc: 0.8000 | Val Acc: 0.2500 | Val P/R/F1 (Macro): 0.2000/0.2500/0.2200 | LR feature/other/discriminator: 5e-6/2e-4/2e-4",
            pseudo(3, "0.1250"),
            "Epoch [3/4] | SVD update #3: pairs=80, sweeps=2, effective_ranks=(64, 64), stabilized_null_dims=(0, 0), class_paired=True, class_balanced=True, target_labels_source=pseudo",
            "Epoch [3/4] | Loss: 0.7 (Cls: 0.7 [src/tgt/mid=0.7/0.0/0.1; labeled-target-observations=0], TAL: -0.1) | OT scale: 0.00 | Source Train Acc: 0.7800 | Val Acc: 0.2600 | Val P/R/F1 (Macro): 0.2400/0.2600/0.2500 | LR feature/other/discriminator: 5e-6/2e-4/2e-4",
            pseudo(4, "0.0000"),
            "Epoch [4/4] | SVD update #4: pairs=80, sweeps=2, effective_ranks=(64, 64), stabilized_null_dims=(0, 0), class_paired=True, class_balanced=True, target_labels_source=pseudo",
            f"Epoch [4/4] | Loss: 0.6 (Cls: 0.6 [src/tgt/mid=0.6/0.0/0.0; labeled-target-observations=0], TAL: -0.1) | OT scale: 0.00 | Source Train Acc: 0.8200 | Val Acc: {final_accuracy:.4f} | Val P/R/F1 (Macro): 0.3100/0.3200/0.3150 | LR feature/other/discriminator: 5e-6/2e-4/2e-4",
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
        "projection_dim": 64,
        "uda_source_warmup_epochs": 0,
        "uda_pairing_mode": "source_prototype_pseudo_class",
        "target_ground_truth_used_for_pairing": False,
        "target_training_item_labels_exposed": False,
        "target_train_samples": 8,
        "visual_finetuning_mode": "legacy_supervised",
        "adaptive_sinkhorn": False,
        "transport_mode": "sinkhorn",
        "ot_sinkhorn_iterations": 50,
        "ot_warmup_epochs": 10,
        "ot_ramp_epochs": 5,
        "class_conditional_ot": True,
        "balance_svd_classes": True,
        "sync_discriminator_lr": False,
        "tensor_loss_weight": 0.12,
        "adv_loss_weight": 0.08,
        "lr_feature": 5e-6,
        "lr_other": 2e-4,
        "weight_decay": 5e-4,
        "modality_reliability_estimator": "leave-one-out-source-prototypes",
        "svd_projection_mode": "legacy_supervised",
        "basis_transport_mode": "legacy_supervised",
        "data_loading": {
            "batch_size": 32,
            "svd_stat_batch_size": 128,
        },
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
        "Tensor projection update: projection_dim=64",
    ]
    parts.extend(
        _iteration(i, iterations, final_accuracy, min_cosine)
        for i in range(1, iterations + 1)
    )
    parts.append("JMDA_RUN_SUMMARY_JSON: " + json.dumps(summary))
    return "\n".join(parts)


class JmdaLogAuditTests(unittest.TestCase):
    def test_expected_batch_count_merges_singleton_tail(self):
        self.assertEqual(
            _expected_training_batches_without_singleton(65, 32),
            2,
        )
        self.assertEqual(
            _expected_training_batches_without_singleton(66, 32),
            3,
        )

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
            expected_projection_dim=64,
            expected_warmup_epochs=0,
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
            result["iterations"][0]["first_effective_ranks"], [64, 64]
        )
        self.assertEqual(
            result["iterations"][0]["first_shared_transport_ranks"],
            [64, 64],
        )

    def test_label_use_and_wrong_projection_fail(self):
        corrupted = _log(iterations=1).replace(
            "projection_dim=64", "projection_dim=16"
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

    def test_low_accuracy_is_a_warning_only(self):
        result = self.audit(
            _log(iterations=1, final_accuracy=0.10, min_cosine=0.05)
        )
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["protocol_errors"], [])
        self.assertGreaterEqual(len(result["performance_warnings"]), 1)

    def test_rank_deficient_first_transition_is_a_performance_warning(self):
        text = _log(iterations=1).replace(
            "effective_ranks=(64, 64), stabilized_null_dims=(0, 0)",
            "effective_ranks=(60, 64), stabilized_null_dims=(0, 0)",
        ).replace(
            "shared_transport_ranks=(64, 64)",
            "shared_transport_ranks=(63, 64)",
        )
        result = self.audit(text)
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(result["protocol_errors"], [])
        self.assertTrue(
            any("rank-deficient" in warning for warning in result["performance_warnings"])
        )

    def test_oracle_summary_must_prove_supervised_parity_modes(self):
        corrupted = _log(iterations=1).replace(
            '"basis_transport_mode": "legacy_supervised"',
            '"basis_transport_mode": "orthogonal_stable"',
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("basis_transport_mode" in error for error in result["protocol_errors"])
        )

    def test_oracle_summary_cannot_claim_all_pairs_training(self):
        corrupted = _log(iterations=1).replace(
            '"uda_pairing_mode": "source_prototype_pseudo_class"',
            '"uda_pairing_mode": "all_pairs"',
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "must use source-prototype pseudo-class pairing" in error
                for error in result["protocol_errors"]
            )
        )

    def test_pseudo_class_counts_must_cover_classifier_and_target_samples(self):
        corrupted = _log(iterations=1).replace(
            "class_counts=[4, 2, 1, 1]",
            "class_counts=[8, 0, 0, 0]",
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "invalid pseudo class_counts" in error
                for error in result["protocol_errors"]
            )
        )

    def test_in_sample_modality_reliability_marker_is_rejected(self):
        corrupted = _log(iterations=1).replace(
            "leave-one-out-source-prototypes",
            "in-sample-source-prototypes",
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "leave-one-out" in error
                or "pseudo-pairing diagnostics" in error
                for error in result["protocol_errors"]
            )
        )

    def test_first_pseudo_labels_cannot_use_random_pre_svd_projection(self):
        corrupted = _log(iterations=1).replace(
            "pseudo_feature_space=native-pre-svd",
            "pseudo_feature_space=aligned-svd",
            1,
        )
        result = self.audit(corrupted)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "pseudo feature space" in error
                for error in result["protocol_errors"]
            )
        )

    def test_ot_schedule_and_fixed_discriminator_lr_are_enforced(self):
        wrong_ot = _log(iterations=1).replace(
            "OT scale: 0.00",
            "OT scale: 0.50",
            1,
        )
        result = self.audit(wrong_ot)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("OT scale expected" in error for error in result["protocol_errors"])
        )

        wrong_discriminator_lr = _log(iterations=1).replace(
            "5e-6/2e-4/2e-4",
            "5e-6/2e-4/1e-4",
            1,
        )
        result = self.audit(wrong_discriminator_lr)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "discriminator LR is not fixed" in error
                for error in result["protocol_errors"]
            )
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
