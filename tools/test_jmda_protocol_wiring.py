"""Lightweight protocol-isolation checks that do not require PyTorch."""

import pathlib
import re
import unittest

from tools.audit_jmda_uda_log import compute_training_code_fingerprint


ROOT = pathlib.Path(__file__).resolve().parents[1]


class JmdaProtocolWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = (ROOT / "scripts" / "run_server.sh").read_text(
            encoding="utf-8"
        )
        cls.preflight = (ROOT / "tools" / "preflight_server.py").read_text(
            encoding="utf-8"
        )
        cls.module = (
            ROOT / "JMDA-Net" / "Ablation" / "module_ablation.py"
        ).read_text(encoding="utf-8")
        cls.fingerprint_tool = (
            ROOT / "tools" / "print_jmda_code_fingerprint.py"
        ).read_text(encoding="utf-8")
        cls.screening = (
            ROOT / "scripts" / "run_jmda_uda_screening.sh"
        ).read_text(encoding="utf-8")
        cls.screening_summary = (
            ROOT / "tools" / "summarize_jmda_screening.py"
        ).read_text(encoding="utf-8")

    def test_strict_and_oracle_launchers_are_separate(self):
        strict_start = self.launcher.index(
            'elif [ "$PROTOCOL" = "jmda_uda" ]'
        )
        oracle_start = self.launcher.index(
            'elif [ "$PROTOCOL" = "jmda_oracle_shared_uda" ]'
        )
        uda_start = self.launcher.index('elif [ "$PROTOCOL" = "uda" ]')

        strict_block = self.launcher[strict_start:oracle_start]
        oracle_block = self.launcher[oracle_start:uda_start]
        self.assertIn("--label_space_mode partial_set", strict_block)
        self.assertNotIn("oracle_shared", strict_block)
        self.assertIn("--label_space_mode oracle_shared", oracle_block)
        self.assertIn("--oracle_shared_labels '$oracle_shared_labels'", oracle_block)
        self.assertIn(
            "--target_train_pair_list '$FAIR_LIST_DIR/${t}_train_pairs.tsv'",
            oracle_block,
        )
        self.assertIn("tools/read_target_class_support.py", oracle_block)
        self.assertIn("--uda_pseudo_class_pairing", oracle_block)
        self.assertIn("--visual_finetuning_mode legacy_supervised", oracle_block)
        self.assertIn("--no_sync_discriminator_lr", oracle_block)
        self.assertIn("--no_adaptive_sinkhorn", oracle_block)
        self.assertIn("--svd_projection_mode legacy_supervised", oracle_block)
        self.assertIn("--basis_transport_mode legacy_supervised", oracle_block)
        self.assertIn("jmda_net_oracle_shared_uda_", oracle_block)

    def test_oracle_results_cannot_be_recorded_as_uda(self):
        self.assertIn('record_protocol="oracle_shared_uda"', self.launcher)
        self.assertIn('label_space_relation="oracle-shared-support"', self.launcher)
        self.assertIn("jmda_oracle_shared_uda", self.preflight)

    def test_module_defaults_to_strict_partial_set(self):
        self.assertIn("choices=['partial_set', 'oracle_shared']", self.module)
        self.assertIn("default='partial_set'", self.module)
        self.assertIn("return 'ORACLE_SHARED_UDA'", self.module)
        self.assertIn("TARGET LABEL SUPPORT ORACLE; NOT STRICT UDA", self.module)

    def test_jmda_runtime_parameters_are_explicitly_forwarded(self):
        strict_start = self.launcher.index(
            'elif [ "$PROTOCOL" = "jmda_uda" ]'
        )
        oracle_start = self.launcher.index(
            'elif [ "$PROTOCOL" = "jmda_oracle_shared_uda" ]'
        )
        uda_start = self.launcher.index('elif [ "$PROTOCOL" = "uda" ]')

        for block in (
            self.launcher[strict_start:oracle_start],
            self.launcher[oracle_start:uda_start],
        ):
            self.assertIn("--batch_size '$JMDA_BATCH_SIZE'", block)
            self.assertIn("--num_workers '$JMDA_NUM_WORKERS'", block)
            self.assertIn("--projection_dim '$JMDA_PROJECTION_DIM'", block)
            self.assertIn("--svd_projection_mode legacy_supervised", block)
            self.assertIn("--basis_transport_mode legacy_supervised", block)
            self.assertIn(
                "--uda_source_warmup_epochs "
                "'$JMDA_UDA_SOURCE_WARMUP_EPOCHS'",
                block,
            )
            self.assertIn(
                "--svd_stat_batch_size '$JMDA_SVD_STAT_BATCH_SIZE'", block
            )
            self.assertIn("--prefetch_factor '$JMDA_PREFETCH_FACTOR'", block)
            self.assertIn("$JMDA_PIN_MEMORY_ARG", block)

        self.assertIn("--num_workers", self.module)
        self.assertIn("--prefetch_factor", self.module)
        self.assertIn("--no_pin_memory", self.module)
        self.assertIn("--uda_pseudo_class_pairing", self.module)
        self.assertIn("source_prototype_pseudo_class", self.module)
        self.assertIn("target_ground_truth_used_for_pairing", self.module)
        self.assertIn("prototype_pseudo_label_assignment", self.module)
        self.assertIn("parse_original_label_csv", self.module)
        self.assertIn("LabelFreePairedTargetDataset", self.module)
        self.assertIn("include_label=(protocol != 'UDA')", self.module)
        self.assertIn("leave-one-out-source-prototypes", self.module)
        self.assertIn("pseudo_projection_ready", self.module)
        self.assertIn("native-pre-svd", self.module)
        self.assertIn(
            "--class_conditional_ot --balance_svd_classes",
            self.launcher,
        )
        self.assertIn(
            '--expected-batch-size "$JMDA_BATCH_SIZE"',
            self.launcher,
        )
        self.assertIn(
            '--expected-svd-stat-batch-size "$JMDA_SVD_STAT_BATCH_SIZE"',
            self.launcher,
        )
        self.assertIn("svd_projection_mode", self.module)
        self.assertIn("basis_transport_mode", self.module)
        self.assertIn(
            'JMDA_PROJECTION_DIM="${JMDA_PROJECTION_DIM:-64}"',
            self.launcher,
        )
        self.assertIn(
            'JMDA_UDA_SOURCE_WARMUP_EPOCHS="'
            '${JMDA_UDA_SOURCE_WARMUP_EPOCHS:-10}"',
            self.launcher,
        )
        self.assertIn(
            'JMDA_SVD_STAT_BATCH_SIZE="${JMDA_SVD_STAT_BATCH_SIZE:-128}"',
            self.launcher,
        )
        self.assertIn(
            "JMDA_UDA_SOURCE_WARMUP_EPOCHS >= EPOCHS",
            self.launcher,
        )
        self.assertIn('EPOCHS < 25', self.launcher)
        self.assertIn(
            'args.uda_source_warmup_epochs',
            self.module,
        )
        self.assertIn(
            "固定进行10轮源域预热",
            self.launcher,
        )
        self.assertIn(
            "otherwise no cross-domain alignment is trained",
            self.module,
        )
        self.assertIn("is_uda_source_warmup", self.module)
        self.assertIn("JMDA_RUN_INVARIANT_TESTS", self.launcher)
        self.assertIn(
            "-m unittest test_tensor_ot_invariants.py", self.launcher
        )
        self.assertIn(
            "tools.test_jmda_protocol_wiring", self.launcher
        )
        self.assertIn("tools.test_audit_jmda_uda_log", self.launcher)
        self.assertIn("tools/audit_jmda_uda_log.py", self.launcher)
        self.assertIn('--expected-protocol "$audit_protocol"', self.launcher)
        self.assertIn(
            '--expected-projection-dim "$JMDA_PROJECTION_DIM"',
            self.launcher,
        )
        self.assertIn(
            '--expected-warmup-epochs "$JMDA_UDA_SOURCE_WARMUP_EPOCHS"',
            self.launcher,
        )
        self.assertIn(
            '--expected-code-fingerprint "$JMDA_EXPECTED_CODE_FINGERPRINT"',
            self.launcher,
        )
        self.assertIn(
            '--expected-num-iterations "$JMDA_NUM_ITERATIONS"',
            self.launcher,
        )
        self.assertIn('--expected-base-seed "$SEED"', self.launcher)
        self.assertIn("training_code_fingerprint", self.module)
        self.assertIn("'code_fingerprint':", self.module)
        for compatibility_shim in (
            "DataLoad.py",
            "Tensor.py",
            "Generator.py",
            "epoch_svd.py",
        ):
            self.assertIn(
                f'module_path.parent / "{compatibility_shim}"',
                self.module,
            )
        self.assertIn("Cannot fingerprint missing JMDA implementation file", self.module)

    def test_launcher_requires_the_reviewed_core_code_fingerprint(self):
        match = re.search(
            r'JMDA_EXPECTED_CODE_FINGERPRINT="([0-9a-f]{16})"',
            self.launcher,
        )
        self.assertIsNotNone(match)
        actual = compute_training_code_fingerprint(ROOT / "JMDA-Net")
        self.assertEqual(match.group(1), actual)
        self.assertIn('.replace(b"\\r\\n", b"\\n")', self.module)
        self.assertIn("tools/print_jmda_code_fingerprint.py", self.launcher)
        self.assertIn(
            'JMDA_ACTUAL_CODE_FINGERPRINT" != '
            '"$JMDA_EXPECTED_CODE_FINGERPRINT',
            self.launcher,
        )
        self.assertIn("compute_training_code_fingerprint", self.fingerprint_tool)

    def test_optional_gpu_memory_gate_is_wired_to_preflight(self):
        self.assertIn("JMDA_MIN_FREE_GPU_MEMORY_MIB", self.launcher)
        self.assertIn("--min-free-gpu-memory-mib", self.launcher)
        self.assertIn("--min-free-gpu-memory-mib", self.preflight)
        self.assertIn("torch.cuda.mem_get_info(0)", self.preflight)

    def test_screening_matrix_is_explicit_and_never_uses_target_ce(self):
        for tag in (
            "source_only_fixed_projection",
            "pseudo_tensor_only",
            "oracle_tensor_only",
            "pseudo_tensor_ot",
            "oracle_tensor_ot",
        ):
            self.assertIn(tag, self.screening)
        self.assertIn("--target_label_ratio 0", self.screening)
        self.assertIn("--target_cls_weight 0", self.screening)
        self.assertIn("--diagnostic_oracle_pairing", self.screening)
        self.assertIn("--diagnostic_source_only", self.screening)
        self.assertIn("summarize_jmda_screening.py", self.screening)
        self.assertIn(
            'EXPECTED_CODE_FINGERPRINT="4dc4ef0b01c7ea70"', self.screening
        )
        self.assertIn("print_jmda_code_fingerprint.py", self.screening)
        self.assertIn("target_ce_max", self.screening_summary)
        self.assertIn("target_observations_max", self.screening_summary)
        self.assertIn("ORACLE_PAIRED_UDA_DIAGNOSTIC", self.module)
        self.assertIn("TARGET SAMPLE LABEL ORACLE ACTIVE FOR PAIRING ONLY", self.module)


if __name__ == "__main__":
    unittest.main()
