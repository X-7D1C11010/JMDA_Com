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
        self.assertIn("--no_uda_source_reweighting", oracle_block)
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
        self.assertIn("16 if adaptation_protocol(args) == 'UDA' else 64", self.module)
        self.assertIn(
            'JMDA_PROJECTION_DIM="${JMDA_PROJECTION_DIM:-16}"',
            self.launcher,
        )
        self.assertIn(
            'JMDA_UDA_SOURCE_WARMUP_EPOCHS="'
            '${JMDA_UDA_SOURCE_WARMUP_EPOCHS:-10}"',
            self.launcher,
        )
        self.assertIn(
            "JMDA_UDA_SOURCE_WARMUP_EPOCHS >= EPOCHS",
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


if __name__ == "__main__":
    unittest.main()
