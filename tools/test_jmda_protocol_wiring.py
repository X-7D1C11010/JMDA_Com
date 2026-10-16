"""Lightweight protocol-isolation checks that do not require PyTorch."""

import pathlib
import unittest


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


if __name__ == "__main__":
    unittest.main()
