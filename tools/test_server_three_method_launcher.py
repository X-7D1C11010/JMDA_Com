import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ThreeMethodServerLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(ROOT, "scripts", "run_server_dann_coral_cdan.sh")
        with open(path, encoding="utf-8") as handle:
            cls.script = handle.read()

    def test_gpu_gate_is_limited_to_zero_and_one(self):
        self.assertIn('GPU_CANDIDATES="${GPU_CANDIDATES:-0,1}"', self.script)
        self.assertIn('--gpus "$GPU_CANDIDATES"', self.script)
        self.assertIn('--min-free-mib "$min_free"', self.script)
        self.assertIn('--max-util-percent "$MAX_GPU_UTIL"', self.script)
        self.assertIn('--stable-checks "$STABLE_CHECKS"', self.script)

    def test_only_requested_methods_are_launched(self):
        self.assertIn('DANN/train_dann.py', self.script)
        self.assertIn('DeepCORAL/train_deepcoral.py', self.script)
        self.assertIn('CDAN/train_cdan.py', self.script)
        for unwanted in ('MDD/train_mdd.py', 'MCC/train_mcc.py', 'PDA/train_pda.py'):
            self.assertNotIn(unwanted, self.script)

    def test_all_update_budgets_are_configurable(self):
        self.assertIn('--dann-iterations', self.script)
        self.assertIn('--deepcoral-iterations', self.script)
        self.assertIn('--cdan-iterations', self.script)
        self.assertIn('--max-iterations "$DANN_ITERATIONS"', self.script)
        self.assertIn('--max-iterations "$DEEPCORAL_ITERATIONS"', self.script)
        self.assertIn('--num-iterations "$CDAN_ITERATIONS"', self.script)

    def test_protocol_and_background_wiring(self):
        self.assertIn('--modality paired', self.script)
        self.assertIn('--multimodal --enhanced-augmentation --no-save-model', self.script)
        self.assertIn('nohup bash', self.script)
        self.assertIn('collect_run_result.py', self.script)


if __name__ == "__main__":
    unittest.main()
