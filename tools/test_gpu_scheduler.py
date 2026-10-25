import unittest

from tools.run_on_available_gpu import GpuState, eligible_gpus, parse_nvidia_smi


class GpuSchedulerTests(unittest.TestCase):
    def test_parse_and_choose_highest_free_eligible_gpu(self):
        states = parse_nvidia_smi(
            "0, GPU-a, 12528, 90\n"
            "1, GPU-b, 31000, 5\n"
            "2, GPU-c, 30500, 3\n"
        )
        eligible = eligible_gpus(states, 30000, 10)
        self.assertEqual([state.index for state in eligible], ["1", "2"])

    def test_memory_and_utilization_are_both_required(self):
        states = [
            GpuState("0", "GPU-a", 32000, 99),
            GpuState("1", "GPU-b", 10000, 0),
        ]
        self.assertEqual(eligible_gpus(states, 30000, 10), [])

    def test_candidate_filter_accepts_index_or_uuid(self):
        states = [
            GpuState("0", "GPU-a", 32000, 0),
            GpuState("1", "GPU-b", 31000, 0),
        ]
        self.assertEqual(
            [state.index for state in eligible_gpus(states, 30000, 10, {"1"})],
            ["1"],
        )
        self.assertEqual(
            [state.index for state in eligible_gpus(states, 30000, 10, {"GPU-a"})],
            ["0"],
        )


if __name__ == "__main__":
    unittest.main()
