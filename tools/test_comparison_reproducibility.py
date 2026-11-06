"""Static reproducibility checks for the retained VDS2Raw comparison."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ComparisonReproducibilityTests(unittest.TestCase):
    def test_tir_upstream_identity_is_pinned(self):
        provenance = (ROOT / "TIR" / "UPSTREAM.md").read_text(encoding="utf-8")
        adapter = (ROOT / "TIR" / "vds2raw_adapter.py").read_text(
            encoding="utf-8"
        )
        commit = "188ff2fcb7373fd5882319212bf712a676e90773"
        self.assertIn("https://github.com/shuaiyi308/TIR", provenance)
        self.assertIn(commit, provenance)
        self.assertIn(f'OFFICIAL_COMMIT = "{commit}"', adapter)
        self.assertIn("did not contain a `LICENSE`", provenance)

    def test_clip_checkpoint_loading_is_strict(self):
        adapter = (ROOT / "TIR" / "vds2raw_adapter.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("missing = sorted", adapter)
        self.assertIn("unexpected = sorted", adapter)
        self.assertIn("shape_mismatch = sorted", adapter)
        self.assertIn("load_state_dict(loadable, strict=True)", adapter)

    def test_mcft_native_resolution_precedes_token_pool(self):
        model = (ROOT / "MCFT" / "mcft_model.py").read_text(encoding="utf-8")
        train = (ROOT / "vds2raw" / "train_mcft.py").read_text(encoding="utf-8")
        self.assertIn("self.token_grid_pool(self.separable1(x1))", model)
        self.assertIn("self.token_grid_pool(self.separable2(x2))", model)
        self.assertGreaterEqual(train.count("image_size=96"), 4)

    def test_weather_mcft_uses_full_target_supervision_without_checkpoints(self):
        train = (ROOT / "MCFT" / "train_mcft.py").read_text(encoding="utf-8")
        self.assertIn("f'{args.target}_train.txt'", train)
        self.assertIn("ConcatDataset((source, target_train))", train)
        self.assertIn("'target_label_budget': 'all available target_train labels'", train)
        self.assertIn("in-memory best state; no model files retained", train)
        self.assertNotIn("torch.save(", train)

    def test_vds2raw_launchers_are_mcft_tir_only(self):
        for relative in (
            "scripts/run_vds2raw_methods.sh",
            "scripts/run_vds2raw_methods.ps1",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertRegex(source, re.compile(r"mcft3", re.I))
            self.assertRegex(source, re.compile(r"tir", re.I))
            self.assertNotRegex(source, re.compile(r"train_(?:ccsa|dsne)", re.I))

    def test_tir_then_weather_mcft_launcher_is_strictly_sequential(self):
        launcher = (
            ROOT / "scripts" / "run_tir_mcft_weather_sequential.sh"
        ).read_text(encoding="utf-8")
        tir_call = 'bash "$HERE/scripts/run_vds2raw_methods.sh" "$MODE"'
        mcft_call = 'bash "$HERE/scripts/run_server.sh" supervised "$MCFT_EPOCHS"'
        self.assertLess(launcher.index(tir_call), launcher.index(mcft_call))
        self.assertIn('export VDS2RAW_METHOD_ORDER="tir"', launcher)
        self.assertIn('/mnt/data/lixiang/data/Data', launcher)
        self.assertIn('MCFT_SEEDS <<< "${COMBINED_MCFT_SEEDS:-42 43 44 45 46}"', launcher)

    def test_efficiency_registry_has_current_methods(self):
        benchmark = (ROOT / "tools" / "benchmark_efficiency.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("'vds_mcft3': adapter_vds_mcft3", benchmark)
        self.assertIn("'vds_tir': adapter_vds_tir", benchmark)
        self.assertIn("96x96 conv", benchmark)


if __name__ == "__main__":
    unittest.main()
