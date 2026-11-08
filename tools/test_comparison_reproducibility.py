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
        self.assertIn("no cross-seed checkpoint inheritance", train)
        self.assertNotIn("torch.save(", train)
        self.assertNotIn("torch.load(", train)

    def test_weather_mcft_regularization_and_best_metric_reporting(self):
        train = (ROOT / "MCFT" / "train_mcft.py").read_text(encoding="utf-8")
        self.assertIn("default=3e-4", train)
        self.assertIn("default=1e-4", train)
        self.assertIn("--label-smoothing", train)
        self.assertIn("default=0.2", train)
        self.assertIn("parser.set_defaults(horizontal_flip=True)", train)
        self.assertIn("CrossEntropyLoss(label_smoothing=args.label_smoothing)", train)
        self.assertIn("format_classification_metrics(final_metrics)", train)

        runner = (ROOT / "scripts" / "run_server.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SUPERVISED_MCFT_LR="${SUPERVISED_MCFT_LR:-0.0003}"', runner)
        self.assertIn(
            'SUPERVISED_MCFT_EARLY_PATIENCE="${SUPERVISED_MCFT_EARLY_PATIENCE:-15}"',
            runner,
        )
        self.assertIn('--horizontal-flip', runner)

    def test_weather_mcft_supervised_protocol_passes_common_preflight(self):
        preflight = (ROOT / "tools" / "preflight_server.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("'uda', 'vlm', 'supervised'", preflight)
        self.assertIn(
            "[SUPERVISED] full sunny_train and every selected target_train",
            preflight,
        )
        runner = (ROOT / "scripts" / "run_server.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--protocol "$PROTOCOL"', runner)
        self.assertIn('"$HERE/MCFT/train_mcft.py"', runner)

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
        self.assertIn('MCFT_EPOCHS="${WEATHER_MCFT_EPOCHS:-100}"', launcher)
        self.assertIn('export SUPERVISED_MCFT_EPOCHS="$MCFT_EPOCHS"', launcher)
        self.assertIn('fresh model; independent run', launcher)

    def test_efficiency_registry_has_current_methods(self):
        benchmark = (ROOT / "tools" / "benchmark_efficiency.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("'vds_mcft3': adapter_vds_mcft3", benchmark)
        self.assertIn("'vds_tir': adapter_vds_tir", benchmark)
        self.assertIn("96x96 conv", benchmark)


if __name__ == "__main__":
    unittest.main()
