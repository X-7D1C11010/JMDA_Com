"""Static protocol checks that run without importing heavyweight ML stacks."""

from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]

COMPARISON_ENTRYPOINTS = (
    "DANN/train_dann.py",
    "DANN/train_dann_sda.py",
    "DeepCORAL/train_deepcoral.py",
    "DeepCORAL/train_deepcoral_sda.py",
    "CDAN/train_cdan.py",
    "CDAN/train_cdan_sda.py",
    "MDD/train_mdd.py",
    "MDD/train_mdd_sda.py",
    "MCC/train_mcc.py",
    "MCC/train_mcc_sda.py",
    "TransAdapter/train_transadapter.py",
    "AD-CLIP/train_adclip.py",
    "Modfinity/train_modfinity.py",
    "Modfinity/train_modfinity_sda.py",
    "Boomda/train_boomda_jmda.py",
    "Boomda/train_boomda_sda.py",
)


class ComparisonReproducibilityTests(unittest.TestCase):
    def test_every_launched_comparison_entrypoint_applies_seed(self):
        for relative_path in COMPARISON_ENTRYPOINTS:
            with self.subTest(entrypoint=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(
                    "from reproducibility import seed_everything",
                    source,
                )
                self.assertIn(
                    "args.seed = seed_everything(args.seed)",
                    source,
                )

    def test_adclip_registers_projector_banks(self):
        source = (ROOT / "AD-CLIP/adclip_model.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("self.linear1 = nn.ModuleList"), 2)
        self.assertIn("self.linear = nn.ModuleList", source)
        self.assertNotIn(".to(device_cuda)", source)
        self.assertIn(
            'self.register_buffer("tokenized_prompts", tokenized_prompts)',
            source,
        )
        self.assertIn("torch.arange(x.shape[0], device=x.device)", source)

    def test_adclip_evaluation_is_target_only(self):
        model_source = (ROOT / "AD-CLIP/adclip_model.py").read_text(
            encoding="utf-8"
        )
        train_source = (ROOT / "AD-CLIP/train_adclip.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def forward_target(self, target_data):", model_source)
        self.assertIn("def forward_target(self, t_image):", model_source)
        self.assertIn("model.forward_target(imgs)", train_source)
        self.assertNotIn("model(imgs, imgs)", train_source)

    def test_multimodal_uda_rejects_label_selected_ais(self):
        for relative_path in (
            "Modfinity/train_modfinity.py",
            "Boomda/train_boomda_jmda.py",
        ):
            with self.subTest(entrypoint=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("if args.use_ais:", source)
                self.assertIn("not valid in strict UDA", source)

    def test_multimodal_pair_augmentation_replays_one_rng_state(self):
        for relative_path in (
            "Modfinity/train_modfinity.py",
            "Modfinity/train_modfinity_sda.py",
            "Boomda/train_boomda_jmda.py",
            "Boomda/train_boomda_sda.py",
        ):
            with self.subTest(entrypoint=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("transform_rng_state = torch.get_rng_state()", source)
                self.assertIn("torch.set_rng_state(transform_rng_state)", source)

    def test_cdan_relative_optimizer_metadata_is_materialized(self):
        module_path = ROOT / "CDAN/optimizer_utils.py"
        spec = importlib.util.spec_from_file_location(
            "cdan_optimizer_utils_for_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        groups = module.materialize_cdan_parameter_groups(
            [
                {"params": [object()], "lr_mult": 1, "decay_mult": 2},
                {"params": [object()], "lr_mult": 10, "decay_mult": 2},
            ],
            base_lr=1e-3,
            weight_decay=1e-4,
        )
        self.assertEqual([group["lr"] for group in groups], [1e-3, 1e-2])
        self.assertEqual(
            [group["weight_decay"] for group in groups],
            [2e-4, 2e-4],
        )


if __name__ == "__main__":
    unittest.main()
