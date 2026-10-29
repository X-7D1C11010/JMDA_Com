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
    "CCSA/train_ccsa.py",
    "d-SNE/train_dsne.py",
    "PDA/train_pda.py",
    "MCFT/train_mcft.py",
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

    def test_sda_entrypoints_resolve_their_declared_label_mode(self):
        ccsa = (ROOT / "CCSA/train_ccsa.py").read_text(encoding="utf-8")
        self.assertIn("_sda_{args.target_labels_per_class}shot_seed", ccsa)

        dsne = (ROOT / "d-SNE/train_dsne.py").read_text(encoding="utf-8")
        self.assertIn("def target_training_list_path", dsne)
        self.assertIn("if labels_per_class == 0", dsne)
        self.assertIn("target_training_list_path(", dsne)

    def test_requested_methods_use_target_driven_class_space_and_save_metrics(self):
        for relative_path in (
            "CCSA/train_ccsa.py", "d-SNE/train_dsne.py", "MCFT/train_mcft.py"
        ):
            with self.subTest(entrypoint=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("uda_label_space(args.data_dir, args.target)", source)
                self.assertIn("save_metric_history(", source)
                self.assertIn("num_classes = len(class_ids)", source)

    def test_released_default_training_protocols_are_explicit(self):
        ccsa = (ROOT / "CCSA/train_ccsa.py").read_text(encoding="utf-8")
        dsne = (ROOT / "d-SNE/train_dsne.py").read_text(encoding="utf-8")
        mcft = (ROOT / "MCFT/train_mcft.py").read_text(encoding="utf-8")
        self.assertIn("default=80", ccsa)
        self.assertIn("negative_ratio=3", ccsa)
        self.assertIn("default=101", dsne)
        self.assertIn("class ResNetV2Features", dsne)
        self.assertIn("--train-src", dsne)
        self.assertIn("default=500", mcft)
        self.assertIn("num_patches=[3, 1]", mcft)

    def test_per_seed_artifacts_cannot_overwrite_each_other(self):
        for relative_path, prefix in (
            ("CCSA/train_ccsa.py", "ccsa"),
            ("d-SNE/train_dsne.py", "dsne"),
            ("MCFT/train_mcft.py", "mcft"),
        ):
            with self.subTest(entrypoint=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(
                    f"run_stem = f'{prefix}_{{args.target}}_seed{{args.seed}}'",
                    source,
                )

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
