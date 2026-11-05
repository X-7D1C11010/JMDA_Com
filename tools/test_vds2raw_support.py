import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from MCFT.mcft_model import MViT
from TIR.vds2raw_adapter import TokenImportanceRecalibration
from vds2raw.common import capture_selected_state, restore_selected_state
from vds2raw.dataset import (
    CLASS_NAMES,
    PARTITIONS,
    VDS2RawDataset,
    VDS2RawTIRDataset,
    compute_image_statistics,
    target_present_class_ids,
)
from vds2raw.models import MCFTWithAIS
from vds2raw.train_mcft import class_balanced_sampler


def write_sample(path, rgb_value, nir_value, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=np.full((3, 96, 96), rgb_value, dtype=np.uint16),
        nir=np.full((1, 96, 96), nir_value, dtype=np.uint16),
        ais=np.arange(13, dtype=np.float32),
        ais_mask=np.ones(13, dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
    )


class _FixtureMixin:
    def make_fixture(self, root):
        for partition in PARTITIONS:
            for class_name in CLASS_NAMES:
                (root / "samples" / partition / class_name).mkdir(
                    parents=True, exist_ok=True
                )
        write_sample(root / "samples/source_train/Cargo/source.npz", 1, 10, 0)
        write_sample(root / "samples/target_train/Fishing/target.npz", 3, 14, 1)
        write_sample(
            root / "samples/target_val/Sailing_Pleasure/val.npz", 1000, 1000, 2
        )
        write_sample(root / "samples/target_test/Cargo/test.npz", 2000, 2000, 0)


class VDS2RawDatasetTests(_FixtureMixin, unittest.TestCase):
    def test_statistics_use_training_partitions_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(root)
            self.assertEqual(statistics.fit_samples, 2)
            self.assertEqual(
                statistics.fit_partitions, ("source_train", "target_train")
            )
            np.testing.assert_allclose(statistics.rgb_mean, [2.0, 2.0, 2.0])
            np.testing.assert_allclose(statistics.nir_mean, [12.0])

    def test_tir_statistics_can_be_fitted_without_source_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(
                root, fit_partitions=("target_train",)
            )
            self.assertEqual(statistics.fit_samples, 1)
            self.assertEqual(statistics.fit_partitions, ("target_train",))
            np.testing.assert_allclose(statistics.rgb_mean, [3.0, 3.0, 3.0])
            np.testing.assert_allclose(statistics.nir_mean, [14.0])

    def test_mcft_and_tir_views_have_declared_native_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(root)
            mcft_item = VDS2RawDataset(
                root, "source_train", statistics, image_size=96
            )[0]
            tir_item = VDS2RawTIRDataset(
                root, "source_train", statistics, augment=False
            )[0]
            self.assertEqual(mcft_item["rgb"].shape, (3, 96, 96))
            self.assertEqual(mcft_item["nir"].shape, (1, 96, 96))
            self.assertEqual(tir_item["rgb"].shape, (3, 224, 224))
            self.assertEqual(tir_item["nir"].shape, (1, 224, 224))
            self.assertEqual(mcft_item["ais"].shape, (4,))
            self.assertEqual(mcft_item["ais_mask"].shape, (4,))
            self.assertEqual(tir_item["ais"].shape, (4,))
            self.assertEqual(tir_item["ais_mask"].shape, (4,))
            torch.testing.assert_close(tir_item["ais"], torch.arange(4).float())

    def test_source_loading_is_driven_by_target_present_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            write_sample(
                root / "samples/source_train/Fishing/source_match.npz", 2, 12, 1
            )
            statistics = compute_image_statistics(root)
            active = target_present_class_ids(root)
            source = VDS2RawDataset(
                root,
                "source_train",
                statistics,
                allowed_class_ids=active,
            )
            target = VDS2RawDataset(
                root,
                "target_train",
                statistics,
                allowed_class_ids=active,
            )
            self.assertEqual(active, (1,))
            self.assertEqual({label for _, label in source.samples}, {1})
            self.assertEqual({label for _, label in target.samples}, {1})


class MCFTAdapterTests(unittest.TestCase):
    def test_native_96_raster_is_convolved_before_15_grid_pooling(self):
        model = MViT(
            patch_size=15,
            num_patches=[3, 1],
            num_classes=3,
            dim=64,
            depth=6,
            heads=4,
            mlp_dim=32,
            dropout=0.0,
            emb_dropout=0.0,
            mode="MViT",
        ).eval()
        observed = {}

        def conv_hook(_module, inputs, output):
            observed["conv_input"] = tuple(inputs[0].shape[-2:])
            observed["conv_output"] = tuple(output.shape[-2:])

        def pool_hook(_module, _inputs, output):
            observed["pool_output"] = tuple(output.shape[-2:])

        first = model.separable1.register_forward_hook(conv_hook)
        second = model.token_grid_pool.register_forward_hook(pool_hook)
        try:
            with torch.no_grad():
                logits, _, _ = model(
                    torch.randn(2, 3, 96, 96),
                    torch.randn(2, 1, 96, 96),
                    label=None,
                    compute_auxiliary=False,
                )
        finally:
            first.remove()
            second.remove()
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(observed["conv_input"], (96, 96))
        self.assertEqual(observed["conv_output"], (96, 96))
        self.assertEqual(observed["pool_output"], (15, 15))

    def test_default_mcft_ais_uses_proxy_safe_gated_values_only(self):
        model = MCFTWithAIS(num_classes=3).eval()
        ais = torch.randn(2, 4)
        mask = torch.zeros_like(ais)
        with torch.no_grad():
            expected = model.encode_ais(torch.zeros_like(ais), mask)
            actual = model.encode_ais(ais, mask)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(model.ais_feature_profile, "continuous")
        self.assertEqual(model.ais_encoder[0].normalized_shape, (4,))

    def test_best_state_is_restored_without_a_model_file(self):
        model = nn.Linear(2, 2)
        original = model.weight.detach().clone()
        snapshot = capture_selected_state(
            model,
            3,
            {"accuracy": 0.5},
        )
        with torch.no_grad():
            model.weight.add_(10.0)
        restore_selected_state(model, snapshot)
        torch.testing.assert_close(model.weight, original)
        self.assertEqual(snapshot["selected_epoch"], 3)


class TIRCoreTests(unittest.TestCase):
    @staticmethod
    def runtime():
        return SimpleNamespace(
            aux_container={
                "visual_ln_post": nn.Identity(),
                "visual_projection": torch.eye(3),
            }
        )

    def test_three_way_rule_matches_official_linear_fewer_class_branch(self):
        tokens = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
                [[-1.0, -1.0, -1.0]],
            ]
        )
        text = torch.eye(3)
        runtime = self.runtime()
        output = TokenImportanceRecalibration()(
            tokens, text, topk_ratio=0.25, temper_weight=0.5, params=runtime
        )
        torch.testing.assert_close(
            runtime.aux_container["tir_last_weights"],
            torch.tensor([[2.0, 2.0, 2.0, 1.0]]),
        )
        self.assertEqual(
            runtime.aux_container["tir_weight_rule"],
            "official_linear_fewer_classes",
        )
        self.assertEqual(output.shape, tokens.shape)

    def test_three_way_all_class_token_uses_official_sum_three_weight(self):
        tokens = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
                [[-1.0, -1.0, -1.0]],
            ]
        )
        text = torch.tensor([[1.0, 0.0, 0.0]] * 3)
        runtime = self.runtime()
        TokenImportanceRecalibration()(
            tokens, text, topk_ratio=0.25, temper_weight=0.5, params=runtime
        )
        torch.testing.assert_close(
            runtime.aux_container["tir_last_weights"],
            torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        )


class _LabelOnlyDataset:
    def __init__(self, labels):
        self.samples = [
            (f"sample_{index}.npz", label) for index, label in enumerate(labels)
        ]


class LauncherAndSamplingTests(unittest.TestCase):
    def test_mcft_sampler_balances_domain_and_class_groups(self):
        source = _LabelOnlyDataset([0, 0, 0, 1, 2])
        target = _LabelOnlyDataset([0, 0, 1, 1, 2])
        sampler, counts = class_balanced_sampler(source, target, seed=42)
        self.assertEqual(counts, [{0: 3, 1: 1, 2: 1}, {0: 2, 1: 2, 2: 1}])
        self.assertEqual(len(sampler.weights), 10)

    def test_launchers_only_offer_mcft_and_tir(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "scripts/run_vds2raw_methods.sh",
            "scripts/run_vds2raw_methods.ps1",
        ):
            source = (root / relative).read_text(encoding="utf-8")
            self.assertIn("mcft3", source)
            self.assertIn("tir", source)
            self.assertNotIn("train_ccsa", source.lower())
            self.assertNotIn("train_dsne", source.lower())
        linux = (root / "scripts/run_vds2raw_methods.sh").read_text(encoding="utf-8")
        self.assertIn('METHOD_ORDER:-mcft3 tir', linux)
        self.assertIn('TIR_EPOCHS="${VDS2RAW_TIR_EPOCHS:-100}"', linux)
        self.assertIn('TIR_BATCH="${VDS2RAW_TIR_BATCH_SIZE:-32}"', linux)
        self.assertIn('--eval-frequency 1', linux)
        self.assertIn('TIR_EARLY_PATIENCE="${VDS2RAW_TIR_EARLY_STOPPING_PATIENCE:-0}"', linux)
        tir_train = (root / "vds2raw/train_tir.py").read_text(encoding="utf-8")
        self.assertIn("parser.set_defaults(augment=False)", tir_train)
        self.assertIn("train_data = ConcatDataset((source, target))", tir_train)
        self.assertIn('"target_label_budget": "all available target_train labels"', tir_train)
        self.assertNotIn("save_checkpoint(", tir_train)


if __name__ == "__main__":
    unittest.main()
