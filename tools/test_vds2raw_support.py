import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from MCFT.mcft_model import MViT
from TIR.loralib.layers import LinearLoRA
from TIR.vds2raw_adapter import TIRVDS2Raw, TokenImportanceRecalibration
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
from vds2raw.train_tir import (
    domain_class_balanced_sampler,
    mild_class_weights,
    normalize_accumulated_gradients,
    warmup_cosine_multiplier,
)


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
            expected_red = (1.0 / 255.0 - VDS2RawTIRDataset.CLIP_MEAN[0]) / (
                VDS2RawTIRDataset.CLIP_STD[0]
            )
            self.assertAlmostEqual(
                float(tir_item["rgb"][0, 0, 0]), expected_red, places=5
            )

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
        model.bias.requires_grad_(False)
        original_frozen_bias = model.bias.detach().clone()
        snapshot = capture_selected_state(
            model,
            3,
            {"accuracy": 0.5},
        )
        with torch.no_grad():
            model.weight.add_(10.0)
            model.bias.add_(20.0)
        restore_selected_state(model, snapshot)
        torch.testing.assert_close(model.weight, original)
        torch.testing.assert_close(model.bias, original_frozen_bias)
        self.assertEqual(snapshot["selected_epoch"], 3)

    def test_full_snapshot_restores_lora_merged_evaluation_weight(self):
        base = nn.Linear(2, 2, bias=True)
        model = LinearLoRA(base, r=1, lora_alpha=1, dropout_rate=0.0)
        inputs = torch.tensor([[1.0, 1.0]])
        with torch.no_grad():
            model.w_lora_A.fill_(0.25)
            model.w_lora_B.fill_(0.5)
        model.eval()
        selected_output = model(inputs).detach().clone()
        snapshot = capture_selected_state(model, 4, {"accuracy": 0.75})

        model.train()
        with torch.no_grad():
            model.w_lora_B.fill_(2.0)
        model.eval()
        self.assertFalse(torch.allclose(model(inputs), selected_output))
        restore_selected_state(model, snapshot)
        torch.testing.assert_close(model(inputs), selected_output)


class TIRCoreTests(unittest.TestCase):
    @staticmethod
    def runtime():
        return SimpleNamespace(
            aux_container={
                "visual_ln_post": nn.Identity(),
                "visual_projection": torch.eye(3),
            }
        )

    def test_ais_gate_is_positive_trainable_and_preserves_initial_value(self):
        model = TIRVDS2Raw.__new__(TIRVDS2Raw)
        nn.Module.__init__(model)
        initial = 1.50
        model.ais_scale = nn.Parameter(
            torch.tensor(np.log(np.expm1(initial)), dtype=torch.float32)
        )
        gate = model.effective_ais_scale()
        self.assertAlmostEqual(float(gate.detach()), initial, places=6)
        (-gate).backward()
        self.assertIsNotNone(model.ais_scale.grad)
        self.assertNotEqual(float(model.ais_scale.grad), 0.0)
        with torch.no_grad():
            model.ais_scale.sub_(model.ais_scale.grad)
        updated = float(model.effective_ais_scale().detach())
        self.assertGreater(updated, initial)
        self.assertGreater(updated, 0.0)

    def test_nir_gate_is_positive_trainable_and_preserves_initial_value(self):
        model = TIRVDS2Raw.__new__(TIRVDS2Raw)
        nn.Module.__init__(model)
        initial = 0.10
        model.nir_scale = nn.Parameter(
            torch.tensor(np.log(np.expm1(initial)), dtype=torch.float32)
        )
        gate = model.effective_nir_scale()
        self.assertAlmostEqual(float(gate.detach()), initial, places=6)
        (-gate).backward()
        self.assertIsNotNone(model.nir_scale.grad)
        self.assertNotEqual(float(model.nir_scale.grad), 0.0)
        self.assertGreater(float(gate.detach()), 0.0)

    def test_three_way_rule_uses_paper_generalized_extremes(self):
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
            "paper_simplified_kway_extremes_bounded",
        )
        self.assertEqual(output.shape, tokens.shape)

    def test_three_way_all_class_token_is_suppressed(self):
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
            torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        )

    def test_five_way_rule_stays_identical_to_released_linear_rule(self):
        tokens = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        )
        text = torch.tensor([[1.0, 0.0, 0.0]] * 5)
        runtime = self.runtime()
        TokenImportanceRecalibration()(
            tokens, text, topk_ratio=0.5, temper_weight=0.5, params=runtime
        )
        torch.testing.assert_close(
            runtime.aux_container["tir_last_weights"],
            torch.tensor([[0.0, 1.0]]),
        )
        self.assertEqual(
            runtime.aux_container["tir_weight_rule"],
            "official_linear_5way",
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

    def test_tir_sampler_balances_domain_and_class_groups(self):
        source = _LabelOnlyDataset([0, 0, 0, 1, 2])
        target = _LabelOnlyDataset([0, 0, 1, 1, 2])
        sampler, counts = domain_class_balanced_sampler(source, target, seed=42)
        self.assertEqual(counts, [{0: 3, 1: 1, 2: 1}, {0: 2, 1: 2, 2: 1}])
        self.assertEqual(len(sampler.weights), 10)

    def test_tir_mild_class_weights_use_combined_counts(self):
        source = _LabelOnlyDataset([0, 0, 0, 1, 2])
        target = _LabelOnlyDataset([0, 0, 1, 1, 2])
        weights, counts = mild_class_weights(source, target, power=0.5)
        self.assertEqual(counts, [5, 3, 2])
        weighted_mean = sum(w * c for w, c in zip(weights.tolist(), counts)) / sum(
            counts
        )
        self.assertAlmostEqual(weighted_mean, 1.0, places=6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))

    def test_tir_domain_specific_weights_do_not_hide_target_minority(self):
        source = _LabelOnlyDataset([0] + [1] * 2 + [2] * 5)
        target = _LabelOnlyDataset([0] * 5 + [1] * 2 + [2])
        source_weights, source_counts = mild_class_weights(source, power=0.5)
        target_weights, target_counts = mild_class_weights(target, power=0.5)
        self.assertEqual(source_counts, [1, 2, 5])
        self.assertEqual(target_counts, [5, 2, 1])
        self.assertGreater(float(target_weights[2]), float(target_weights[0]))
        self.assertLess(float(source_weights[2]), float(source_weights[0]))

    def test_tir_warmup_cosine_scheduler(self):
        values = [warmup_cosine_multiplier(i, 60, 5, 0.1) for i in range(61)]
        self.assertAlmostEqual(values[0], 0.2)
        self.assertAlmostEqual(values[4], 1.0)
        self.assertAlmostEqual(values[5], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(all(0.1 <= value <= 1.0 for value in values))

    def test_tir_full_epoch_gradient_normalization(self):
        parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
        parameter.grad = torch.tensor([6.0, 10.0])
        normalize_accumulated_gradients({"group": [parameter]}, denominator=2)
        torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 5.0]))

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
        self.assertIn('METHOD_ORDER:-tir', linux)
        self.assertIn('TIR_EPOCHS="${VDS2RAW_TIR_EPOCHS:-100}"', linux)
        self.assertIn('TIR_BATCH="${VDS2RAW_TIR_BATCH_SIZE:-32}"', linux)
        self.assertIn('--eval-frequency 1', linux)
        self.assertIn('TIR_LORA_LR="${VDS2RAW_TIR_LORA_LR:-0.0002}"', linux)
        self.assertIn('TIR_LORA_RANK="${VDS2RAW_TIR_LORA_RANK:-16}"', linux)
        self.assertIn('--scheduler none', linux)
        self.assertIn('--natural-train-sampling', linux)
        self.assertIn('--minibatch-updates', linux)
        self.assertIn('TIR_GATE_LR="${VDS2RAW_TIR_GATE_LR:-0.01}"', linux)
        self.assertIn(
            'TIR_MODALITY_WARMUP_EPOCHS="${VDS2RAW_TIR_MODALITY_WARMUP_EPOCHS:-25}"',
            linux,
        )
        self.assertIn('TIR_NIR_AUX_WEIGHT="${VDS2RAW_TIR_NIR_AUX_LOSS_WEIGHT:-0.2}"', linux)
        self.assertIn('TIR_AIS_AUX_WEIGHT="${VDS2RAW_TIR_AIS_AUX_LOSS_WEIGHT:-0.2}"', linux)
        self.assertIn('--lora-encoder both --lora-params q k v', linux)
        self.assertIn('MCFT_BATCH=2; TIR_BATCH=32', linux)
        self.assertIn('MCFT_MAX_BATCH_ARGS=(--max-train-batches 1)', linux)
        self.assertNotIn('TIR_MAX_BATCH_ARGS=(--max-train-batches 1)', linux)
        self.assertIn('TIR_EARLY_PATIENCE="${VDS2RAW_TIR_EARLY_STOPPING_PATIENCE:-15}"', linux)
        self.assertIn('TIR_EARLY_MIN_EPOCHS="${VDS2RAW_TIR_EARLY_STOPPING_MIN_EPOCHS:-60}"', linux)
        tir_train = (root / "vds2raw/train_tir.py").read_text(encoding="utf-8")
        self.assertIn("domain_class_balanced_train=False", tir_train)
        self.assertIn("accumulate_full_epoch=False", tir_train)
        self.assertIn("nir_logits, ais_logits = model", tir_train)
        self.assertIn("args.nir_aux_loss_weight * nir_aux_loss", tir_train)
        self.assertIn("args.ais_aux_loss_weight * ais_aux_loss", tir_train)
        self.assertIn("DomainTaggedDataset(source, 0)", tir_train)
        self.assertIn("DomainTaggedDataset(target, 1)", tir_train)
        self.assertIn('"domain_separated_supervised_loss": True', tir_train)
        self.assertIn('"target_label_budget": "all available target_train labels"', tir_train)
        self.assertNotIn("save_checkpoint(", tir_train)


if __name__ == "__main__":
    unittest.main()
