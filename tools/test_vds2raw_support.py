import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from MCFT.mcft_model import MViT
from tools.domain_pairing import CrossDomainPairDataset, DSNEClassCoverageBatchSampler
from vds2raw.common import (
    evaluate_multimodal,
    save_checkpoint,
    save_prediction_artifacts,
)
from vds2raw.dataset import (
    CLASS_NAMES,
    NaturalImageMultimodalDataset,
    NaturalImageRGBDataset,
    PARTITIONS,
    VDS2RawDataset,
    compute_image_statistics,
    fixed_samples_per_class,
    load_or_compute_statistics,
)
from vds2raw.models import MCFTWithAIS
from vds2raw.summarize import training_curve_diagnostics
from vds2raw.train_ccsa import (
    adaptation_parameter_groups as ccsa_adaptation_parameter_groups,
    class_weights,
)
from vds2raw.train_dsne import (
    adaptation_parameter_groups as dsne_adaptation_parameter_groups,
    cosine_lr_factor,
)
from vds2raw.train_mcft import class_balanced_sampler


def write_sample(path, rgb_value, nir_value, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=np.full((3, 96, 96), rgb_value, dtype=np.uint16),
        nir=np.full((1, 96, 96), nir_value, dtype=np.uint16),
        ais=np.zeros(13, dtype=np.float32),
        ais_mask=np.zeros(13, dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
    )


class VDS2RawStatisticsTests(unittest.TestCase):
    def make_fixture(self, root):
        for partition in PARTITIONS:
            for class_name in CLASS_NAMES:
                (root / "samples" / partition / class_name).mkdir(
                    parents=True, exist_ok=True
                )
        write_sample(root / "samples/source_train/Cargo/source.npz", 1, 10, 0)
        write_sample(root / "samples/target_train/Fishing/target.npz", 3, 14, 1)
        # These extreme values must not affect fitted statistics.
        write_sample(
            root / "samples/target_val/Sailing_Pleasure/val.npz", 1000, 1000, 2
        )
        write_sample(root / "samples/target_test/Cargo/test.npz", 2000, 2000, 0)

    def test_image_statistics_exclude_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(root)
            self.assertEqual(statistics.fit_samples, 2)
            self.assertEqual(
                statistics.fit_partitions, ("source_train", "target_train")
            )
            np.testing.assert_allclose(statistics.rgb_mean, [2.0, 2.0, 2.0])
            np.testing.assert_allclose(statistics.rgb_std, [1.0, 1.0, 1.0])
            np.testing.assert_allclose(statistics.rgb_min, [1.0, 1.0, 1.0])
            np.testing.assert_allclose(statistics.rgb_max, [3.0, 3.0, 3.0])
            np.testing.assert_allclose(statistics.nir_mean, [12.0])
            np.testing.assert_allclose(statistics.nir_std, [2.0])
            np.testing.assert_allclose(statistics.nir_min, [10.0])
            np.testing.assert_allclose(statistics.nir_max, [14.0])

    def test_natural_image_view_uses_train_only_raw_dn_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(root)
            source = NaturalImageRGBDataset(
                root, "source_train", statistics,
                augment=False, normalization="half",
            )
            validation = NaturalImageRGBDataset(
                root, "target_val", statistics,
                augment=False, normalization="half",
            )
            source_rgb, _ = source[0]
            validation_rgb, _ = validation[0]
            self.assertAlmostEqual(
                float(source_rgb.max()), 2.0 / 255.0 - 1.0, places=5
            )
            self.assertAlmostEqual(float(validation_rgb.min()), 1.0, places=5)

    def test_zero_per_class_uses_every_available_target_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(root)
            target = NaturalImageRGBDataset(
                root, "target_train", statistics,
                augment=False, normalization="half",
            )
            selected = fixed_samples_per_class(target, 0, seed=42)
            self.assertIs(selected, target)
            self.assertEqual(len(selected), len(target))
            with self.assertRaisesRegex(ValueError, "cannot be negative"):
                fixed_samples_per_class(target, -1, seed=42)

        from vds2raw.train_dsne import DEFAULT_TARGET_SAMPLES_PER_CLASS

        self.assertEqual(DEFAULT_TARGET_SAMPLES_PER_CLASS, 0)

    def test_natural_multimodal_view_returns_all_three_modalities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(root)
            dataset = NaturalImageMultimodalDataset(
                root,
                "source_train",
                statistics,
                augment=False,
                normalization="half",
            )
            inputs, label = dataset[0]
            self.assertEqual(label, 0)
            self.assertEqual(inputs["rgb"].shape, (3, 224, 224))
            self.assertEqual(inputs["nir"].shape, (1, 224, 224))
            self.assertEqual(inputs["ais"].shape, (13,))
            self.assertEqual(inputs["ais_mask"].shape, (13,))

    def test_dataset_returns_float32_and_respects_resize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            statistics = compute_image_statistics(root)
            dataset = VDS2RawDataset(root, "source_train", statistics, image_size=15)
            item = dataset[0]
            self.assertEqual(item["rgb"].shape, (3, 15, 15))
            self.assertEqual(item["nir"].shape, (1, 15, 15))
            self.assertEqual(item["rgb"].dtype, torch.float32)
            self.assertEqual(item["nir"].dtype, torch.float32)
            self.assertEqual(item["ais"].shape, (13,))
            self.assertEqual(item["ais_mask"].shape, (13,))

    def test_paired_flips_transform_ais_heading_consistently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            path = root / "samples/source_train/Cargo/source.npz"
            ais = np.zeros(13, dtype=np.float32)
            ais[:2] = (0.6, -0.8)
            np.savez_compressed(
                path,
                rgb=np.full((3, 96, 96), 1, dtype=np.uint16),
                nir=np.full((1, 96, 96), 10, dtype=np.uint16),
                ais=ais,
                ais_mask=np.ones(13, dtype=np.float32),
                label=np.asarray(0, dtype=np.int64),
            )
            statistics = compute_image_statistics(root)
            dataset = VDS2RawDataset(
                root, "source_train", statistics, image_size=15, augment=True
            )
            with patch(
                "vds2raw.dataset.torch.rand",
                side_effect=(torch.tensor(0.0), torch.tensor(0.0)),
            ):
                item = dataset[0]
            self.assertAlmostEqual(float(item["ais"][0]), -0.6, places=6)
            self.assertAlmostEqual(float(item["ais"][1]), 0.8, places=6)

    def test_cached_statistics_reject_wrong_fit_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            stats_path = root / "stats.json"
            payload = compute_image_statistics(root).to_dict()
            payload["fit_partitions"] = ["source_train", "target_val"]
            stats_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not fitted"):
                load_or_compute_statistics(root, stats_path)


class _LabelOnlyDataset:
    def __init__(self, labels):
        self.samples = [(f"sample_{index}.npz", label) for index, label in enumerate(labels)]


class VDS2RawTunedTrainingTests(unittest.TestCase):
    def test_adaptation_parameter_groups_cover_model_once_and_protect_backbone(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Linear(3, 4)
                self.head = nn.Linear(4, 2)

        for builder in (
            ccsa_adaptation_parameter_groups,
            dsne_adaptation_parameter_groups,
        ):
            model = TinyModel()
            groups = builder(model, lr=0.2, backbone_multiplier=0.1)
            self.assertAlmostEqual(groups[0]["lr"], 0.02)
            self.assertAlmostEqual(groups[1]["lr"], 0.2)
            grouped = [parameter for group in groups for parameter in group["params"]]
            self.assertEqual(len(grouped), len(list(model.parameters())))
            self.assertEqual(len({id(parameter) for parameter in grouped}), len(grouped))

    def test_ccsa_class_weights_are_inverse_frequency_and_mean_one(self):
        weights = class_weights(
            _LabelOnlyDataset([0, 0, 0, 0, 1, 1, 2]),
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))

    def test_dsne_cosine_multiplier_preserves_configured_floor(self):
        self.assertAlmostEqual(cosine_lr_factor(0, 20, 0.05), 1.0)
        self.assertAlmostEqual(cosine_lr_factor(20, 20, 0.05), 0.05)
        self.assertAlmostEqual(cosine_lr_factor(100, 20, 0.05), 0.05)

    def test_dsne_balanced_epoch_covers_every_available_target_sample(self):
        source = _LabelOnlyDataset(
            [label for label, count in ((0, 16), (1, 41), (2, 59)) for _ in range(count)]
        )
        target = _LabelOnlyDataset(
            [label for label, count in ((0, 46), (1, 33), (2, 15)) for _ in range(count)]
        )
        pairs = CrossDomainPairDataset(source, target, negative_ratio=1, seed=42)
        sampler = DSNEClassCoverageBatchSampler(pairs, batch_size=32, seed=42)
        sampled_pair_indices = [index for batch in sampler for index in batch]
        sampled_target_indices = {
            pairs.pairs[pair_index][1] for pair_index in sampled_pair_indices
        }
        self.assertEqual(sampled_target_indices, set(range(len(target.samples))))

    def test_mcft_sampler_has_equal_expected_weight_per_class(self):
        source = _LabelOnlyDataset([0, 0, 0, 1, 2])
        target = _LabelOnlyDataset([0, 0, 1, 1, 2])
        sampler, counts = class_balanced_sampler(source, target, seed=42)
        labels = [label for dataset in (source, target) for _, label in dataset.samples]
        totals = {
            label: sum(
                float(weight)
                for weight, sample_label in zip(sampler.weights, labels)
                if sample_label == label
            )
            for label in range(3)
        }
        self.assertEqual(counts, [{0: 3, 1: 1, 2: 1}, {0: 2, 1: 2, 2: 1}])
        self.assertAlmostEqual(totals[0], totals[1], places=6)
        self.assertAlmostEqual(totals[1], totals[2], places=6)
        offset = len(source.samples)
        for start, dataset in ((0, source), (offset, target)):
            domain_totals = {
                label: sum(
                    float(sampler.weights[start + index])
                    for index, (_, sample_label) in enumerate(dataset.samples)
                    if sample_label == label
                )
                for label in range(3)
            }
            self.assertAlmostEqual(domain_totals[0], domain_totals[1], places=6)
            self.assertAlmostEqual(domain_totals[1], domain_totals[2], places=6)

    def test_mcft_hard_negative_mask_keeps_real_class_zero_candidates(self):
        weights = torch.ones(3, 3)
        labels = torch.tensor([0, 1, 2])
        masked, eligible = MViT._mask_same_class_negatives(weights, labels)
        torch.testing.assert_close(masked.diag(), torch.zeros(3))
        self.assertEqual(float(masked[1, 0]), 1.0)
        self.assertEqual(float(masked[2, 0]), 1.0)
        self.assertTrue(bool(eligible[1, 0]))
        self.assertFalse(bool(eligible[0, 0]))

    def test_launchers_enable_all_tuned_paths(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "scripts/run_vds2raw_methods.sh",
            "scripts/run_vds2raw_methods.ps1",
        ):
            script = (root / relative).read_text(encoding="utf-8")
            for argument in (
                "--class-balanced-ce",
                "--source-lr-min-ratio",
                "--label-smoothing",
                "--dropout",
                "--class-balanced-train",
                "--loss-itm-weight",
            ):
                self.assertIn(argument, script)
        linux_script = (root / "scripts/run_vds2raw_methods.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('MCFT_EPOCHS="${VDS2RAW_MCFT_EPOCHS:-200}"', linux_script)

        pilot_script = (
            root / "scripts/run_vds2raw_tuned_pilot_background.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('VDS2RAW_SEEDS="$PILOT_SEED"', pilot_script)
        self.assertIn('VDS2RAW_METHOD_ORDER="dsne ccsa mcft3"', pilot_script)
        self.assertIn("VDS2RAW_RUN_COMPLEXITY_BENCHMARK=0", pilot_script)
        self.assertIn("run_vds2raw_background.sh\" formal", pilot_script)
        self.assertIn("must retrain source stages", pilot_script)

    def test_training_summary_flags_early_peak_and_reports_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = root / "target_test_metrics.csv"
            history.write_text(
                "epoch,ACC,Precision_macro,Recall_macro,F1_macro\n"
                "1,0.70,0.70,0.70,0.70\n"
                "10,0.80,0.65,0.65,0.60\n",
                encoding="utf-8",
            )
            payload = {
                "selected_epoch": 2,
                "target_test": {"accuracy": 0.90, "f1_macro": 0.80},
                "metadata": {"epochs": 20, "completed_epochs": 10},
                "target_test_history_csv": "/server/target_test_metrics.csv",
            }
            diagnostic = training_curve_diagnostics(payload, root)
            self.assertTrue(diagnostic["early_peak"])
            self.assertEqual(diagnostic["early_peak_threshold_epoch"], 5)
            self.assertEqual(diagnostic["last_evaluated_epoch"], 10)
            self.assertAlmostEqual(diagnostic["peak_to_last_ACC_drop_pp"], 10.0)
            self.assertAlmostEqual(diagnostic["peak_to_last_F1_drop_pp"], 20.0)

            payload["metadata"]["epochs"] = 1
            self.assertFalse(training_curve_diagnostics(payload, root)["early_peak"])


class _OneMultimodalSample(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "rgb": torch.zeros(3, 15, 15),
            "nir": torch.zeros(1, 15, 15),
            "ais": torch.zeros(13),
            "ais_mask": torch.zeros(13),
            "label": 1,
        }


class _NoLabelInferenceModel(nn.Module):
    def forward(
        self,
        rgb,
        nir,
        ais,
        ais_mask,
        labels=None,
        compute_auxiliary=True,
    ):
        if labels is not None or compute_auxiliary:
            raise AssertionError("evaluation leaked labels into the model")
        logits = torch.tensor([[0.0, 1.0, 0.0]]).expand(rgb.shape[0], -1)
        zero = logits.new_zeros(())
        return logits, zero, zero


class VDS2RawEvaluationTests(unittest.TestCase):
    def test_multimodal_inference_does_not_pass_ground_truth_to_model(self):
        metrics = evaluate_multimodal(
            _NoLabelInferenceModel(),
            DataLoader(_OneMultimodalSample(), batch_size=1),
            torch.device("cpu"),
        )
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_test_selected_checkpoint_and_prediction_artifacts_are_explicit(self):
        model = nn.Linear(2, 3)
        val_metrics = {
            "accuracy": 0.5,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
            "f1_macro": 0.5,
            "val_samples": 2,
            "val_present_classes": 2,
            "val_total_classes": 3,
        }
        test_metrics = dict(val_metrics, accuracy=0.75, f1_macro=0.7)
        records = [
            {
                "sample_index": 0,
                "sample_id": "sample.npz",
                "y_true": 1,
                "y_pred": 1,
                "correct": 1,
                "logits": [0.0, 1.0, 0.0],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pth"
            save_checkpoint(
                checkpoint,
                model,
                4,
                test_metrics,
                {"selection_partition": "target_test"},
                target_val_metrics=val_metrics,
                target_test_metrics=test_metrics,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["selection_partition"], "target_test")
            self.assertEqual(payload["target_val_metrics"]["accuracy"], 0.5)
            self.assertEqual(payload["target_test_metrics"]["accuracy"], 0.75)

            artifacts = save_prediction_artifacts(directory, "selected", records)
            for path in artifacts.values():
                self.assertTrue(Path(path).is_file())
            confusion = json.loads(
                Path(artifacts["confusion_matrix_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(confusion["matrix"][1][1], 1)

    def test_mcft_mask_is_validity_gate_not_classifier_feature(self):
        model = MCFTWithAIS(
            ais_feature_profile="continuous",
            ais_dropout_probability=0.0,
        ).eval()
        ais = torch.zeros(2, 13)
        # With zero values, changing validity bits alone cannot change the
        # encoded feature because mask is not concatenated as an input.
        masks = torch.stack((torch.zeros(13), torch.ones(13)))
        encoded = model.encode_ais(ais, masks)
        torch.testing.assert_close(encoded[0], encoded[1])

        # Navigation-status features are excluded by the robust default.
        ais[1, 6] = 1.0  # nav_engaged_in_fishing
        encoded = model.encode_ais(ais, torch.ones_like(ais))
        torch.testing.assert_close(encoded[0], encoded[1])
        self.assertEqual(model.ais_encoder[0].normalized_shape, (4,))

    def test_mcft_full_ais_profile_remains_explicit_opt_in(self):
        model = MCFTWithAIS(
            ais_feature_profile="full",
            ais_dropout_probability=0.0,
        )
        self.assertEqual(model.ais_encoder[0].normalized_shape, (13,))
        with self.assertRaisesRegex(ValueError, "unknown AIS feature profile"):
            MCFTWithAIS(ais_feature_profile="unknown")
        with self.assertRaisesRegex(ValueError, "must be in"):
            MCFTWithAIS(ais_dropout_probability=1.0)


if __name__ == "__main__":
    unittest.main()
