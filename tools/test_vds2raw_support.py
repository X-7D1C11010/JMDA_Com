import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vds2raw.common import evaluate_multimodal
from vds2raw.dataset import (
    CLASS_NAMES,
    NaturalImageRGBDataset,
    PARTITIONS,
    VDS2RawDataset,
    compute_image_statistics,
    fixed_samples_per_class,
    load_or_compute_statistics,
)


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


if __name__ == "__main__":
    unittest.main()
