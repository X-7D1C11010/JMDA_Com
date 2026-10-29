"""Leakage-safe VDS2Raw dataset loading and audit utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ("Cargo", "Fishing", "Sailing_Pleasure")
DISPLAY_CLASS_NAMES = ("Cargo", "Fishing", "Sailing & Pleasure")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
PARTITIONS = ("source_train", "target_train", "target_val", "target_test")
TRAIN_PARTITIONS = ("source_train", "target_train")
EXPECTED_COUNTS = {
    "source_train": 116,
    "target_train": 94,
    "target_val": 28,
    "target_test": 33,
}
REQUIRED_KEYS = {"rgb", "nir", "ais", "ais_mask", "label"}


@dataclass(frozen=True)
class ImageStatistics:
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    rgb_min: tuple[float, float, float]
    rgb_max: tuple[float, float, float]
    nir_mean: tuple[float]
    nir_std: tuple[float]
    nir_min: tuple[float]
    nir_max: tuple[float]
    fit_partitions: tuple[str, str] = TRAIN_PARTITIONS
    fit_samples: int = 210

    def to_dict(self):
        return {
            "rgb_mean": list(self.rgb_mean),
            "rgb_std": list(self.rgb_std),
            "rgb_min": list(self.rgb_min),
            "rgb_max": list(self.rgb_max),
            "nir_mean": list(self.nir_mean),
            "nir_std": list(self.nir_std),
            "nir_min": list(self.nir_min),
            "nir_max": list(self.nir_max),
            "fit_partitions": list(self.fit_partitions),
            "fit_samples": self.fit_samples,
            "rule": "statistics fitted on source_train + target_train only",
        }

    @classmethod
    def from_dict(cls, payload):
        if tuple(payload["fit_partitions"]) != TRAIN_PARTITIONS:
            raise ValueError(
                "normalization statistics were not fitted on the two training partitions"
            )
        required_ranges = ("rgb_min", "rgb_max", "nir_min", "nir_max")
        missing_ranges = [key for key in required_ranges if key not in payload]
        if missing_ranges:
            raise ValueError(
                "normalization statistics predate raw-DN range tracking; "
                f"regenerate the stats file (missing {missing_ranges})"
            )
        return cls(
            rgb_mean=tuple(map(float, payload["rgb_mean"])),
            rgb_std=tuple(map(float, payload["rgb_std"])),
            rgb_min=tuple(map(float, payload["rgb_min"])),
            rgb_max=tuple(map(float, payload["rgb_max"])),
            nir_mean=tuple(map(float, payload["nir_mean"])),
            nir_std=tuple(map(float, payload["nir_std"])),
            nir_min=tuple(map(float, payload["nir_min"])),
            nir_max=tuple(map(float, payload["nir_max"])),
            fit_partitions=tuple(payload["fit_partitions"]),
            fit_samples=int(payload["fit_samples"]),
        )


def partition_files(root, partition):
    root = Path(root)
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    files = []
    for class_name in CLASS_NAMES:
        class_dir = root / "samples" / partition / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class directory: {class_dir}")
        files.extend(sorted(class_dir.glob("*.npz")))
    return files


def compute_image_statistics(root):
    """Fit RGB/NIR channel statistics using training partitions only."""
    sums = np.zeros(4, dtype=np.float64)
    squared = np.zeros(4, dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    minima = np.full(4, np.inf, dtype=np.float64)
    maxima = np.full(4, -np.inf, dtype=np.float64)
    sample_count = 0
    for partition in TRAIN_PARTITIONS:
        for path in partition_files(root, partition):
            with np.load(path, allow_pickle=False) as item:
                arrays = (item["rgb"], item["nir"])
                channels = np.concatenate(
                    [arrays[0].reshape(3, -1), arrays[1].reshape(1, -1)], axis=0
                ).astype(np.float64, copy=False)
            sums += channels.sum(axis=1)
            squared += np.square(channels).sum(axis=1)
            counts += channels.shape[1]
            minima = np.minimum(minima, channels.min(axis=1))
            maxima = np.maximum(maxima, channels.max(axis=1))
            sample_count += 1
    means = sums / counts
    variances = np.maximum(squared / counts - np.square(means), 1e-12)
    stds = np.sqrt(variances)
    return ImageStatistics(
        rgb_mean=tuple(map(float, means[:3])),
        rgb_std=tuple(map(float, stds[:3])),
        rgb_min=tuple(map(float, minima[:3])),
        rgb_max=tuple(map(float, maxima[:3])),
        nir_mean=(float(means[3]),),
        nir_std=(float(stds[3]),),
        nir_min=(float(minima[3]),),
        nir_max=(float(maxima[3]),),
        fit_samples=sample_count,
    )


def load_or_compute_statistics(root, stats_file=None):
    stats_path = Path(stats_file) if stats_file else None
    if stats_path and stats_path.is_file():
        return ImageStatistics.from_dict(
            json.loads(stats_path.read_text(encoding="utf-8"))
        )
    statistics = compute_image_statistics(root)
    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = stats_path.with_suffix(stats_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(statistics.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(stats_path)
    return statistics


def _manifest_rows(root):
    manifest = Path(root) / "dataset_manifest.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_dataset(root, check_every_sample=True):
    """Independently verify the stated partition and anti-leakage contract."""
    root = Path(root).expanduser().resolve()
    required_metadata = (
        "dataset_manifest.csv",
        "feature_schema.json",
        "build_report.json",
        "band_order_verification.json",
        "validation_report.json",
    )
    for name in required_metadata:
        if not (root / name).is_file():
            raise FileNotFoundError(f"missing VDS2Raw metadata file: {root / name}")

    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("validation_report.json status is not PASS")
    if tuple(schema.get("train_partitions", ())) != TRAIN_PARTITIONS:
        raise ValueError(
            "feature_schema.json permits non-training partitions in statistics"
        )
    if schema.get("feature_dim") != 13:
        raise ValueError("AIS feature_dim must be 13")

    counts = {}
    disk_relpaths = set()
    for partition in PARTITIONS:
        files = partition_files(root, partition)
        counts[partition] = len(files)
        if counts[partition] != EXPECTED_COUNTS[partition]:
            raise ValueError(
                f"{partition} count mismatch: {counts[partition]} != "
                f"{EXPECTED_COUNTS[partition]}"
            )
        for path in files:
            disk_relpaths.add(path.relative_to(root).as_posix())
            if not check_every_sample:
                continue
            class_id = CLASS_TO_ID[path.parent.name]
            with np.load(path, allow_pickle=False) as item:
                if set(item.files) != REQUIRED_KEYS:
                    raise ValueError(f"invalid NPZ keys in {path}: {item.files}")
                if item["rgb"].shape != (3, 96, 96) or item["rgb"].dtype != np.uint16:
                    raise ValueError(f"invalid RGB tensor in {path}")
                if item["nir"].shape != (1, 96, 96) or item["nir"].dtype != np.uint16:
                    raise ValueError(f"invalid NIR tensor in {path}")
                if item["ais"].shape != (13,) or item["ais_mask"].shape != (13,):
                    raise ValueError(f"invalid AIS tensor in {path}")
                if not np.isfinite(item["ais"]).all():
                    raise ValueError(f"non-finite AIS values in {path}")
                if not np.isin(item["ais_mask"], (0.0, 1.0)).all():
                    raise ValueError(f"non-binary AIS mask in {path}")
                if int(item["label"]) != class_id:
                    raise ValueError(f"folder/NPZ label mismatch in {path}")

    rows = _manifest_rows(root)
    if len(rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"manifest row count is {len(rows)}, expected 271")
    manifest_relpaths = {row["npz_relpath"].replace("\\", "/") for row in rows}
    if manifest_relpaths != disk_relpaths:
        raise ValueError("manifest NPZ file set does not exactly match samples/")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("manifest sample_id values are not unique")

    overlap = {}
    for field in ("scene", "mmsi"):
        values = {
            partition: {
                row[field]
                for row in rows
                if row["partition"] == partition and row[field].strip()
            }
            for partition in PARTITIONS
        }
        for left, right in combinations(PARTITIONS, 2):
            common = values[left] & values[right]
            overlap[f"{field}:{left}__{right}"] = len(common)
            if common:
                raise ValueError(
                    f"{field} leakage between {left} and {right}: {common}"
                )

    return {
        "status": "PASS",
        "root": str(root),
        "counts": counts,
        "class_names": list(DISPLAY_CLASS_NAMES),
        "train_statistics_partitions": list(TRAIN_PARTITIONS),
        "overlap_counts": overlap,
        "checked_every_sample": bool(check_every_sample),
    }


class VDS2RawDataset(Dataset):
    """Return normalized RGB/NIR, standardized AIS+mask, and label."""

    def __init__(self, root, partition, statistics, image_size=None, augment=False):
        self.root = Path(root).expanduser().resolve()
        self.partition = partition
        self.statistics = statistics
        self.image_size = image_size
        self.augment = augment
        self.paths = partition_files(self.root, partition)
        self.samples = [
            (str(path), CLASS_TO_ID[path.parent.name]) for path in self.paths
        ]
        self.rgb_mean = torch.tensor(statistics.rgb_mean, dtype=torch.float32).view(
            3, 1, 1
        )
        self.rgb_std = torch.tensor(statistics.rgb_std, dtype=torch.float32).view(
            3, 1, 1
        )
        self.nir_mean = torch.tensor(statistics.nir_mean, dtype=torch.float32).view(
            1, 1, 1
        )
        self.nir_std = torch.tensor(statistics.nir_std, dtype=torch.float32).view(
            1, 1, 1
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as item:
            rgb = torch.from_numpy(item["rgb"].astype(np.float32))
            nir = torch.from_numpy(item["nir"].astype(np.float32))
            ais = torch.from_numpy(item["ais"].astype(np.float32))
            ais_mask = torch.from_numpy(item["ais_mask"].astype(np.float32))
            label = int(item["label"])
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        nir = (nir - self.nir_mean) / self.nir_std
        if self.image_size and self.image_size != 96:
            size = (self.image_size, self.image_size)
            rgb = F.interpolate(
                rgb.unsqueeze(0), size=size, mode="bilinear", align_corners=False
            ).squeeze(0)
            nir = F.interpolate(
                nir.unsqueeze(0), size=size, mode="bilinear", align_corners=False
            ).squeeze(0)
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                rgb, nir = rgb.flip(-1), nir.flip(-1)
            if bool(torch.rand(()) < 0.5):
                rgb, nir = rgb.flip(-2), nir.flip(-2)
        return {
            "rgb": rgb,
            "nir": nir,
            "ais": ais,
            "ais_mask": ais_mask,
            "label": label,
            "path": str(path),
        }


class RGBView(Dataset):
    """Pair-table compatible RGB view of :class:`VDS2RawDataset`."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.samples = dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return item["rgb"], item["label"]


class NaturalImageRGBDataset(Dataset):
    """RGB view for ImageNet-initialized CCSA and d-SNE backbones.

    The d-SNE release resizes to 256, crops to 224, applies horizontal flip
    and colour jitter, then normalizes around 0.5.  A vertical flip is added
    for this overhead-imagery dataset; it does not change band semantics.
    """

    def __init__(self, root, partition, statistics, augment=False, normalization="half"):
        self.root = Path(root).expanduser().resolve()
        self.partition = partition
        self.paths = partition_files(self.root, partition)
        self.samples = [
            (str(path), CLASS_TO_ID[path.parent.name]) for path in self.paths
        ]
        self.statistics = statistics
        if normalization == "half":
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif normalization == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise ValueError(f"unknown RGB normalization {normalization!r}")
        if augment:
            spatial = [
                transforms.Resize(256, antialias=True),
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
            ]
        else:
            spatial = [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
            ]
        self.transform = transforms.Compose(
            spatial + [transforms.Normalize(mean=mean, std=std)]
        )
        self.normalization = normalization

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as item:
            rgb = torch.from_numpy(item["rgb"].astype(np.float32))
            label = int(item["label"])
        # More than 98.5% of every training RGB channel remains within the
        # conventional 8-bit image range.  Preserve RGB colour balance with a
        # fixed 255 scale and clip the sparse high-DN outliers, rather than
        # compressing the whole image by a per-channel extreme maximum.
        rgb = rgb.div_(255.0)
        rgb = self.transform(rgb.clamp_(0.0, 1.0))
        return rgb, label


class FixedPerClassSubset(Dataset):
    """Deterministic, class-balanced subset that keeps ``samples`` visible."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(map(int, indices))
        self.samples = [dataset.samples[index] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def fixed_samples_per_class(dataset, samples_per_class, seed):
    """Select a fixed per-class subset, or return all samples when set to zero."""
    import random

    if samples_per_class < 0:
        raise ValueError("samples_per_class cannot be negative")
    if samples_per_class == 0:
        return dataset
    by_class = {}
    for index, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(int(label), []).append(index)
    rng = random.Random(int(seed))
    chosen = []
    for label in sorted(by_class):
        candidates = by_class[label]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"class {label} has {len(candidates)} samples, fewer than "
                f"requested {samples_per_class}"
            )
        chosen.extend(rng.sample(candidates, samples_per_class))
    chosen.sort()
    return FixedPerClassSubset(dataset, chosen)


__all__ = [
    "CLASS_NAMES",
    "DISPLAY_CLASS_NAMES",
    "EXPECTED_COUNTS",
    "PARTITIONS",
    "TRAIN_PARTITIONS",
    "FixedPerClassSubset",
    "ImageStatistics",
    "NaturalImageRGBDataset",
    "RGBView",
    "VDS2RawDataset",
    "audit_dataset",
    "compute_image_statistics",
    "fixed_samples_per_class",
    "load_or_compute_statistics",
    "partition_files",
]
