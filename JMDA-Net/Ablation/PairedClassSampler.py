"""Epoch-covering supervised source/target class pairing."""

import math
import random
from collections import defaultdict

import torch


def _batch_ranges_without_singleton(sample_count, batch_size):
    """Partition an epoch without emitting a final one-sample batch.

    The neural-OT correction network contains BatchNorm and therefore cannot
    train on a singleton batch.  Preserve every balanced observation by
    merging a one-sample tail into the preceding batch instead of dropping it.
    """
    sample_count = int(sample_count)
    batch_size = int(batch_size)
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    starts = list(range(0, sample_count, batch_size))
    if len(starts) > 1 and sample_count - starts[-1] == 1:
        starts.pop()
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else sample_count)
        for index, start in enumerate(starts)
    ]


class PairedClassSampler:
    """Cover every target sample once and pair it with same-class source data.

    The previous sampler drew both domains with replacement.  With the small
    target weather sets this left roughly one third of the labeled target
    observations unseen in an epoch.  Target indices are now shuffled without
    replacement; source indices cycle within the corresponding class.
    """

    def __init__(self, src_ds, tgt_ds, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = int(batch_size)
        self.src_indices = self._build_index(src_ds)
        self.tgt_indices = self._build_index(tgt_ds)
        self.classes = sorted(set(self.src_indices) & set(self.tgt_indices))
        if not self.classes:
            raise ValueError("Source and target domains have no common classes.")

        missing_target_classes = sorted(set(self.tgt_indices) - set(self.src_indices))
        if missing_target_classes:
            raise ValueError(
                "Supervised pairing cannot cover target-only classes: "
                f"{missing_target_classes}"
            )
        self.target_indices = [
            index
            for class_label in self.classes
            for index in self.tgt_indices[class_label]
        ]
        self.samples_per_class = max(
            len(self.tgt_indices[class_label]) for class_label in self.classes
        )
        self.epoch_sample_count = self.samples_per_class * len(self.classes)
        print(
            "配对采样器已初始化，共 "
            f"{len(self.classes)} 个共有类别；每个epoch覆盖全部 "
            f"{len(self.target_indices)} 个目标样本，并按类别平衡扩展为 "
            f"{self.epoch_sample_count} 个训练观测。"
        )

    @staticmethod
    def _build_index(dataset):
        indices = defaultdict(list)
        for index, label in enumerate(dataset.labels):
            indices[int(label)].append(index)
        return indices

    def __iter__(self):
        target_order = []
        for class_label in self.classes:
            pool = list(self.tgt_indices[class_label])
            random.shuffle(pool)
            repeats, remainder = divmod(self.samples_per_class, len(pool))
            target_order.extend(pool * repeats + pool[:remainder])
        random.shuffle(target_order)

        source_pools = {}
        source_offsets = {}
        for class_label in self.classes:
            pool = list(self.src_indices[class_label])
            random.shuffle(pool)
            source_pools[class_label] = pool
            source_offsets[class_label] = 0

        for start, end in _batch_ranges_without_singleton(
            len(target_order), self.batch_size
        ):
            target_batch_indices = target_order[start:end]
            source_batch_indices = []
            for target_index in target_batch_indices:
                class_label = int(self.tgt_ds.labels[target_index])
                pool = source_pools[class_label]
                offset = source_offsets[class_label]
                if offset >= len(pool):
                    random.shuffle(pool)
                    offset = 0
                source_batch_indices.append(pool[offset])
                source_offsets[class_label] = offset + 1

            source_batch = torch.utils.data.default_collate(
                [self.src_ds[index] for index in source_batch_indices]
            )
            target_batch = torch.utils.data.default_collate(
                [self.tgt_ds[index] for index in target_batch_indices]
            )
            target_batch["sample_index"] = torch.tensor(
                target_batch_indices,
                dtype=torch.long,
            )
            yield source_batch, target_batch

    def __len__(self):
        return len(
            _batch_ranges_without_singleton(
                self.epoch_sample_count,
                self.batch_size,
            )
        )


class PseudoPairedClassSampler:
    """Class-balanced UDA pairing driven only by target pseudo labels.

    Target ground-truth labels are deliberately never inspected here.  Source
    labels are mapped into the classifier's contiguous label space and paired
    with externally inferred target pseudo labels in that same space.  Every
    returned source/target row therefore has the same *pseudo* class, which is
    the label-free analogue of :class:`PairedClassSampler`.
    """

    def __init__(
        self,
        src_ds,
        tgt_ds,
        target_pseudo_labels,
        batch_size,
        expected_classes=None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        pseudo_labels = torch.as_tensor(
            target_pseudo_labels,
            dtype=torch.long,
        ).reshape(-1).cpu()
        if pseudo_labels.numel() != len(tgt_ds):
            raise ValueError(
                "target_pseudo_labels must contain exactly one label per "
                f"target sample: expected {len(tgt_ds)}, found "
                f"{pseudo_labels.numel()}."
            )
        if pseudo_labels.numel() == 0 or int(pseudo_labels.min()) < 0:
            raise ValueError("target_pseudo_labels must be non-empty and non-negative.")

        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = int(batch_size)
        self.target_pseudo_labels = pseudo_labels
        self.src_indices = self._build_mapped_source_index(src_ds)
        self.tgt_indices = defaultdict(list)
        for index, label in enumerate(pseudo_labels.tolist()):
            self.tgt_indices[int(label)].append(index)

        self.classes = sorted(self.tgt_indices)
        if expected_classes is not None:
            expected = sorted({int(label) for label in expected_classes})
            if self.classes != expected:
                raise ValueError(
                    "Pseudo-label assignment must cover the complete configured "
                    f"class space: expected {expected}, found {self.classes}."
                )
        missing_source_classes = sorted(set(self.classes) - set(self.src_indices))
        if missing_source_classes:
            raise ValueError(
                "Pseudo-paired UDA cannot cover target pseudo classes absent "
                f"from the source domain: {missing_source_classes}."
            )

        self.samples_per_class = max(
            len(self.tgt_indices[class_label]) for class_label in self.classes
        )
        self.epoch_sample_count = self.samples_per_class * len(self.classes)

    @staticmethod
    def _build_mapped_source_index(dataset):
        indices = defaultdict(list)
        for index, original_label in enumerate(dataset.labels):
            try:
                mapped_label = int(dataset.label_map[int(original_label)])
            except KeyError as error:
                raise ValueError(
                    "Source sample label is absent from the configured label map: "
                    f"{original_label}."
                ) from error
            indices[mapped_label].append(index)
        return indices

    def __iter__(self):
        target_order = []
        for class_label in self.classes:
            pool = list(self.tgt_indices[class_label])
            random.shuffle(pool)
            repeats, remainder = divmod(self.samples_per_class, len(pool))
            target_order.extend(pool * repeats + pool[:remainder])
        random.shuffle(target_order)

        source_pools = {}
        source_offsets = {}
        for class_label in self.classes:
            pool = list(self.src_indices[class_label])
            random.shuffle(pool)
            source_pools[class_label] = pool
            source_offsets[class_label] = 0

        for start, end in _batch_ranges_without_singleton(
            len(target_order), self.batch_size
        ):
            target_batch_indices = target_order[start:end]
            target_batch_pseudo_labels = [
                int(self.target_pseudo_labels[index])
                for index in target_batch_indices
            ]
            source_batch_indices = []
            for class_label in target_batch_pseudo_labels:
                pool = source_pools[class_label]
                offset = source_offsets[class_label]
                if offset >= len(pool):
                    random.shuffle(pool)
                    offset = 0
                source_batch_indices.append(pool[offset])
                source_offsets[class_label] = offset + 1

            source_batch = torch.utils.data.default_collate(
                [self.src_ds[index] for index in source_batch_indices]
            )
            target_batch = torch.utils.data.default_collate(
                [self.tgt_ds[index] for index in target_batch_indices]
            )
            # Defense in depth: UDA target datasets are configured not to
            # return labels at all.  Keep this removal so a future dataset
            # implementation cannot silently re-expose target ground truth.
            target_batch.pop("label", None)
            target_batch["sample_index"] = torch.tensor(
                target_batch_indices,
                dtype=torch.long,
            )
            target_batch["pseudo_label"] = torch.tensor(
                target_batch_pseudo_labels,
                dtype=torch.long,
            )
            # This is a protocol invariant, not an accuracy claim: the paired
            # source labels must equal the target pseudo labels row by row.
            if not torch.equal(source_batch["label"], target_batch["pseudo_label"]):
                raise RuntimeError(
                    "Pseudo-class sampler produced a cross-class source/target pair."
                )
            yield source_batch, target_batch

    def __len__(self):
        return len(
            _batch_ranges_without_singleton(
                self.epoch_sample_count,
                self.batch_size,
            )
        )


__all__ = ["PairedClassSampler", "PseudoPairedClassSampler"]
