"""Cross-domain pair tables used by the official CCSA and d-SNE recipes.

Both releases build a labelled source/target pair table before training.  The
table contains every positive (same-class) pair and a configurable number of
randomly selected negative pairs.  Keeping this logic in one small module
prevents the two ports from silently drifting to independently shuffled domain
mini-batches, which is not the released training protocol.
"""

import random

from torch.utils.data import Dataset


class CrossDomainPairDataset(Dataset):
    """Materialise the released exhaustive positive/negative pair protocol.

    ``source`` and ``target`` must expose ``samples`` as ``(path, label)`` and
    return ``(image, label)`` from ``__getitem__``.  ``negative_ratio=3`` is
    CCSA's released setting; ``negative_ratio=1`` is d-SNE's default.
    """

    def __init__(self, source, target, negative_ratio, seed=0):
        if negative_ratio < 0:
            raise ValueError('negative_ratio must be non-negative')
        if not hasattr(source, 'samples') or not hasattr(target, 'samples'):
            raise TypeError('source and target datasets must expose samples')

        positives = []
        negatives = []
        for source_index, (_, source_label) in enumerate(source.samples):
            for target_index, (_, target_label) in enumerate(target.samples):
                item = (source_index, target_index, int(source_label == target_label))
                (positives if item[2] else negatives).append(item)

        if not positives:
            raise ValueError('cross-domain pair table contains no positive pair')

        rng = random.Random(int(seed))
        rng.shuffle(negatives)
        if negative_ratio:
            negatives = negatives[:min(len(negatives), negative_ratio * len(positives))]
        # d-SNE defines ratio=0 as retaining all negative pairs.
        self.pairs = positives + negatives
        rng.shuffle(self.pairs)
        self.source = source
        self.target = target
        self.positive_pairs = len(positives)
        self.negative_pairs = len(negatives)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source_index, target_index, pair_label = self.pairs[index]
        source_image, source_label = self.source[source_index]
        target_image, target_label = self.target[target_index]
        return source_image, source_label, target_image, target_label, pair_label


class ControlledCrossDomainPairDataset(Dataset):
    """Epoch-resampled, class-controlled CCSA pair table.

    Pair counts are fixed and split evenly across target classes.  Within a
    class, target examples are traversed round-robin before any is reused, so
    no target item is accidentally repeated against the whole source set.
    Positive and negative pair definitions are unchanged from CCSA.
    """

    def __init__(
        self,
        source,
        target,
        pairs_per_epoch,
        negative_ratio=3,
        seed=0,
    ):
        if pairs_per_epoch <= 0:
            raise ValueError('pairs_per_epoch must be positive')
        if negative_ratio <= 0:
            raise ValueError('negative_ratio must be positive')
        if not hasattr(source, 'samples') or not hasattr(target, 'samples'):
            raise TypeError('source and target datasets must expose samples')
        denominator = negative_ratio + 1
        if pairs_per_epoch % denominator:
            raise ValueError(
                'pairs_per_epoch must be divisible by negative_ratio + 1'
            )

        self.source = source
        self.target = target
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.negative_ratio = int(negative_ratio)
        self.seed = int(seed)
        self.source_by_class = self._group(source.samples)
        self.target_by_class = self._group(target.samples)
        if set(self.target_by_class) - set(self.source_by_class):
            raise ValueError('every target class must be present in source')
        self.classes = tuple(sorted(self.target_by_class))
        positive_count = self.pairs_per_epoch // denominator
        negative_count = self.pairs_per_epoch - positive_count
        if positive_count % len(self.classes) or negative_count % len(self.classes):
            raise ValueError(
                'positive and negative pair counts must divide evenly across classes'
            )
        self.positive_pairs = positive_count
        self.negative_pairs = negative_count
        self.pairs = []
        self.set_epoch(0)

    @staticmethod
    def _group(samples):
        grouped = {}
        for index, (_, label) in enumerate(samples):
            grouped.setdefault(int(label), []).append(index)
        return grouped

    @staticmethod
    def _balanced_pairs(rng, target_indices, source_indices, count, pair_label):
        targets = list(target_indices)
        sources = list(source_indices)
        rng.shuffle(targets)
        rng.shuffle(sources)
        pairs = []
        for offset in range(count):
            if offset and offset % len(targets) == 0:
                rng.shuffle(targets)
            if offset and offset % len(sources) == 0:
                rng.shuffle(sources)
            pairs.append(
                (
                    sources[offset % len(sources)],
                    targets[offset % len(targets)],
                    pair_label,
                )
            )
        return pairs

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + int(epoch) * 1000003)
        positives_per_class = self.positive_pairs // len(self.classes)
        negatives_per_class = self.negative_pairs // len(self.classes)
        pairs = []
        for label in self.classes:
            target_indices = self.target_by_class[label]
            pairs.extend(
                self._balanced_pairs(
                    rng,
                    target_indices,
                    self.source_by_class[label],
                    positives_per_class,
                    1,
                )
            )
            negative_sources = [
                index
                for source_label, indices in self.source_by_class.items()
                if source_label != label
                for index in indices
            ]
            pairs.extend(
                self._balanced_pairs(
                    rng,
                    target_indices,
                    negative_sources,
                    negatives_per_class,
                    0,
                )
            )
        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source_index, target_index, pair_label = self.pairs[index]
        source_image, source_label = self.source[source_index]
        target_image, target_label = self.target[target_index]
        return source_image, source_label, target_image, target_label, pair_label


__all__ = ['ControlledCrossDomainPairDataset', 'CrossDomainPairDataset']
