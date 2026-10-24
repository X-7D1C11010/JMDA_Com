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
        else:
            # d-SNE defines ratio=0 as retaining all negative pairs.
            negatives = negatives
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


__all__ = ['CrossDomainPairDataset']
