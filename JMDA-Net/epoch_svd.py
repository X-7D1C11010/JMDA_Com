"""Epoch-boundary feature collection for Tensor SVD projection updates."""

from collections import defaultdict
import numpy as np
import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _seed_data_worker(worker_id):
    """Seed Python and NumPy from the worker seed assigned by DataLoader."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class _PairedFeatureDataset(Dataset):
    """Expose fixed source/target index pairs to worker processes."""

    def __init__(self, source_dataset, target_dataset, pairs):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source_index, target_index = self.pairs[index]
        return self.source_dataset[source_index], self.target_dataset[target_index]


def _loader_options(batch_size, num_workers, prefetch_factor, pin_memory, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_data_worker,
        "generator": generator,
    }
    if num_workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return options


def _indices_by_class(dataset, labels_override=None) -> Dict[int, List[int]]:
    labels = dataset.labels if labels_override is None else labels_override
    if len(labels) != len(dataset):
        raise ValueError(
            "labels_override must contain exactly one class per dataset sample."
        )
    indices: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices[int(label)].append(index)
    return indices


def build_epoch_pairs(
    source_dataset,
    target_dataset,
    seed: int,
    class_paired: bool,
    balance_classes: bool = False,
    target_labels_override=None,
) -> List[Tuple[int, int]]:
    """Return deterministic pairs while covering each target sample once.

    In supervised adaptation, every target sample is paired with a source
    sample of the same class.  Source samples cycle within a class when the
    domain sizes differ.  The unpaired fallback never inspects class labels.
    """
    rng = random.Random(int(seed))

    if not class_paired:
        source_indices = list(range(len(source_dataset)))
        target_indices = list(range(len(target_dataset)))
        if not source_indices or not target_indices:
            raise ValueError("Source and target datasets must both be non-empty.")
        rng.shuffle(source_indices)
        rng.shuffle(target_indices)
        return [
            (source_indices[position % len(source_indices)], target_index)
            for position, target_index in enumerate(target_indices)
        ]

    source_by_class = _indices_by_class(source_dataset)
    target_by_class = _indices_by_class(
        target_dataset,
        labels_override=target_labels_override,
    )
    target_classes = set(target_by_class)
    missing_classes = sorted(target_classes - set(source_by_class))
    if missing_classes:
        raise ValueError(
            "Cannot construct supervised SVD statistics because target "
            f"classes are absent from the source domain: {missing_classes}."
        )

    pairs: List[Tuple[int, int]] = []
    samples_per_class = (
        max(len(indices) for indices in target_by_class.values())
        if balance_classes
        else None
    )
    for class_label in sorted(target_classes):
        source_indices = list(source_by_class[class_label])
        target_indices = list(target_by_class[class_label])
        rng.shuffle(source_indices)
        rng.shuffle(target_indices)
        if samples_per_class is not None:
            repeats, remainder = divmod(samples_per_class, len(target_indices))
            target_indices = (
                target_indices * repeats + target_indices[:remainder]
            )
        pairs.extend(
            (
                source_indices[position % len(source_indices)],
                target_index,
            )
            for position, target_index in enumerate(target_indices)
        )

    rng.shuffle(pairs)
    if not pairs:
        raise ValueError("No source-target pairs were available for the SVD update.")
    return pairs


@torch.no_grad()
def update_epoch_projections(
    tal_module,
    encoders: Sequence[torch.nn.Module],
    modality_keys: Sequence[str],
    source_dataset,
    target_dataset,
    batch_size: int,
    device: torch.device,
    seed: int,
    class_paired: bool = True,
    balance_classes: bool = False,
    target_labels_override=None,
    source_class_weights: torch.Tensor = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
):
    """Extract complete feature banks and perform one projection update.

    Supervised adaptation retains the original same-class paired statistics.
    Unsupervised adaptation collects each domain independently and lets the
    Tensor module infer an all-pairs soft transport correspondence; it never
    constructs an arbitrary positional source-target pairing.
    """
    if len(encoders) != len(modality_keys):
        raise ValueError("encoders and modality_keys must have the same length.")
    if len(encoders) != tal_module.num_modalities:
        raise ValueError(
            "The number of encoders must match tal_module.num_modalities."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive.")
    non_blocking = bool(pin_memory and device.type == "cuda")

    previous_training_states = [encoder.training for encoder in encoders]
    for encoder in encoders:
        encoder.eval()

    source_features: List[List[torch.Tensor]] = [[] for _ in encoders]
    target_features: List[List[torch.Tensor]] = [[] for _ in encoders]
    source_labels: List[torch.Tensor] = []

    try:
        if class_paired:
            pairs = build_epoch_pairs(
                source_dataset,
                target_dataset,
                seed=seed,
                class_paired=True,
                balance_classes=balance_classes,
                target_labels_override=target_labels_override,
            )
            paired_loader = DataLoader(
                _PairedFeatureDataset(source_dataset, target_dataset, pairs),
                **_loader_options(
                    batch_size,
                    num_workers,
                    prefetch_factor,
                    pin_memory,
                    seed,
                ),
            )
            for source_batch, target_batch in paired_loader:
                for mode, (encoder, key) in enumerate(
                    zip(encoders, modality_keys)
                ):
                    source_input = source_batch[key].to(
                        device, non_blocking=non_blocking
                    )
                    target_input = target_batch[key].to(
                        device, non_blocking=non_blocking
                    )
                    source_features[mode].append(encoder(source_input).float())
                    target_features[mode].append(encoder(target_input).float())
        else:
            if len(source_dataset) == 0 or len(target_dataset) == 0:
                raise ValueError("Source and target datasets must both be non-empty.")
            for dataset, feature_chunks, keep_labels in (
                (source_dataset, source_features, True),
                (target_dataset, target_features, False),
            ):
                loader_seed = seed if keep_labels else seed + 1
                loader = DataLoader(
                    dataset,
                    **_loader_options(
                        batch_size,
                        num_workers,
                        prefetch_factor,
                        pin_memory,
                        loader_seed,
                    ),
                )
                for batch in loader:
                    for mode, (encoder, key) in enumerate(
                        zip(encoders, modality_keys)
                    ):
                        inputs = batch[key].to(
                            device, non_blocking=non_blocking
                        )
                        feature_chunks[mode].append(encoder(inputs).float())
                    if keep_labels:
                        source_labels.append(
                            batch["label"].to(
                                device, non_blocking=non_blocking
                            )
                        )
    finally:
        for encoder, was_training in zip(encoders, previous_training_states):
            encoder.train(was_training)

    source_bank = [torch.cat(chunks, dim=0) for chunks in source_features]
    target_bank = [torch.cat(chunks, dim=0) for chunks in target_features]
    if class_paired:
        update_info = tal_module.update_projections(source_bank, target_bank)
    else:
        source_weights = None
        if source_class_weights is not None:
            class_weights = source_class_weights.to(device=device)
            labels = torch.cat(source_labels, dim=0).long()
            if labels.min().item() < 0 or labels.max().item() >= class_weights.numel():
                raise ValueError("Source labels exceed source_class_weights.")
            # Convert a desired class marginal into per-observation mass.  A
            # direct lookup would multiply every class by its source sample
            # count once more and would therefore not realize the requested
            # partial-set prior.
            class_counts = torch.bincount(
                labels, minlength=class_weights.numel()
            ).to(dtype=class_weights.dtype).clamp_min(1.0)
            source_weights = class_weights[labels] / class_counts[labels]
        update_info = tal_module.update_unpaired_projections(
            source_bank,
            target_bank,
            source_weights=source_weights,
        )
    update_info["class_paired"] = bool(class_paired)
    update_info["class_balanced"] = bool(class_paired and balance_classes)
    update_info["target_labels_source"] = (
        "pseudo" if target_labels_override is not None else "ground_truth"
    ) if class_paired else "none"
    return update_info


def format_svd_update(update_info: Dict[str, object]) -> str:
    """Create a compact, reproducible projection-update log line."""
    relative_change = float(update_info["relative_singular_change"])
    change_text = "inf" if relative_change == float("inf") else f"{relative_change:.3e}"
    basis_text = ""
    if "source_subspace_overlap" in update_info:
        basis_text = (
            f", basis_overlap_src/tgt="
            f"{update_info['source_subspace_overlap']:.3f}/"
            f"{update_info['target_subspace_overlap']:.3f}, "
            f"basis_min_cos_src/tgt="
            f"{update_info.get('source_min_principal_cosine', float('nan')):.3f}/"
            f"{update_info.get('target_min_principal_cosine', float('nan')):.3f}, "
            f"basis_nonorth={update_info['basis_map_non_orthogonality']:.3e}, "
            f"basis_nonorth_src/tgt="
            f"{update_info.get('source_basis_map_non_orthogonality', float('nan')):.3e}/"
            f"{update_info.get('target_basis_map_non_orthogonality', float('nan')):.3e}, "
            f"transport_nonorth="
            f"{update_info.get('shared_transport_non_orthogonality', float('nan')):.3e}, "
            f"basis_domain_gap="
            f"{update_info.get('source_target_basis_map_gap', float('nan')):.3e}, "
            f"shared_transport_ranks="
            f"{update_info.get('shared_transport_effective_ranks', ())}, "
            f"shared_transport_rank_thresholds="
            f"{update_info.get('shared_transport_rank_thresholds', ())}, "
            f"classifier_basis={update_info.get('classifier_basis_mode', 'shared')}, "
            f"basis_transport={update_info.get('basis_transport_mode', 'unknown')}"
        )
    plan_text = ""
    if "unlabeled_plan_marginal_residual" in update_info:
        plan_text = (
            f", unlabeled_plan_residual="
            f"{update_info['unlabeled_plan_marginal_residual']:.2e}, "
            f"unlabeled_plan_entropy="
            f"{update_info['unlabeled_plan_entropy']:.4f}"
        )
    return (
        f"SVD update #{update_info['update_count']}: "
        f"pairs={update_info['sample_count']}, sweeps={update_info['sweeps']}, "
        f"converged={update_info['converged']}, rel_change={change_text}, "
        f"effective_ranks={update_info['effective_ranks']}, "
        f"stabilized_null_dims={update_info.get('stabilized_null_dims', ())}, "
        f"rank_thresholds={update_info.get('rank_thresholds', ())}, "
        f"class_paired={update_info['class_paired']}, "
        f"class_balanced={update_info.get('class_balanced', False)}, "
        f"target_labels_source={update_info.get('target_labels_source', 'unknown')}"
        f"{basis_text}{plan_text}"
    )


__all__ = [
    "build_epoch_pairs",
    "update_epoch_projections",
    "format_svd_update",
]
