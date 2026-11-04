"""Small recursive helpers for tensor or multimodal mini-batches."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return type(batch)((key, batch_to_device(value, device)) for key, value in batch.items())
    if isinstance(batch, tuple):
        return tuple(batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [batch_to_device(value, device) for value in batch]
    raise TypeError(f"unsupported mini-batch container: {type(batch).__name__}")


def batch_slice(batch, start, stop):
    if torch.is_tensor(batch):
        return batch[start:stop]
    if isinstance(batch, Mapping):
        return type(batch)(
            (key, batch_slice(value, start, stop)) for key, value in batch.items()
        )
    if isinstance(batch, tuple):
        return tuple(batch_slice(value, start, stop) for value in batch)
    if isinstance(batch, list):
        return [batch_slice(value, start, stop) for value in batch]
    raise TypeError(f"unsupported mini-batch container: {type(batch).__name__}")


def batch_size(batch):
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        if not batch:
            raise ValueError("multimodal mini-batch cannot be empty")
        sizes = {batch_size(value) for value in batch.values()}
    elif isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("multimodal mini-batch cannot be empty")
        sizes = {batch_size(value) for value in batch}
    else:
        raise TypeError(f"unsupported mini-batch container: {type(batch).__name__}")
    if len(sizes) != 1:
        raise ValueError(f"inconsistent modality batch sizes: {sorted(sizes)}")
    return sizes.pop()


__all__ = ["batch_size", "batch_slice", "batch_to_device"]
