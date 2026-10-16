"""Small, shared reproducibility helpers for comparison entry points."""

import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed the RNGs used by the training scripts and return the normalized seed.

    This intentionally does not force deterministic CUDA kernels: doing so can
    change supported operations and efficiency measurements.  It does ensure
    that the ``--seed`` argument actually controls Python, NumPy, CPU PyTorch,
    and all visible CUDA generators before datasets, loaders, and models are
    constructed.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


__all__ = ["seed_everything"]
