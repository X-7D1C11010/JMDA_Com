"""Small, shared reproducibility helpers for comparison entry points."""

import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=None):
    """Seed the RNGs used by the training scripts and return the normalized seed.

    Deterministic CUDA kernels are opt-in via ``JMDA_DETERMINISTIC=1`` (or the
    explicit argument) because they can affect throughput. Formal comparison
    runs enable this option so repeated method/seed runs are comparable.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic is None:
        deterministic = os.environ.get("JMDA_DETERMINISTIC", "0").strip().lower() \
            in {"1", "true", "yes", "on"}
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"Deterministic kernels: {bool(deterministic)}")
    return seed


__all__ = ["seed_everything"]
