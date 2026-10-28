"""Shared DataLoader settings for the comparison methods.

Evaluation uses a separate worker budget.  This matters on Windows where every
worker is a fresh ``spawn`` process that imports PyTorch and its CUDA DLLs.  A
source loader, target loader and validation loader with four workers each can
otherwise consume many GiB of committed host memory before doing useful work.
"""

import argparse
import torch


def add_data_loader_args(parser, default_workers, default_eval_workers=0):
    parser.add_argument(
        "--num-workers", type=int, default=default_workers,
        help="DataLoader worker processes (server launcher defaults to 8)",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=4,
        help="batches prefetched by each worker; ignored when num_workers=0",
    )
    parser.add_argument(
        "--eval-num-workers", type=int, default=default_eval_workers,
        help=(
            "validation DataLoader workers; default 0 avoids starting an "
            "additional PyTorch process pool while training workers are alive"
        ),
    )
    parser.add_argument(
        "--worker-timeout-seconds", type=int, default=300,
        help=(
            "fail a multi-process DataLoader when a batch is unavailable for "
            "this many seconds instead of waiting indefinitely"
        ),
    )
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="use page-locked host buffers for CUDA transfers",
    )
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction,
        default=True,
        help="keep DataLoader workers alive between epochs",
    )


def _loader_kwargs(args, workers, persistent_workers):
    workers = int(workers)
    if workers < 0:
        raise ValueError("DataLoader workers must be >= 0")
    if int(args.prefetch_factor) < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    timeout = int(args.worker_timeout_seconds)
    if timeout < 1:
        raise ValueError("--worker-timeout-seconds must be >= 1")
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(args.pin_memory),
    }
    if workers > 0:
        kwargs.update(
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=bool(persistent_workers),
            timeout=timeout,
        )
    return kwargs


def data_loader_kwargs(args):
    """Training-loader arguments."""
    return _loader_kwargs(args, args.num_workers, args.persistent_workers)


def evaluation_loader_kwargs(args):
    """Validation-loader arguments.

    Validation workers are deliberately non-persistent: validation sets are
    small and keeping another process pool alive adds memory without improving
    the optimization path.
    """
    return _loader_kwargs(args, args.eval_num_workers, False)


__all__ = [
    "add_data_loader_args",
    "data_loader_kwargs",
    "evaluation_loader_kwargs",
]
