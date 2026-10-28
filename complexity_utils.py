"""Unified model complexity utilities for five UDA baselines on weather / M4-SAR data.

Metrics captured (all numbers are for the ORIGINAL training flow; nothing in the
methods' core logic or hyper-parameters is modified):

  1. Params:  total trainable + frozen parameters (count(*)).
  2. FLOPs / MACs: thop.profile(model, inputs=(...)) * 2  ->  single forward pass
                  FLOPs; then multiplied by (batches_per_epoch * 2, accounting
                  for backward roughly ~ 2x forward for non-trivial nets) to
                  give ``flops_per_epoch_approx``.  Raw single-pass FLOPs are
                  always reported as ``flops_single_forward``.  Methods whose
                  train-time forward differs from inference-time (e.g. SCORE
                  freezes the encoder but still runs it during both) use the
                  training-time forward signature.
  3. Total training wall-clock time:  time.perf_counter() around the whole
      outer loop (all epochs, all runs if multi-run).  Per-epoch breakdown
      also recorded.
  4. Inference throughput (samples / sec): model.eval(), warm-up then 50
      batches timed with CUDA events.
  5. Peak GPU memory (MB): torch.cuda.reset_peak_memory_stats() at the start
      of each epoch / inference run; max_memory_allocated() at the end.

Usage (in a training script):

    from complexity_utils import ComplexityTracker
    tracker = ComplexityTracker(device='cuda', log_dir=...)

    # after model is fully constructed:
    tracker.capture_model(model, dummy_inputs, tag='train-time')

    # training loop - wrap each epoch:
    tracker.epoch_start()
    train_one_epoch(...)
    tracker.epoch_end()

    # after training:
    tracker.capture_inference(model, val_loader)
    tracker.finalize()
    tracker.save('outputs/complexity.json')

The module is deliberately dependency-light: PyTorch + thop (pip install thop).
If thop is unavailable, FLOPs fields are set to None and no crash occurs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _try_profile_flops(
    model: nn.Module,
    inputs: Union[torch.Tensor, Tuple[Any, ...]],
    model_name: str = "",
) -> Optional[float]:
    """Return FLOPs for a SINGLE forward pass, or None if thop missing/fails.

    Multiplies thop's MACs by 2 (MAC -> FLOP convention).
    """
    try:
        import thop  # type: ignore
    except Exception:
        return None

    # thop expects a tuple even for a single tensor
    if isinstance(inputs, torch.Tensor):
        args = (inputs,)
    else:
        args = tuple(inputs)

    try:
        model.eval()
        with torch.no_grad():
            macs, _params = thop.profile(
                model, inputs=args, verbose=False
            )
        return float(macs) * 2.0  # FLOPs
    except Exception as e:
        sys.stderr.write(
            f"[complexity_utils] thop.profile FAILED for {model_name}: {e!r}\n"
        )
        return None


def _format_count(n: float) -> str:
    """Return a short human form like '1.23 G' or '456 M'."""
    if n is None:
        return "N/A"
    for unit, div in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n/div:.3f} {unit}"
    return f"{n:.1f}"


# --------------------------------------------------------------------------- #
#  tracker
# --------------------------------------------------------------------------- #

@dataclass
class EpochTiming:
    epoch: int
    elapsed_sec: float
    peak_mem_train_mb: float
    peak_mem_infer_mb: float


@dataclass
class ComplexityReport:
    method: str = ""
    dataset: str = ""
    model_tag: str = ""

    # params
    params_total: int = 0
    params_trainable: int = 0

    # FLOPs (single forward pass during training)
    flops_single_forward: Optional[float] = None
    flops_per_epoch_approx: Optional[float] = None
    macs_single_forward: Optional[float] = None

    # batch config used for flops_per_epoch_approx
    batches_per_epoch: Optional[int] = None
    batch_size: Optional[int] = None

    # training wall-clock
    total_training_sec: float = 0.0
    num_epochs: int = 0
    num_runs: int = 1
    per_epoch_timings: List[Dict[str, float]] = field(default_factory=list)

    # peak GPU memory
    peak_train_mem_mb: float = 0.0
    peak_infer_mem_mb: float = 0.0

    # inference speed
    infer_samples_per_sec: Optional[float] = None
    infer_batch_size: Optional[int] = None
    infer_warmup_batches: int = 10
    infer_timed_batches: int = 50
    infer_total_sec: float = 0.0
    infer_total_samples: int = 0

    # misc
    notes: Dict[str, Any] = field(default_factory=dict)

    # convenience getters for pretty printing
    @property
    def params_total_str(self) -> str:   return _format_count(self.params_total)
    @property
    def flops_single_str(self) -> str:   return _format_count(self.flops_single_forward)
    @property
    def flops_epoch_str(self) -> str:    return _format_count(self.flops_per_epoch_approx)
    @property
    def total_time_str(self) -> str:     return f"{self.total_training_sec:.1f} s ({self.total_training_sec/60:.1f} min)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComplexityTracker:
    """Thin stateful wrapper that owns a ComplexityReport and exposes hooks.

    Only CUDA-specific behaviour is guarded with torch.cuda.is_available() so
    the same code path works on CPU for correctness checks.
    """

    def __init__(
        self,
        method: str = "unknown",
        dataset: str = "unknown",
        device: Union[str, torch.device] = "cuda",
        log_dir: Optional[str] = None,
    ):
        self.report = ComplexityReport(method=method, dataset=dataset)
        self._device = torch.device(device) if isinstance(device, str) else device
        self._cuda_ok = self._device.type == "cuda" and torch.cuda.is_available()

        # epoch state
        self._epoch_start_ts: Optional[float] = None
        self._epoch_timings: List[Dict[str, float]] = []
        self._epoch_count = 0
        self._train_mem_peak_mb = 0.0

        # timing guard
        self._total_train_start: Optional[float] = None
        self._total_train_end: Optional[float] = None

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir

    # --- model capture ------------------------------------------------------

    def capture_model(
        self,
        model: nn.Module,
        dummy_inputs: Union[torch.Tensor, Tuple[Any, ...]],
        model_tag: str = "main",
        batches_per_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Run once after construction; records Params + single-forward FLOPs."""
        self.report.model_tag = model_tag
        self.report.params_total = _count_params(model)
        self.report.params_trainable = _count_trainable_params(model)

        flops = _try_profile_flops(model, dummy_inputs, model_name=model_tag)
        self.report.flops_single_forward = flops
        if flops is not None:
            self.report.macs_single_forward = flops / 2.0

        self.report.batches_per_epoch = batches_per_epoch
        self.report.batch_size = batch_size
        if flops is not None and batches_per_epoch is not None:
            # backward ~ 2x forward for most nets
            self.report.flops_per_epoch_approx = flops * batches_per_epoch * 3

        _print(
            f"[complexity] {self.report.method}/{model_tag}: "
            f"Params={self.report.params_total_str}, "
            f"FLOPs/forward={self.report.flops_single_str}, "
            f"FLOPs/epoch≈{self.report.flops_epoch_str}"
        )

    # --- epoch hooks -------------------------------------------------------

    def training_start(self) -> None:
        self._total_train_start = time.perf_counter()

    def training_end(self) -> None:
        self._total_train_end = time.perf_counter()
        if self._total_train_start is not None and self._total_train_end is not None:
            self.report.total_training_sec += (
                self._total_train_end - self._total_train_start
            )

    def epoch_start(self) -> None:
        if self._cuda_ok:
            torch.cuda.reset_peak_memory_stats(self._device)
            # release any stale allocations that won't be reused
            torch.cuda.empty_cache()
        self._epoch_start_ts = time.perf_counter()

    def epoch_end(self) -> None:
        if self._epoch_start_ts is None:
            return
        elapsed = time.perf_counter() - self._epoch_start_ts
        self._epoch_count += 1

        peak_mb = 0.0
        if self._cuda_ok:
            peak_mb = torch.cuda.max_memory_allocated(self._device) / 1024**2
        self._train_mem_peak_mb = max(self._train_mem_peak_mb, peak_mb)

        self._epoch_timings.append({
            "epoch": self._epoch_count,
            "elapsed_sec": elapsed,
            "peak_mem_mb": peak_mb,
        })

    # --- multi-run support -------------------------------------------------

    def new_run(self, run_index: int) -> None:
        """Call before each independent run in N-run experiments."""
        self.report.num_runs = max(self.report.num_runs, run_index + 1)
        self._epoch_start_ts = None

    # --- inference benchmark ----------------------------------------------

    def capture_inference(
        self,
        model: nn.Module,
        dataloader: Iterable,
        num_batches: Optional[int] = None,
        warmup_batches: int = 10,
        timed_batches: int = 50,
        run_on_gpu: bool = True,
    ) -> None:
        """Run model.eval() through dataloader once to measure throughput."""
        model.eval()
        if run_on_gpu and self._cuda_ok:
            model.to(self._device)
        else:
            self._device = torch.device("cpu")
            model.to(self._device)
        device = self._device

        loader_iter = iter(dataloader)
        if num_batches is None:
            num_batches = warmup_batches + timed_batches

        # figure out the collate format: expect dict or tuple
        def _to_device(batch):
            if isinstance(batch, dict):
                return {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
            if isinstance(batch, (list, tuple)):
                return tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
            return batch.to(device) if torch.is_tensor(batch) else batch

        # warmup
        with torch.no_grad():
            for _ in range(warmup_batches):
                try:
                    batch = _to_device(next(loader_iter))
                except StopIteration:
                    loader_iter = iter(dataloader)
                    batch = _to_device(next(loader_iter))
                _run_forward(model, batch)

        if self._cuda_ok:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        start = time.perf_counter()
        total_samples = 0
        with torch.no_grad():
            for _ in range(timed_batches):
                try:
                    batch = _to_device(next(loader_iter))
                except StopIteration:
                    loader_iter = iter(dataloader)
                    batch = _to_device(next(loader_iter))
                out = _run_forward(model, batch)
                bsize = _infer_batch_size(out, batch)
                total_samples += bsize

        if self._cuda_ok:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        peak_mb = 0.0
        if self._cuda_ok:
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2

        self.report.infer_total_sec += elapsed
        self.report.infer_total_samples += total_samples
        if elapsed > 0:
            self.report.infer_samples_per_sec = (
                self.report.infer_total_samples / self.report.infer_total_sec
            )
        self.report.infer_warmup_batches = warmup_batches
        self.report.infer_timed_batches = timed_batches
        self.report.peak_infer_mem_mb = max(self.report.peak_infer_mem_mb, peak_mb)

        # infer batch_size - try to pull from first batch
        try:
            loader_iter2 = iter(dataloader)
            first = _to_device(next(loader_iter2))
            bsize_guess = _infer_batch_size(None, first)
            self.report.infer_batch_size = bsize_guess
        except Exception:
            pass

        _print(
            f"[complexity] inference: "
            f"{self.report.infer_samples_per_sec:.1f} samples/sec, "
            f"peak GPU {peak_mb:.1f} MB"
        )

    # --- finalise ----------------------------------------------------------

    def finalize(self) -> None:
        self.report.num_epochs = self._epoch_count
        self.report.peak_train_mem_mb = self._train_mem_peak_mb
        self.report.per_epoch_timings = self._epoch_timings
        # if total_training_sec still 0 but per-epoch timings exist, derive
        if self.report.total_training_sec == 0.0 and self._epoch_timings:
            self.report.total_training_sec = sum(
                e["elapsed_sec"] for e in self._epoch_timings
            )
        _print(self._summary_line())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, indent=2, default=str)
        _print(f"[complexity] saved -> {path}")

    def _summary_line(self) -> str:
        r = self.report
        return (
            f"[complexity] SUMMARY {r.method}@{r.dataset} "
            f"Params={r.params_total_str} "
            f"FLOPs/forward={r.flops_single_str} "
            f"FLOPs/epoch≈{r.flops_epoch_str} "
            f"train={r.total_time_str} "
            f"peak_train_mem={r.peak_train_mem_mb:.1f}MB "
            f"peak_infer_mem={r.peak_infer_mem_mb:.1f}MB "
            f"infer_speed={r.infer_samples_per_sec:.1f}s/s"
            if r.infer_samples_per_sec is not None else ""
        )


# --------------------------------------------------------------------------- #
#  helpers used by capture_inference
# --------------------------------------------------------------------------- #

def _run_forward(model: nn.Module, batch: Any) -> Any:
    """Dispatch batch to model.forward().

    Image-classification loaders commonly return (img_tensor, label_tensor)
    tuples; we only feed the first element (inputs) to model.forward.
    """
    _LABEL_KEYS = {"label", "labels", "target", "targets", "y", "y_gt",
                   "gt", "ground_truth", "class_id"}

    if isinstance(batch, dict):
        # strip obvious label / metadata fields, then try **kwargs; if it
        # still fails (e.g. model expects specific positional args), fall
        # back to feeding every tensor value positionally.
        filtered = {k: v for k, v in batch.items()
                    if k not in _LABEL_KEYS and torch.is_tensor(v)}
        try:
            return model(**filtered)
        except TypeError:
            return model(*filtered.values())
    if isinstance(batch, (list, tuple)):
        # common (img, label) or (img,) from torchvision/DataLoader
        if len(batch) >= 1 and torch.is_tensor(batch[0]):
            return model(batch[0])
        return model(*batch)
    return model(batch)


def _infer_batch_size(out: Any, batch: Any) -> int:
    """Try to recover batch size from output or input."""
    # 1) from output - tensor or dict['logits']
    if out is not None:
        if torch.is_tensor(out):
            return out.shape[0]
        if isinstance(out, dict):
            for key in ("logits", "pred", "preds", "y", "y_hat"):
                v = out.get(key)
                if torch.is_tensor(v):
                    return v.shape[0]
        if isinstance(out, (list, tuple)):
            for v in out:
                if torch.is_tensor(v):
                    return v.shape[0]

    # 2) from batch
    if isinstance(batch, dict):
        for key in ("img", "image", "x", "rgb", "ir", "input"):
            v = batch.get(key)
            if torch.is_tensor(v):
                return v.shape[0]
        # fallback: first tensor value
        for v in batch.values():
            if torch.is_tensor(v):
                return v.shape[0]
    if isinstance(batch, (list, tuple)) and batch:
        v = batch[0]
        if torch.is_tensor(v):
            return v.shape[0]
    if torch.is_tensor(batch):
        return batch.shape[0]
    return 1


def _print(msg: str) -> None:
    print(msg, flush=True)
