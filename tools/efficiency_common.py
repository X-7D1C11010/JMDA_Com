"""
Shared efficiency measurement utilities for all DA methods.

This module provides:
  * EfficiencyLogger   – lightweight per-epoch / per-step timer for training loops
  * count_parameters   – total / trainable params + non-trainable buffer size
  * get_state_dict_size_mb
  * measure_flops      – MACs + GFLOPs via thop (1 MAC = 2 FLOPs), with unsupported_ops
  * measure_inference_latency – mean/std/p50/p95 with CUDA sync
  * measure_inference_memory  – peak allocated / reserved
  * measure_train_step       – forward/loss/backward/optimizer timing + peak memory
  * save_environment_json
  * save_metrics_json
  * append_efficiency_csv
  * generate_summary_files    – efficiency_summary.csv + efficiency_table.md

Convention: 1 MAC = 2 FLOPs (stated explicitly in all outputs).
"""
import os
import csv
import json
import time
import uuid
import platform
import numpy as np
import torch
import torch.nn as nn


# ── helpers ────────────────────────────────────────────────────────────────
def get_run_id():
    return uuid.uuid4().hex[:12]


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _bytes_to_mb(b):
    return b / (1024 * 1024)


def _bytes_to_gb(b):
    return b / (1024 ** 3)


# ── parameter counting ─────────────────────────────────────────────────────
def count_parameters(model):
    """Return (total_params, trainable_params, buffer_bytes)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return total, trainable, buffer_bytes


def get_state_dict_size_mb(model):
    """Size of state_dict in MB (on CPU)."""
    tmp = io.BytesIO()
    torch.save(model.state_dict(), tmp)
    return tmp.getbuffer().nbytes / (1024 * 1024)


import io  # needed by get_state_dict_size_mb


# ── FLOPs (thop) ───────────────────────────────────────────────────────────
def measure_flops(model, inputs, device, forward_fn=None):
    """
    Measure MACs and GFLOPs using thop.
    Convention: 1 MAC = 2 FLOPs  ->  GFLOPs = MACs * 2 / 1e9.

    forward_fn(model, inputs) -> output  (optional; if None, uses default dispatch)
    Returns dict with: MACs, inference_GFLOPs, total_params_thop, unsupported_ops
    """
    from thop import profile
    model_eval = model.eval()

    # Determine how to call the model for thop
    if forward_fn is not None:
        # Use a wrapper that thop can profile
        class _FlopsWrapper(nn.Module):
            def __init__(self, model, fwd_fn, inp):
                super().__init__()
                self.model = model
                self.fwd_fn = fwd_fn
                self.inp = inp
            def forward(self):
                return self.fwd_fn(self.model, self.inp)
        wrapper = _FlopsWrapper(model_eval, forward_fn, inputs)
        try:
            macs, params = profile(wrapper, inputs=(), verbose=False)
        except Exception:
            macs, params = 0, 0
    else:
        # Default: unpack dict as positional, tuple as *args
        if isinstance(inputs, dict):
            inputs_tuple = tuple(inputs.values())
        elif isinstance(inputs, (list, tuple)):
            inputs_tuple = tuple(inputs)
        else:
            inputs_tuple = (inputs,)
        try:
            macs, params = profile(model_eval, inputs=inputs_tuple, verbose=False)
        except Exception:
            macs, params = 0, 0

    # Clean up thop hooks to prevent interference with subsequent runs
    for name, m in model_eval.named_modules():
        # Remove hooks that thop added (they reference total_ops/total_params)
        hook_ids = list(m._forward_hooks.keys())
        for hid in hook_ids:
            # Check if this is a thop hook by inspecting the closure
            hook_fn = m._forward_hooks[hid]
            if 'total_ops' in getattr(hook_fn, '__code__', type('',(),{'co_names':()})).co_names or \
               hasattr(hook_fn, '__wrapped__'):
                m._forward_hooks.pop(hid)
        pre_ids = list(m._forward_pre_hooks.keys())
        for hid in pre_ids:
            m._forward_pre_hooks.pop(hid)

    gflops = macs * 2 / 1e9  # 1 MAC = 2 FLOPs

    # Detect unsupported custom ops
    unsupported = []
    for name, m in model_eval.named_modules():
        mname = type(m).__name__.lower()
        if 'svd' in mname or 'sinkhorn' in mname or 'tensor' in mname.lower():
            unsupported.append(f"{name}({type(m).__name__})")

    return {
        'MACs': int(macs),
        'inference_GFLOPs': round(gflops, 4),
        'total_params_thop': int(params),
        'unsupported_ops': unsupported,
    }


# ── inference latency ──────────────────────────────────────────────────────
def measure_inference_latency(model, inputs, device,
                              warmup=100, iters=500, forward_fn=None):
    """
    Measure per-sample inference latency (ms) with CUDA sync.
    forward_fn(model, inputs) -> output  (optional custom dispatch)

    Returns dict: mean_ms, std_ms, p50_ms, p95_ms, throughput_samples_s
    """
    model.eval()
    # Prepare input on device
    if isinstance(inputs, dict):
        inp = {k: v.to(device) if torch.is_tensor(v) else v
               for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inp = tuple(v.to(device) if torch.is_tensor(v) else v for v in inputs)
    else:
        inp = inputs.to(device)

    def _call():
        if forward_fn is not None:
            return forward_fn(model, inp)
        if isinstance(inp, dict):
            return model(inp)  # pass dict as single arg
        elif isinstance(inp, tuple):
            return model(*inp)
        else:
            return model(inp)

    # Warmup — use no_grad (not inference_mode) when forward_fn is provided,
    # because some models (e.g. Boomda) need enable_grad() internally for hooks.
    grad_ctx = torch.no_grad() if forward_fn is not None else torch.inference_mode()
    with grad_ctx:
        for _ in range(warmup):
            _call()
    _cuda_sync()

    # Measure
    times_ms = []
    with grad_ctx:
        for _ in range(iters):
            _cuda_sync()
            t0 = time.perf_counter()
            _call()
            _cuda_sync()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    arr = np.array(times_ms)
    return {
        'latency_mean_ms': round(float(arr.mean()), 4),
        'latency_std_ms': round(float(arr.std()), 4),
        'latency_p50_ms': round(float(np.percentile(arr, 50)), 4),
        'latency_p95_ms': round(float(np.percentile(arr, 95)), 4),
        'throughput_samples_s': round(1000.0 / float(arr.mean()), 2),
    }


# ── inference memory ───────────────────────────────────────────────────────
def measure_inference_memory(model, inputs, device, batch_size=1, forward_fn=None):
    """Peak GPU memory during inference (batch_size=1 by default)."""
    model.eval()
    _reset_memory()
    if isinstance(inputs, dict):
        inp = {k: v.to(device) if torch.is_tensor(v) else v
               for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inp = tuple(v.to(device) if torch.is_tensor(v) else v for v in inputs)
    else:
        inp = inputs.to(device)

    grad_ctx = torch.no_grad() if forward_fn is not None else torch.inference_mode()
    with grad_ctx:
        if forward_fn is not None:
            forward_fn(model, inp)
        elif isinstance(inp, dict):
            model(inp)
        elif isinstance(inp, tuple):
            model(*inp)
        else:
            model(inp)
    _cuda_sync()

    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated()
        reserv = torch.cuda.max_memory_reserved()
    else:
        alloc = reserv = 0
    return {
        'peak_infer_allocated_gb': round(_bytes_to_gb(alloc), 4),
        'peak_infer_reserved_gb': round(_bytes_to_gb(reserv), 4),
    }


# ── train step timing ───────────────────────────────────────────────────────
def measure_train_step(model, inputs, optimizer, forward_loss_fn,
                       device, warmup=10, iters=50):
    """
    Time a full training step (forward + loss + backward + optimizer.step).

    forward_loss_fn(model, inputs) -> loss tensor

    Returns dict with per-step mean timings (ms) and peak memory (GB).
    """
    model.train()
    # Move inputs to device once
    if isinstance(inputs, dict):
        inp = {k: v.to(device) if torch.is_tensor(v) else v
               for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inp = tuple(v.to(device) if torch.is_tensor(v) else v for v in inputs)
    else:
        inp = inputs.to(device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        loss = forward_loss_fn(model, inp)
        loss.backward()
        optimizer.step()
    _cuda_sync()

    # Measure
    fwd_times, loss_times, bwd_times, opt_times, total_times = [], [], [], [], []
    _reset_memory()
    for _ in range(iters):
        optimizer.zero_grad()
        _cuda_sync()
        t0 = time.perf_counter()
        loss = forward_loss_fn(model, inp)
        _cuda_sync()
        t1 = time.perf_counter()
        # loss already computed inside forward_loss_fn; measure backward
        _cuda_sync()
        t2 = time.perf_counter()
        loss.backward()
        _cuda_sync()
        t3 = time.perf_counter()
        optimizer.step()
        _cuda_sync()
        t4 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)
        loss_times.append(max(0, (t2 - t1)) * 1000)  # often ~0 if loss is part of fwd
        bwd_times.append((t3 - t2) * 1000)
        opt_times.append((t4 - t3) * 1000)
        total_times.append((t4 - t0) * 1000)

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserv = torch.cuda.max_memory_reserved()
    else:
        peak_alloc = peak_reserv = 0

    def _m(lst):
        a = np.array(lst)
        return round(float(a.mean()), 4)

    return {
        'train_fwd_ms': _m(fwd_times),
        'train_loss_ms': _m(loss_times),
        'train_bwd_ms': _m(bwd_times),
        'train_opt_ms': _m(opt_times),
        'train_step_mean_ms': _m(total_times),
        'peak_train_allocated_gb': round(_bytes_to_gb(peak_alloc), 4),
        'peak_train_reserved_gb': round(_bytes_to_gb(peak_reserv), 4),
    }


# ── environment ────────────────────────────────────────────────────────────
def save_environment_json(output_dir, extra=None):
    info = {
        'python': platform.python_version(),
        'pytorch': torch.__version__,
        'cuda': torch.version.cuda or 'N/A',
        'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A',
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'platform': platform.platform(),
        'device': str(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')),
    }
    if extra:
        info.update(extra)
    path = os.path.join(output_dir, 'environment.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    return info


# ── save metrics.json ───────────────────────────────────────────────────────
def save_metrics_json(method_dir, run_id, metrics):
    """Save metrics.json under <output_dir>/<method>/<run_id>/."""
    # Sanitize method name for Windows directory (no * / \ etc.)
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in method_dir.rstrip(os.sep).split(os.sep)[-1])
    d = os.path.join(os.path.dirname(method_dir), safe, run_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return d


# ── CSV append ──────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    'timestamp', 'run_id', 'method', 'protocol', 'modalities', 'seed',
    'device', 'gpu_name', 'pytorch', 'cuda', 'precision', 'batch_size',
    'vis_shape', 'ir_shape', 'ais_shape',
    'target_label_count', 'target_labels_per_class', 'backbone', 'pretraining',
    'total_params', 'trainable_params', 'buffer_mb', 'model_size_mb',
    'MACs', 'inference_GFLOPs', 'unsupported_ops',
    'latency_mean_ms', 'latency_std_ms', 'latency_p50_ms', 'latency_p95_ms',
    'throughput_samples_s',
    'train_step_mean_ms', 'train_fwd_ms', 'train_bwd_ms', 'train_opt_ms',
    'train_epoch_mean_s', 'train_epoch_std_s', 'total_train_time_s',
    'peak_train_allocated_gb', 'peak_train_reserved_gb',
    'peak_infer_allocated_gb', 'peak_infer_reserved_gb',
    'svd_update_ms', 'sinkhorn_ms',
    'encoder_forward_ms', 'tensor_forward_ms', 'backward_ms', 'optimizer_step_ms',
]


def append_efficiency_csv(output_dir, row):
    path = os.path.join(output_dir, 'efficiency_runs.csv')
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row)


# ── summary generation ─────────────────────────────────────────────────────
def generate_summary_files(output_dir):
    """Generate efficiency_summary.csv and efficiency_table.md from runs CSV."""
    csv_path = os.path.join(output_dir, 'efficiency_runs.csv')
    if not os.path.exists(csv_path):
        return

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    # Group by method, compute mean ± std for numeric fields
    from collections import defaultdict
    numeric_fields = [
        'total_params', 'trainable_params', 'buffer_mb', 'model_size_mb',
        'MACs', 'inference_GFLOPs',
        'latency_mean_ms', 'latency_std_ms', 'latency_p50_ms', 'latency_p95_ms',
        'throughput_samples_s', 'train_step_mean_ms',
        'train_epoch_mean_s', 'train_epoch_std_s', 'total_train_time_s',
        'peak_train_allocated_gb', 'peak_train_reserved_gb',
        'peak_infer_allocated_gb', 'peak_infer_reserved_gb',
        'svd_update_ms', 'sinkhorn_ms',
    ]
    groups = defaultdict(list)
    for r in rows:
        groups[r.get('method', 'unknown')].append(r)

    def _group_meta(rs, key, default=''):
        for r in rs:
            v = r.get(key, '')
            if v not in ('', None):
                return v
        return default

    # Write summary CSV (grouped by method; protocol/label-budget columns kept)
    summary_path = os.path.join(output_dir, 'efficiency_summary.csv')
    meta_fields = ['protocol', 'modalities', 'target_labels_per_class',
                   'backbone', 'pretraining']
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['method'] + meta_fields + \
                 [f'{fld}_mean' for fld in numeric_fields] + \
                 [f'{fld}_std' for fld in numeric_fields]
        w.writerow(header)
        for method, method_rows in sorted(groups.items()):
            vals = {}
            for fld in numeric_fields:
                nums = []
                for r in method_rows:
                    v = r.get(fld, '')
                    try:
                        nums.append(float(v))
                    except (ValueError, TypeError):
                        pass
                if nums:
                    vals[f'{fld}_mean'] = round(np.mean(nums), 4)
                    vals[f'{fld}_std'] = round(np.std(nums), 4)
                else:
                    vals[f'{fld}_mean'] = ''
                    vals[f'{fld}_std'] = ''
            row_out = [method] + [_group_meta(method_rows, mf) for mf in meta_fields] + \
                      [vals[f'{fld}_mean'] for fld in numeric_fields] + \
                      [vals[f'{fld}_std'] for fld in numeric_fields]
            w.writerow(row_out)

    # Write markdown tables, split by protocol (SDA / UDA results are never
    # mixed into the same table)
    md_path = os.path.join(output_dir, 'efficiency_table.md')
    with open(md_path, 'w') as f:
        f.write('# Efficiency Benchmark Results\n\n')
        f.write('> Convention: 1 MAC = 2 FLOPs. All timings on single GPU.\n')
        f.write('> Tables are split by supervision protocol: SDA (labeled target subset) '
                'vs UDA (no target labels). Do not compare rows across protocols.\n\n')

        def _protocol_of(method):
            return _group_meta(groups[method], 'protocol', 'UDA') or 'UDA'

        protocol_order = []
        for m in groups:
            p = _protocol_of(m)
            if p not in protocol_order:
                protocol_order.append(p)
        # SDA (paper main results) first, then UDA, then anything else
        protocol_order.sort(key=lambda p: {'SDA': 0, 'UDA': 1}.get(p, 2))

        for protocol in protocol_order:
            proto_methods = [m for m in groups if _protocol_of(m) == protocol]
            f.write(f'## Protocol: {protocol}\n\n')

            trimodal_methods = [m for m in proto_methods
                                if _group_meta(groups[m], 'modalities') in
                                ('VIS+IR+AIS', 'VIS+IR')]
            vlm_methods = [m for m in proto_methods
                           if _group_meta(groups[m], 'modalities') in ('VIS', 'CLIP')]
            jmda_methods = [m for m in proto_methods
                            if 'jmda' in m.lower() or 'JMDA' in m]

            def _fmt(method, fld):
                rows_m = groups.get(method, [])
                nums = []
                for r in rows_m:
                    v = r.get(fld, '')
                    try:
                        nums.append(float(v))
                    except (ValueError, TypeError):
                        pass
                if nums:
                    m_val = np.mean(nums)
                    s_val = np.std(nums)
                    if abs(m_val) < 0.01 and abs(s_val) < 0.01:
                        return f"{m_val:.4f}"
                    return f"{m_val:.2f}±{s_val:.2f}"
                return '–'

            if trimodal_methods:
                f.write(f'### Table ({protocol}): Trimodal / Dual-modal Methods (VIS+IR[+AIS])\n\n')
                cols = ['total_params', 'inference_GFLOPs', 'latency_mean_ms',
                        'throughput_samples_s', 'train_step_mean_ms',
                        'peak_train_allocated_gb', 'peak_infer_allocated_gb']
                col_names = ['Params', 'GFLOPs', 'Latency(ms)', 'Throughput(s/s)',
                             'Train Step(ms)', 'Train Mem(GB)', 'Infer Mem(GB)']
                f.write('| Method | ' + ' | '.join(col_names) + ' |\n')
                f.write('|--------|' + '|'.join(['---'] * len(col_names)) + '|\n')
                for m in sorted(trimodal_methods):
                    vals = [_fmt(m, c) for c in cols]
                    f.write(f'| {m} | ' + ' | '.join(vals) + ' |\n')
                f.write('\n')

            if jmda_methods:
                f.write(f'### Table ({protocol}): JMDA-Net Module-Level Ablation\n\n')
                cols2 = ['encoder_forward_ms', 'tensor_forward_ms', 'svd_update_ms',
                        'sinkhorn_ms', 'backward_ms', 'optimizer_step_ms',
                        'inference_GFLOPs', 'latency_mean_ms']
                col_names2 = ['Encoder(ms)', 'Tensor(ms)', 'SVD(ms)', 'Sinkhorn(ms)',
                              'Backward(ms)', 'Opt(ms)', 'GFLOPs', 'Latency(ms)']
                f.write('| Method | ' + ' | '.join(col_names2) + ' |\n')
                f.write('|--------|' + '|'.join(['---'] * len(col_names2)) + '|\n')
                for m in sorted(jmda_methods):
                    vals = [_fmt(m, c) for c in cols2]
                    f.write(f'| {m} | ' + ' | '.join(vals) + ' |\n')
                f.write('\n')
                f.write('> Sinkhorn is training-only; no inference-time Sinkhorn overhead.\n')
                f.write('> SVD is updated once per epoch; per-step amortized time shown separately.\n')
                f.write('> If projection matrices participate in inference, their linear map is included in GFLOPs.\n\n')

            if vlm_methods:
                f.write(f'### Table ({protocol}): Visual / VLM Methods (CLIP / Swin backbone)\n\n')
                f.write('> Listed separately due to different backbone and input paradigm.\n\n')
                cols3 = ['total_params', 'inference_GFLOPs', 'latency_mean_ms',
                         'throughput_samples_s', 'train_step_mean_ms',
                         'peak_train_allocated_gb', 'peak_infer_allocated_gb']
                col_names3 = ['Params', 'GFLOPs', 'Latency(ms)', 'Throughput(s/s)',
                              'Train Step(ms)', 'Train Mem(GB)', 'Infer Mem(GB)']
                f.write('| Method | ' + ' | '.join(col_names3) + ' |\n')
                f.write('|--------|' + '|'.join(['---'] * len(col_names3)) + '|\n')
                for m in sorted(vlm_methods):
                    vals = [_fmt(m, c) for c in cols3]
                    f.write(f'| {m} | ' + ' | '.join(vals) + ' |\n')
                f.write('\n')


# ── EfficiencyLogger (training-loop layer) ──────────────────────────────────
class EfficiencyLogger:
    """
    Lightweight per-epoch / per-step timer for training loops.

    Usage in a training script::

        from tools.efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(output_dir, 'DANN', seed=42)
        for epoch in range(epochs):
            eff.epoch_start(epoch)
            for batch in loader:
                eff.step_start()           # before DataLoader yield
                # ... training step ...
                eff.step_end()             # after optimizer.step()
            eff.epoch_end()

        eff.finalize(extra_metrics={...})
    """

    def __init__(self, output_dir, method_name, seed=0,
                 device='cuda:0', modalities='VIS', protocol='UDA',
                 target_label_count=None, target_labels_per_class=None,
                 backbone='', pretraining=''):
        self.output_dir = output_dir
        self.method_name = method_name
        self.seed = seed
        self.device = device
        self.modalities = modalities
        # Experiment protocol: 'UDA' (strictly unsupervised target) or 'SDA'
        # (supervised DA on the shared labeled target subset).
        self.protocol = protocol
        self.target_label_count = target_label_count if target_label_count is not None else ''
        self.target_labels_per_class = target_labels_per_class if target_labels_per_class is not None else ''
        self.backbone = backbone
        self.pretraining = pretraining
        self.run_id = get_run_id()

        self.method_dir = os.path.join(output_dir, method_name)
        self.run_dir = os.path.join(self.method_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self._epoch_times = []
        self._step_times = []
        self._dl_times = []
        self._epoch_compute_times = []
        self._current_epoch_steps = []
        self._current_epoch_dl = []
        self._epoch_start = 0
        self._step_start = 0
        self._dl_end = 0
        self._total_train_time = 0
        self._epoch_peak_mem = []

        # JMDA-Net specific
        self._module_times = defaultdict(list)  # name -> list of ms

        self._csv_path = os.path.join(self.run_dir, 'epoch_timing.csv')
        with open(self._csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'epoch_time_s', 'train_compute_s',
                        'dataloader_wait_s', 'avg_step_ms',
                        'peak_alloc_gb', 'peak_reserved_gb'])

    def epoch_start(self, epoch):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._epoch_start = time.perf_counter()
        self._current_epoch_steps = []
        self._current_epoch_dl = []

    def step_start(self):
        self._step_start = time.perf_counter()

    def dataloader_done(self):
        """Call right after getting the batch from DataLoader."""
        self._dl_end = time.perf_counter()

    def step_end(self):
        t_end = time.perf_counter()
        step_ms = (t_end - self._step_start) * 1000
        dl_ms = (self._dl_end - self._step_start) * 1000 if self._dl_end > self._step_start else 0
        compute_ms = step_ms - dl_ms
        self._current_epoch_steps.append(step_ms)
        self._current_epoch_dl.append(dl_ms)
        self._step_start = 0
        self._dl_end = 0

    def record_module(self, name, duration_ms):
        """Record a JMDA-Net module-level timing (e.g. 'encoder_forward', 'svd_update')."""
        self._module_times[name].append(duration_ms)

    def epoch_end(self, epoch=None):
        t_end = time.perf_counter()
        epoch_time = t_end - self._epoch_start
        self._epoch_times.append(epoch_time)
        compute_time = sum(self._current_epoch_steps) / 1000  # ms->s
        self._epoch_compute_times.append(compute_time)
        dl_wait = sum(self._current_epoch_dl) / 1000
        self._dl_times.append(dl_wait)
        self._total_train_time += epoch_time

        avg_step = np.mean(self._current_epoch_steps) if self._current_epoch_steps else 0
        if torch.cuda.is_available():
            peak_a = torch.cuda.max_memory_allocated()
            peak_r = torch.cuda.max_memory_reserved()
        else:
            peak_a = peak_r = 0

        self._epoch_peak_mem.append((_bytes_to_gb(peak_a), _bytes_to_gb(peak_r)))

        with open(self._csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, round(epoch_time, 4), round(compute_time, 4),
                        round(dl_wait, 4), round(avg_step, 4),
                        round(_bytes_to_gb(peak_a), 4),
                        round(_bytes_to_gb(peak_r), 4)])

    @property
    def epoch_count(self):
        return len(self._epoch_times)

    def get_module_means(self):
        return {k: round(float(np.mean(v)), 4) if v else 0
                for k, v in self._module_times.items()}

    def finalize(self, extra_metrics=None):
        """Write metrics.json and append to efficiency_runs.csv."""
        # Skip first warmup epoch for stable stats if > 2 epochs
        if len(self._epoch_times) > 2:
            stable = self._epoch_times[1:]
        else:
            stable = self._epoch_times

        epoch_mean = float(np.mean(stable)) if stable else 0
        epoch_std = float(np.std(stable)) if stable else 0
        peak_alloc = max(m[0] for m in self._epoch_peak_mem) if self._epoch_peak_mem else 0
        peak_reserv = max(m[1] for m in self._epoch_peak_mem) if self._epoch_peak_mem else 0

        metrics = {
            'run_id': self.run_id,
            'method': self.method_name,
            'protocol': self.protocol,
            'seed': self.seed,
            'device': self.device,
            'modalities': self.modalities,
            'target_label_count': self.target_label_count,
            'target_labels_per_class': self.target_labels_per_class,
            'backbone': self.backbone,
            'pretraining': self.pretraining,
            'train_epoch_mean_s': round(epoch_mean, 4),
            'train_epoch_std_s': round(epoch_std, 4),
            'total_train_time_s': round(self._total_train_time, 4),
            'peak_train_allocated_gb': round(peak_alloc, 4),
            'peak_train_reserved_gb': round(peak_reserv, 4),
        }
        # JMDA module-level timings
        mod_means = self.get_module_means()
        for k, v in mod_means.items():
            metrics[f'{k}_ms'] = v
        if extra_metrics:
            metrics.update(extra_metrics)

        # Save metrics.json
        with open(os.path.join(self.run_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Append to master CSV
        row = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'run_id': self.run_id,
            'method': self.method_name,
            'protocol': self.protocol,
            'modalities': self.modalities,
            'seed': self.seed,
            'device': self.device,
            'target_label_count': self.target_label_count,
            'target_labels_per_class': self.target_labels_per_class,
            'backbone': self.backbone,
            'pretraining': self.pretraining,
            'train_epoch_mean_s': round(epoch_mean, 4),
            'train_epoch_std_s': round(epoch_std, 4),
            'total_train_time_s': round(self._total_train_time, 4),
            'peak_train_allocated_gb': round(peak_alloc, 4),
            'peak_train_reserved_gb': round(peak_reserv, 4),
        }
        for k, v in mod_means.items():
            row[f'{k}_ms'] = v
        if extra_metrics:
            row.update(extra_metrics)

        append_efficiency_csv(self.output_dir, row)
        return metrics


from collections import defaultdict  # needed by EfficiencyLogger


# ── torch.profiler summary ──────────────────────────────────────────────────
def run_profiler_summary(model, inputs, device, run_dir,
                         forward_fn=None, iters=10):
    """Run torch.profiler briefly and save profiler_summary.txt.

    Captures CPU + CUDA activity for a few forward passes.
    Does NOT modify model state.
    """
    model.eval()
    if isinstance(inputs, dict):
        inp = {k: v.to(device) if torch.is_tensor(v) else v
               for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inp = tuple(v.to(device) if torch.is_tensor(v) else v for v in inputs)
    else:
        inp = inputs.to(device)

    def _call():
        if forward_fn is not None:
            return forward_fn(model, inp)
        if isinstance(inp, dict):
            return model(inp)
        elif isinstance(inp, tuple):
            return model(*inp)
        else:
            return model(inp)

    try:
        from torch.profiler import profile, ProfilerActivity
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=False) as prof:
            with torch.no_grad():
                # warmup
                for _ in range(3):
                    _call()
                _cuda_sync()
                for _ in range(iters):
                    _call()
                _cuda_sync()

        summary = prof.key_averages().table(sort_by="cuda_time_total",
                                             row_limit=30)
    except Exception as e:
        summary = f"Profiler failed: {e}\n"

    path = os.path.join(run_dir, 'profiler_summary.txt')
    with open(path, 'w') as f:
        f.write(f"torch.profiler summary (forward-only, {iters} iters)\n")
        f.write(f"Model: {type(model).__name__}\n")
        f.write(f"Device: {device}\n\n")
        f.write(summary)
    return path
