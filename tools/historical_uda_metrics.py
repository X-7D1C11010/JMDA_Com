"""Measure the models used by the pinned VIS+IR training suite."""

import json
from statistics import pstdev
import time

import torch
from torch import nn

from classification_metrics import classification_metrics as base_classification_metrics
from efficiency_common import EfficiencyLogger, count_parameters, measure_inference_latency


def classification_metrics(labels, predictions, num_classes):
    metrics = base_classification_metrics(labels, predictions, num_classes)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for label, prediction in zip(labels, predictions):
        matrix[int(label)][int(prediction)] += 1
    metrics["confusion_matrix"] = matrix
    return metrics


def report_validation(metrics, class_ids):
    matrix = metrics["confusion_matrix"]
    support = [sum(row) for row in matrix]
    record = {
        "original_class_ids": list(class_ids),
        "support": support,
        "recall": [round(matrix[i][i] / n, 4) if n else None
                   for i, n in enumerate(support)],
        "confusion_matrix_rows_truth": matrix,
    }
    for key in ("vis_zeroed_acc", "ir_zeroed_acc"):
        if key in metrics:
            record[key] = metrics[key]
    print("UDA_VALIDATION_DIAGNOSTICS: " + json.dumps(record))


def classify_dann(model, images):
    return model.class_classifier(model.encoder(images))


def target_prediction_stats(logits):
    scores = logits.detach().softmax(dim=1)
    confidence, labels = scores.max(dim=1)
    return {
        "mean_max_probability": round(float(confidence.mean()), 4),
        "predicted_counts_compact": labels.bincount(minlength=scores.size(1)).tolist(),
    }


def report_coral_gradients(step, head, classification_loss, weighted_coral_loss):
    weight = head.weight
    cls, = torch.autograd.grad(classification_loss, weight, retain_graph=True)
    coral, = torch.autograd.grad(weighted_coral_loss, weight, retain_graph=True)
    cls_norm, coral_norm = float(cls.norm()), float(coral.norm())
    ratio = coral_norm / max(cls_norm, 1e-12)
    print(f"CORAL head gradient at update {step}: "
          f"CE={cls_norm:.6g} weighted_CORAL={coral_norm:.6g} ratio={ratio:.6g}")


def report_mcc_gradients(epoch, head, classification_loss, weighted_transfer_loss):
    parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    cls = torch.autograd.grad(classification_loss, parameters, retain_graph=True)
    transfer = torch.autograd.grad(weighted_transfer_loss, parameters, retain_graph=True)
    cls_flat = torch.cat([gradient.flatten() for gradient in cls])
    transfer_flat = torch.cat([gradient.flatten() for gradient in transfer])
    cls_norm, transfer_norm = float(cls_flat.norm()), float(transfer_flat.norm())
    cosine = float(torch.nn.functional.cosine_similarity(cls_flat, transfer_flat, dim=0))
    print(f"MCC head gradient at epoch {epoch}: "
          f"CE={cls_norm:.6g} weighted_MCC={transfer_norm:.6g} "
          f"ratio={transfer_norm / max(cls_norm, 1e-12):.6g} cosine={cosine:.6g}")


def forward_flops(model, inputs, forward_fn):
    """Count multiply-add FLOPs for Conv2d and Linear modules actually executed."""
    macs = 0

    def count(module, _inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs += output.numel() * (module.in_channels // module.groups) * kh * kw
        else:
            macs += output.numel() * module.in_features

    handles = [layer.register_forward_hook(count) for layer in model.modules()
               if isinstance(layer, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode():
            forward_fn(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    return macs


class HistoricalEfficiencyLogger(EfficiencyLogger):
    """Time completed training windows and benchmark the deployed classifier."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            self.device = f"cuda:{torch.cuda.current_device()}"
        self.pretraining_metrics = {}
        self.inference_metrics = {}
        self.calibration_metrics = {}

    def epoch_end(self, epoch=None):
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        super().epoch_end(epoch)

    def record_pretraining(self, metrics):
        self.pretraining_metrics = metrics

    def record_calibration(self, metrics):
        self.calibration_metrics = metrics

    def benchmark(self, model, forward_fn, *, method, training_module=None,
                  training_unit_updates=None):
        device = torch.device(self.device)
        total, trainable, buffer_bytes = count_parameters(model)
        if training_module is not None:
            extra_total, extra_trainable, extra_buffers = count_parameters(training_module)
            total += extra_total
            trainable += extra_trainable
            buffer_bytes += extra_buffers
        metrics = {
            "total_params": total,
            "trainable_params": trainable,
            "buffer_mb": round(buffer_bytes / 1024**2, 4),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "pytorch": str(torch.__version__),
            "cuda": torch.version.cuda if device.type == "cuda" else None,
            "precision": str(next(model.parameters()).dtype).removeprefix("torch."),
            "batch_size": 1,
            "inference_input_shape": [1, 6, 224, 224],
            "vis_shape": "3x224x224",
            "ir_shape": "3x224x224",
            "inference_flops_scope": "executed Conv2d and Linear; 1 MAC = 2 FLOPs",
            "validation_views": 10 if method == "cdan" else 1,
            "training_time_unit": (f"{training_unit_updates} optimizer updates"
                                   if training_unit_updates is not None else "epoch"),
        }
        try:
            model.eval()
            inputs = next(model.parameters()).new_zeros(metrics["inference_input_shape"])
            macs = forward_flops(model, inputs, forward_fn)
            metrics.update(MACs=macs, inference_GFLOPs=round(macs * 2 / 1e9, 6))
            metrics.update(measure_inference_latency(
                model, inputs, device, warmup=20, iters=100, forward_fn=forward_fn,
            ))
            metrics["complexity_status"] = "complete"
        except (RuntimeError, ValueError) as exc:
            metrics["complexity_status"] = "inference_benchmark_failed"
            metrics["complexity_error"] = str(exc)
        self.inference_metrics = metrics

    def finalize(self, extra_metrics=None):
        pretrain = self.pretraining_metrics
        cuda = torch.device(self.device).type == "cuda"
        peak_allocated = max((m[0] for m in self._epoch_peak_mem), default=0.0)
        peak_reserved = max((m[1] for m in self._epoch_peak_mem), default=0.0)
        metrics = {
            **(extra_metrics or {}),
            **self.inference_metrics,
            **pretrain,
            **self.calibration_metrics,
            "train_epoch_mean_s": round(sum(self._epoch_times) / self.epoch_count, 4),
            "train_epoch_std_s": round(pstdev(self._epoch_times), 4),
            "total_train_time_s": round(
                self._total_train_time + pretrain.get("ir_pretrain_time_s", 0.0), 4),
            "peak_train_allocated_gb": round(max(
                peak_allocated, pretrain.get("ir_pretrain_peak_allocated_gb", 0.0)), 4) if cuda else None,
            "peak_train_reserved_gb": round(max(
                peak_reserved, pretrain.get("ir_pretrain_peak_reserved_gb", 0.0)), 4) if cuda else None,
            "training_time_scope": "training windows plus IR pretraining, excluding validation and GPU queue",
            "timed_training_units": self.epoch_count,
            "total_training_and_calibration_time_s": round(
                self._total_train_time + pretrain.get("ir_pretrain_time_s", 0.0) +
                self.calibration_metrics.get("bn_calibration_total_s", 0.0), 4),
        }
        result = super().finalize(metrics)
        print("UDA_COMPLEXITY: " + json.dumps(result))
        return result


def start_stage_timer(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def finish_stage_timer(device, started):
    metrics = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        metrics.update(
            ir_pretrain_peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 1024**3,
            ir_pretrain_peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 1024**3,
        )
    metrics["ir_pretrain_time_s"] = time.perf_counter() - started
    return metrics
