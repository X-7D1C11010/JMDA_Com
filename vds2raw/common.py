"""Shared VDS2Raw evaluation, provenance, and result helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from tools.classification_metrics import (
    classification_metrics,
)
from tools.metrics_io import save_metric_history

NUM_CLASSES = 3


def _prediction_record(sample_index, sample_id, truth, prediction, logits):
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "y_true": int(truth),
        "y_pred": int(prediction),
        "correct": int(truth == prediction),
        "logits": [float(value) for value in logits],
    }


def evaluate_rgb(model, loader, device, return_predictions=False):
    model.eval()
    labels, predictions, records = [], [], []
    sample_paths = getattr(loader.dataset, "paths", None)
    sample_offset = 0
    with torch.no_grad():
        for images, truth in loader:
            logits = model(images.to(device))
            batch_truth = truth.tolist()
            batch_predictions = logits.argmax(1).cpu().tolist()
            batch_logits = logits.detach().cpu().tolist()
            labels.extend(batch_truth)
            predictions.extend(batch_predictions)
            if return_predictions:
                for index, (expected, predicted, scores) in enumerate(
                    zip(batch_truth, batch_predictions, batch_logits)
                ):
                    absolute_index = sample_offset + index
                    sample_id = (
                        sample_paths[absolute_index]
                        if sample_paths is not None
                        else absolute_index
                    )
                    records.append(
                        _prediction_record(
                            absolute_index, sample_id, expected, predicted, scores
                        )
                    )
            sample_offset += len(batch_truth)
    metrics = classification_metrics(labels, predictions, num_classes=NUM_CLASSES)
    if return_predictions:
        return metrics, records
    return metrics


def evaluate_multimodal(model, loader, device, return_predictions=False):
    model.eval()
    labels, predictions, records = [], [], []
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            truth = batch["label"]
            logits, _, _ = model(
                batch["rgb"].to(device),
                batch["nir"].to(device),
                batch["ais"].to(device),
                batch["ais_mask"].to(device),
                labels=None,
                compute_auxiliary=False,
            )
            batch_truth = truth.tolist()
            batch_predictions = logits.argmax(1).cpu().tolist()
            batch_logits = logits.detach().cpu().tolist()
            labels.extend(batch_truth)
            predictions.extend(batch_predictions)
            if return_predictions:
                batch_paths = batch.get("path")
                for index, (expected, predicted, scores) in enumerate(
                    zip(batch_truth, batch_predictions, batch_logits)
                ):
                    absolute_index = sample_offset + index
                    sample_id = (
                        batch_paths[index]
                        if batch_paths is not None
                        else absolute_index
                    )
                    records.append(
                        _prediction_record(
                            absolute_index, sample_id, expected, predicted, scores
                        )
                    )
            sample_offset += len(batch_truth)
    metrics = classification_metrics(labels, predictions, num_classes=NUM_CLASSES)
    if return_predictions:
        return metrics, records
    return metrics


def metric_line(split, metrics):
    return (
        f"{split} accuracy: {metrics['accuracy']:.4f}\n"
        f"  Precision (Macro): {metrics['precision_macro']:.4f} | "
        f"Recall (Macro): {metrics['recall_macro']:.4f} | "
        f"F1 (Macro): {metrics['f1_macro']:.4f} | "
        f"class coverage: {metrics['val_present_classes']}/"
        f"{metrics['val_total_classes']} | samples: {metrics['val_samples']}"
    )


def better_metrics(candidate, incumbent):
    if incumbent is None:
        return True
    return (candidate["accuracy"], candidate["f1_macro"]) > (
        incumbent["accuracy"],
        incumbent["f1_macro"],
    )


def save_checkpoint(
    path,
    model,
    epoch,
    selection_metrics,
    metadata,
    *,
    target_val_metrics=None,
    target_test_metrics=None,
):
    """Atomically save a checkpoint with explicit selection provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    val_metrics = target_val_metrics or selection_metrics
    test_metrics = target_test_metrics or selection_metrics
    torch.save(
        {
            "model_state": model.state_dict(),
            "selected_epoch": int(epoch),
            "selection_partition": metadata.get("selection_partition"),
            "selection_metrics": dict(selection_metrics),
            "target_val_metrics": dict(val_metrics),
            "target_test_metrics": dict(test_metrics),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_selected_checkpoint(path, model, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_record_history(save_dir, stem, records, metadata=None):
    """Persist arbitrary per-epoch records as CSV and JSON."""
    if not records:
        return None, None
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    csv_path = save_dir / f"{stem}.csv"
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in record.items()
                }
            )
    temporary_csv.replace(csv_path)
    json_path = save_dir / f"{stem}.json"
    _atomic_json(json_path, {"metadata": metadata or {}, "history": records})
    return csv_path, json_path


def save_prediction_artifacts(save_dir, stem, records, class_names=None):
    """Save selected-checkpoint predictions and a confusion matrix."""
    if not records:
        raise ValueError("prediction records cannot be empty")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(class_names or [str(index) for index in range(NUM_CLASSES)])
    if len(class_names) != NUM_CLASSES:
        raise ValueError("class_names must match NUM_CLASSES")

    prediction_csv = save_dir / f"{stem}_predictions.csv"
    fields = [
        "sample_index",
        "sample_id",
        "y_true",
        "y_pred",
        "correct",
    ] + [f"logit_{index}" for index in range(NUM_CLASSES)]
    temporary = prediction_csv.with_suffix(prediction_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields[:5]}
            row.update(
                {
                    f"logit_{index}": record["logits"][index]
                    for index in range(NUM_CLASSES)
                }
            )
            writer.writerow(row)
    temporary.replace(prediction_csv)

    matrix = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    for record in records:
        matrix[record["y_true"]][record["y_pred"]] += 1
    confusion_csv = save_dir / f"{stem}_confusion_matrix.csv"
    temporary = confusion_csv.with_suffix(confusion_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + class_names)
        for index, row in enumerate(matrix):
            writer.writerow([class_names[index]] + row)
    temporary.replace(confusion_csv)
    confusion_json = save_dir / f"{stem}_confusion_matrix.json"
    _atomic_json(
        confusion_json,
        {
            "class_names": class_names,
            "orientation": "rows=true, columns=predicted",
            "matrix": matrix,
        },
    )
    return {
        "predictions_csv": str(prediction_csv.resolve()),
        "confusion_matrix_csv": str(confusion_csv.resolve()),
        "confusion_matrix_json": str(confusion_json.resolve()),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repository, *arguments):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def build_run_provenance(args, data_root, input_files=()):
    """Capture code, command, dataset metadata, and explicit weight hashes."""
    repository = Path(__file__).resolve().parents[1]
    data_root = Path(data_root)
    dataset_files = (
        "dataset_manifest.csv",
        "feature_schema.json",
        "build_report.json",
        "band_order_verification.json",
        "validation_report.json",
    )
    dataset_hashes = {}
    for name in dataset_files:
        path = data_root / name
        dataset_hashes[name] = _sha256(path) if path.is_file() else None
    explicit_inputs = {}
    for value in input_files:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        explicit_inputs[str(path)] = _sha256(path) if path.is_file() else None
    portable_manifest = repository / "pretrained_weights" / "manifest.json"
    if portable_manifest.is_file():
        explicit_inputs[str(portable_manifest)] = _sha256(portable_manifest)
    return {
        "argv": list(sys.argv),
        "arguments": vars(args),
        "git_head": _git_output(repository, "rev-parse", "HEAD"),
        "git_status_short": _git_output(repository, "status", "--short"),
        "dataset_metadata_sha256": dataset_hashes,
        "explicit_input_sha256": explicit_inputs,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def save_run_config(save_dir, stem, args, metadata, data_root, input_files=()):
    path = Path(save_dir) / f"{stem}_run_config.json"
    payload = {
        "protocol_warning": (
            "target_test is evaluated during training and selects the checkpoint; "
            "reported target_test is a test-set-selected peak, not an unbiased estimate"
        ),
        "metadata": metadata,
        "provenance": build_run_provenance(args, data_root, input_files),
    }
    _atomic_json(path, payload)
    return path


def save_experiment_results(
    save_dir,
    stem,
    val_history,
    test_history,
    selected_epoch,
    selected_val,
    selected_test,
    metadata,
    *,
    training_history=None,
    prediction_artifacts=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    val_csv, val_json = save_metric_history(
        str(save_dir), stem + "_target_val", val_history, metadata=metadata
    )
    test_csv, test_json = save_metric_history(
        str(save_dir), stem + "_target_test", test_history, metadata=metadata
    )
    train_csv, train_json = save_record_history(
        save_dir, stem + "_train_history", training_history or [], metadata
    )
    result = {
        "protocol": "cross-day fully supervised domain adaptation",
        "model_selection": (
            "best target_test ACC, macro-F1 tie-break; test-set-selected peak"
        ),
        "selection_partition": "target_test",
        "test_selection_bias": True,
        "test_evaluations_for_selection": len(test_history),
        "selected_checkpoint_report_evaluations": 1,
        "selected_epoch": int(selected_epoch),
        "target_val": dict(selected_val),
        "target_test": dict(selected_test),
        "metadata": metadata,
        "target_val_history_csv": str(Path(val_csv).resolve()),
        "target_val_history_json": str(Path(val_json).resolve()),
        "target_test_history_csv": str(Path(test_csv).resolve()),
        "target_test_history_json": str(Path(test_json).resolve()),
        "training_history_csv": str(Path(train_csv).resolve()) if train_csv else None,
        "training_history_json": str(Path(train_json).resolve()) if train_json else None,
        "prediction_artifacts": prediction_artifacts or {},
    }
    json_path = save_dir / f"{stem}_result.json"
    _atomic_json(json_path, result)

    csv_path = save_dir / f"{stem}_result.csv"
    fields = (
        "method",
        "seed",
        "selected_epoch",
        "selection_partition",
        "test_selection_bias",
        "split",
        "ACC",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "samples",
    )
    rows = []
    for split, metrics in (
        ("target_val_at_selected_epoch", selected_val),
        ("target_test_selected_peak", selected_test),
    ):
        rows.append(
            {
                "method": metadata["method"],
                "seed": metadata["seed"],
                "selected_epoch": selected_epoch,
                "selection_partition": "target_test",
                "test_selection_bias": True,
                "split": split,
                "ACC": metrics["accuracy"],
                "Precision_macro": metrics["precision_macro"],
                "Recall_macro": metrics["recall_macro"],
                "F1_macro": metrics["f1_macro"],
                "samples": metrics["val_samples"],
            }
        )
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)
    print(
        "FINAL_TEST_METRICS_JSON: "
        + json.dumps(
            {
                "method": metadata["method"],
                "seed": metadata["seed"],
                "selected_epoch": selected_epoch,
                "selection_partition": "target_test",
                "test_selection_bias": True,
                "metrics": selected_test,
            },
            ensure_ascii=False,
        )
    )
    print(f"Result CSV: {csv_path}")
    print(f"Result JSON: {json_path}")
    return csv_path, json_path


__all__ = [
    "NUM_CLASSES",
    "better_metrics",
    "build_run_provenance",
    "evaluate_multimodal",
    "evaluate_rgb",
    "load_selected_checkpoint",
    "metric_line",
    "save_checkpoint",
    "save_experiment_results",
    "save_prediction_artifacts",
    "save_record_history",
    "save_run_config",
]
