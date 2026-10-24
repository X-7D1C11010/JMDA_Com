"""Shared training, model-selection, and final-test result helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from tools.classification_metrics import (
    classification_metrics,
    format_classification_metrics,
)
from tools.metrics_io import save_metric_history

NUM_CLASSES = 3


def evaluate_rgb(model, loader, device):
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for images, truth in loader:
            logits = model(images.to(device))
            labels.extend(truth.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(labels, predictions, num_classes=NUM_CLASSES)


def evaluate_multimodal(model, loader, device):
    model.eval()
    labels, predictions = [], []
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
            labels.extend(truth.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(labels, predictions, num_classes=NUM_CLASSES)


def metric_line(split, metrics):
    return (
        f"{split} accuracy: {metrics['accuracy']:.4f}\n"
        f"  {format_classification_metrics(metrics)}"
    )


def better_metrics(candidate, incumbent):
    if incumbent is None:
        return True
    return (candidate["accuracy"], candidate["f1_macro"]) > (
        incumbent["accuracy"],
        incumbent["f1_macro"],
    )


def save_checkpoint(path, model, epoch, val_metrics, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "selected_epoch": int(epoch),
            "target_val_metrics": dict(val_metrics),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_selected_checkpoint(path, model, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return payload


def save_experiment_results(
    save_dir,
    stem,
    val_history,
    selected_epoch,
    selected_val,
    test_metrics,
    metadata,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history_csv, history_json = save_metric_history(
        str(save_dir), stem + "_target_val", val_history, metadata=metadata
    )
    result = {
        "protocol": "cross-day supervised domain adaptation",
        "model_selection": "best target_val ACC, macro-F1 tie-break",
        "test_evaluations": 1,
        "selected_epoch": int(selected_epoch),
        "target_val": dict(selected_val),
        "target_test": dict(test_metrics),
        "metadata": metadata,
        "validation_history_csv": str(Path(history_csv).resolve()),
        "validation_history_json": str(Path(history_json).resolve()),
    }
    json_path = save_dir / f"{stem}_result.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(json_path)

    csv_path = save_dir / f"{stem}_result.csv"
    fields = (
        "method",
        "seed",
        "selected_epoch",
        "split",
        "ACC",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "samples",
    )
    rows = []
    for split, metrics in (
        ("target_val_selected", selected_val),
        ("target_test", test_metrics),
    ):
        rows.append(
            {
                "method": metadata["method"],
                "seed": metadata["seed"],
                "selected_epoch": selected_epoch,
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
                "metrics": test_metrics,
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
    "evaluate_multimodal",
    "evaluate_rgb",
    "load_selected_checkpoint",
    "metric_line",
    "save_checkpoint",
    "save_experiment_results",
]
