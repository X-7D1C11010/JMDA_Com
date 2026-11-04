"""Combine VDS2Raw per-seed final-test records without re-evaluating test data."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
EARLY_PEAK_FRACTION = 0.25
MIN_BUDGET_FOR_EARLY_PEAK = 10


def _local_artifact_path(results_dir, recorded_path):
    """Resolve copied artifacts even when JSON contains a server-side path."""
    if not recorded_path:
        return None
    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded
    filename = str(recorded_path).replace("\\", "/").rsplit("/", 1)[-1]
    candidate = results_dir / filename
    return candidate if candidate.is_file() else None


def training_curve_diagnostics(payload, results_dir):
    """Describe where the selected test peak occurs within the planned budget."""
    metadata = payload["metadata"]
    selected_epoch = int(payload["selected_epoch"])
    history_path = _local_artifact_path(
        results_dir, payload.get("target_test_history_csv")
    )
    history = []
    if history_path is not None:
        with history_path.open(encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))

    planned_epochs = int(
        metadata.get("adaptation_budget_epochs", metadata.get("epochs", 0)) or 0
    )
    last_epoch = int(history[-1]["epoch"]) if history else int(
        metadata.get("completed_epochs", selected_epoch)
    )
    # The recorded history is authoritative when early stopping shortens a run.
    completed_epochs = last_epoch
    early_threshold = (
        max(3, math.ceil(planned_epochs * EARLY_PEAK_FRACTION))
        if planned_epochs >= MIN_BUDGET_FOR_EARLY_PEAK
        else 0
    )
    early_peak = bool(early_threshold and selected_epoch <= early_threshold)
    last_acc = float(history[-1]["ACC"]) if history else None
    last_f1 = float(history[-1]["F1_macro"]) if history else None
    selected_acc = float(payload["target_test"]["accuracy"])
    selected_f1 = float(payload["target_test"]["f1_macro"])
    return {
        "planned_epochs": planned_epochs,
        "completed_epochs": completed_epochs,
        "last_evaluated_epoch": last_epoch,
        "selected_epoch_fraction_of_budget": (
            selected_epoch / planned_epochs if planned_epochs else None
        ),
        "early_peak_threshold_epoch": early_threshold,
        "early_peak": early_peak,
        "selected_is_last_evaluation": selected_epoch == last_epoch,
        "last_ACC": last_acc,
        "last_F1_macro": last_f1,
        "peak_to_last_ACC_drop_pp": (
            100.0 * (selected_acc - last_acc) if last_acc is not None else None
        ),
        "peak_to_last_F1_drop_pp": (
            100.0 * (selected_f1 - last_f1) if last_f1 is not None else None
        ),
    }


def _mean_available(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = []
    for path in sorted(results_dir.glob("*_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        test = payload["target_test"]
        record = {
            "method": metadata["method"],
            "seed": int(metadata["seed"]),
            "selected_epoch": int(payload["selected_epoch"]),
            "selection_partition": payload.get("selection_partition", "unknown"),
            "test_selection_bias": bool(payload.get("test_selection_bias", False)),
            "test_evaluations_for_selection": int(
                payload.get("test_evaluations_for_selection", 0)
            ),
            "ACC": float(test["accuracy"]),
            "Precision_macro": float(test["precision_macro"]),
            "Recall_macro": float(test["recall_macro"]),
            "F1_macro": float(test["f1_macro"]),
            "test_samples": int(test["val_samples"]),
            "result_json": str(path),
        }
        record.update(training_curve_diagnostics(payload, results_dir))
        records.append(record)
    if not records:
        raise RuntimeError(f"no *_result.json files found in {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.csv"
    with runs_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)

    groups = defaultdict(list)
    for record in records:
        groups[record["method"]].append(record)
    summary_rows = []
    key_map = {
        "accuracy": "ACC",
        "precision_macro": "Precision_macro",
        "recall_macro": "Recall_macro",
        "f1_macro": "F1_macro",
    }
    for method, rows in sorted(groups.items()):
        summary = {
            "method": method,
            "num_seeds": len(rows),
            "seeds": " ".join(
                str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])
            ),
            "selection_partition": rows[0]["selection_partition"],
            "test_selection_bias": rows[0]["test_selection_bias"],
            "selected_epoch_mean": statistics.mean(
                row["selected_epoch"] for row in rows
            ),
            "selected_epoch_std": statistics.pstdev(
                row["selected_epoch"] for row in rows
            ),
            "early_peak_seeds": sum(bool(row["early_peak"]) for row in rows),
            "early_peak_rate": statistics.mean(
                bool(row["early_peak"]) for row in rows
            ),
            "peak_to_last_ACC_drop_pp_mean": _mean_available(
                rows, "peak_to_last_ACC_drop_pp"
            ),
            "peak_to_last_F1_drop_pp_mean": _mean_available(
                rows, "peak_to_last_F1_drop_pp"
            ),
        }
        for metric in METRICS:
            values = [row[key_map[metric]] for row in rows]
            summary[f"{key_map[metric]}_mean"] = statistics.mean(values)
            summary[f"{key_map[metric]}_std"] = statistics.pstdev(values)
        summary_rows.append(summary)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(runs_path)
    print(summary_path)
    for record in records:
        if record["early_peak"]:
            print(
                "[EARLY-PEAK] "
                f"{record['method']} seed={record['seed']}: selected epoch "
                f"{record['selected_epoch']}/{record['planned_epochs']} "
                f"(threshold <= {record['early_peak_threshold_epoch']}); "
                "treat the optimization schedule as needing review"
            )


if __name__ == "__main__":
    main()
