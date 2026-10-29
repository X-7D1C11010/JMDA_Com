"""Combine VDS2Raw per-seed final-test records without re-evaluating test data."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


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
        records.append(
            {
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
        )
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


if __name__ == "__main__":
    main()
