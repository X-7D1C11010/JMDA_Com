"""Summarize held-out test evaluations without mixing UDA/SDA profiles."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result["selection_partition"] != "target_val" or result["test_selection_bias"]
                or result["test_evaluations"] != 1 or result["test_evaluations_for_selection"] != 0):
            raise ValueError(f"Invalid held-out protocol: {path}")
        protocol = result["protocol"]
        if protocol == "UDA" and result.get("target_training_labels_used") is not False:
            raise ValueError(f"UDA result lacks its label-free training contract: {path}")
        row = {"method": result["method"], "seed": result["seed"],
               "optimization_profile": result.get("optimization_profile", "baseline"),
               "selected_epoch": result["selected_epoch"], "completed_epochs": result["completed_epochs"],
               "stop_reason": result.get("stop_reason", "not_recorded"),
               "stopped_early": result.get("stopped_early"),
               "final_lr_scale": result.get("validation_control", {}).get("final_lr_scale"),
               "early_stopping_monitor": result.get("validation_control", {}).get("early_stopping_monitor"),
               "selection_partition": "target_val", "protocol": protocol, "smoke": result["smoke"],
               "val_ACC": result["target_val"]["accuracy"],
               **{key: result["target_test"][metric] for key, metric in (
                   ("ACC", "accuracy"), ("Precision_macro", "precision_macro"),
                   ("Recall_macro", "recall_macro"), ("F1_macro", "f1_macro"))},
               "test_samples": result["target_test"]["val_samples"],
               **{key: result["complexity"][key] for key in (
                   "total_params", "trainable_params", "inference_GFLOPs", "total_train_time_s",
                   "train_epoch_mean_s", "latency_mean_ms", "throughput_samples_s", "peak_train_reserved_gb")},
               "weights_RGB_NIR_AIS": json.dumps(result["modality_weights_at_selected_epoch"]),
               "result_json": str(path)}
        rows.append(row)
    if not rows:
        raise ValueError(f"No results found in {root}")
    if len({row["protocol"] for row in rows}) != 1:
        raise ValueError("Do not merge historical SDA and new UDA results into one summary")
    if len({row["optimization_profile"] for row in rows}) != 1:
        raise ValueError("Do not merge baseline and optimized configurations into one summary")
    output = root/"summary"
    output.mkdir(exist_ok=True)
    with (output/"runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        row = {"method": method, "protocol": selected[0]["protocol"],
               "optimization_profile": selected[0]["optimization_profile"],
               "seeds": " ".join(str(x["seed"]) for x in selected), "num_seeds": len(selected)}
        for key in ("ACC", "Precision_macro", "Recall_macro", "F1_macro"):
            values = [x[key] for x in selected]
            row[key+"_mean"], row[key+"_std"] = statistics.mean(values), statistics.pstdev(values)
        summary.append(row)
    with (output/"summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Per-run held-out test results: {output/'runs.csv'}", flush=True)
    print(f"Across-seed held-out test summary: {output/'summary.csv'}", flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    summarize(p.parse_args().results_dir)
