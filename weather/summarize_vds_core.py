"""Summarize completed weather runs without mixing held-out test metrics."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    rows = []
    keys = ("total_params", "trainable_params", "inference_GFLOPs", "total_train_time_s",
            "train_epoch_mean_s", "latency_mean_ms", "throughput_samples_s", "peak_train_reserved_gb")
    for path in sorted(root.glob("*_seed*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("core") != "vds2raw.adaptation.AdaptationModel":
            raise ValueError(f"Unexpected training core: {path}")
        if result.get("target_training_labels_used") is not False:
            raise ValueError(f"Unexpected supervision: {path}")
        complete = json.loads((path.parent/"completeness.json").read_text(encoding="utf-8"))
        if complete["status"] != "PASS":
            raise ValueError(f"Incomplete run: {path}")
        metrics = result["target_val"]
        rows.append({"method": result["method"], "target": result["target"], "seed": result["seed"],
                     "weather_profile": result.get("weather_profile", "baseline"),
                     "domain_forward": result.get("domain_forward", "reference"),
                     "stop_reason": result.get("stop_reason", "not_recorded"),
                     "selected_epoch": result["selected_epoch"], "completed_epochs": result["completed_epochs"],
                     "optimizer_updates": result["optimizer_updates"],
                     "ACC_percent": 100*metrics["accuracy"], "Precision_macro_percent": 100*metrics["precision_macro"],
                     "Recall_macro_percent": 100*metrics["recall_macro"], "F1_macro_percent": 100*metrics["f1_macro"],
                     "val_samples": metrics["val_samples"], "evaluation_protocol": result["evaluation_protocol"],
                     "smoke": result["smoke"], **{key: result["complexity"][key] for key in keys},
                     "result_file": str(path.resolve())})
    if not rows:
        raise FileNotFoundError(f"No completed weather results: {root}")
    if len({row["weather_profile"] for row in rows}) != 1:
        raise ValueError("Do not merge baseline and timing_v2 weather profiles")
    output = root/"summary"
    output.mkdir(exist_ok=True)
    with (output/"runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Weather VIS+IR UDA; peak target_val within each run (no separate target_test).", "",
             "| Method | Target | Seed | Best epoch | ACC | Precision | Recall | F1 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['target']} | {row['seed']} | {row['selected_epoch']} | "
                     + " | ".join(f"{row[k]:.2f}" for k in ("ACC_percent", "Precision_macro_percent", "Recall_macro_percent", "F1_macro_percent"))+" |")
    (output/"results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Weather best-epoch results: {output/'runs.csv'}", flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    summarize(p.parse_args().results_dir)
