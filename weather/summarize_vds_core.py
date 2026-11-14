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
                     "image_backbone": result.get("image_backbone", "resnet18"),
                     "freeze_image_encoders": result.get("freeze_image_encoders", False),
                     "encoder_training_scope": result.get("encoder_training_scope", "frozen" if result.get("freeze_image_encoders", False) else "all"),
                     "feature_normalization": result.get("feature_normalization", "none"),
                     "unit_method_features": result.get("unit_method_features", False),
                     "classifier_layout": result.get("classifier_layout", "reference"),
                     "FDA_beta": result.get("input_augmentation", {}).get("FDA_beta", 0.),
                     "source_zoom_min_area": result.get("input_augmentation", {}).get("source_zoom_min_area", 1.),
                     "optimizer": result.get("optimizer", "not_recorded"),
                     "domain_forward": result.get("domain_forward", "reference"),
                     "stop_reason": result.get("stop_reason", "not_recorded"),
                     "selected_epoch": result["selected_epoch"], "completed_epochs": result["completed_epochs"],
                     "optimizer_updates": result["optimizer_updates"],
                     "ACC_percent": 100*metrics["accuracy"], "Precision_macro_percent": 100*metrics["precision_macro"],
                     "Recall_macro_percent": 100*metrics["recall_macro"], "F1_macro_percent": 100*metrics["f1_macro"],
                     "val_samples": metrics["val_samples"], "evaluation_protocol": result["evaluation_protocol"],
                     "smoke": result["smoke"], **{key: result["complexity"][key] for key in keys},
                     "screening_only": result.get("screening_only", False), "planned_epochs": result.get("planned_epochs", result["completed_epochs"]),
                     "result_file": str(path.resolve())})
    if not rows:
        raise FileNotFoundError(f"No completed weather results: {root}")
    if len({row["weather_profile"] for row in rows}) != 1:
        raise ValueError("Do not merge different weather profiles")
    if len({row["screening_only"] for row in rows}) != 1:
        raise ValueError("Do not merge prefix screens and complete experiments")
    signatures = {(row["image_backbone"], row["encoder_training_scope"], row["feature_normalization"],
                   row["classifier_layout"], row["FDA_beta"], row["source_zoom_min_area"]) for row in rows}
    if len(signatures) != 1:
        raise ValueError("Do not merge different encoder or fusion-normalization protocols")
    identities = [(row["method"], row["target"], row["seed"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated method/target/seed; keep parameter trials in separate summaries")
    output = root/"summary"
    output.mkdir(exist_ok=True)
    with (output/"runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    signature = rows[0]
    lines = ["Weather VIS+IR UDA; peak target_val within each run (no separate target_test).",
             f"Encoder={signature['image_backbone']}; frozen={signature['freeze_image_encoders']}; fusion normalization={signature['feature_normalization']}.", "",
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
