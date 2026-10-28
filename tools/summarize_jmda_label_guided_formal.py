"""Aggregate the four-weather, five-seed label-guided JMDA experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics


SUMMARY_RE = re.compile(r"(?m)^JMDA_RUN_SUMMARY_JSON:\s*(\{.*\})\s*$")
TARGET_CN = {
    "backlight": "逆光",
    "rainy": "雨天",
    "foggy": "雾天",
    "dark": "黑天",
}
METRICS = (
    ("accuracy", "ACC"),
    ("precision_macro", "Precision"),
    ("recall_macro", "Recall"),
    ("f1_macro", "F1"),
)
EFFICIENCY = (
    ("training_total_params", "Parameters"),
    ("training_trainable_params", "Trainable parameters"),
    ("inference_gflops", "GFLOPs"),
    ("total_training_time_s", "Training time (s)"),
    ("train_epoch_mean_s", "Epoch time (s)"),
    ("inference_latency_mean_ms", "Latency (ms/sample)"),
    ("inference_throughput_samples_s", "Throughput (samples/s)"),
    ("peak_train_allocated_gb", "Peak train GPU (GB)"),
    ("peak_inference_allocated_gb", "Peak infer GPU (GB)"),
)


def _stats(values):
    return {
        "values": [float(value) for value in values],
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
    }


def _read_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = SUMMARY_RE.findall(text)
    if not matches:
        raise ValueError(f"JMDA_RUN_SUMMARY_JSON missing: {path}")
    return json.loads(matches[-1])


def summarize(log_dir: Path, targets: list[str], expected_iterations: int):
    rows = []
    full = {}
    errors = []
    for target in targets:
        path = log_dir / f"{target}_label_guided_tensor_ot.log"
        if not path.is_file():
            errors.append(f"missing formal log: {path}")
            continue
        try:
            summary = _read_summary(path)
        except (ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
            continue
        if summary.get("protocol") != "LABEL_GUIDED_DA_NO_TARGET_CE":
            errors.append(f"{target}: unexpected protocol={summary.get('protocol')!r}")
        if summary.get("num_iterations") != expected_iterations:
            errors.append(
                f"{target}: iterations={summary.get('num_iterations')}, "
                f"expected={expected_iterations}"
            )
        if summary.get("report_strategy") != "best":
            errors.append(
                f"{target}: report_strategy={summary.get('report_strategy')!r}, "
                "expected 'best'"
            )

        row = {"target": target, "target_cn": TARGET_CN[target]}
        for key, label in METRICS:
            stats = summary.get("metrics", {}).get(key, {})
            values = stats.get("values", [])
            if len(values) != expected_iterations:
                errors.append(
                    f"{target}: {key} values={len(values)}, "
                    f"expected={expected_iterations}"
                )
                continue
            row[f"{key}_mean"] = float(stats["mean"])
            row[f"{key}_std"] = float(stats["std"])

        efficiency = summary.get("efficiency", {})
        runs = efficiency.get("per_iteration", [])
        if not efficiency.get("enabled") or len(runs) != expected_iterations:
            errors.append(
                f"{target}: complete efficiency evidence is missing "
                f"({len(runs)}/{expected_iterations})"
            )
        for key, _ in EFFICIENCY:
            values = [run.get(key) for run in runs]
            if (
                len(values) != expected_iterations
                or any(value is None for value in values)
            ):
                errors.append(f"{target}: incomplete efficiency field {key}")
                continue
            stats = _stats([float(value) for value in values])
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
        row["log"] = str(path.resolve())
        rows.append(row)
        full[target] = summary
    return rows, full, errors


def write_outputs(output_dir: Path, rows: list[dict], full: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "formal_results.csv"
    json_path = output_dir / "formal_results.json"
    md_path = output_dir / "formal_results.md"

    csv_fields = ["target", "target_cn"]
    for key, _ in METRICS:
        csv_fields.extend([f"{key}_mean", f"{key}_std"])
    for key, _ in EFFICIENCY:
        csv_fields.extend([f"{key}_mean", f"{key}_std"])
    csv_fields.append("log")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    overall_metrics = {}
    for key, _ in METRICS:
        values = [row[f"{key}_mean"] for row in rows]
        overall_metrics[key] = _stats(values)
    payload = {
        "protocol": "LABEL_GUIDED_DA_NO_TARGET_CE",
        "report_strategy": "best_validation_accuracy_epoch",
        "metric_coherence": "precision/recall/F1 come from the ACC-selected epoch",
        "weather_count": len(rows),
        "weather_results": rows,
        "weather_mean_metrics": overall_metrics,
        "raw_summaries": full,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# JMDA formal label-guided experiment",
        "",
        "> Each seed reports the epoch with maximum validation ACC; Precision, "
        "Recall and F1 are taken from that same epoch.",
        "",
        "## Classification metrics",
        "",
        "| Target | ACC | Precision (macro) | Recall (macro) | F1 (macro) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = []
        for key, _ in METRICS:
            values.append(
                f"{100 * row[f'{key}_mean']:.2f} ± "
                f"{100 * row[f'{key}_std']:.2f}"
            )
        lines.append(f"| {row['target_cn']} | " + " | ".join(values) + " |")
    overall = []
    for key, _ in METRICS:
        stats = overall_metrics[key]
        overall.append(f"{100 * stats['mean']:.2f} ± {100 * stats['std']:.2f}")
    aggregate_label = (
        "Four-weather mean" if len(rows) == 4 else "Selected-target mean"
    )
    lines.append(f"| {aggregate_label} | " + " | ".join(overall) + " |")

    lines.extend([
        "",
        "## Computational complexity and efficiency",
        "",
        "| Target | Params | GFLOPs | Training time (s) | Latency (ms) | "
        "Throughput (samples/s) | Peak train GPU (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['target_cn']} | "
            f"{row['training_total_params_mean']:.0f} | "
            f"{row['inference_gflops_mean']:.4f} | "
            f"{row['total_training_time_s_mean']:.2f} ± "
            f"{row['total_training_time_s_std']:.2f} | "
            f"{row['inference_latency_mean_ms_mean']:.4f} ± "
            f"{row['inference_latency_mean_ms_std']:.4f} | "
            f"{row['inference_throughput_samples_s_mean']:.2f} ± "
            f"{row['inference_throughput_samples_s_std']:.2f} | "
            f"{row['peak_train_allocated_gb_mean']:.4f} ± "
            f"{row['peak_train_allocated_gb_std']:.4f} |"
        )
    lines.extend([
        "",
        "> FLOPs use 1 MAC = 2 FLOPs. Inference timing uses batch size 1 and "
        "excludes data loading. Training time includes pairing, epoch SVD, "
        "Tensor/OT optimization and validation.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["backlight", "rainy", "foggy", "dark"],
        choices=sorted(TARGET_CN),
    )
    parser.add_argument("--expected-iterations", type=int, default=5)
    args = parser.parse_args()

    rows, full, errors = summarize(
        args.log_dir,
        args.targets,
        args.expected_iterations,
    )
    for error in errors:
        print(f"[ERROR] {error}")
    if errors:
        return 1
    paths = write_outputs(args.output_dir, rows, full)
    print("[OK] formal JMDA summary completed")
    for path in paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
