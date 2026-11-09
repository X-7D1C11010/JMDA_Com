"""Summarize independent TIR weather runs from their per-epoch metric JSON."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ("ACC", "Precision_macro", "Recall_macro", "F1_macro")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(results_dir.glob("tir_weather_*_seed*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        best = payload.get("best_by_ACC_then_F1") or payload.get("best_by_ACC")
        if not best:
            raise ValueError(f"missing best metrics in {path}")
        rows.append(
            {
                "target": metadata.get("target"),
                "seed": int(metadata.get("seed")),
                "best_epoch": int(best["epoch"]),
                **{name: float(best[name]) for name in METRICS},
                "metrics_json": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no TIR weather metric JSON files in {results_dir}")

    runs_path = output_dir / "runs.csv"
    with runs_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for target in sorted({row["target"] for row in rows}):
        target_rows = [row for row in rows if row["target"] == target]
        summary = {"target": target, "runs": len(target_rows)}
        for name in ("best_epoch", *METRICS):
            values = [float(row[name]) for row in target_rows]
            summary[f"{name}_mean"] = statistics.fmean(values)
            summary[f"{name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary_rows.append(summary)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"TIR weather run table: {runs_path}")
    print(f"TIR weather mean/std summary: {summary_path}")


if __name__ == "__main__":
    main()
