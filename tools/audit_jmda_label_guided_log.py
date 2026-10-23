"""Audit a completed label-guided Tensor+OT run with target CE disabled."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


SUMMARY_RE = re.compile(r"(?m)^JMDA_RUN_SUMMARY_JSON:\s*(\{.*\})\s*$")
EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] \| Loss: .*?"
    r"src/tgt/mid=([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+); "
    r"labeled-target-observations=(\d+)"
)
SVD_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] \| SVD update #\d+:.*?"
    r"class_balanced=(True|False), "
    r"target_labels_source=([^,\s]+)"
)
OT_RE = re.compile(
    r"OT scale: ([0-9.]+) \| OT inner objective: ([-+0-9.eE]+) \| "
    r"Marginal residual row/col/max: "
    r"([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+)"
)


def audit_text(text: str, expected_iterations: int) -> tuple[dict, list[str]]:
    errors: list[str] = []
    summaries = SUMMARY_RE.findall(text)
    if not summaries:
        return {}, ["JMDA_RUN_SUMMARY_JSON is missing"]
    summary = json.loads(summaries[-1])

    expected_fields = {
        "protocol": "LABEL_GUIDED_DA_NO_TARGET_CE",
        "adaptation_protocol": "LABEL_GUIDED_DA",
        "classification_protocol": "UDA",
        "label_space_mode": "oracle_shared",
        "uda_pairing_mode": "target_ground_truth_label_guided",
        "target_ground_truth_used_for_pairing": True,
        "target_training_item_labels_exposed": False,
        "label_guided_alignment": True,
        "class_conditional_ot": True,
        "balance_svd_classes": True,
        "transport_mode": "sinkhorn",
    }
    for name, expected in expected_fields.items():
        if summary.get(name) != expected:
            errors.append(
                f"summary {name}={summary.get(name)!r}, expected {expected!r}"
            )
    if summary.get("num_iterations") != expected_iterations:
        errors.append(
            "summary num_iterations="
            f"{summary.get('num_iterations')!r}, expected {expected_iterations}"
        )

    epochs = list(EPOCH_RE.finditer(text))
    if not epochs:
        errors.append("epoch loss records are missing")
    else:
        target_ce_max = max(abs(float(match.group(4))) for match in epochs)
        labeled_observations = max(int(match.group(6)) for match in epochs)
        if target_ce_max > 1e-12:
            errors.append(f"target classification CE is non-zero: {target_ce_max}")
        if labeled_observations != 0:
            errors.append(
                "target classification labels reached CE: "
                f"labeled-target-observations={labeled_observations}"
            )

    svd_rows = list(SVD_RE.finditer(text))
    if not svd_rows:
        errors.append("label-guided SVD updates are missing")
    for row in svd_rows:
        if row.group(3) != "True":
            errors.append(
                f"epoch {row.group(1)} SVD statistics are not class-balanced"
            )
        if row.group(4) != "target-ground-truth-label-guided":
            errors.append(
                f"epoch {row.group(1)} SVD label source is {row.group(4)!r}"
            )

    ot_rows = list(OT_RE.finditer(text))
    active_ot = [row for row in ot_rows if float(row.group(1)) > 0.0]
    if not active_ot:
        errors.append("no active Sinkhorn/OT epoch was recorded")
    for row in active_ot:
        values = [float(row.group(index)) for index in range(2, 6)]
        if not all(math.isfinite(value) for value in values):
            errors.append("non-finite OT objective or marginal residual detected")
            break

    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--expected-iterations", type=int, default=1)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    summary, errors = audit_text(text, args.expected_iterations)
    for error in errors:
        print(f"[ERROR] {error}")
    if errors:
        return 1

    metrics = summary["metrics"]
    print("[OK] label-guided protocol audit passed")
    print(
        "final mean ACC/P/R/F1="
        f"{metrics['accuracy']['mean']:.6f}/"
        f"{metrics['precision_macro']['mean']:.6f}/"
        f"{metrics['recall_macro']['mean']:.6f}/"
        f"{metrics['f1_macro']['mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
