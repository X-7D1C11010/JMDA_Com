"""Summarize the controlled JMDA UDA fault-localization matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean


SUMMARY_RE = re.compile(r"(?m)^JMDA_RUN_SUMMARY_JSON:\s*(\{.*\})\s*$")
TAG_RE = re.compile(r"(?m)^JMDA_SCREEN_EXPERIMENT=(\S+)\s*$")
TARGET_RE = re.compile(r"(?m)^JMDA_SCREEN_TARGET=(\S+)\s*$")
EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] \| Loss: .*?"
    r"src/tgt/mid=([-+0-9.eE]+)/([-+0-9.eE]+)/([-+0-9.eE]+); "
    r"labeled-target-observations=(\d+).*?"
    r"Source Train Acc: ([0-9.]+) \| Val Acc: ([0-9.]+) \| "
    r"Val P/R/F1 \(Macro\): ([0-9.]+)/([0-9.]+)/([0-9.]+)"
)
SVD_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] \| SVD update #1:.*?"
    r"effective_ranks=\((\d+), (\d+)\).*?"
    r"basis_overlap_src/tgt=([0-9.]+)/([0-9.]+).*?"
    r"basis_min_cos_src/tgt=([0-9.]+)/([0-9.]+).*?"
    r"basis_nonorth=([-+0-9.eE]+)"
)


def average_epoch(rows: list[dict], epoch: int) -> dict | None:
    selected = [row for row in rows if row["epoch"] == epoch]
    if not selected:
        return None
    return {
        key: mean(row[key] for row in selected)
        for key in ("source_acc", "val_acc", "precision", "recall", "f1")
    }


def metric(summary: dict, name: str) -> float:
    return float(summary.get("metrics", {}).get(name, {}).get("mean", math.nan))


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    tag_match = TAG_RE.search(text)
    target_match = TARGET_RE.search(text)
    summary_matches = SUMMARY_RE.findall(text)
    if not tag_match or not target_match or not summary_matches:
        raise ValueError("missing screening marker or JMDA_RUN_SUMMARY_JSON")
    summary = json.loads(summary_matches[-1])

    epoch_rows = []
    for match in EPOCH_RE.finditer(text):
        epoch_rows.append(
            {
                "epoch": int(match.group(1)),
                "total": int(match.group(2)),
                "source_loss": float(match.group(3)),
                "target_loss": float(match.group(4)),
                "mid_loss": float(match.group(5)),
                "target_observations": int(match.group(6)),
                "source_acc": float(match.group(7)),
                "val_acc": float(match.group(8)),
                "precision": float(match.group(9)),
                "recall": float(match.group(10)),
                "f1": float(match.group(11)),
            }
        )
    if not epoch_rows:
        raise ValueError("no epoch metrics found")

    total_epochs = max(row["total"] for row in epoch_rows)
    warmup = average_epoch(epoch_rows, 10)
    transition = average_epoch(epoch_rows, 11)
    final = average_epoch(epoch_rows, total_epochs)
    first_svd = SVD_RE.search(text)
    target_ce_max = max(abs(row["target_loss"]) for row in epoch_rows)
    target_observations_max = max(row["target_observations"] for row in epoch_rows)
    target_labels_exposed = bool(summary.get("target_training_item_labels_exposed", True))
    protocol_safe = (
        target_ce_max <= 1e-12
        and target_observations_max == 0
        and not target_labels_exposed
    )

    return {
        "target": target_match.group(1),
        "experiment": tag_match.group(1),
        "protocol": summary.get("protocol"),
        "pairing_mode": summary.get("uda_pairing_mode"),
        "target_truth_for_pairing": bool(
            summary.get("target_ground_truth_used_for_pairing", False)
        ),
        "epochs": total_epochs,
        "iterations": int(summary.get("num_iterations", 0)),
        "warmup_acc": warmup["val_acc"] if warmup else math.nan,
        "transition_acc": transition["val_acc"] if transition else math.nan,
        "transition_source_acc": transition["source_acc"] if transition else math.nan,
        "final_acc": metric(summary, "accuracy"),
        "precision_macro": metric(summary, "precision_macro"),
        "recall_macro": metric(summary, "recall_macro"),
        "f1_macro": metric(summary, "f1_macro"),
        "delta_from_warmup": (
            metric(summary, "accuracy") - warmup["val_acc"]
            if warmup else math.nan
        ),
        "target_ce_max": target_ce_max,
        "target_observations_max": target_observations_max,
        "target_training_labels_exposed": target_labels_exposed,
        "protocol_safe": protocol_safe,
        "first_svd_source_rank": int(first_svd.group(3)) if first_svd else None,
        "first_svd_target_rank": int(first_svd.group(4)) if first_svd else None,
        "first_svd_source_overlap": float(first_svd.group(5)) if first_svd else None,
        "first_svd_target_overlap": float(first_svd.group(6)) if first_svd else None,
        "first_svd_source_min_cos": float(first_svd.group(7)) if first_svd else None,
        "first_svd_target_min_cos": float(first_svd.group(8)) if first_svd else None,
        "first_svd_nonorth": float(first_svd.group(9)) if first_svd else None,
        "log": str(path.resolve()),
    }


def pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def diagnosis(rows: list[dict], target: str) -> list[str]:
    by_name = {row["experiment"]: row for row in rows if row["target"] == target}
    required = {
        "source_only_fixed_projection",
        "pseudo_tensor_only",
        "oracle_tensor_only",
        "pseudo_tensor_ot",
        "oracle_tensor_ot",
    }
    if not required.issubset(by_name):
        missing = sorted(required - set(by_name))
        return [f"实验矩阵不完整，缺少：{', '.join(missing)}。"]

    source = by_name["source_only_fixed_projection"]["final_acc"]
    pseudo_tensor = by_name["pseudo_tensor_only"]["final_acc"]
    oracle_tensor = by_name["oracle_tensor_only"]["final_acc"]
    pseudo_full = by_name["pseudo_tensor_ot"]["final_acc"]
    oracle_full = by_name["oracle_tensor_ot"]["final_acc"]
    statements = []
    warmup_values = [by_name[name]["warmup_acc"] for name in sorted(required)]
    finite_warmups = [value for value in warmup_values if math.isfinite(value)]
    if finite_warmups and max(finite_warmups) - min(finite_warmups) > 1e-6:
        statements.append(
            "警告：五组实验在第10轮预热末的准确率不一致，说明初始化或数据顺序"
            "未完全受控，本组因果比较暂不可信。"
        )
    else:
        statements.append("五组实验的源域预热终点一致，可进行后续单变量比较。")
    oracle_gain = oracle_tensor - pseudo_tensor
    if oracle_gain >= 0.05 and oracle_tensor >= source - 0.03:
        statements.append(
            "真实类别配对显著优于伪标签配对，且接近/超过 source-only："
            "主要瓶颈是伪标签对应。"
        )
    elif oracle_tensor < source - 0.05:
        statements.append(
            "即使使用真实目标类别，Tensor/SVD 仍明显低于 source-only："
            "问题不只来自伪标签，还包括投影更新、基变换或配对采样流程。"
        )
    else:
        statements.append(
            "真实类别配对带来的增益不足以单独确定瓶颈，需要结合首次SVD后的"
            "准确率和子空间诊断继续判断。"
        )

    pseudo_ot_delta = pseudo_full - pseudo_tensor
    oracle_ot_delta = oracle_full - oracle_tensor
    if oracle_ot_delta < -0.05:
        statements.append("在真实类别对应下加入 OT 仍显著下降，OT 路径存在独立负迁移。")
    elif oracle_ot_delta > 0.03:
        statements.append("在真实类别对应下 OT 有正增益，OT 本身可工作，其效果依赖正确语义对应。")
    else:
        statements.append("真实类别对应下 OT 的增益很小，暂不能证明 OT 有效。")
    statements.append(
        "定量差值：oracle-vs-pseudo Tensor="
        f"{pct(oracle_gain)}，pseudo OT增量={pct(pseudo_ot_delta)}，"
        f"oracle OT增量={pct(oracle_ot_delta)}。"
    )
    return statements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    errors = []
    for path in sorted(args.log_dir.glob("*.log")):
        try:
            rows.append(parse_log(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{path.name}: {error}")

    fieldnames = [
        "target", "experiment", "protocol", "pairing_mode",
        "target_truth_for_pairing", "epochs", "iterations", "warmup_acc",
        "transition_acc", "transition_source_acc", "final_acc",
        "precision_macro", "recall_macro", "f1_macro", "delta_from_warmup",
        "target_ce_max", "target_observations_max",
        "target_training_labels_exposed", "protocol_safe",
        "first_svd_source_rank", "first_svd_target_rank",
        "first_svd_source_overlap", "first_svd_target_overlap",
        "first_svd_source_min_cos", "first_svd_target_min_cos",
        "first_svd_nonorth", "log",
    ]
    csv_path = args.output_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# JMDA UDA 筛选实验汇总",
        "",
        "> 这些结果仅用于定位伪标签、SVD/Tensor 和 OT 的故障来源，不能进入论文的 SDA/UDA 对比表。",
        "",
        "| 目标域 | 实验 | 配对来源 | 预热末Acc | 第11轮Acc | 最终Acc | Macro-F1 | 相对预热 | 目标CE=0 |",
        "|---|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['experiment']} | {row['pairing_mode']} | "
            f"{pct(row['warmup_acc'])} | {pct(row['transition_acc'])} | "
            f"{pct(row['final_acc'])} | {pct(row['f1_macro'])} | "
            f"{pct(row['delta_from_warmup'])} | "
            f"{'是' if row['protocol_safe'] else '否'} |"
        )

    for target in sorted({row["target"] for row in rows}):
        lines.extend(["", f"## {target} 诊断", ""])
        lines.extend(f"- {item}" for item in diagnosis(rows, target))
    if errors:
        lines.extend(["", "## 未成功解析的日志", ""])
        lines.extend(f"- {error}" for error in errors)

    md_path = args.output_prefix.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    for error in errors:
        print(f"[WARN] {error}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
