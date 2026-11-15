"""Collect completed local weather screens, without mixing probes and UDA scores."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ("remaining_weather_v10_calibrated_r18_seed42", "remaining_weather_v11_joint_rainy_seed42")


def collect(directory):
    plan = json.loads((directory/"candidate_plan.json").read_text(encoding="utf-8"))
    status = json.loads((directory/"queue_status.json").read_text(encoding="utf-8"))
    if status["status"] != "PASS":
        raise ValueError(f"Queue is incomplete: {directory}")
    rows = []
    for path in sorted(directory.glob("*/*_seed*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        args = json.loads((path.parent/"run_config.json").read_text(encoding="utf-8"))["arguments"]
        if result["target_training_labels_used"] is not False or result["smoke"]:
            raise ValueError(f"Unexpected supervision/smoke result: {path}")
        events = [json.loads(line) for line in (path.parent/"evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
        validation = [event for event in events if event["partition"] == "target_val"]
        source = [event for event in events if event["partition"] == "source_train"]
        after = [event for event in validation if event["epoch"] > args["adaptation_warmup_epochs"]]
        best_after = max(after, key=lambda event: (event["metrics"]["accuracy"], event["metrics"]["f1_macro"])) if after else None
        metrics = result["target_val"]
        rows.append({"variant": plan["hypothesis"], "target": result["target"], "method": result["method"],
                     "seed": result["seed"], "selected_epoch": result["selected_epoch"],
                     "completed_epochs": result["completed_epochs"], "planned_epochs": result["planned_epochs"],
                     **{name: 100*metrics[key] for name, key in (("ACC_percent", "accuracy"),
                        ("Precision_percent", "precision_macro"), ("Recall_percent", "recall_macro"), ("F1_percent", "f1_macro"))},
                     "selected_before_adaptation": result["selected_epoch"] <= args["adaptation_warmup_epochs"],
                     "last_ACC_percent": 100*validation[-1]["metrics"]["accuracy"],
                     "post_adaptation_peak_ACC_percent": 100*best_after["metrics"]["accuracy"] if best_after else None,
                     "last_source_ACC_percent": 100*source[-1]["metrics"]["accuracy"],
                     "result_file": path.relative_to(ROOT).as_posix()})
    if len(rows) != len(plan["candidates"]):
        raise ValueError(f"Missing candidate results: {directory}")
    return rows


def main():
    rows = [row for run in DEFAULT_RUNS for row in collect(ROOT/"runs/weather_local_screen"/run)]
    rows.sort(key=lambda row: (row["variant"], row["target"], row["method"]))
    output = ROOT/"results"
    with (output/"remaining_weather_seed42_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    record = {"seed": 42, "completed_real_training_runs": len(rows), "formal_release_ready": False,
              "selection_rule": "peak target_val ACC; macro F1 tie-break",
              "source_probe_CV_caveat": "fixed-encoder linear classifier CV only; encoder may have seen all source labels",
              "runs": rows}
    (output/"remaining_weather_seed42_results.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    lines = ["# 其余天气本地 seed42 实测（2026-11-14）", "",
             "完成五方法×雨/雾/黑15项真实UDA筛选，另完成雨天DANN/Deep CORAL两项追加测试，共17项。服务器任务由用户启动，本轮没有连接、暂停或修改服务器任务。", "",
             "## 首轮15项最高验证ACC（%）", "", "| 方法 | 雨天 | 雾天 | 黑天 |", "|---|---:|---:|---:|"]
    for method in ("dann", "deepcoral", "cdan", "mdd", "mcc"):
        selected = [next(row for row in rows if row["variant"] == "calibrated_r18_v10" and row["method"] == method and row["target"] == target)
                    for target in ("rainy", "foggy", "dark")]
        lines.append("| "+method+" | "+" | ".join(f"{row['ACC_percent']:.2f}" for row in selected)+" |")
    lines += ["", "雨/雾为60轮计划的前20轮，黑天为前30轮，保留完整LR/GRL时钟。本轮筛选未通过，尚不能替代历史最优；完整60轮的后期表现未验证。完整Precision/Recall/F1、最佳epoch、末轮和适配阶段峰值见CSV/JSON。", "",
              "## 两项追加测试", ""]
    for row in rows:
        if row["variant"] == "joint_r18_v11":
            lines.append(f"- 雨天{row['method']}：最高ACC {row['ACC_percent']:.2f}%，F1 {row['F1_percent']:.2f}%，最佳epoch{row['selected_epoch']}；适配后峰值{row['post_adaptation_peak_ACC_percent']:.2f}%，末轮{row['last_ACC_percent']:.2f}%。")
    lines += ["", "追加测试先冻结骨干5轮训练分类器，再解冻联合适配，骨干学习率提高到0.0003。峰值仍在解冻/适配之前，不能归因于适配能力提高。参数记录证明冻结期间骨干更新为0、解冻后非0。", "",
              "首轮雨天/黑天候选最高ACC均发生在适配启动前；雾天多数在适配初期。仍按用户约定报告全程最高ACC，另存适配后峰值用于分析，未修改评价规则。", "",
              "## 诊断与检查", "",
              "- 核对17次训练实际返回的214730条源域/验证标签，均与物理类别文件夹及紧凑映射一致；127190条目标训练预测记录没有真实类别标签。证据：results/remaining_weather_returned_label_audit.json。",
              "- 源域能够拟合，但目标域仍低。源域标签拟合的分支线性探针也出现跨天气下降，当前编码表征与适配训练尚未解决目标域类别区分，单改融合权重缺乏依据。",
              "- source_CV仅是固定编码器上的线性分类器交叉验证，编码器可能已使用全部源域标签，不能当作整个网络的独立源域泛化评估。target_val探针仅作诊断，没有替代正式UDA模型指标。",
              "- 提高骨干更新后，最终编码器的分支诊断仍未改善，不能只用更新次数不足解释。", "",
              "## 自动裁剪检查", "",
              "使用[官方YOLO11s权重](https://github.com/ultralytics/assets/releases/tag/v8.3.0)，在44张固定样本上按同一规则做CPU检测。雨天多数原图已紧密包含船体；IR提议不稳定，多船源图有选错对象风险。本版裁剪没有用于分类训练。预览见runs/weather_local_screen/boat_roi_inspection_seed42；结论见results/boat_roi_inspection_decision.json。", "",
              "## 状态", "", "这批候选未达到80%，筛选指标也未优于各天气历史最佳，不建议服务器重跑这批配置。已保留日志、逐样本预测、混淆矩阵、源域拟合、梯度、参数更新、分支探针和复杂性数据。相关完整回归110项通过。"]
    (ROOT/"REMAINING_WEATHER_SEED42_20261114.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Collected real training runs:", len(rows))
    for row in rows:
        print(row["variant"], row["target"], row["method"], round(row["ACC_percent"], 2), "post-DA", round(row["post_adaptation_peak_ACC_percent"], 2))


if __name__ == "__main__":
    main()
