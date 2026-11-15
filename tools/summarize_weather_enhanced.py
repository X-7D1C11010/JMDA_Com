"""Audit enhanced training runs without mixing them with original method scores."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_weather_goal import read_run


def main():
    rows = []
    for path in sorted((ROOT / "runs/weather_enhanced_screen").rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        enhancement = result.get("enhancement", {})
        if not enhancement.get("enabled"):
            raise ValueError(f"Unexpected original-method run: {path}")
        row = read_run(path, path.parents[2].name)
        row.update({
            "enhancement_family": enhancement["variant_family"],
            "teacher_weight": enhancement.get("cluster_teacher_weight"),
            "prediction_teacher_weight":(enhancement.get("prediction_teacher") or {}).get("probability_weight",0.),
            "prediction_distillation_weight":enhancement.get("prediction_distillation_weight",0.),
            "prediction_distillation_encoder_lr_scale":enhancement.get("prediction_distillation_encoder_lr_scale",1.),
            "prediction_distillation_start_epoch":enhancement.get("prediction_distillation_start_epoch"),
            "prediction_distillation_sharpening":enhancement.get("prediction_distillation_sharpening"),
            "prediction_teacher_temperature":(enhancement.get("prediction_teacher") or {}).get("teacher_temperature",1.),
            "prediction_student_temperature":(enhancement.get("prediction_teacher") or {}).get("student_temperature",1.),
            "crossmodal_association": enhancement.get("crossmodal_association", {}).get("enabled", False),
            "initial_source_ACC_percent": 100 * enhancement["source_initialization_ACC"] if "source_initialization_ACC" in enhancement else None,
            "source_crossmodal_weight":enhancement.get("source_supervised_crossmodal_weight",0.),
            "source_prototype_weight":enhancement.get("source_prototype_weight",0.),
            "domain_confusion_weight":enhancement.get("domain_confusion_weight",0.),
            "source_IR_statistics_probability":(result.get("input_augmentation") or {}).get("source_ir_statistics_probability",0.),
            "source_IR_statistics_strength":(result.get("input_augmentation") or {}).get("source_ir_statistics_strength",1.),
            "target_representation_anchor_weight":enhancement.get("target_representation_anchor_weight",0.),
            "target_feature_style_probability":(enhancement.get("target_feature_style") or {}).get("probability",0.),
            "target_feature_style_source_max":(enhancement.get("target_feature_style") or {}).get("source_max",.5),
            "source_object_ce_weight":enhancement.get("source_object_ce_weight",0.),
            "source_object_consistency_weight":enhancement.get("source_object_consistency_weight",0.),
            "target_crossmodal_weight":enhancement.get("target_instance_crossmodal_weight",0.),
            "target_crossmodal_objective":enhancement.get("target_crossmodal_objective","contrastive"),
            "crossmodal_projection_dim":enhancement.get("projection",{}).get("width",0),
            "crossmodal_stop_gradient":enhancement.get("stop_contrastive_gradient_at_encoder","none"),
            "source_IR_inversion_probability":result.get("input_augmentation",{}).get("source_ir_inversion_probability",0.),
        })
        rows.append(row)
    payload = {
        "updated_UTC": datetime.now(timezone.utc).isoformat(),
        "completed_real_enhanced_runs": len(rows),
        "original_method_results": False,
        "target_training_labels_used": False,
        "initialization_or_diagnostic_scores_counted_as_training_results": False,
        "evaluation": "within-run peak target_val; no independent weather test",
        "runs": rows,
    }
    (ROOT / "results/weather_enhanced_runs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        with (ROOT / "results/weather_enhanced_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 天气数据集增强版本实测", "", "本表单独记录统计校准/聚类蒸馏及跨模态配对约束等增强版本，具体版本不能混为原方法成绩。",
             "源域标签可见，目标训练标签不可见；统计仅拟合源训练集和无标签目标训练集。",
             "初始化诊断、冻结特征诊断、CUDA预检均不计为训练成绩。以下指标来自同一选定轮次。", "",
             "| 配置 | 方法 | 目标域 | ACC% | Precision% | Recall% | F1% | 最佳/完成轮次 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["prediction_distillation_weight"]:
            name=f"缓存教师蒸馏，权重{row['prediction_distillation_weight']:g}，推理仅学生"
            if row["prediction_distillation_encoder_lr_scale"]!=1:name+=f"，编码器LR×{row['prediction_distillation_encoder_lr_scale']:g}"
        elif row["prediction_teacher_weight"]:
            name=f"冻结聚类教师概率集成，权重{row['prediction_teacher_weight']:g}"
        elif row["teacher_weight"] is not None:
            name = ("跨模态关联" if row["crossmodal_association"] else "直接关联") + f"，蒸馏权重{row['teacher_weight']:g}"
        elif row["domain_confusion_weight"] and not row["source_crossmodal_weight"] and not row["target_crossmodal_weight"] and not row["source_prototype_weight"]:
            name = f"原核心＋编码器域混淆，权重{row['domain_confusion_weight']:g}"
        elif row["source_object_ce_weight"] or row["source_object_consistency_weight"]:
            name=f"源域多视图：CE={row['source_object_ce_weight']:g}/一致性={row['source_object_consistency_weight']:g}"
        elif row["target_feature_style_probability"]:
            name=f"原核心＋目标特征风格，p={row['target_feature_style_probability']:g}"
        else:
            name = f"跨模态约束：源{row['source_crossmodal_weight']:g}/目标{row['target_crossmodal_weight']:g}/源原型{row['source_prototype_weight']:g}"
            if row["target_crossmodal_objective"]=="vicreg":name += "/目标VICReg"
            if row["domain_confusion_weight"]:name += f"/域混淆{row['domain_confusion_weight']:g}"
        if row["source_IR_statistics_probability"]:
            name+=f"/源IR统计增强p={row['source_IR_statistics_probability']:g},强度{row['source_IR_statistics_strength']:g}"
        name+=f"/原核心权重{row['core_trade_off']:g}"
        if row['target_representation_anchor_weight']:
            name+=f"/初始目标表征约束{row['target_representation_anchor_weight']:g}"
        if row['source_crossmodal_weight'] or row['target_crossmodal_weight']:
            name+=f"/跨模态停梯度{row['crossmodal_stop_gradient']}"
        lines.append(f"| {name} | {row['method']} | {row['target']} | {row['ACC_percent']:.2f} | "
                     f"{row['Precision_percent']:.2f} | {row['Recall_percent']:.2f} | {row['F1_percent']:.2f} | "
                     f"{row['selected_epoch']}/{row['completed_epochs']} |")
    lines += ["", f"共核验{len(rows)}项真实训练；混淆矩阵、轮次峰值及完整性检查一致。",
              "此表不代表五方法四天气全部达到80%。聚类版本相同的峰值主要来自共同初始化，不能据此声称各自的域适配已取得相同提升。",
              "详细配置及原始结果路径见 results/weather_enhanced_runs.csv。"]
    (ROOT / "WEATHER_ENHANCED_20261114.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"real_runs": len(rows), "best_ACC_percent": max((x["ACC_percent"] for x in rows), default=None)}))


if __name__ == "__main__":
    main()
