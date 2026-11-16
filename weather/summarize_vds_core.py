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
        metrics = result.get('target_test') or result["target_val"]
        rows.append({"method": result["method"], "target": result["target"], "seed": result["seed"],
                     'selection_metric':result.get('selection_metric','accuracy_f1'),
                     'reporting_partition':'target_test' if result.get('target_test') is not None else 'target_val',
                     'selected_validation_ACC_percent':100*result['target_val']['accuracy'],
                     'peak_validation_ACC_percent':100*result.get('peak_target_val',result['target_val'])['accuracy'],
                     "enhancement_family":(result.get("enhancement") or {}).get("variant_family","none"),
                     "calibration_components":((result.get("enhancement") or {}).get("calibration") or {}).get("components"),
                     "cluster_teacher_weight":(result.get("enhancement") or {}).get("cluster_teacher_weight",0.),
                     "prediction_teacher_weight":((result.get("enhancement") or {}).get("prediction_teacher") or {}).get("probability_weight",0.),
                     "prediction_distillation_weight":(result.get("enhancement") or {}).get("prediction_distillation_weight",0.),
                     "prediction_distillation_encoder_lr_scale":(result.get("enhancement") or {}).get("prediction_distillation_encoder_lr_scale",1.),
                     "prediction_distillation_start_epoch":(result.get("enhancement") or {}).get("prediction_distillation_start_epoch"),
                     "prediction_distillation_sharpening":(result.get("enhancement") or {}).get("prediction_distillation_sharpening"),
                     "prediction_teacher_temperature":((result.get("enhancement") or {}).get("prediction_teacher") or {}).get("teacher_temperature",1.),
                     "prediction_student_temperature":((result.get("enhancement") or {}).get("prediction_teacher") or {}).get("student_temperature",1.),
                     "source_crossmodal_weight":(result.get("enhancement") or {}).get("source_supervised_crossmodal_weight",0.),
                     "source_prototype_weight":(result.get("enhancement") or {}).get("source_prototype_weight",0.),
                     "domain_confusion_weight":(result.get("enhancement") or {}).get("domain_confusion_weight",0.),
                     "target_feature_style_probability":((result.get("enhancement") or {}).get("target_feature_style") or {}).get("probability",0.),
                     "target_feature_style_source_max":((result.get("enhancement") or {}).get("target_feature_style") or {}).get("source_max",.5),
                     "source_object_ce_weight":(result.get("enhancement") or {}).get("source_object_ce_weight",0.),
                     "source_object_consistency_weight":(result.get("enhancement") or {}).get("source_object_consistency_weight",0.),
                     "target_crossmodal_weight":(result.get("enhancement") or {}).get("target_instance_crossmodal_weight",0.),
                     "target_crossmodal_objective":(result.get("enhancement") or {}).get("target_crossmodal_objective","contrastive"),
                     "crossmodal_projection_dim":((result.get("enhancement") or {}).get("projection") or {}).get("width",0),
                     "crossmodal_stop_gradient":(result.get("enhancement") or {}).get("stop_contrastive_gradient_at_encoder","none"),
                     "source_IR_inversion_probability":result.get("input_augmentation",{}).get("source_ir_inversion_probability",0.),
                     "source_IR_statistics_probability":result.get("input_augmentation",{}).get("source_ir_statistics_probability",0.),
                     "source_IR_statistics_strength":result.get("input_augmentation",{}).get("source_ir_statistics_strength",1.),
                     "target_representation_anchor_weight":(result.get("enhancement") or {}).get("target_representation_anchor_weight",0.),
                     "weather_profile": result.get("weather_profile", "baseline"),
                     "image_backbone": result.get("image_backbone", "resnet18"),
                     "ir_backbone": result.get("ir_backbone", result.get("image_backbone", "resnet18")),
                     "ir_feature_mode": (result.get("infrared_pretraining") or {}).get("feature_mode", "shared"),
                     "image_adaptation_lr_scale": result.get("image_adaptation_lr_scale", 1.),
                     "freeze_image_encoders": result.get("freeze_image_encoders", False),
                     "encoder_training_scope": result.get("encoder_training_scope", "frozen" if result.get("freeze_image_encoders", False) else "all"),
                     "feature_normalization": result.get("feature_normalization", "none"),
                     "unit_method_features": result.get("unit_method_features", False),
                     "classifier_layout": result.get("classifier_layout", "reference"),
                     "FDA_beta": result.get("input_augmentation", {}).get("FDA_beta", 0.),
                     "source_zoom_min_area": result.get("input_augmentation", {}).get("source_zoom_min_area", 1.),
                     "source_weather_probability": result.get("input_augmentation", {}).get("source_weather_probability", 0.),
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
    if len({(row['selection_metric'],row['reporting_partition']) for row in rows})!=1:
        raise ValueError('Do not merge peak-validation and CE-selected heldout evaluation')
    if len({row["weather_profile"] for row in rows}) != 1:
        raise ValueError("Do not merge different weather profiles")
    if len({row["screening_only"] for row in rows}) != 1:
        raise ValueError("Do not merge prefix screens and complete experiments")
    signatures = {(row["image_backbone"], row["encoder_training_scope"], row["feature_normalization"],
                   row["classifier_layout"], row["FDA_beta"], row["source_zoom_min_area"],
                   row["ir_backbone"], row["ir_feature_mode"], row["source_weather_probability"],
                   row["image_adaptation_lr_scale"],row["enhancement_family"],row["calibration_components"],
                   row["cluster_teacher_weight"],row["prediction_teacher_weight"],row["prediction_teacher_temperature"],row["prediction_student_temperature"],
                   row["prediction_distillation_weight"],row["prediction_distillation_start_epoch"],row["prediction_distillation_sharpening"],
                   row["prediction_distillation_encoder_lr_scale"],
                   row["source_crossmodal_weight"],row["target_crossmodal_weight"],row["crossmodal_projection_dim"],
                   row["crossmodal_stop_gradient"],row["target_crossmodal_objective"],row["source_IR_inversion_probability"],row["source_prototype_weight"],row["domain_confusion_weight"],
                   row["target_feature_style_probability"],row["target_feature_style_source_max"],
                   row["source_object_ce_weight"],row["source_object_consistency_weight"],
                   row["source_IR_statistics_probability"],row["source_IR_statistics_strength"],
                   row["target_representation_anchor_weight"]) for row in rows}
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
    caption=("Weather VIS+IR UDA; peak target_val within each run (no separate target_test)."
             if signature['selection_metric']=='accuracy_f1' and signature['reporting_partition']=='target_val'
             else f"Weather VIS+IR UDA; selection={signature['selection_metric']}; report={signature['reporting_partition']}.")
    lines = [caption,
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
