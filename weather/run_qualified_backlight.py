"""Run the five locally screened backlight presets; other weathers are not released."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

from tools.download_official_uda_weights import ensure_weight


# Each entry is tied to an actual seed42 run, not a target-score promise.
PRESETS = {
    "dann": {"lr": .0003, "trade_off": .05, "weight_decay": .0005,
             "feature_normalization": "l2", "gate_lr_multiplier": 1., "optimizer": "adam",
             "evidence": "representation_v6c_seed42/dann_warmup_fixed/dann_backlight_seed42"},
    "deepcoral": {"lr": .01, "trade_off": .05, "weight_decay": .002,
                  "feature_normalization": "unit", "gate_lr_multiplier": .01, "gate_weight_decay": 0.,
                  "evidence": "representation_v6d_requalified_seed42/deepcoral_unit/deepcoral_backlight_seed42"},
    "cdan": {"lr": .01, "trade_off": .05, "weight_decay": .001,
             "feature_normalization": "unit", "gate_lr_multiplier": .01, "gate_weight_decay": 0.,
             "unit_method_features": True,
             "evidence": "representation_v6d_requalified_seed42/cdan_unit/cdan_backlight_seed42"},
    "mdd": {"lr": .1, "trade_off": .2, "weight_decay": .002,
            "feature_normalization": "unit", "gate_lr_multiplier": .01, "gate_weight_decay": 0.,
            "classifier_layout": "linear",
            "evidence": "representation_v8_linear_deterministic_backlight_seed42/mdd_linear/mdd_backlight_seed42"},
    "mcc": {"lr": .1, "trade_off": .02, "weight_decay": .002,
            "feature_normalization": "unit", "gate_lr_multiplier": .01, "gate_weight_decay": 0.,
            "classifier_layout": "linear",
            "evidence": "representation_v8_linear_deterministic_backlight_seed42/mcc_linear/mcc_backlight_seed42"},
}


def training_argv(method, lists, output, weights, seed):
    command = [sys.executable, "-B", "-u", str(RUNTIME/"weather/train_vds_core.py"),
               "--method", method, "--target", "backlight", "--data-dir", str(lists),
               "--save-dir", str(output/method), "--weights", str(weights), "--seed", str(seed),
               "--image-backbone", "convnext_small", "--encoder-training-scope", "frozen",
               "--epochs", "60", "--batch-size", "16", "--source-class-balance",
               "--initial-vis-weight", ".85", "--adaptation-warmup-epochs", "3", "--adaptation-ramp-epochs", "10",
               "--lr-gamma", ".0002" if method == "mdd" else ".001",
               "--num-workers", "2", "--cpu-threads", "2", "--cache-evaluation",
               "--modality-diagnostics", "--log-feature-moments", "--early-stopping-patience", "0"]
    for key, value in PRESETS[method].items():
        if key == "evidence":
            continue
        command.append("--"+key.replace("_", "-"))
        if not isinstance(value, bool):
            command.append(str(value))
    return command


def summarize(output, seed):
    rows = []
    for method in PRESETS:
        path = output/method/f"{method}_backlight_seed{seed}"/"result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if (r["target"] != "backlight" or r["target_training_labels_used"] is not False
                or r["screening_only"] or r["smoke"]):
            raise ValueError(f"Unexpected result protocol: {path}")
        row = {"method": method, "target": "backlight", "seed": r["seed"],
               "selected_epoch": r["selected_epoch"], "completed_epochs": r["completed_epochs"],
               **{k: 100*r["target_val"][v] for k, v in (("ACC", "accuracy"), ("Precision", "precision_macro"),
                                                        ("Recall", "recall_macro"), ("F1", "f1_macro"))},
               "feature_normalization": r["feature_normalization"], "classifier_layout": r["classifier_layout"],
               "image_backbone": r["image_backbone"], "encoder_training_scope": r["encoder_training_scope"],
               "optimizer": r["optimizer"], "FDA_beta": r["input_augmentation"]["FDA_beta"],
               "source_zoom_min_area": r["input_augmentation"].get("source_zoom_min_area", 1.),
               "evaluation_partition": "target_val", "result_file": str(path), **r["complexity"]}
        row.pop("inference_input", None)
        rows.append(row)
    if rows:
        with (output/"results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("JMDA_DATA_ROOT", "/mnt/data/lixiang/data/Data")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Print commands without downloading, preparing data or training")
    args = p.parse_args()
    if args.seed < 0:
        p.error("seed must be nonnegative")
    root = args.project_root.resolve()
    output = (args.output_dir or root/"runs/weather_qualified_v8"/f"backlight_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}").resolve()
    weights = root/"pretrained_weights/torch/hub/checkpoints/convnext_small-0c510722.pth"
    lists = output/"prepared_exact_pairs"
    commands = {method: training_argv(method, lists, output, weights, args.seed) for method in PRESETS}
    plan = {"target": "backlight", "seed": args.seed, "methods": PRESETS, "argv": commands,
            "local_evidence_budget": "first 20 of 60 planned epochs; seed42 only", "formal_budget_epochs": 60,
            "target_training_labels_used": False, "FDA_beta": 0., "trained_model_files_saved": False,
            "evaluation": "peak target_val ACC, then macro F1; no separate weather test set",
            "other_weathers_released": False}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite previous trial: {output}")
    output.mkdir(parents=True)
    (output/"logs").mkdir()
    (output/"run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Run directory: {output}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", JMDA_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    ensure_weight("convnext_small", root/"pretrained_weights")
    subprocess.run([sys.executable, "-B", "-u", str(RUNTIME/"tools/prepare_paired_protocol_lists.py"),
                    "--data-root", str(args.data_root), "--output-dir", str(lists), "--targets", "backlight",
                    "--modality", "paired"], env=env, check=True)
    failures = []
    for method, command in commands.items():
        log = output/"logs"/(method+".log")
        print(f">>> {method}; GPU queue/training log: {log}", flush=True)
        gpu = [sys.executable, "-B", "-u", str(RUNTIME/"tools/run_on_available_gpu.py"),
               "--gpus", env.get("JMDA_GPU_CANDIDATES", "0,1"),
               "--min-free-mib", env.get("WEATHER_V8_MIN_FREE_MIB", "3072"),
               "--max-util-percent", env.get("JMDA_GPU_MAX_UTIL_PERCENT", "80"),
               "--stable-checks", env.get("JMDA_GPU_STABLE_CHECKS", "2"),
               "--poll-seconds", env.get("JMDA_GPU_POLL_SECONDS", "15"), "--tag", f"v8_{method}_backlight", "--"]
        with log.open("w", encoding="utf-8") as stream:
            status = subprocess.run(gpu+command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        print(f"    {method}: exit={status}", flush=True)
        if status:
            failures.append(method)
        else:
            expected = output/method/f"{method}_backlight_seed{args.seed}"/"completeness.json"
            if not expected.is_file() or json.loads(expected.read_text(encoding="utf-8"))["status"] != "PASS":
                raise RuntimeError(f"Training exited successfully without complete records: {method}")
            summarize(output, args.seed)
    (output/"queue_status.json").write_text(json.dumps({"failed_methods": failures, "status": "FAIL" if failures else "PASS"}), encoding="utf-8")
    print(f"Finished; summary={output/'results.csv'}; failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
