"""Repair backlight seeds42/43 and run rainy five-method seeds42--46."""
import argparse
import csv
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))
from tools.download_official_uda_weights import require_weight
from weather.run_qualified_backlight import training_argv as backlight_argv
from weather.rainy_presets import arguments_for, EVIDENCE

METHODS = ("dann", "deepcoral", "cdan", "mdd", "mcc")
METRICS = {"ACC_percent": "accuracy", "Precision_macro_percent": "precision_macro",
           "Recall_macro_percent": "recall_macro", "F1_macro_percent": "f1_macro"}


def required_weights(scope):
    return ("convnext_small",) if scope == "backlight" else ("convnext_small", "resnet18", "unip_tiny")


def dependency_preflight(scope):
    modules = {"torch": "torch", "torchvision": "torchvision", "PIL": "Pillow"}
    if scope != "backlight":
        modules.update(timm="timm", scipy="scipy", sklearn="scikit-learn", threadpoolctl="threadpoolctl")
    missing = []
    for name, package in modules.items():
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError("Install missing dependencies before training: "
                           + sys.executable + " -m pip install " + " ".join(missing))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; server suite will not silently fall back to CPU")
    # Import train-only components now rather than failing late in CDAN's job.
    if scope != "backlight":
        for name in ("prediction_teacher", "prediction_distillation", "statistical_calibration",
                     "encoder_schedule", "unip_encoder", "branch_probe", "feature_cache",
                     "training_image_cache", "weather_augmentation", "source_aspect"):
            importlib.import_module("weather." + name)


def build_plan(project_root, output, scope="both", backlight_seeds=(42, 43), rainy_seeds=tuple(range(42, 47))):
    if scope not in {"both", "backlight", "rainy"}:
        raise ValueError("Unknown scope")
    from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
    root, output = Path(project_root).resolve(), Path(output).resolve()
    jobs = []
    for target, seeds in (("backlight", backlight_seeds), ("rainy", rainy_seeds)):
        if scope != "both" and scope != target:
            continue
        if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError("Seeds must be unique nonnegative integers")
        for seed in seeds:
            for method in METHODS:
                parent = output/target/f"seed{seed}"
                lists = output/"prepared_exact_pairs"/target
                if target == "backlight":
                    weights = root/"pretrained_weights/torch/hub/checkpoints/convnext_small-0c510722.pth"
                    command = backlight_argv(method, lists, parent, weights, seed)
                    budget, minimum = 60, 60
                    profile = "qualified_backlight_v8"
                else:
                    values = arguments_for(method, root, lists, parent/method, seed)
                    command = [sys.executable, "-B", "-u", str(RUNTIME/"weather/train_vds_core.py"),
                               *recorded_arguments_to_cli(values)]
                    budget, minimum = values["epochs"], values["early_stopping_min_epochs"]
                    profile = "rainy_cached_target_distillation"
                jobs.append({"tag": f"{target}_{method}_seed{seed}", "target": target, "method": method,
                             "seed": seed, "profile": profile, "epochs": budget, "minimum_epochs": minimum,
                             "argv": command,
                             "result_file": str(parent/method/f"{method}_{target}_seed{seed}"/"result.json"),
                             "min_free_mib": 4096 if target == "rainy" else 3072})
    return {"scope": scope, "requested": len(jobs), "jobs": jobs,
            "required_weights": list(required_weights(scope)), "rainy_local_evidence": EVIDENCE,
            "target_training_labels_used": False, "student_only_inference": True,
            "trained_model_checkpoints_saved": False, "VDS_repeated": False,
            "evaluation": "same-epoch peak target_val ACC/P/R/F1; no weather target_test",
            "backlight_prior_completed_seeds_not_repeated": [44, 45, 46]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def summarize(output, plan):
    rows = []
    for job in plan["jobs"]:
        path = Path(job["result_file"])
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        complete = json.loads((path.parent/"completeness.json").read_text(encoding="utf-8"))
        if (complete.get("status") != "PASS" or result.get("protocol") != "UDA"
                or result.get("target_training_labels_used") is not False or result.get("smoke")
                or result.get("screening_only") is not False
                or any(result[k] != job[k] for k in ("method", "target", "seed"))
                or not job["minimum_epochs"] <= result["completed_epochs"] <= job["epochs"]
                or not 1 <= result["selected_epoch"] <= result["completed_epochs"]):
            raise ValueError(f"Incomplete or wrong-protocol result: {path}")
        if job["target"] == "rainy":
            enhancement = result.get("enhancement", {})
            if (enhancement.get("prediction_distillation_weight") != 4.
                    or enhancement.get("prediction_distillation", {}).get("inference_includes_teacher") is not False):
                raise ValueError(f"Rain result is not the selected student-only variant: {path}")
        rows.append({**{k: job[k] for k in ("target", "method", "seed", "profile")},
                     "selected_epoch": result["selected_epoch"], "completed_epochs": result["completed_epochs"],
                     **{k: 100*result["target_val"][v] for k, v in METRICS.items()},
                     **{k: v for k, v in result.get("complexity", {}).items() if isinstance(v, (int, float))},
                     "result_file": str(path)})
    destination = Path(output)/"summary"
    destination.mkdir(exist_ok=True)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (destination/"runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)
        means = []
        for target, method in sorted({(r["target"], r["method"]) for r in rows}):
            group = [r for r in rows if (r["target"], r["method"]) == (target, method)]
            row = {"target": target, "method": method, "seeds": " ".join(str(r["seed"]) for r in group),
                   "num_seeds": len(group), "metric_partition": "target_val", "std_ddof": 0}
            for key in METRICS:
                values = [r[key] for r in group]
                row[key+"_mean"], row[key+"_std"] = statistics.mean(values), statistics.pstdev(values)
            means.append(row)
        with (destination/"summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, list(means[0])); writer.writeheader(); writer.writerows(means)
    missing = [j["tag"] for j in plan["jobs"] if not any(
        (r["target"], r["method"], r["seed"]) == (j["target"], j["method"], j["seed"]) for r in rows)]
    write_json(destination/"completeness.json", {"requested": plan["requested"], "completed": len(rows),
                                               "missing": missing, "status": "PASS" if not missing else "INCOMPLETE"})
    return len(rows), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("JMDA_DATA_ROOT", "/mnt/data/lixiang/data/Data")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scope", choices=("both", "backlight", "rainy"), default="both")
    parser.add_argument("--backlight-seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--rainy-seeds", type=int, nargs="+", default=list(range(42, 47)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-environment", action="store_true", help="Verify weights/dependencies/CUDA without starting training")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root/"runs/weather_backlight_rainy"/
              f"{args.scope}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}").resolve()
    plan = build_plan(root, output, args.scope, args.backlight_seeds, args.rainy_seeds)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    # Global offline preflight: no first-seed download and no hidden network
    # access halfway through the experiment queue.
    for weight_id in plan["required_weights"]:
        require_weight(weight_id, root/"pretrained_weights")
    dependency_preflight(args.scope)
    if args.check_environment:
        print("[PREFLIGHT PASS] weights, runtime imports and CUDA; no training started", flush=True)
        return
    (output/"logs").mkdir(parents=True)
    write_json(output/"run_plan.json", plan)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TORCH_HOME=str(root/"pretrained_weights/torch"),
               JMDA_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", JMDA_DATA_ROOT=str(args.data_root))
    from weather.train_vds_core import prepare_datasets
    os.environ["JMDA_DATA_ROOT"] = str(args.data_root.resolve())
    for target in dict.fromkeys(j["target"] for j in plan["jobs"]):
        lists = output/"prepared_exact_pairs"/target
        subprocess.run([sys.executable, "-B", "-u", str(RUNTIME/"tools/prepare_paired_protocol_lists.py"),
                        "--data-root", str(args.data_root), "--output-dir", str(lists),
                        "--targets", target, "--modality", "paired"], env=env, check=True, cwd=root)
        _, audit = prepare_datasets(lists, target, True)
        write_json(output/f"{target}_dataset_preflight.json", audit)
    print(f"Run directory: {output}\nRequested: {plan['requested']}; scope={args.scope}; weights/data preflight PASS", flush=True)
    failures = []
    for job in plan["jobs"]:
        log = output/"logs"/(job["tag"]+".log")
        gpu = [sys.executable, "-B", "-u", str(RUNTIME/"tools/run_on_available_gpu.py"),
               "--gpus", env.get("JMDA_GPU_CANDIDATES", "0,1"),
               "--min-free-mib", env.get("WEATHER_QUALIFIED_MIN_FREE_MIB", str(job["min_free_mib"])),
               "--max-util-percent", env.get("JMDA_GPU_MAX_UTIL_PERCENT", "80"),
               "--stable-checks", env.get("JMDA_GPU_STABLE_CHECKS", "2"),
               "--poll-seconds", env.get("JMDA_GPU_POLL_SECONDS", "15"), "--tag", job["tag"], "--"]
        print(f">>> {job['tag']}; live GPU/training log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as f:
            status = subprocess.run(gpu+job["argv"], cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if status:
            failures.append(job["tag"])
        completed, missing = summarize(output, plan)
        write_json(output/"queue_status.json", {"status": "RUNNING", "completed": completed,
                                               "requested": plan["requested"], "failed": failures})
        print(f"    exit={status}; completed={completed}/{plan['requested']}", flush=True)
    success = not failures and not missing
    write_json(output/"queue_status.json", {"status": "PASS" if success else "FAIL", "completed": completed,
                                           "requested": plan["requested"], "failed": failures, "missing": missing})
    print(f"Finished; summary={output/'summary'}; status={'PASS' if success else 'FAIL'}", flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
