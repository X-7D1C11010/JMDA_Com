"""Queue VDS MCC/CDAN and the five qualified backlight presets, seeds 42--46."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

RUNTIME = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(42, 47))
WEATHER_METHODS = ("dann", "deepcoral", "cdan", "mdd", "mcc")
COMPLEXITY = ("total_params", "trainable_params", "inference_GFLOPs", "total_train_time_s",
              "train_epoch_mean_s", "latency_mean_ms", "throughput_samples_s", "peak_train_reserved_gb")


def build_plan(project_root, output, scope="both"):
    if scope not in {"both", "vds", "backlight"}:
        raise ValueError(f"Unknown scope: {scope}")
    jobs, experiments = [], []
    if scope != "backlight":
        jobs.append({"tag": "vds_mcc_cdan_seeds42_46", "argv": [
            "bash", str(RUNTIME/"scripts/run_pending_adaptation_v2.sh"), "42", "vds-only"]})
        experiments.extend({"dataset": "VDS2Raw", "target": "non_day3", "method": method, "seed": seed,
                            "profile": profile, "metric_partition": "target_test"}
                           for method, profile in (("mcc", "mcc_v3"), ("cdan", "cdan_safe_v4")) for seed in SEEDS)
    if scope != "vds":
        for seed in SEEDS:
            jobs.append({"tag": f"backlight_seed{seed}", "argv": [
                sys.executable, "-B", "-u", str(RUNTIME/"weather/run_qualified_backlight.py"),
                "--project-root", str(project_root), "--seed", str(seed),
                "--output-dir", str(output/"weather"/f"backlight_seed{seed}")]})
            experiments.extend({"dataset": "weather", "target": "backlight", "method": method, "seed": seed,
                                "profile": "qualified_backlight_v8", "metric_partition": "target_val"}
                               for method in WEATHER_METHODS)
    return {"scope": scope, "seeds": list(SEEDS), "experiments": experiments, "jobs": jobs,
            "requested_experiments": len(experiments), "other_weathers_included": False,
            "target_training_labels_used": False, "trained_model_files_saved": False}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def summarize(output):
    plan = json.loads((output/"run_plan.json").read_text(encoding="utf-8"))
    expected = {(e["dataset"], e["target"], e["method"], e["seed"]) for e in plan["experiments"]}
    rows, seen = [], set()
    for folder, dataset, partition in (("vds", "VDS2Raw", "target_test"), ("weather", "weather", "target_val")):
        for path in sorted((output/folder).rglob("result.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            target = "non_day3" if dataset == "VDS2Raw" else r.get("target")
            identity = (dataset, target, r["method"], r["seed"])
            if identity not in expected or identity in seen:
                raise ValueError(f"Unexpected/duplicate experiment: {identity}")
            if r.get("protocol") != "UDA" or r.get("target_training_labels_used") is not False or r.get("smoke"):
                raise ValueError(f"Unexpected supervision or smoke result: {path}")
            if dataset == "VDS2Raw":
                profile = "mcc_v3" if r["method"] == "mcc" else "cdan_safe_v4"
                if (r.get("ais_protocol") != "heading_size" or r.get("optimization_profile") != profile
                        or r.get("selection_partition") != "target_val" or r.get("test_selection_bias") is not False
                        or r.get("test_evaluations") != 1 or r.get("test_evaluations_for_selection") != 0):
                    raise ValueError(f"Unexpected VDS input/selection/test protocol: {path}")
            else:
                complete = json.loads((path.parent/"completeness.json").read_text(encoding="utf-8"))
                if (complete.get("status") != "PASS" or r.get("screening_only") is not False
                        or r.get("completed_epochs") != 60 or r.get("image_backbone") != "convnext_small"
                        or r.get("encoder_training_scope") != "frozen"):
                    raise ValueError(f"Weather result did not complete the qualified budget/protocol: {path}")
                profile = "qualified_backlight_v8"
            metrics = r[partition]
            rows.append({"dataset": dataset, "target": target, "method": r["method"], "seed": r["seed"],
                         "profile": profile, "metric_partition": partition, "selected_epoch": r["selected_epoch"],
                         "completed_epochs": r["completed_epochs"],
                         **{name: 100*metrics[key] for name, key in (("ACC_percent", "accuracy"),
                            ("Precision_macro_percent", "precision_macro"), ("Recall_macro_percent", "recall_macro"),
                            ("F1_macro_percent", "f1_macro"))},
                         **{key: r["complexity"][key] for key in COMPLEXITY}, "result_file": str(path)})
            seen.add(identity)
    destination = output/"summary"
    destination.mkdir(exist_ok=True)
    if rows:
        with (destination/"runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        means = []
        keys = sorted({(r["dataset"], r["target"], r["method"], r["profile"], r["metric_partition"]) for r in rows})
        for dataset, target, method, profile, partition in keys:
            selected = [r for r in rows if (r["dataset"], r["target"], r["method"]) == (dataset, target, method)]
            row = {"dataset": dataset, "target": target, "method": method, "profile": profile,
                   "metric_partition": partition, "seeds": " ".join(str(r["seed"]) for r in selected), "num_seeds": len(selected)}
            for key in ("ACC_percent", "Precision_macro_percent", "Recall_macro_percent", "F1_macro_percent"):
                values = [r[key] for r in selected]
                row[key+"_mean"], row[key+"_std"] = statistics.mean(values), statistics.pstdev(values)
            means.append(row)
        with (destination/"summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(means[0])); writer.writeheader(); writer.writerows(means)
    missing = [list(identity) for identity in sorted(expected-seen)]
    write_json(destination/"completeness.json", {"expected": len(expected), "completed": len(seen),
                                               "missing": missing, "all_results_present": not missing})
    return len(seen), missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--scope", choices=("both", "vds", "backlight"), default="both")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root/"runs/vds_backlight_complete"/f"{args.scope}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}").resolve()
    plan = build_plan(root, output, args.scope)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    (output/"logs").mkdir(parents=True)
    write_json(output/"run_plan.json", plan)
    env = dict(os.environ, JMDA_PROJECT_ROOT=str(root), PYTHON=sys.executable, PYTHONUNBUFFERED="1",
               JMDA_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               VDS2RAW_DA_OUTPUT_ROOT=str(output/"vds"), TORCH_HOME=str(root/"pretrained_weights/torch"))
    env.setdefault("JMDA_GPU_CANDIDATES", "0,1")
    env.setdefault("VDS2RAW_GPU_CANDIDATES", env["JMDA_GPU_CANDIDATES"])
    env.setdefault("JMDA_GPU_MAX_UTIL_PERCENT", "80")
    env.setdefault("VDS2RAW_GPU_MAX_UTIL_PERCENT", env["JMDA_GPU_MAX_UTIL_PERCENT"])
    env.setdefault("VDS2RAW_GPU_STABLE_CHECKS", "2")
    env.setdefault("VDS2RAW_GPU_POLL_SECONDS", "15")
    print(f"Run directory: {output}\nRequested experiments: {plan['requested_experiments']}; seeds42-46; scope={args.scope}", flush=True)
    failures = []
    for job in plan["jobs"]:
        log = output/"logs"/(job["tag"]+".log")
        print(f">>> {job['tag']}; live queue/training log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            status = subprocess.run(job["argv"], cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if status:
            failures.append(job["tag"])
        completed, missing = summarize(output)
        write_json(output/"queue_status.json", {"status": "RUNNING", "failed_groups": failures,
                                               "completed": completed, "requested": plan["requested_experiments"]})
        print(f"    exit={status}; completed={completed}/{plan['requested_experiments']}", flush=True)
    success = not failures and not missing
    write_json(output/"queue_status.json", {"status": "PASS" if success else "FAIL", "failed_groups": failures,
                                           "completed": completed, "requested": plan["requested_experiments"]})
    print(f"Finished: status={'PASS' if success else 'FAIL'}; summary={output/'summary'}", flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
