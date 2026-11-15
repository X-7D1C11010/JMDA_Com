"""Single-seed remaining-weather candidates; no server/backlight preset changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot

METHODS = ("dann", "deepcoral", "cdan", "mdd", "mcc")
# Calibrated unit-norm classifier plus low-LR image adaptation. These are
# hypotheses to measure, not qualified settings or promised accuracies.
WEATHER = {
    "rainy": {"image_lr": .00001, "vis": .5, "warmup": 5, "ramp": 10, "prefix": 20,
              "trade_off": dict(dann=.1, deepcoral=.1, cdan=.1, mdd=.5, mcc=1.)},
    "foggy": {"image_lr": .00003, "vis": .5, "warmup": 5, "ramp": 10, "prefix": 20,
              "trade_off": dict(dann=.1, deepcoral=.1, cdan=.1, mdd=.5, mcc=1.)},
    "dark": {"image_lr": .00005, "vis": .3, "warmup": 8, "ramp": 15, "prefix": 30,
             "trade_off": dict(dann=.2, deepcoral=.2, cdan=.1, mdd=1., mcc=1.)},
}


def settings_for(target, variant):
    setting = dict(WEATHER[target])
    setting["encoder_warmup_freeze_epochs"] = 0
    if variant == "joint_r18_v11":
        setting["image_lr"] = .001 if target == "dark" else .0003
        setting["encoder_warmup_freeze_epochs"] = setting["warmup"]
    return setting


def commands(snapshot, data, output, weights, target, methods, variant="calibrated_r18_v10"):
    setting = settings_for(target, variant)
    for method in methods:
        name = f"{method}_{target}_"+("joint_r18" if variant == "joint_r18_v11" else "calibrated_r18")
        argv = [sys.executable, "-B", "-u", str(snapshot/"weather/train_vds_core.py"),
                "--method", method, "--target", target, "--data-dir", str(data), "--save-dir", str(output/name),
                "--weights", str(weights), "--image-backbone", "resnet18", "--encoder-training-scope", "all",
                "--classifier-layout", "linear", "--feature-normalization", "unit", "--optimizer", "sgd",
                "--lr", str(.01 if method in {"deepcoral", "cdan"} else .1),
                "--image-lr-override", str(setting["image_lr"]),
                "--weight-decay", str(.001 if method == "cdan" else .002),
                "--trade-off", str(setting["trade_off"][method]),
                "--initial-vis-weight", str(setting["vis"]), "--gate-lr-multiplier", ".01", "--gate-weight-decay", "0",
                "--freeze-bn-after-epochs", "0", "--freeze-bn-scope", "encoders", "--source-class-balance",
                "--adaptation-warmup-epochs", str(setting["warmup"]), "--adaptation-ramp-epochs", str(setting["ramp"]),
                "--epochs", "60", "--stop-after-epochs", str(setting["prefix"]), "--early-stopping-patience", "0",
                "--batch-size", "16", "--seed", "42", "--num-workers", "2", "--cpu-threads", "2",
                "--cache-evaluation", "--modality-diagnostics", "--log-feature-moments", "--source-branch-probe"]
        if variant == "joint_r18_v11":
            argv.extend(["--encoder-warmup-freeze-epochs", str(setting["encoder_warmup_freeze_epochs"]), "--log-parameter-updates"])
        yield name, argv


def summarize(output, completed=None):
    rows = []
    for path in sorted(output.glob("*/*_seed*/result.json")):
        if completed is not None and path.parent.parent.name not in completed:
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"method": r["method"], "target": r["target"], "seed": r["seed"],
                     "selected_epoch": r["selected_epoch"], "completed_epochs": r["completed_epochs"],
                     "planned_epochs": r["planned_epochs"], "screening_only": r["screening_only"],
                     "metrics": r["target_val"], "result_file": str(path)})
    (output/"screen_results.json").write_text(json.dumps({"formal_release": False, "runs": rows}, indent=2)+"\n", encoding="utf-8")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", choices=tuple(WEATHER), required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-parallel", type=int, choices=(1, 2), default=1)
    p.add_argument("--variant", choices=("calibrated_r18_v10", "joint_r18_v11"), default="calibrated_r18_v10")
    args = p.parse_args()
    root, output = args.project_root.resolve(), args.output.resolve()
    data = root/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/prepared_exact_pairs"
    weights = root/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if len(set(args.targets)) != len(args.targets) or len(set(args.methods)) != len(args.methods):
        p.error("Do not repeat targets or methods")
    if not data.is_dir() or not weights.is_file():
        raise FileNotFoundError("Existing exact-pair lists and ResNet18 ImageNet weights are required")
    snapshot = root if args.dry_run else prepare_snapshot(root)
    jobs = [item for target in args.targets for item in commands(snapshot, data, output, weights, target, args.methods, args.variant)]
    plan = {"started_at": datetime.now().isoformat(), "hypothesis": args.variant,
            "snapshot": str(snapshot), "seed": 42, "target_training_labels_used": False,
            "formal_release": False, "max_parallel": args.max_parallel,
            "weather_parameters": {t: settings_for(t, args.variant) for t in args.targets},
            "candidates": [{"name": name, "argv": argv} for name, argv in jobs]}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    output.mkdir(parents=True, exist_ok=False)
    (output/"candidate_plan.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JMDA_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    print_lock = Lock()
    def announce(message):
        with print_lock:
            print(message, flush=True)
    def execute(name, argv):
        announce("START="+name)
        with (output/(name+".log")).open("x", encoding="utf-8") as stream:
            status = subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if status == 0 and len(list((output/name).glob("*_seed*/result.json"))) != 1:
            status = 2
        return name, status
    failures, completed = [], set()
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = [pool.submit(execute, name, argv) for name, argv in jobs]
        for future in as_completed(futures):
            name, status = future.result()
            if status:
                failures.append(name)
                announce(f"FAILED={name}; exit={status}")
            else:
                completed.add(name)
                announce(f"COMPLETE={name}")
            count = len(summarize(output, completed))
            (output/"queue_status.json").write_text(json.dumps({"status": "RUNNING", "failed": failures,
                "last_finished": name, "completed_candidates": count, "requested_candidates": len(jobs)}, indent=2)+"\n", encoding="utf-8")
    (output/"queue_status.json").write_text(json.dumps({"status": "FAIL" if failures else "PASS", "failed": failures,
        "completed_candidates": len(summarize(output, completed)), "requested_candidates": len(jobs)}, indent=2)+"\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
