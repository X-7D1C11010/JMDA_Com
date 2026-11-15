"""Time-bounded real-image screening; no claimed success without result files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from threading import Lock, Condition
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("unip_stage", "convnext_stage", "unip_stage_weather", "convnext_stage_weather", "convnext_stage_weather_aspect", "full_tiny_weather_aspect"), default="unip_stage")
    p.add_argument("--methods", nargs="+", default=["dann", "deepcoral"], choices=("dann", "deepcoral", "cdan", "mdd", "mcc"))
    p.add_argument("--targets", nargs="+", default=["dark", "rainy", "foggy"], choices=("dark", "rainy", "foggy"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parallel", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--share-with-pid", type=int, default=0,
                   help="At most one light worker while this owned heavy runner remains live")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--optimizer",choices=("adam","adamw","sgd"),default=None,
                   help="Explicit optimizer candidate; does not change any named DA loss")
    p.add_argument("--adversarial-lr-override",type=float,default=None,
                   help="Single-factor discriminator/head LR candidate; keeps all other suite settings")
    p.add_argument("--wait-for-pid", type=int, default=0, help="Wait for a specific live predecessor process before using CUDA")
    args = p.parse_args()
    if len(set(args.targets)) != len(args.targets) or len(set(args.methods)) != len(args.methods):
        p.error("Repeated candidates are not allowed")
    if args.adversarial_lr_override is not None and (args.adversarial_lr_override<=0 or
            any(method not in {"dann","cdan","mdd"} for method in args.methods)):
        p.error("A positive adversarial LR applies only to DANN/CDAN/MDD")
    if args.parallel==3 and args.suite!="full_tiny_weather_aspect":
        p.error("Three-way concurrency is only memory-checked for the BF16 tiny variant")
    if args.wait_for_pid and not args.dry_run:
        import psutil
        try:
            predecessor=psutil.Process(args.wait_for_pid)
            created=predecessor.create_time()
            command=predecessor.cmdline()
            if not any(Path(x).name=="run_weather_goal_screen.py" for x in command):
                raise ValueError("Predecessor is not the expected owned weather runner")
            print(f"WAIT_LIVE_PID={args.wait_for_pid}; created={created}",flush=True)
            while predecessor.is_running() and predecessor.create_time()==created:
                time.sleep(5)
        except psutil.NoSuchProcess:
            pass
    snapshot = ROOT if args.dry_run else prepare_snapshot(ROOT)
    data = ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/prepared_exact_pairs"
    weight = ROOT/"pretrained_weights/torch/hub/checkpoints/convnext_small-0c510722.pth"
    if not weight.is_file(): weight = Path.home()/".cache/torch/hub/checkpoints/convnext_small-0c510722.pth"
    plans = []
    for target in args.targets:
        for method in args.methods:
            name = f"{method}_{target}_{args.suite}"
            vis = {"rainy": .2, "foggy": .85, "dark": .5}[target]
            command = [sys.executable, "-B", "-u", str(snapshot/"weather/train_vds_core.py"),
                "--method", method, "--target", target, "--data-dir", str(data), "--save-dir", str(args.output.resolve()/name),
                "--weights", str(weight), "--image-backbone", "convnext_small", "--encoder-training-scope", "last_stages",
                "--classifier-layout", "linear", "--feature-normalization", "unit", "--source-class-balance",
                "--optimizer", "adam", "--lr", ".0003" if method in {"deepcoral", "cdan"} else ".001",
                "--image-lr-override", ".0001", "--image-adaptation-lr-scale", ".1", "--auxiliary-source-weight", ".5",
                "--weight-decay", ".0001", "--label-smoothing", ".05", "--gradient-clip-norm", "5",
                "--initial-vis-weight", str(vis), "--gate-lr-multiplier", ".1", "--gate-weight-decay", "0",
                "--trade-off", {"dann":".05", "deepcoral":".1", "cdan":".1", "mdd":".2", "mcc":".05"}[method],
                "--adaptation-warmup-epochs", "8", "--adaptation-ramp-epochs", "12", "--epochs", "40",
                "--early-stopping-patience", "12", "--early-stopping-min-epochs", "24",
                "--batch-size", "16", "--seed", "42", "--num-workers", "2", "--cpu-threads", "2",
                "--cache-evaluation", "--log-feature-moments", "--modality-diagnostics", "--source-branch-probe", "--log-parameter-updates"]
            if args.suite.startswith("unip_stage"):
                command += ["--ir-backbone", "unip_small", "--ir-weights", str(ROOT/"pretrained_weights/unip/unip_vit_small_by_ibot_large.pth"),
                            "--ir-feature-mode", "cls" if target == "rainy" else "patch_mean"]
            if "_weather" in args.suite:
                command += ["--source-weather-probability", ".75"]
            if args.suite.endswith("_aspect"):
                command += ["--source-aspect-min", ".35" if target=="dark" else ".5",
                            "--source-aspect-max", "2.85" if target=="dark" else "2."]
            if args.suite=="full_tiny_weather_aspect":
                replacements={"--weights":str(weight),
                    "--image-backbone":"convnext_small","--encoder-training-scope":"last_stages","--batch-size":"8",
                    "--epochs":"20","--adaptation-warmup-epochs":"4","--adaptation-ramp-epochs":"6",
                    "--early-stopping-patience":"6","--early-stopping-min-epochs":"12",
                    "--image-lr-override":".00005","--initial-vis-weight":".5"}
                for flag,value in replacements.items():command[command.index(flag)+1]=value
                command += ["--ir-backbone","unip_tiny","--ir-weights",str(ROOT/"pretrained_weights/unip/unip_vit_tiny_by_ibot_large.pth"),
                    "--ir-feature-mode","cls" if target=="rainy" else "patch_mean","--ir-lr-override",".0001",
                    "--encoder-precision","bf16","--ir-training-scope","all_used_layers",
                    "--cuda-memory-fraction",".16"]
            plans.append({"name": name, "command": command})
            if args.optimizer:command[command.index("--optimizer")+1]=args.optimizer
            if args.adversarial_lr_override is not None:
                command += ["--adversarial-lr-override",str(args.adversarial_lr_override)]
    plan = {"created_UTC": datetime.now(timezone.utc).isoformat(), "suite": args.suite,
            "launcher_pid": os.getpid(),
            "seed": 42, "target_training_labels_used": False, "qualified": False,
            "source_stage": "4epochs full tiny /8epochs last stages; source classification plus independent branch CE",
            "adaptation_stage": "unchanged named DA loss ramp; encoder LR reduced10x, head LR retained",
            "jobs": plans}
    if args.dry_run: print(json.dumps(plan, indent=2)); return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    lock = Lock()
    capacity=Condition();running=0;active_paths={};first_epoch_seen=False
    def shared_runner_live():
        if not args.share_with_pid:return False
        import psutil
        try:
            other=psutil.Process(args.share_with_pid)
            if not any(Path(x).name=="run_weather_goal_screen.py" for x in other.cmdline()):
                raise ValueError("Shared PID is not the expected owned runner")
            return other.is_running()
        except psutil.NoSuchProcess:return False
    def announce(s):
        with lock: print(s, flush=True)
    def run(job):
        nonlocal running,first_epoch_seen
        with capacity:
            while True:
                first_epoch_seen = first_epoch_seen or any(p.exists() for p in active_paths.values())
                limit=1 if shared_runner_live() else args.parallel
                if args.parallel==3 and not first_epoch_seen:limit=1
                enough=True
                if args.parallel==3:
                    import psutil
                    free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0])
                    startup=sum(2048 for p in active_paths.values() if not p.exists())
                    enough=free-startup>=2048 and psutil.virtual_memory().available>=3*1024**3
                if running<limit and enough:break
                capacity.wait(timeout=3)
            running+=1
            cmd=job["command"];out=Path(cmd[cmd.index("--save-dir")+1])
            active_paths[job["name"]]=out/f"{cmd[cmd.index('--method')+1]}_{cmd[cmd.index('--target')+1]}_seed42"/"training.jsonl"
        try:
            announce("START="+job["name"])
            with (args.output/(job["name"]+".log")).open("x", encoding="utf-8") as stream:
                env = dict(os.environ, JMDA_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
                status = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            return job["name"], status
        finally:
            with capacity:
                running-=1;active_paths.pop(job["name"],None);capacity.notify_all()
    finished = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run, job) for job in plans]
        for future in as_completed(futures):
            name, status = future.result()
            finished.append({"name": name, "exit": status})
            announce(f"FINISHED={name}; exit={status}")
            (args.output/"queue_status.json").write_text(json.dumps({"finished": finished, "requested": len(plans),
                "status": "RUNNING" if len(finished)<len(plans) else "FAIL" if any(x["exit"] for x in finished) else "PASS"}, indent=2), encoding="utf-8")
    if any(x["exit"] for x in finished): raise SystemExit(1)


if __name__ == "__main__": main()
