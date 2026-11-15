"""Authorized enhanced UDA screening, separated from the original method runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--target",choices=("rainy","foggy","dark","backlight"),default="rainy")
    p.add_argument("--methods",nargs="+",choices=("dann","deepcoral","cdan","mdd","mcc"),default=["mcc"])
    p.add_argument("--teacher-weights",nargs="+",type=float,default=[1.,4.])
    p.add_argument("--vis-weight",type=float,default=.75)
    p.add_argument("--components",type=int,default=32)
    p.add_argument("--association",choices=("source_votes","source_prototypes","source_neighbors","source_trimmed"),default="source_votes")
    p.add_argument("--crossmodal-association",action="store_true")
    p.add_argument("--association-vis-weight",type=float,default=.75)
    p.add_argument("--teacher-temperature",type=float,default=.1)
    p.add_argument("--teacher-confidence",type=float,default=.4)
    p.add_argument("--learning-rate-scale",type=float,default=1.)
    p.add_argument("--adapter-scale",type=float,default=.1)
    p.add_argument("--minimum-epochs",type=int,default=12)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--parallel",type=int,choices=(1,2),default=2)
    args=p.parse_args()
    if len(set(args.methods))!=len(args.methods) or len(set(args.teacher_weights))!=len(args.teacher_weights):p.error("Duplicate trials")
    import psutil
    free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0])
    if free<3000 or psutil.virtual_memory().available<4*1024**3:raise RuntimeError("Insufficient headroom for frozen-encoder candidates")
    snapshot=prepare_snapshot(ROOT)
    data=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/prepared_exact_pairs"
    weights=ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
    jobs=[]
    for method in args.methods:
        for teacher_weight in args.teacher_weights:
            name=f"{method}_{args.target}_teacher{teacher_weight:g}"
            cmd=[sys.executable,"-B","-u",str(snapshot/"weather/train_vds_core.py"),
                "--method",method,"--target",args.target,"--data-dir",str(data),
                "--save-dir",str(args.output.resolve()/name),"--weights",str(weights),"--image-backbone","resnet18",
                "--freeze-image-encoders","--classifier-layout","linear","--feature-normalization","unit",
                "--feature-calibration","--calibration-components",str(args.components),"--cluster-association",args.association,
                "--cluster-teacher-weight",str(teacher_weight),"--cluster-teacher-temperature",str(args.teacher_temperature),
                "--cluster-teacher-confidence",str(args.teacher_confidence),"--calibration-adapter-scale",str(args.adapter_scale),
                "--source-class-balance","--label-smoothing",".05","--auxiliary-source-weight","0",
                "--initial-vis-weight",str(args.vis_weight),"--gate-lr-multiplier",".1","--gate-weight-decay","0",
                "--optimizer","adamw","--lr",str(args.learning_rate_scale*(.00005 if method in {"cdan","deepcoral"} else .0005)),
                "--weight-decay",".0001","--gradient-clip-norm","5","--mcc-temperature","1",
                "--trade-off",{"dann":".1","deepcoral":".1","cdan":".1","mdd":".2","mcc":".2"}[method],
                "--adaptation-warmup-epochs","0","--adaptation-ramp-epochs","5","--epochs",str(args.epochs),
                "--early-stopping-patience",str(args.patience),"--early-stopping-min-epochs",str(args.minimum_epochs),
                "--batch-size","16","--seed","42","--num-workers","2","--cpu-threads","2",
                "--cache-evaluation","--log-feature-moments","--modality-diagnostics","--log-parameter-updates",
                "--cuda-memory-fraction",".2"]
            if method in {"dann","cdan","mdd"}:cmd += ["--adversarial-lr-override",".0001"]
            if args.crossmodal_association:cmd += ["--cluster-crossmodal-association",
                "--cluster-association-vis-weight",str(args.association_vis_weight)]
            jobs.append({"name":name,"command":cmd})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"plan.json").write_text(json.dumps({"created_UTC":datetime.now(timezone.utc).isoformat(),
        "launcher_pid":os.getpid(),"variant_family":"core_plus_train_statistics_cluster_initialization",
        "target_training_labels_used":False,"jobs":jobs,"user_authorized_additional_losses_and_calibration":True},indent=2),encoding="utf-8")
    def execute(job,preflight=False):
        cmd=job["command"].copy();name=job["name"]
        if preflight:
            name="actual_CUDA_preflight"
            path=ROOT/"runs/weather_local_preflight"/(args.output.name+"_calibrated")
            cmd[cmd.index("--save-dir")+1]=str(path)
            cmd += ["--max-train-batches","1","--stop-after-epochs","1"]
        print("START="+name,flush=True)
        with (args.output/(name+".log")).open("x",encoding="utf-8") as stream:
            env=dict(os.environ,JMDA_DETERMINISTIC="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONUNBUFFERED="1",LOKY_MAX_CPU_COUNT="2")
            status=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        print(f"FINISHED={name}; exit={status}",flush=True)
        return {"name":name,"exit":status}
    preflight=execute(jobs[0],True)
    if preflight["exit"]:
        (args.output/"preflight_failure.json").write_text(json.dumps(preflight),encoding="utf-8")
        raise SystemExit(preflight["exit"])
    finished=[]
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures=[executor.submit(execute,job) for job in jobs]
        for future in as_completed(futures):
            finished.append(future.result())
            (args.output/"queue_status.json").write_text(json.dumps({"requested":len(jobs),"finished":finished,
                "status":"RUNNING" if len(finished)<len(jobs) else "FAIL" if any(x["exit"] for x in finished) else "PASS"},indent=2),encoding="utf-8")
    if any(x["exit"] for x in finished):raise SystemExit(1)


if __name__=="__main__":main()
