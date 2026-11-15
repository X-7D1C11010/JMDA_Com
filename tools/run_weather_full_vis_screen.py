"""Preflight full ConvNeXt VIS tuning, then two controlled real candidates."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot
from tools.summarize_weather_goal import read_run


def change(command,overrides):
    cmd=command.copy()
    for flag,value in overrides.items():
        if flag in cmd:
            if value is None:continue
            cmd[cmd.index(flag)+1]=str(value)
        else:
            cmd.append(flag)
            if value is not None:cmd.append(str(value))
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args()
    import psutil
    free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0])
    if free<4800 or psutil.virtual_memory().available<5*1024**3:
        raise RuntimeError("Wait for the owned small-model queue to finish before the full-VIS preflight")
    baseline=ROOT/"runs/weather_local_screen/goal3h_mcc_temperature_seed42"
    plan=json.loads((baseline/"plan.json").read_text(encoding="utf-8"))
    available=[]
    for job in plan["jobs"]:
        result=baseline/job["name"]/"mcc_foggy_seed42/result.json"
        score=read_run(result,baseline.name)
        available.append((score["ACC_percent"],score["F1_percent"],job))
    _,_,chosen=max(available,key=lambda x:x[:2])
    snapshot=prepare_snapshot(ROOT)
    command=chosen["command"].copy()
    index=next(i for i,x in enumerate(command) if Path(x).name=="train_vds_core.py")
    command[index]=str(snapshot/"weather/train_vds_core.py")
    preflight=ROOT/"runs/weather_local_preflight"/(args.output.name+"_fullVIS")
    if preflight.exists():raise FileExistsError(preflight)
    jobs=[]
    for name,overrides in (("mcc_foggy_fullVIS",{"--encoder-training-scope":"all","--checkpoint-vis-blocks":None,"--cuda-memory-fraction":".65"}),
                           ("mcc_foggy_VISprior90",{"--initial-vis-weight":".9","--cuda-memory-fraction":".25"})):
        cmd=change(command,{**overrides,"--save-dir":str(args.output.resolve()/name)})
        jobs.append({"name":name,"command":cmd,"changed_learning_factor":overrides})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"plan.json").write_text(json.dumps({"created_UTC":datetime.now(timezone.utc).isoformat(),
        "launcher_pid":os.getpid(),"baseline_plan":str(baseline/"plan.json"),"chosen_baseline":chosen["name"],
        "jobs":jobs,"preflight_directory":str(preflight),"target_training_labels_used":False,"qualified":False},indent=2),encoding="utf-8")
    def execute(name,cmd):
        print("START="+name,flush=True)
        with (args.output/(name+".log")).open("x",encoding="utf-8") as stream:
            env=dict(os.environ,JMDA_DETERMINISTIC="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONUNBUFFERED="1")
            status=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        print(f"FINISHED={name}; exit={status}",flush=True)
        return status
    cmd=change(jobs[0]["command"],{"--save-dir":str(preflight),"--stop-after-epochs":"1","--max-train-batches":"1"})
    status=execute("fullVIS_preflight",cmd)
    if status:
        (args.output/"preflight_failure.json").write_text(json.dumps({"exit":status,"formal_training_started":False}),encoding="utf-8")
        raise SystemExit(status)
    r=json.loads((preflight/"mcc_foggy_seed42/result.json").read_text(encoding="utf-8"))
    if not r["smoke"]:raise ValueError("Preflight must remain separate from accuracy results")
    print(f"PREFLIGHT_PASS reserved_GiB={r['complexity']['peak_train_reserved_gb']:.3f}; not an accuracy result",flush=True)
    if args.preflight_only:return
    finished=[]
    for job in jobs:
        finished.append({"name":job["name"],"exit":execute(job["name"],job["command"])})
        (args.output/"queue_status.json").write_text(json.dumps({"requested":2,"finished":finished,
            "status":"RUNNING" if len(finished)<2 else "FAIL" if any(x["exit"] for x in finished) else "PASS"},indent=2),encoding="utf-8")
    if any(x["exit"] for x in finished):raise SystemExit(1)


if __name__=="__main__":main()
