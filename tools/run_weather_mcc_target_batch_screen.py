"""Single-factor MCC target-batch candidate from the measured seed42 baseline."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--target-batch-size",type=int,default=32)
    p.add_argument("--wait-for-pid",type=int,default=0)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    if args.target_batch_size<2:p.error("Target batch size must be at least two")
    if args.wait_for_pid and not args.dry_run:
        import psutil
        try:
            previous=psutil.Process(args.wait_for_pid)
            created=previous.create_time()
            if not any(Path(x).name=="run_weather_goal_screen.py" for x in previous.cmdline()):
                raise ValueError("Predecessor must be the owned local weather runner")
            print(f"WAIT_OWNED_RUNNER={args.wait_for_pid}",flush=True)
            while previous.is_running() and previous.create_time()==created:time.sleep(5)
        except psutil.NoSuchProcess:pass
    baseline=ROOT/"runs/weather_local_screen/goal3h_fullIR_adamw_seed42/plan.json"
    original=json.loads(baseline.read_text(encoding="utf-8"))
    snapshot=ROOT if args.dry_run else prepare_snapshot(ROOT)
    jobs=[]
    for job in original["jobs"]:
        cmd=job["command"].copy()
        if cmd[cmd.index("--method")+1]!="mcc":continue
        target=cmd[cmd.index("--target")+1]
        name=f"mcc_{target}_targetBatch{args.target_batch_size}"
        script_index=next(i for i,x in enumerate(cmd) if Path(x).name=="train_vds_core.py")
        cmd[script_index]=str(snapshot/"weather/train_vds_core.py")
        cmd[cmd.index("--save-dir")+1]=str(args.output.resolve()/name)
        cmd[cmd.index("--cuda-memory-fraction")+1]=".5"
        cmd += ["--target-batch-size",str(args.target_batch_size)]
        jobs.append({"name":name,"command":cmd})
    if len(jobs)!=3:raise ValueError("Expected exactly three prior MCC candidates")
    plan={"created_UTC":datetime.now(timezone.utc).isoformat(),"launcher_pid":os.getpid(),
          "baseline_plan":str(baseline),"seed":42,"jobs":jobs,
          "changed_training_factor":"target batch8 ->32; source batch8 and source sampler/update budget retained",
          "unchanged_loss":"released minimum class confusion on target logits only",
          "target_training_labels_used":False,"qualified":False}
    if args.dry_run:print(json.dumps(plan,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    finished=[]
    for job in jobs:
        import psutil
        while True:
            free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free",
                     "--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0])
            if free>=3500 and psutil.virtual_memory().available>=4*1024**3:break
            print(f"WAIT_RESOURCE free_GPU_MiB={free}",flush=True);time.sleep(15)
        print("START="+job["name"],flush=True)
        with (args.output/(job["name"]+".log")).open("x",encoding="utf-8") as stream:
            env=dict(os.environ,JMDA_DETERMINISTIC="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONUNBUFFERED="1")
            status=subprocess.run(job["command"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        finished.append({"name":job["name"],"exit":status})
        (args.output/"queue_status.json").write_text(json.dumps({"requested":3,"finished":finished,
             "status":"RUNNING" if len(finished)<3 else "FAIL" if any(x["exit"] for x in finished) else "PASS"},indent=2),encoding="utf-8")
        print(f"FINISHED={job['name']}; exit={status}",flush=True)
    if any(x["exit"] for x in finished):raise SystemExit(1)


if __name__=="__main__":main()
