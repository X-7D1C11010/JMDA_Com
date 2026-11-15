"""Two controlled loss-weight candidates following measured gradient audits."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse
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
    p.add_argument("--suite",choices=("loss_weight","mcc_temperature"),default="loss_weight")
    args=p.parse_args()
    import psutil
    free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0])
    if free<4500 or psutil.virtual_memory().available<5*1024**3:
        raise RuntimeError("Insufficient resource headroom for the two small controlled candidates")
    snapshot=prepare_snapshot(ROOT)
    jobs=[]
    scenarios=([("goal3h_fullIR_slowAdversary_seed42","dann","dann_foggy_stronger_DA",{"--trade-off":"1.0"}),
                ("goal3h_mcc_targetBatch32_seed42","mcc","mcc_foggy_stronger_DA",{"--trade-off":"1.0"})]
               if args.suite=="loss_weight" else
               [("goal3h_stronger_adaptation_seed42","mcc","mcc_foggy_temperature05",{"--mcc-temperature":".5"}),
                ("goal3h_stronger_adaptation_seed42","mcc","mcc_foggy_temperature1",{"--mcc-temperature":"1."})])
    for variant,method,name,overrides in scenarios:
        baseline=ROOT/"runs/weather_local_screen"/variant/"plan.json"
        plan=json.loads(baseline.read_text(encoding="utf-8"))
        matches=[x for x in plan["jobs"] if x["command"][x["command"].index("--method")+1]==method and
                 x["command"][x["command"].index("--target")+1]=="foggy"]
        if len(matches)!=1:raise ValueError("Expected one completed baseline candidate")
        cmd=matches[0]["command"].copy()
        index=next(i for i,x in enumerate(cmd) if Path(x).name=="train_vds_core.py")
        cmd[index]=str(snapshot/"weather/train_vds_core.py")
        for flag,value in overrides.items():
            if flag in cmd:cmd[cmd.index(flag)+1]=value
            else:cmd += [flag,value]
        cmd[cmd.index("--cuda-memory-fraction")+1]=".25"
        out=args.output.resolve()/name
        cmd[cmd.index("--save-dir")+1]=str(out)
        jobs.append({"name":name,"method":method,"command":cmd,"baseline_plan":str(baseline),"overrides":overrides,
                     "first_epoch":str(out/f"{method}_foggy_seed42/training.jsonl")})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"plan.json").write_text(json.dumps({"created_UTC":datetime.now(timezone.utc).isoformat(),
        "launcher_pid":os.getpid(),"jobs":jobs,"seed":42,"target_training_labels_used":False,
        "changed_training_factor":("adaptation loss weight .05 ->1.0" if args.suite=="loss_weight" else "MCC temperature2.5 ->.5 or1.0")+
                                  "; all other learning parameters retained",
        "qualified":False},indent=2),encoding="utf-8")
    def execute(job):
        print("START="+job["name"],flush=True)
        with (args.output/(job["name"]+".log")).open("x",encoding="utf-8") as stream:
            env=dict(os.environ,JMDA_DETERMINISTIC="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONUNBUFFERED="1")
            status=subprocess.run(job["command"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        return {"name":job["name"],"exit":status}
    finished=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(execute,jobs[0])]
        while not Path(jobs[0]["first_epoch"]).is_file() and not futures[0].done():time.sleep(3)
        futures.append(executor.submit(execute,jobs[1]))
        for future in as_completed(futures):
            row=future.result();finished.append(row)
            print(f"FINISHED={row['name']}; exit={row['exit']}",flush=True)
            (args.output/"queue_status.json").write_text(json.dumps({"finished":finished,"requested":2,
                "status":"RUNNING" if len(finished)<2 else "FAIL" if any(x["exit"] for x in finished) else "PASS"},indent=2),encoding="utf-8")
    if any(x["exit"] for x in finished):raise SystemExit(1)


if __name__=="__main__":main()
