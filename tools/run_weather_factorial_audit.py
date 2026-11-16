"""Real seed42 encoder/DA controls for dark DANN and foggy MCC.

Four cells keep the same architecture, initialization, cohorts and clocks:
trainable/frozen image encoders x native DA enabled/disabled. Source auxiliary
losses remain fixed. Controls without native DA are not reported as DA methods.
No synthetic logs, model checkpoints or server jobs are used.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot

REFERENCES={
    "dark":"runs/weather_local_screen/remaining_weather_v10_calibrated_r18_seed42/dann_dark_calibrated_r18/dann_dark_seed42/run_config.json",
    "foggy":"runs/weather_enhanced_screen/no_target_pair_mcc_foggy_seed42_v1/mcc_foggy_source0.1_target0/mcc_foggy_seed42/run_config.json",
}
CELLS=("train_source","frozen_source","train_da","frozen_da")


def build_jobs(output,snapshot,weathers,cells):
    jobs=[]
    from weather.train_vds_core import parser
    defaults={a.dest:a.default for a in parser()._actions}
    for weather in weathers:
        reference_path=ROOT/REFERENCES[weather]
        config=json.loads(reference_path.read_text(encoding="utf-8"))
        reference=config["arguments"]
        for cell in cells:
            values=dict(reference)
            # Dark retains the original planned60/observed30 prefix and its
            # schedules; fog retains full20. Do not silently restart clocks.
            common={"save_dir":str(output/weather/cell),"seed":42,"device":"cuda",
                    "cuda_memory_fraction":.4,"log_parameter_updates":True,
                    "early_stopping_patience":0}
            values.update(common)
            if cell.startswith("frozen"):
                values.update(encoder_training_scope="frozen",freeze_image_encoders=True,
                              ir_training_scope="inherit")
            if cell.endswith("source"):
                values["trade_off"]=0.
            # All four cells keep source supervised crossmodal/auxiliary CE,
            # batch size, augmentation and target forward behavior identical.
            changes={k:{"reference":reference.get(k,defaults.get(k)),"candidate":v}
                     for k,v in values.items() if reference.get(k,defaults.get(k))!=v}
            command=[sys.executable,"-B","-u",str(snapshot/"weather/train_vds_core.py"),
                     *recorded_arguments_to_cli(values)]
            jobs.append({"weather":weather,"cell":cell,"method":values["method"],
                "reference_config":str(reference_path),"changes":changes,"command":command,
                "result_file":str(Path(values["save_dir"])/f"{values['method']}_{weather}_seed42"/"result.json"),
                "native_DA_enabled":values["trade_off"]>0,
                "control_is_formal_DA_score":not cell.endswith("source")})
    return jobs


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def summarize(output,jobs):
    from tools.summarize_weather_goal import read_run
    rows=[]
    for job in jobs:
        p=Path(job["result_file"])
        if not p.is_file():continue
        row=read_run(p,"factorial_"+job["cell"])
        row.update(cell=job["cell"],native_DA_enabled=job["native_DA_enabled"],
                   control_is_formal_DA_score=job["control_is_formal_DA_score"])
        evaluation=[json.loads(s) for s in (p.parent/"evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
        source=[r for r in evaluation if r["partition"]=="source_train"]
        target=[r for r in evaluation if r["partition"]=="target_val"]
        matrix=next(r["metrics"]["confusion_matrix"] for r in target if r["epoch"]==row["selected_epoch"])
        counts=[sum(r[i] for r in matrix) for i in range(len(matrix))]
        row.update(source_final_ACC_percent=100*source[-1]["metrics"]["accuracy"],
                   target_final_ACC_percent=100*target[-1]["metrics"]["accuracy"],
                   peak_is_single_class_collapse=sum(n>0 for n in counts)==1,
                   predicted_counts_at_peak=counts)
        rows.append(row)
    write(output/"factorial_results.json",{"real_training_only":True,"simulated":False,
        "target_training_labels_used":False,"source_only_controls_separated":True,"rows":rows})
    return rows


def execute(job,output,smoke=False):
    from weather.local_capacity import local_training_lease,commit_free_bytes
    import psutil
    with local_training_lease(ROOT/"runs/weather_local_screen/.local_training.lock",timeout_seconds=120):
        deadline=time.monotonic()+120
        while True:
            free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0])
            commit=commit_free_bytes()
            if free>=2200 and psutil.virtual_memory().available>=6*1024**3 and (commit is None or commit>=6*1024**3):break
            if time.monotonic()>deadline:raise RuntimeError("Insufficient local memory headroom; no additional job started")
            print("WAITING_LOCAL_HEADROOM",flush=True);time.sleep(5)
        name=job["weather"]+"_"+job["cell"]+("_preflight" if smoke else "")
        command=job["command"].copy()
        if smoke:
            command[command.index("--save-dir")+1]=str(output/"preflight"/name)
            command += ["--max-train-batches","1","--stop-after-epochs","1",
                        "--adaptation-warmup-epochs","0","--adaptation-ramp-epochs","1"]
        env=dict(os.environ,JMDA_DATA_ROOT="D:/Code/JMDA-Net/Data",JMDA_DETERMINISTIC="1",
                 CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONUNBUFFERED="1",LOKY_MAX_CPU_COUNT="2")
        print("START="+name,flush=True)
        with (output/(name+".log")).open("x",encoding="utf-8") as f:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        print(f"FINISHED={name}; exit={code}",flush=True)
        if code:raise RuntimeError(f"{name} failed; see {output/(name+'.log')}")
        return name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--weathers",nargs="+",choices=tuple(REFERENCES),default=list(REFERENCES))
    parser.add_argument("--cells",nargs="+",choices=CELLS,default=list(CELLS))
    parser.add_argument("--execute",action="store_true",help="Otherwise write plan only")
    args=parser.parse_args()
    if any(len(x)!=len(set(x)) for x in [args.weathers,args.cells]):parser.error("Duplicate trials")
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/"runs/weather_local_screen"):
        parser.error("This is a local-only audit; output must be under runs/weather_local_screen")
    if output.exists():raise FileExistsError(output)
    snapshot=prepare_snapshot(ROOT)
    jobs=build_jobs(output,snapshot,args.weathers,args.cells)
    output.mkdir(parents=True)
    write(output/"factorial_plan.json",{"created_UTC":datetime.now(timezone.utc).isoformat(),
        "seed":42,"simulated":False,"target_train_labels_used":False,"jobs":jobs,
        "encoder_factor_includes_trainability_and_encoder_training_mode":True,
        "source_auxiliary_losses_fixed":True,"trained_model_checkpoints_saved":False,
        "server_jobs_accessed":False})
    if not args.execute:print("PLAN_ONLY="+str(output),flush=True);return
    finished=[]
    try:
        for job in jobs:
            execute(job,output,True)
            finished.append(execute(job,output))
            rows=summarize(output,jobs)
            write(output/"queue_status.json",{"status":"RUNNING","finished":finished,"requested":len(jobs)})
        write(output/"queue_status.json",{"status":"PASS","finished":finished,"requested":len(jobs)})
    except Exception as exc:
        write(output/"queue_status.json",{"status":"FAIL","finished":finished,"error":str(exc)})
        raise
    print("COMPLETED_REAL_CONTROLS="+str(len(rows)),flush=True)


if __name__=="__main__":main()
