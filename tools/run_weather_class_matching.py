"""Explicit label-guided queue, preserving supplied reference configurations.

Default writes a reviewable plan only. --execute trains sequentially. Target
true classes control sampling, never an additional target classification CE.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot
from weather.class_matching import protocol_metadata


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build_values(config,output,batch_size,seed=None,data_dir=None,stop_after=None,smoke=False):
    values=dict(config['arguments'])
    if values.get('target_class_matching','none')!='none':
        raise ValueError('Supply the original UDA reference, not a previous guided trial')
    values.update(save_dir=str(output),target_class_matching='target_truth',
                  batch_size=batch_size,target_batch_size=batch_size,max_train_batches=1 if smoke else 0)
    if seed is not None:values['seed']=seed
    if data_dir is not None:values['data_dir']=str(data_dir)
    if stop_after is not None:values['stop_after_epochs']=stop_after
    if smoke:values['stop_after_epochs']=1
    if values.get('stop_after_epochs',0)>values['epochs']:
        raise ValueError('Prefix exceeds the reference epoch budget')
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-config',nargs='+',type=Path,required=True)
    p.add_argument('--batch-size',type=int,required=True,help='Explicit equal source/target batch size')
    p.add_argument('--seed',type=int)
    p.add_argument('--matched-alignment-weight',type=float)
    p.add_argument('--matched-alignment-margin',type=float)
    p.add_argument('--matched-alignment-ramp-epochs',type=int)
    p.add_argument('--adaptation-warmup-epochs',type=int)
    p.add_argument('--adaptation-ramp-epochs',type=int)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--stop-after-epochs',type=int)
    p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true',help='One update/epoch; excluded from performance summaries')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.batch_size<2 or (args.stop_after_epochs is not None and args.stop_after_epochs<0):
        p.error('Batch size must be >=2 and prefix >=0')
    if args.matched_alignment_weight is not None and (not math.isfinite(args.matched_alignment_weight) or args.matched_alignment_weight<0):
        p.error('Alignment weight must be finite and nonnegative')
    if args.matched_alignment_margin is not None and not 0<=args.matched_alignment_margin<2:
        p.error('Alignment cosine margin must lie in [0,2)')
    if any(v is not None and v<0 for v in (args.matched_alignment_ramp_epochs,args.adaptation_warmup_epochs,args.adaptation_ramp_epochs)):
        p.error('Warmup/ramp epochs must be nonnegative')
    output=(args.output or ROOT/'runs/weather_label_guided'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if not output.is_relative_to(ROOT/'runs/weather_label_guided'):
        p.error('Keep this separate protocol under runs/weather_label_guided')
    if output.exists():raise FileExistsError(output)
    snapshot=prepare_snapshot(ROOT)
    jobs=[];identities=set()
    for path in args.reference_config:
        config=json.loads(path.read_text(encoding='utf-8'))
        values=build_values(config,output,args.batch_size,args.seed,args.data_dir,args.stop_after_epochs,args.smoke)
        for key in ('matched_alignment_weight','matched_alignment_margin','matched_alignment_ramp_epochs',
                    'adaptation_warmup_epochs','adaptation_ramp_epochs'):
            if getattr(args,key) is not None:values[key]=getattr(args,key)
        identity=(values['method'],values['target'],values['seed'])
        if identity in identities:raise ValueError('Repeated method/target/seed')
        identities.add(identity)
        name=f'{identity[0]}_{identity[1]}_seed{identity[2]}'
        original=config['arguments']
        changes={k:{'reference':original.get(k),'candidate':v} for k,v in values.items() if original.get(k)!=v}
        jobs.append({'name':name,'reference_config':str(path.resolve()),'changes':changes,
            'result_file':str(output/name/'result.json'),
            'command':[sys.executable,'-B','-u',str(snapshot/'weather/train_vds_core.py'),*recorded_arguments_to_cli(values)]})
    output.mkdir(parents=True)
    supervision=protocol_metadata(True,args.matched_alignment_weight or 0.)
    write(output/'plan.json',{**supervision,'smoke':args.smoke,'jobs':jobs,
        'all_other_reference_parameters_preserved':True,'trained_model_checkpoints_saved':False,
        'batch_size_is_explicit_protocol_change':True})
    print(f"{supervision['protocol']}; output={output}; jobs={len(jobs)}",flush=True)
    if not args.execute:print('PLAN_ONLY; add --execute to train',flush=True);return
    from weather.local_capacity import local_training_lease
    from weather.summarize_class_matching import summarize
    completed=[]
    try:
        for job in jobs:
            write(output/'queue_status.json',{'status':'RUNNING','completed':len(completed),
                                             'total':len(jobs),'active':job['name'],'smoke':args.smoke})
            with local_training_lease(ROOT/'runs/weather_local_screen/.local_training.lock',timeout_seconds=120):
                print('START='+job['name'],flush=True)
                with (output/(job['name']+'.log')).open('x',encoding='utf-8') as stream:
                    code=subprocess.run(job['command'],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1'),
                                        stdout=stream,stderr=subprocess.STDOUT).returncode
                if code:raise RuntimeError(f"{job['name']} failed (exit={code}); see its log")
            completed.append(job['result_file'])
            if not args.smoke:summarize(completed,output/'summary')
            write(output/'queue_status.json',{'status':'RUNNING','completed':len(completed),'total':len(jobs)})
            print('COMPLETED='+job['name'],flush=True)
    except BaseException as error:
        write(output/'queue_status.json',{'status':'FAIL','completed':len(completed),'error':str(error)})
        raise
    write(output/'queue_status.json',{'status':'PASS','completed':len(completed),'total':len(jobs),'smoke':args.smoke})


if __name__=='__main__':main()
