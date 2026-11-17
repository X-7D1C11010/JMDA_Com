"""Sequential real seed screens, explicitly separate from UDA experiments."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weather.label_guided_presets import REFERENCE_PARAMETERS,RECORDED_PEAK_RESERVED_GIB,build_preset
from weather.class_matching import ALIGNMENT_PROTOCOL,protocol_metadata
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from weather.fewshot_presets import PAIRS as SUITE_PAIRS,ALLOCATION,build_fewshot_preset


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def alignment_overrides(weight,profile='reference',labels_per_class=0):
    """Independent coefficient variant; zero disables the additional loss only."""
    if not isinstance(labels_per_class,int) or labels_per_class<0:raise ValueError('Invalid target label budget')
    if weight is None:
        if not labels_per_class:return {}
        weight=5.
    if not math.isfinite(weight) or weight<0:
        raise ValueError('Alignment weight must be finite and nonnegative')
    tag=format(weight,'.12g').replace('.','p')
    values={'matched_alignment_weight':weight,'log_parameter_updates':True,
            'configuration_tag':f'{profile}_alignment{tag}_v1'}
    if labels_per_class:
        if weight<=0:raise ValueError('Few-shot guidance needs a positive alignment coefficient')
        values.update(target_labels_per_class=labels_per_class,
                      configuration_tag=f'{profile}_alignment{tag}_{labels_per_class}shot_v1')
    return values


def preflight_command(command,save_dir):
    command=list(command)
    command[command.index('--save-dir')+1]=str(save_dir)
    if '--holdout-list' in command:
        position=command.index('--holdout-list')
        del command[position:position+2]
    no_extra=('--matched-alignment-weight' in command and
              float(command[command.index('--matched-alignment-weight')+1])==0.)
    fewshot=('--target-labels-per-class' in command and
             int(command[command.index('--target-labels-per-class')+1])>0)
    if fewshot:
        return command+['--max-train-batches','4','--stop-after-epochs','2','--selection-min-epoch','1',
                        '--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','0',
                        '--encoder-warmup-freeze-epochs','0']
    if no_extra:
        return command+['--max-train-batches','2','--stop-after-epochs','2','--selection-min-epoch','1',
                        '--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','0',
                        '--encoder-warmup-freeze-epochs','0']
    return command+['--max-train-batches','1','--stop-after-epochs','1','--selection-min-epoch','1']


def build_jobs(output,data_dir,pairs,seeds,snapshot,complete_budget=False,overrides=None,evaluation_mode='validation_peak',parameter_profile='reference',unified_fewshot=False):
    jobs=[]
    for pair in pairs:
        method,target=pair.rsplit('_',1)
        for seed in seeds:
            parent=output/pair/f'seed{seed}'
            values=(build_fewshot_preset(ROOT,data_dir,parent,method,target,seed) if unified_fewshot else
                    build_preset(ROOT,data_dir,parent,method,target,seed,parameter_profile))
            if complete_budget:values['stop_after_epochs']=0
            if evaluation_mode in ('holdout_ce','validation_ce'):
                values.update(selection_metric='cross_entropy',holdout_list=(str(Path(data_dir)/f'{target}_test.txt') if evaluation_mode=='holdout_ce' else None),
                    selection_min_epoch=values['adaptation_warmup_epochs']+1,
                    early_stopping_monitor='cross_entropy',early_stopping_patience=8,early_stopping_min_epochs=12,
                    validation_control_start_epoch=1,val_plateau_patience=4)
            elif evaluation_mode!='validation_peak':raise ValueError('Unknown evaluation mode')
            if parameter_profile in ('conservative','lr_only'):
                values.update(early_stopping_patience=6,early_stopping_min_epochs=12,
                              early_stopping_monitor='cross_entropy',validation_control_start_epoch=1,val_plateau_patience=3)
            values.update(overrides or {})
            if unified_fewshot:
                values.update(epochs=40,stop_after_epochs=0,early_stopping_patience=0,
                    target_labels_per_class=3,matched_alignment_weight=.1,
                    configuration_tag='fewshot3_alignment0p1_40epoch_v1')
            if values.get('early_stopping_min_epochs',0)>values['epochs']:
                raise ValueError('Early-stopping minimum exceeds the planned epoch budget')
            name=f'{pair}_seed{seed}'
            jobs.append({'name':name,'pair':pair,'values':values,
                'required_free_MiB':4096 if target=='rainy' else max(2048,math.ceil(RECORDED_PEAK_RESERVED_GIB[pair]*1280+512)),
                'source_reference':('weather/rainy_presets.py; teacher losses removed; effective encoder LR preserved' if target=='rainy' else REFERENCE_PARAMETERS[pair]['reference']),
                'result_file':str(parent/f'{method}_{target}_seed{seed}'/'result.json'),
                'command':[sys.executable,'-B','-u',str(snapshot/'weather/train_vds_core.py'),
                           *recorded_arguments_to_cli(values)]})
    return jobs


def write_pair_summary(output,rows,expected_seeds):
    groups=defaultdict(list)
    for row in rows:groups[(row['method'],row['target'])].append(row)
    if len({(r.get('selection_metric','accuracy_f1'),r.get('target_test_available',False),r.get('configuration_tag','reference')) for r in rows})>1:
        raise ValueError('Do not mix evaluation protocols in a seed summary')
    if len({r.get('protocol',ALIGNMENT_PROTOCOL) for r in rows})>1:
        raise ValueError('Do not mix alignment and sampling-only protocols')
    summary=[]
    for (method,target),group in sorted(groups.items()):
        if len({x['seed'] for x in group})!=len(group):raise ValueError('Repeated seed in evidence inventory')
        record={'protocol':group[0].get('protocol',ALIGNMENT_PROTOCOL),'method':method,'target':target,'completed_seeds':len(group),
            'expected_seeds':len(expected_seeds),'all_requested_seeds_completed':{x['seed'] for x in group}==set(expected_seeds),
            'std_ddof':0,'screening_only':group[0]['screening_only'],
            'configuration_tag':group[0].get('configuration_tag','reference'),
            'selection_metric':group[0].get('selection_metric','accuracy_f1'),
            'target_test_available':group[0].get('target_test_available',False)}
        record['reporting_partition']=group[0].get('reporting_partition','target_val')
        for metric in ('ACC_percent','Precision_macro_percent','Recall_macro_percent','F1_macro_percent'):
            values=[x[metric] for x in group]
            record[metric+'_mean']=statistics.mean(values)
            record[metric+'_std']=statistics.pstdev(values) if len(values)>1 else None
            if group[0].get('target_test_available'):
                tests=[x['heldout_'+metric] for x in group]
                record['heldout_'+metric+'_mean']=statistics.mean(tests)
                record['heldout_'+metric+'_std']=statistics.pstdev(tests) if len(tests)>1 else None
        for metric in ('peak_validation_ACC_percent','selected_validation_ACC_percent'):
            if all(metric in x for x in group):
                values=[x[metric] for x in group]
                record[metric+'_mean']=statistics.mean(values)
                record[metric+'_std']=statistics.pstdev(values) if len(values)>1 else None
        summary.append(record)
    with (output/'summary_by_pair.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    return summary


def write_peak_summary(output,rows,expected_seeds):
    """All four peak metrics come from the single maximum-ACC epoch."""
    converted=[]
    names=('ACC_percent','Precision_macro_percent','Recall_macro_percent','F1_macro_percent')
    for row in rows:
        converted.append({**{k:row[k] for k in ('method','target','seed','protocol','configuration_tag','screening_only')},
            'selection_metric':'accuracy_f1','target_test_available':False,'reporting_partition':'target_val',
            'selected_epoch':row['peak_validation_epoch'],
            **{key:row['peak_validation_'+key] for key in names}})
    output=Path(output)
    temporary=output/'peak_acc_summary'
    temporary.mkdir(exist_ok=True)
    summary=write_pair_summary(temporary,converted,expected_seeds)
    for filename,records in (('peak_acc_runs.csv',converted),('peak_acc_summary.csv',summary)):
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--pairs',nargs='+',choices=sorted(SUITE_PAIRS),default=sorted(REFERENCE_PARAMETERS))
    p.add_argument('--unified-fewshot',action='store_true',help='Fixed 3 labels/class, alignment .1, 40 epochs, no teacher losses')
    p.add_argument('--machine-split',choices=('local','server'))
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--local',action='store_true',help='Use the shared local GPU lease and memory headroom guard')
    p.add_argument('--local-lease-timeout-seconds',type=float,default=120.,help='0 waits indefinitely for the existing local training queue')
    p.add_argument('--matched-alignment-weight',type=float,help='Nonnegative independent variant; zero keeps truth-guided sampling without extra alignment')
    p.add_argument('--target-labels-per-class',type=int,default=0)
    p.add_argument('--complete-budget',action='store_true',help='Remove local reference prefixes; train each preset full planned budget')
    p.add_argument('--early-stopping-patience',type=int)
    p.add_argument('--early-stopping-min-epochs',type=int)
    p.add_argument('--parameter-profile',choices=('reference','conservative','lr_only'),default='reference')
    p.add_argument('--evaluation-mode',choices=('validation_peak','validation_ce','holdout_ce'),default='validation_ce')
    p.add_argument('--gpus',default=os.environ.get('JMDA_GPU_CANDIDATES','0,1'))
    p.add_argument('--execute',action='store_true',help='Otherwise write exact commands only')
    args=p.parse_args()
    if args.unified_fewshot and (args.target_labels_per_class!=3 or args.matched_alignment_weight!=.1 or args.evaluation_mode!='validation_ce' or args.parameter_profile!='reference'):
        p.error('Unified few-shot requires --target-labels-per-class 3 --matched-alignment-weight 0.1 and reference/validation_ce')
    if not args.unified_fewshot and any(pair not in REFERENCE_PARAMETERS for pair in args.pairs):
        p.error('Rainy is available only through the explicit unified few-shot variant')
    if args.machine_split and (not args.unified_fewshot or set(args.pairs)!=set(ALLOCATION[args.machine_split])):
        p.error('Method/weather list differs from the requested machine allocation')
    try:variant_overrides=alignment_overrides(args.matched_alignment_weight,args.parameter_profile,args.target_labels_per_class)
    except ValueError as error:p.error(str(error))
    if not math.isfinite(args.local_lease_timeout_seconds) or args.local_lease_timeout_seconds<0:
        p.error('Local lease timeout must be finite and nonnegative')
    if len(set(args.pairs))!=len(args.pairs) or len(set(args.seeds))!=len(args.seeds):p.error('Repeated trials')
    if any(v is not None and v<0 for v in (args.early_stopping_patience,args.early_stopping_min_epochs)):
        p.error('Early-stopping settings must be nonnegative')
    output=(args.output or ROOT/'runs/weather_label_guided'/('matrix_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if not output.is_relative_to(ROOT/'runs/weather_label_guided'):p.error('Use the separate runs/weather_label_guided directory')
    if output.exists():raise FileExistsError(output)
    if not args.data_dir.is_dir():raise FileNotFoundError(args.data_dir)
    snapshot=prepare_snapshot(ROOT)
    overrides={key:getattr(args,key) for key in ('early_stopping_patience','early_stopping_min_epochs') if getattr(args,key) is not None}
    overrides.update(variant_overrides)
    if args.evaluation_mode in ('holdout_ce','validation_ce') and not (args.data_dir/'holdout_protocol_manifest.json').is_file():
        raise FileNotFoundError('Prepare the locked retraining split before CE-based evaluation')
    jobs=build_jobs(output,args.data_dir.resolve(),args.pairs,args.seeds,snapshot,args.complete_budget,overrides,args.evaluation_mode,args.parameter_profile,args.unified_fewshot)
    for job in jobs:
        a=job['values']
        for key in ('weights','ir_weights','holdout_list'):
            if a.get(key) and not Path(a[key]).is_file():raise FileNotFoundError(f"{job['name']}: missing {key}: {a[key]}")
    output.mkdir(parents=True)
    supervision=protocol_metadata(True,jobs[0]['values']['matched_alignment_weight'],args.target_labels_per_class)
    write(output/'plan.json',{**supervision,'target_classifier_CE_used':False,
        'simulated':False,'evaluation_mode':args.evaluation_mode,'parameter_profile':args.parameter_profile,
        'matched_alignment_weight_override':args.matched_alignment_weight,
        'target_labels_per_class':args.target_labels_per_class,
        'unified_fewshot':args.unified_fewshot,
        'machine_split':args.machine_split,
        'expected_by_weather':{t:sum(j['values']['target']==t for j in jobs) for t in sorted({j['values']['target'] for j in jobs})},
        'accuracy_cap_applied':False,'original_UDA_results_not_overwritten':True,'jobs':jobs})
    print(f"{supervision['protocol']}; jobs={len(jobs)}; output={output}",flush=True)
    if not args.execute:print('PLAN_ONLY',flush=True);return
    completed=[]
    def run(job,smoke):
        stage_started=time.perf_counter()
        stage_utc=datetime.now().astimezone().isoformat()
        command=job['command'].copy()
        name=job['name']+('_smoke' if smoke else '')
        if smoke:
            command=preflight_command(command,output/'preflight'/job['name'])
        env=dict(os.environ,PYTHONUNBUFFERED='1')
        from contextlib import nullcontext
        if args.local:
            from weather.local_capacity import local_training_lease,commit_free_bytes
            import psutil
            lease=local_training_lease(ROOT/'runs/weather_local_screen/.local_training.lock',
                timeout_seconds=args.local_lease_timeout_seconds or None)
            write(output/'queue_status.json',{'status':'RUNNING','active':job['name'],
                'phase':'waiting_for_local_slot','completed':len(completed),'total':len(jobs)})
        else:
            lease=nullcontext()
            command=[sys.executable,'-B','-u',str(snapshot/'tools/run_on_available_gpu.py'),
                '--min-free-mib',str(job['required_free_MiB']),'--max-util-percent',os.environ.get('JMDA_GPU_MAX_UTIL','60'),
                '--gpus',args.gpus,'--stable-checks','3','--poll-seconds','10','--tag',name,'--',*command]
        with lease:
            if args.local:
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                commit=commit_free_bytes()
                if (free<job['required_free_MiB'] or psutil.virtual_memory().available<6*1024**3
                        or commit is not None and commit<6*1024**3):
                    raise RuntimeError('Insufficient local memory headroom; no job started')
            write(output/'queue_status.json',{'status':'RUNNING','active':job['name'],
                'phase':'preflight' if smoke else 'formal_training','completed':len(completed),'total':len(jobs)})
            print('START='+name,flush=True)
            with (output/(name+'.log')).open('x',encoding='utf-8') as stream:
                code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError(f'{name} exit={code}; see its log')
            elapsed=time.perf_counter()-stage_started
            with (output/'timings.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'job':job['name'],'phase':'preflight' if smoke else 'formal',
                    'method':job['values']['method'],'target':job['values']['target'],'seed':job['values']['seed'],
                    'start_time':stage_utc,'end_time':datetime.now().astimezone().isoformat(),
                    'wall_s_including_gpu_wait':elapsed})+'\n')
            print('COMPLETED='+name,flush=True)
    try:
        for job in jobs:
            write(output/'queue_status.json',{'status':'RUNNING','active':job['name'],'completed':len(completed),'total':len(jobs)})
            run(job,True)
            a=job['values']
            smoke_path=output/'preflight'/job['name']/f"{a['method']}_{a['target']}_seed{a['seed']}"/'result.json'
            smoke_result=json.loads(smoke_path.read_text(encoding='utf-8'))
            if smoke_result.get('target_test_available') or smoke_result.get('test_evaluations',0):
                raise RuntimeError('Preflight must not inspect heldout test predictions')
            for key,value in supervision.items():
                if smoke_result.get(key)!=value:raise RuntimeError('Preflight supervision mismatch: '+key)
            if args.target_labels_per_class:
                sampling=smoke_result['label_guided_sampling']
                expected=args.target_labels_per_class*len(sampling['labelled_target_class_counts'])
                if sampling['labelled_target_samples']!=expected or sampling['unselected_truth_used_for_pairing_or_alignment']:
                    raise RuntimeError('Few-shot label budget mismatch')
                events=[json.loads(s) for s in (smoke_path.parent/'updates.jsonl').read_text().splitlines()]
                known={x['path']:x['label'] for x in sampling['selected_labelled_samples']}
                if not all(x['target_known_alignment_labels']==[known.get(p,-1) for p in x['target_batch_paths']] for x in events):
                    raise RuntimeError('Unselected target labels leaked into preflight')
                if not any(x.get('weighted_matched_alignment',0)>0 for x in events):
                    raise RuntimeError('Few-shot class alignment was not exercised')
                if not any(x.get('core_adaptation_graph_in_loss') and abs(x.get('weighted_core_transfer',0))>0 for x in events):
                    raise RuntimeError('Native DA was not exercised with the label budget')
            if a['matched_alignment_weight']==0:
                events=[json.loads(s) for s in (smoke_path.parent/'updates.jsonl').read_text().splitlines()]
                if any(x.get('weighted_matched_alignment',0)!=0 for x in events):
                    raise RuntimeError('Additional class alignment was not disabled')
                if not all(x.get('same_true_class_at_every_position') is True for x in events):
                    raise RuntimeError('Truth-guided pairing was not preserved')
                if not any(x.get('core_adaptation_graph_in_loss') and abs(x.get('weighted_core_transfer',0))>0 for x in events):
                    raise RuntimeError('Native adaptation was not exercised in preflight')
            run(job,False)
            from weather.summarize_class_matching import summarize
            rows=summarize([job['result_file']],Path(job['result_file']).parent.parent/'summary')
            row=rows[0]
            row['planned_epochs']=job['values']['epochs']
            completed.append(row)
            write_pair_summary(output,completed,args.seeds)
            if args.unified_fewshot:write_peak_summary(output,completed,args.seeds)
            # This is an evidence inventory, not a mean across differing budgets.
            write(output/'real_results_inventory.json',{'protocol':supervision['protocol'],'aggregation_performed':False,
                'simulated':False,'completed':len(completed),'planned':len(jobs),'rows':completed})
            print('REAL_SELECTED='+json.dumps({k:row[k] for k in ['method','target','seed','ACC_percent','selected_epoch','completed_epochs',
                'selection_metric','peak_validation_ACC_percent','heldout_ACC_percent']}),flush=True)
    except BaseException as error:
        write(output/'queue_status.json',{'status':'FAIL','completed':len(completed),'total':len(jobs),'error':str(error)})
        raise
    write(output/'queue_status.json',{'status':'PASS','completed':len(completed),'total':len(jobs)})


if __name__=='__main__':main()
