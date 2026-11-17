"""One weather queue with explicitly separated UDA and label-guided protocols."""
import argparse
import csv
from datetime import datetime
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_weather_stability import (build_jobs as uda_jobs,preflight_values,runtime_audit,
                                        completed_result as completed_uda,check_environment)
from tools.run_label_guided_matrix import build_jobs as guided_jobs
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from weather.class_matching import ALIGNMENT_PROTOCOL

METHODS=('dann','deepcoral','cdan','mdd','mcc')
METRICS={'ACC':'accuracy','Precision':'precision_macro','Recall':'recall_macro','F1':'f1_macro'}


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build_plan(output,snapshot,seeds=tuple(range(42,47)),rainy_weights=(1.,.5),backlight_lr=1e-4,
               targets=('backlight','rainy','dark','foggy')):
    targets=tuple(targets)
    if not targets or len(set(targets))!=len(targets) or not set(targets)<={'backlight','rainy','dark','foggy'}:
        raise ValueError('Unknown or repeated weather targets')
    output=Path(output)
    jobs=uda_jobs(ROOT,output,'both',seeds,seeds,rainy_weights,False,backlight_lr,('cdan','mcc'))
    jobs=[j for j in jobs if j['target'] in targets]
    for job in jobs:job.update(protocol='UDA',target_training_labels_used=False)
    pairs=[f'{m}_{target}' for target in ('dark','foggy') if target in targets for m in METHODS]
    labelled=guided_jobs(output/'label_guided',output/'lists/guided',pairs,seeds,Path(snapshot),True,
                         evaluation_mode='validation_ce',parameter_profile='reference')
    for item in labelled:
        values=item['values'];values['log_parameter_updates']=True
        jobs.append({'name':'guided_'+item['name'],'method':values['method'],'target':values['target'],
            'seed':values['seed'],'variant':'label_guided_reference','comparison_role':'label_guided_adaptation',
            'protocol':ALIGNMENT_PROTOCOL,'target_training_labels_used':True,'values':values,
            'minimum_epochs':values['early_stopping_min_epochs'],
            'required_free_MiB':item['required_free_MiB'],'result_file':item['result_file']})
    return {'scope':'all_four_weathers' if len(targets)==4 else 'selected_weathers','targets':list(targets),
            'seeds':list(seeds),'rainy_weights':list(rainy_weights),
            'backlight_methods':['cdan','mcc'],'other_weather_methods':list(METHODS),
            'protocols':{t:('UDA' if t in ('backlight','rainy') else ALIGNMENT_PROTOCOL) for t in targets},
            'score_caps_applied':False,'snapshot':str(snapshot),'jobs':jobs,'total':len(jobs),
            'counts_by_weather':{t:sum(j['target']==t for j in jobs) for t in targets}}


def check_all(data_root,targets=('backlight','rainy','dark','foggy')):
    if not Path(data_root).is_dir():raise FileNotFoundError(data_root)
    check_environment('both')
    from tools.download_official_uda_weights import require_weight
    if 'foggy' in targets:require_weight('unip_small',ROOT/'pretrained_weights')


def validate_formal(job):
    path=Path(job['result_file'])
    if job['protocol']=='UDA':return completed_uda(job)
    if not path.is_file():
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError('Partial formal output exists: '+str(path.parent))
        return None
    r=json.loads(path.read_text());c=json.loads((path.parent/'completeness.json').read_text())
    if (c['status']!='PASS' or r['smoke'] or r['screening_only']
            or r['target_training_labels_used'] is not True or r['protocol']!=ALIGNMENT_PROTOCOL
            or r.get('target_test_available') or r.get('test_evaluations',0)
            or not job['minimum_epochs']<=r['completed_epochs']<=job['values']['epochs']
            or any(r[k]!=job[k] for k in ('method','target','seed'))):
        raise ValueError('Invalid label-guided completed run: '+str(path))
    actual=json.loads((path.parent/'run_config.json').read_text())['arguments']
    for key in ('lr','image_lr_override','ir_lr_override','weight_decay','label_smoothing','gradient_clip_norm',
                'encoder_training_scope','matched_alignment_weight','matched_alignment_ramp_epochs',
                'target_class_matching','selection_metric','early_stopping_patience','data_dir','save_dir'):
        if actual.get(key)!=job['values'].get(key):raise ValueError('Guided saved parameter changed: '+key)
    from weather.summarize_class_matching import summarize
    summarize([path],path.parent/'checked_summary')
    return r


def validate_probe(job,path):
    if job['protocol']=='UDA':return runtime_audit(path,True,job['target']=='rainy')
    r=json.loads(Path(path).read_text())
    if (r['protocol']!=ALIGNMENT_PROTOCOL or r['target_training_labels_used'] is not True
            or r.get('target_true_labels_used_in_alignment_loss') is not True
            or r.get('target_true_labels_used_in_classification_loss') is not False
            or r.get('target_test_available') or r.get('test_evaluations',0)):
        raise ValueError('Wrong label-guided probe protocol')
    updates=[json.loads(s) for s in (Path(path).parent/'updates.jsonl').read_text().splitlines()]
    if not updates or not all(x.get('same_true_class_at_every_position') is True for x in updates):
        raise ValueError('Probe class matching failed')
    if not any(x.get('weighted_matched_alignment',0)>0 for x in updates):
        raise ValueError('Probe label-guided alignment was not active')
    return {'status':'PASS','protocol':ALIGNMENT_PROTOCOL,'target_training_labels_used':True,'test_evaluations':0}


def make_row(job,result):
    return {k:job[k] for k in ('target','method','seed','variant','protocol','target_training_labels_used')} | {
        'selection_metric':result.get('selection_metric','accuracy_f1'),'metric_partition':'target_val',
        'selected_epoch':result['selected_epoch'],'completed_epochs':result['completed_epochs'],
        **{k:100*result['target_val'][v] for k,v in METRICS.items()},
        'peak_validation_ACC':100*result.get('peak_target_val',result['target_val'])['accuracy'],
        'pseudo_probability_sha256':result.get('enhancement',{}).get('prediction_distillation',{}).get('probability_values_sha256'),
        'result_file':job['result_file']}


def make_diagnostic(job,result):
    root=Path(job['result_file']).parent
    selected=json.loads((root/'selected_target_val_predictions.json').read_text())
    selected=sorted(selected,key=lambda x:x['sample_id'])
    predictions=[{k:x[k] for k in ('sample_id','y_true','y_pred','probabilities')} for x in selected]
    classes=[(x['sample_id'],x['y_pred']) for x in predictions]
    probabilities=[(x['sample_id'],x['probabilities']) for x in predictions]
    gradients=[json.loads(s) for s in (root/'gradients.jsonl').read_text().splitlines()]
    updates=[json.loads(s) for s in (root/'parameter_updates.jsonl').read_text().splitlines()]
    g=next(x for x in gradients if x['epoch']==result['selected_epoch'])
    u=next(x for x in updates if x['epoch']==result['selected_epoch'])
    config=json.loads((root/'run_config.json').read_text())
    training=([json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
              if (root/'training.jsonl').is_file() else [])
    evaluations=([json.loads(s) for s in (root/'evaluation.jsonl').read_text().splitlines()]
                 if (root/'evaluation.jsonl').is_file() else [])
    return {k:job[k] for k in ('method','target','seed','variant','protocol')} | {
        'selected_epoch':result['selected_epoch'],'configuration':config['arguments'],
        'enhancement':result.get('enhancement',{}),
        'complexity':result.get('complexity',{}),'dataset_audit':config.get('dataset_audit',{}),
        'selected_training':next((x for x in training if x['epoch']==result['selected_epoch']),None),
        'selected_evaluations':[x for x in evaluations if x['epoch']==result['selected_epoch']],
        'metrics':result['target_val'],'modality_weights':result['modality_weights_at_selected_epoch'],
        'predicted_class_sha256':hashlib.sha256(json.dumps(classes).encode()).hexdigest(),
        'probability_sha256':hashlib.sha256(json.dumps(probabilities).encode()).hexdigest(),
        'selected_gradients':g,'selected_parameter_updates':u,'selected_predictions':predictions,
        'graph_activity_is_not_proof_of_performance_gain':True}


def write_prediction_comparisons(output,diagnostics):
    groups={};comparisons=[]
    for record in diagnostics:groups.setdefault((record['protocol'],record['target'],record['seed']),[]).append(record)
    for (protocol,target,seed),group in groups.items():
        for a,b in combinations(group,2):
            x,y=a['selected_predictions'],b['selected_predictions']
            if [(p['sample_id'],p['y_true']) for p in x]!=[(p['sample_id'],p['y_true']) for p in y]:
                raise ValueError('Validation cohort or class mapping differs across methods')
            changed=sum(p['y_pred']!=q['y_pred'] for p,q in zip(x,y))
            delta=max(abs(v-w) for p,q in zip(x,y) for v,w in zip(p['probabilities'],q['probabilities']))
            comparisons.append({'protocol':protocol,'target':target,'seed':seed,'method_a':a['method'],
                'variant_a':a['variant'],'method_b':b['method'],'variant_b':b['variant'],
                'same_class_predictions':changed==0,'changed_class_predictions':changed,
                'same_probabilities':delta==0,'max_probability_difference':delta,
                'all_four_metrics_equal':all(a['metrics'][m]==b['metrics'][m] for m in METRICS.values())})
    if comparisons:
        with (Path(output)/'prediction_comparisons.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    return comparisons


def summarize(output,rows,diagnostics,jobs):
    if not rows:return
    output=Path(output)
    with (output/'runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={};expected={}
    def key(x):return (x['protocol'],x['target'],x['method'],x['variant'])
    for job in jobs:expected.setdefault(key(job),set()).add(job['seed'])
    for row in rows:groups.setdefault(key(row),[]).append(row)
    means=[]
    for (protocol,target,method,variant),group in groups.items():
        if len({x['seed'] for x in group})!=len(group):raise ValueError('Repeated seed in summary')
        if len({x['selection_metric'] for x in group})!=1:raise ValueError('Mixed selection rules')
        row=dict(protocol=protocol,target=target,method=method,variant=variant,
            target_training_labels_used=group[0]['target_training_labels_used'],
            selection_metric=group[0]['selection_metric'],metric_partition='target_val',
            seeds=' '.join(str(n) for n in sorted(x['seed'] for x in group)),num_seeds=len(group),
            expected_seeds=len(expected[(protocol,target,method,variant)]),
            all_requested_seeds_completed={x['seed'] for x in group}==expected[(protocol,target,method,variant)],std_ddof=0)
        for metric in METRICS:
            scores=[x[metric] for x in group];row[metric+'_mean']=statistics.mean(scores)
            row[metric+'_std']=statistics.pstdev(scores) if len(scores)>1 else None
        means.append(row)
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(means[0]));writer.writeheader();writer.writerows(means)
    with (output/'diagnostics.jsonl').open('w',encoding='utf-8') as f:
        for d in diagnostics:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    write_prediction_comparisons(output,diagnostics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,47)))
    p.add_argument('--targets',nargs='+',choices=('backlight','rainy','dark','foggy'),default=None)
    p.add_argument('--local',action='store_true',help='Local backlight/foggy allocation with shared GPU lease and memory guards')
    p.add_argument('--rainy-weights',nargs='+',type=float,default=[1.,.5])
    p.add_argument('--backlight-image-lr',type=float,default=1e-4)
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('JMDA_DATA_ROOT',r'D:\Code\JMDA-Net\Data')))
    p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--check-environment',action='store_true')
    p.add_argument('--export-diagnostics-from',type=Path,help='Export selected predictions/gradients from an existing server queue; no training')
    args=p.parse_args()
    if args.export_diagnostics_from:
        folder=args.export_diagnostics_from.resolve()
        if not folder.is_relative_to(ROOT/'runs'):p.error('Export only a project run directory')
        old=json.loads((folder/'plan.json').read_text());records=[]
        for item in old['jobs']:
            file=Path(item['result_file']).resolve()
            if not file.is_relative_to(folder):raise ValueError('Run this export where the original result files exist')
            result=json.loads(file.read_text());item=dict(item,protocol=result['protocol'])
            records.append(make_diagnostic(item,result))
        with (folder/'diagnostics.jsonl').open('w',encoding='utf-8') as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        write_prediction_comparisons(folder,records)
        print('EXPORTED='+str(len(records))+'; '+str(folder/'diagnostics.jsonl'),flush=True);return
    args.targets=args.targets or (['backlight','foggy'] if args.local else ['rainy','dark'])
    allocation={'backlight','foggy'} if args.local else {'rainy','dark'}
    if (args.execute or args.check_environment) and not args.resume and not set(args.targets)<=allocation:
        p.error('Local allocation is backlight/foggy; server allocation is rainy/dark')
    if args.check_environment:
        check_all(args.data_root,args.targets);print('[PREFLIGHT PASS] selected weather dependencies and weights',flush=True);return
    if args.resume and args.output:p.error('Use either --resume or --output')
    if args.resume and any(x in sys.argv for x in ('--targets','--seeds','--rainy-weights','--backlight-image-lr','--data-root')):
        p.error('Resume uses the saved plan; no parameter overrides')
    output=(args.resume or args.output or ROOT/'runs/weather_all'/(('local_' if args.local else 'server_')+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if not output.is_relative_to(ROOT/'runs/weather_all'):p.error('Use runs/weather_all for this separate queue')
    from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot,MANIFEST,_verified
    if args.resume:
        plan=json.loads((output/'plan.json').read_text());snapshot=Path(plan['snapshot']);args.data_root=Path(plan['data_root'])
        args.targets=plan.get('targets',list(plan['counts_by_weather']))
        if args.execute and not set(args.targets)<=allocation:
            p.error('Stored plan crosses the new machine allocation; start a new split queue instead')
        manifest=json.loads((snapshot/MANIFEST).read_text())
        if not _verified(snapshot,manifest['files']):raise ValueError('Stored model snapshot changed')
    else:
        if output.exists():raise FileExistsError(output)
        snapshot=prepare_snapshot(ROOT);plan=build_plan(output,snapshot,args.seeds,args.rainy_weights,args.backlight_image_lr,args.targets)
        plan['execution_location']='local' if args.local else 'server'
        plan['data_root']=str(args.data_root.resolve());output.mkdir(parents=True);write(output/'plan.json',plan)
    jobs=plan['jobs']
    if any(not Path(j['result_file']).resolve().is_relative_to(output) for j in jobs):raise ValueError('Result path escaped this queue')
    print('Weather scope='+json.dumps(plan['counts_by_weather'])+'; total='+str(len(jobs))+'; output='+str(output),flush=True)
    if not args.execute:print('PLAN_ONLY',flush=True);return
    env=dict(os.environ,JMDA_DATA_ROOT=str(args.data_root.resolve()),PYTHONUNBUFFERED='1',
             JMDA_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_HOME=str(ROOT/'pretrained_weights/torch'))
    rows=[];diagnostics=[];reused=set();probability_hashes={}
    attempt='resume_'+datetime.now().strftime('%Y%m%d_%H%M%S') if args.resume else 'initial'
    def remember_cache(job,result):
        if job['target']!='rainy':return
        value=result['enhancement']['prediction_distillation']['probability_values_sha256'];key=(job['method'],job['seed'])
        if key in probability_hashes and probability_hashes[key]!=value:
            raise ValueError('Different teacher bootstrap probabilities across distillation-weight controls')
        probability_hashes[key]=value
    def progress(status,job=None,phase=None,error=None):
        counts={t:sum(r['target']==t for r in rows) for t in plan['counts_by_weather']}
        return {'status':status,'active':job['name'] if job else None,'protocol':job['protocol'] if job else None,
                'phase':phase,'completed':len(rows),'total':len(jobs),'completed_by_weather':counts,
                'expected_by_weather':plan['counts_by_weather'],'error':error}
    def worker(job,values,label):
        command=[sys.executable,'-B','-u',str(snapshot/'weather/train_vds_core.py'),*recorded_arguments_to_cli(values)]
        if args.local:
            import psutil
            from weather.local_capacity import commit_free_bytes
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            commit=commit_free_bytes()
            if free<job['required_free_MiB'] or psutil.virtual_memory().available<6*1024**3 or commit is not None and commit<6*1024**3:
                raise RuntimeError('Insufficient local memory headroom; job not started')
        else:
            command=[sys.executable,'-B','-u',str(snapshot/'tools/run_on_available_gpu.py'),
                '--gpus',os.environ.get('JMDA_GPU_CANDIDATES','0,1'),'--min-free-mib',str(job['required_free_MiB']),
                '--max-util-percent',os.environ.get('JMDA_GPU_MAX_UTIL','60'),'--stable-checks','3','--poll-seconds','10',
                '--tag',label,'--',*command]
        print('START='+label,flush=True)
        with (output/(label+'.log')).open('x',encoding='utf-8') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        write(output/'queue_status.json',progress('PREPARING',phase='environment_and_data'))
        check_all(args.data_root,args.targets)
        if not args.resume:
            for target in ('backlight','rainy'):
                if target not in args.targets:continue
                subprocess.run([sys.executable,'-B',str(snapshot/'tools/prepare_paired_protocol_lists.py'),
                    '--data-root',str(args.data_root),'--output-dir',str(output/'lists'/target),
                    '--targets',target,'--modality','paired'],env=env,cwd=ROOT,check=True)
            guided_targets=[t for t in ('dark','foggy') if t in args.targets]
            if guided_targets:
                subprocess.run([sys.executable,'-B',str(snapshot/'tools/prepare_paired_protocol_lists.py'),
                    '--data-root',str(args.data_root),'--output-dir',str(output/'lists/guided_base'),
                    '--targets',*guided_targets,'--modality','paired'],env=env,cwd=ROOT,check=True)
                subprocess.run([sys.executable,'-B',str(snapshot/'tools/prepare_weather_generalization_split.py'),
                    '--input-dir',str(output/'lists/guided_base'),'--output-dir',str(output/'lists/guided'),
                    '--targets',*guided_targets,'--test-fraction','.2','--split-seed','20261115'],env=env,cwd=ROOT,check=True)
        else:
            for job in jobs:
                result=validate_formal(job)
                if result is not None:
                    remember_cache(job,result)
                    rows.append(make_row(job,result));diagnostics.append(make_diagnostic(job,result));reused.add(job['name'])
                    print('REUSED_COMPLETED='+job['name'],flush=True)
            summarize(output,rows,diagnostics,jobs)
        for job in jobs:
            if job['name'] in reused:continue
            write(output/'queue_status.json',progress('RUNNING',job,'preflight'))
            from contextlib import nullcontext
            if args.local:
                from weather.local_capacity import local_training_lease
                lease=local_training_lease(ROOT/'runs/weather_local_screen/.local_training.lock',timeout_seconds=120)
            else:lease=nullcontext()
            with lease:
                probe=preflight_values(job['values'],output/'preflight'/attempt/job['name'])
                worker(job,probe,job['name']+'_probe_'+attempt)
                probe_file=Path(probe['save_dir'])/f"{job['method']}_{job['target']}_seed{job['seed']}"/'result.json'
                write(output/(job['name']+'_probe_audit_'+attempt+'.json'),validate_probe(job,probe_file))
                write(output/'queue_status.json',progress('RUNNING',job,'formal_training'))
                worker(job,job['values'],job['name'])
            result=validate_formal(job)
            if result is None:raise ValueError('Training returned without complete result')
            remember_cache(job,result)
            rows.append(make_row(job,result));diagnostics.append(make_diagnostic(job,result))
            summarize(output,rows,diagnostics,jobs);print('COMPLETED='+job['name'],flush=True)
    except BaseException as error:
        write(output/'queue_status.json',progress('FAIL',error=str(error)));raise
    write(output/'queue_status.json',progress('PASS',phase='complete'))


if __name__=='__main__':main()
