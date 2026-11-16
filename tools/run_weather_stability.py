"""Controlled backlight encoder and rainy distillation trials; no score caps."""
import argparse
import csv
import hashlib
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weather.run_qualified_backlight import training_argv
from weather.rainy_presets import arguments_for
from weather.train_vds_core import parser as training_parser
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli

METHODS=('dann','deepcoral','cdan','mdd','mcc')


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build_jobs(project_root,output,scope='both',backlight_seeds=(42,),rainy_seeds=(42,46),
               rainy_weights=(1.,.5),source_controls=False,backlight_image_lr=1e-5):
    if scope not in ('both','backlight','rainy'):raise ValueError('Unknown scope')
    for seeds in (backlight_seeds,rainy_seeds):
        if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:raise ValueError('Invalid seeds')
    if not rainy_weights or len(set(rainy_weights))!=len(rainy_weights) or any(not math.isfinite(w) or w<=0 for w in rainy_weights):
        raise ValueError('Distillation weights must be unique and positive')
    if not math.isfinite(backlight_image_lr) or backlight_image_lr<=0:
        raise ValueError('Backlight encoder learning rate must be finite and positive')
    root,output=Path(project_root),Path(output);jobs=[]
    def append(method,target,seed,variant,values,role='adaptation'):
        parent=output/target/variant/f'seed{seed}'/method
        values.update(save_dir=str(parent),configuration_tag=variant,target_class_matching='none',
                      matched_alignment_weight=0.,log_parameter_updates=True)
        jobs.append({'name':f'{target}_{method}_{variant}_seed{seed}','method':method,'target':target,
                     'seed':seed,'variant':variant,'comparison_role':role,'values':values,
                     'minimum_epochs':60 if target=='backlight' else values['early_stopping_min_epochs'],
                     'required_free_MiB':4608 if target=='backlight' else 4096,
                     'result_file':str(parent/f'{method}_{target}_seed{seed}'/'result.json')})
    if scope in ('both','backlight'):
        for seed in backlight_seeds:
            for method in ('deepcoral','mdd','mcc'):
                argv=training_argv(method,output/'lists/backlight',output,
                    root/'pretrained_weights/torch/hub/checkpoints/convnext_small-0c510722.pth',seed)
                values=vars(training_parser().parse_args(argv[4:]))
                # The old backbone LR was unused while frozen. Give the newly
                # trainable stage its own small LR, retaining every head rate.
                values.update(encoder_training_scope='last_stages',image_lr_override=backlight_image_lr)
                variant=('backlight_last_stage_v9' if backlight_image_lr==1e-5
                         else 'backlight_last_stage_lr'+format(backlight_image_lr,'g').replace('.','p')+'_v1')
                append(method,'backlight',seed,variant,dict(values))
                if source_controls:
                    control=dict(values,trade_off=0.)
                    control_variant=('backlight_last_stage_source_control' if backlight_image_lr==1e-5
                                     else variant+'_source_control')
                    append(method,'backlight',seed,control_variant,control,'source_only_control')
    if scope in ('both','rainy'):
        for seed in rainy_seeds:
            for method in METHODS:
                for weight in rainy_weights:
                    values=arguments_for(method,root,output/'lists/rainy',output,seed)
                    values['prediction_distillation_weight']=weight
                    tag='rainy_distill_'+format(weight,'g').replace('.','p')+'_v1'
                    append(method,'rainy',seed,tag,values)
    return jobs


def preflight_values(values,output):
    return dict(values,save_dir=str(output),max_train_batches=2,stop_after_epochs=2,
                adaptation_warmup_epochs=0,adaptation_ramp_epochs=0,selection_min_epoch=1,
                encoder_warmup_freeze_epochs=0,early_stopping_patience=0,holdout_list=None)


def runtime_audit(result_file,require_core_gradient=True,require_distillation=False):
    path=Path(result_file);r=json.loads(path.read_text(encoding='utf-8'))
    if (r['protocol']!='UDA' or r['target_training_labels_used'] is not False
            or r.get('test_evaluations',0) or r.get('target_test_available',False)):
        raise ValueError('Wrong training/evaluation protocol: '+str(path))
    config=json.loads((path.parent/'run_config.json').read_text(encoding='utf-8'))
    if config['arguments'].get('target_class_matching','none')!='none' or config['arguments'].get('matched_alignment_weight',0):
        raise ValueError('Instance-label guidance is prohibited in this UDA queue')
    gradients=[json.loads(s) for s in (path.parent/'gradients.jsonl').read_text().splitlines()]
    updates=[json.loads(s) for s in (path.parent/'parameter_updates.jsonl').read_text().splitlines()]
    evidence={}
    for modality in ('VIS','IR'):
        source=max(x['gradients'][modality]['CE_norm'] for x in gradients)
        native=max(x.get('enhancement_components',{}).get('core',x['gradients']).get(modality,
                   x.get('enhancement_components',{}).get('core',{}).get('RGB' if modality=='VIS' else 'NIR',{}))
                   .get('weighted_DA_norm',0.) for x in gradients)
        # Encoder update logs are a named dictionary inside each epoch event.
        samples=[]
        for record in updates:
            groups=record['sampled_parameters']
            item=groups.get(modality+'_last_trainable_matrix',{})
            if item:samples.append(item)
        changed=max((x.get('update_L2',0.) for x in samples),default=0.)
        if not source>0 or not changed>0 or (require_core_gradient and not native>0):
            raise ValueError(f'Inactive {modality}: CE={source}, core={native}, update={changed}; {path}')
        evidence[modality]={'source_gradient_max':source,'native_gradient_max':native,'sampled_update_max':changed}
    if require_distillation:
        events=[json.loads(s) for s in (path.parent/'updates.jsonl').read_text().splitlines()]
        if not any(x.get('weighted_prediction_distillation',0)>0 for x in events):
            raise ValueError('The distillation phase was never exercised')
    return {'status':'PASS','encoder_evidence':evidence,'target_training_labels_used':False,
            'test_evaluations':0,'result_file':str(path)}


def summarize(output,rows):
    for name,data in (('runs',rows),):
        with (output/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    groups={}
    for row in rows:groups.setdefault((row['target'],row['method'],row['variant'],row['comparison_role']),[]).append(row)
    means=[]
    for (target,method,variant,role),values in groups.items():
        row=dict(target=target,method=method,variant=variant,comparison_role=role,num_seeds=len(values),std_ddof=0,
                 metric_partition='target_val')
        for key in ('ACC','Precision','Recall','F1'):
            scores=[x[key] for x in values];row[key+'_mean']=statistics.mean(scores)
            row[key+'_std']=statistics.pstdev(scores) if len(scores)>1 else None
        means.append(row)
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(means[0]));writer.writeheader();writer.writerows(means)


def completed_result(job):
    """Only skip verified complete jobs, never infer completion from a log."""
    path=Path(job['result_file'])
    if not path.is_file():
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError('Partial formal output exists; use a new output directory: '+str(path.parent))
        return None
    result=json.loads(path.read_text())
    complete=json.loads((path.parent/'completeness.json').read_text())
    if (complete['status']!='PASS' or result['smoke'] or result['screening_only']
            or not job['minimum_epochs']<=result['completed_epochs']<=job['values']['epochs']
            or any(result[k]!=job[k] for k in ('method','target','seed'))):
        raise ValueError('Incomplete or wrong-identity formal result: '+str(path))
    from weather.train_vds_core import resolve_weather_settings
    expected=training_parser().parse_args(recorded_arguments_to_cli(job['values']))
    resolve_weather_settings(expected)
    actual=json.loads((path.parent/'run_config.json').read_text())['arguments']
    keys=('method','target','seed','data_dir','save_dir','weights','ir_weights','epochs','batch_size',
          'lr','image_lr_override','ir_lr_override','adversarial_lr_override','optimizer','weight_decay',
          'label_smoothing','gradient_clip_norm','encoder_training_scope','encoder_warmup_freeze_epochs',
          'trade_off','adaptation_warmup_epochs','adaptation_ramp_epochs','prediction_distillation_weight',
          'prediction_distillation_start_epoch','prediction_distillation_encoder_lr_scale','configuration_tag')
    for key in keys:
        if actual.get(key)!=getattr(expected,key):raise ValueError('Saved configuration differs: '+key+'; '+str(path))
    runtime_audit(path,job['comparison_role']=='adaptation',job['target']=='rainy')
    return result


def result_row(job,result):
    probability_hash=(result['enhancement']['prediction_distillation']['probability_values_sha256']
                      if job['target']=='rainy' else None)
    return {k:job[k] for k in ('target','method','seed','variant','comparison_role')} | {
        'selected_epoch':result['selected_epoch'],'completed_epochs':result['completed_epochs'],
        **{k:100*result['target_val'][v] for k,v in (('ACC','accuracy'),('Precision','precision_macro'),
            ('Recall','recall_macro'),('F1','f1_macro'))},'pseudo_probability_sha256':probability_hash,
        'result_file':job['result_file']}


def check_environment(scope):
    """Offline, before any queue dispatch; missing imports cannot fail late."""
    from tools.run_backlight_rainy_suite import dependency_preflight
    from tools.download_official_uda_weights import require_weight
    dependency_preflight(scope)
    for key in (('convnext_small',) if scope=='backlight' else ('convnext_small','resnet18','unip_tiny')):
        require_weight(key,ROOT/'pretrained_weights')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=('both','backlight','rainy'),default='both')
    p.add_argument('--backlight-seeds',nargs='+',type=int,default=[42])
    p.add_argument('--rainy-seeds',nargs='+',type=int,default=[42,46])
    p.add_argument('--rainy-weights',nargs='+',type=float,default=[1.,.5])
    p.add_argument('--backlight-image-lr',type=float,default=1e-5,
                   help='Explicit encoder LR; does not change the method classifier/adversary rates')
    p.add_argument('--source-controls',action='store_true')
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('JMDA_DATA_ROOT',r'D:\Code\JMDA-Net\Data')))
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',type=Path,help='Reuse the exact stored plan and verified complete trials')
    p.add_argument('--local',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--check-environment',action='store_true',help='Verify offline weights/imports/CUDA, then exit without creating a run')
    args=p.parse_args()
    if args.resume and args.output:p.error('--resume and --output cannot be combined')
    if args.resume and any(option in sys.argv for option in ('--scope','--backlight-seeds','--rainy-seeds','--rainy-weights','--backlight-image-lr','--source-controls','--data-root')):
        p.error('Resume uses the stored plan; do not supply new trial/data overrides')
    if args.check_environment:
        if args.resume:p.error('--check-environment cannot be combined with --resume')
        build_jobs(ROOT,ROOT/'runs/weather_stability/_environment_check',args.scope,args.backlight_seeds,
                   args.rainy_seeds,args.rainy_weights,args.source_controls,args.backlight_image_lr)
        if not args.data_root.is_dir():raise FileNotFoundError(args.data_root)
        check_environment(args.scope)
        print('[PREFLIGHT PASS] imports, CUDA, weights and data root; no training or run creation',flush=True)
        return
    output=(args.resume or args.output or ROOT/'runs/weather_stability'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if not output.is_relative_to(ROOT/'runs/weather_stability'):p.error('Use runs/weather_stability for separate evidence')
    attempt='resume_'+datetime.now().strftime('%Y%m%d_%H%M%S') if args.resume else None
    if args.resume:
        plan=json.loads((output/'plan.json').read_text());jobs=plan['jobs'];snapshot=Path(plan['snapshot'])
        args.data_root=Path(plan['data_root'])
        from tools.prepare_mcc_weather_v2_snapshot import MANIFEST,_verified
        manifest=json.loads((snapshot/MANIFEST).read_text())
        if not _verified(snapshot,manifest['files']):raise ValueError('Stored immutable runtime snapshot changed')
        if any(not Path(j['result_file']).resolve().is_relative_to(output) for j in jobs):
            raise ValueError('Stored result path escaped this experiment directory')
    else:
        if output.exists():raise FileExistsError(output)
        jobs=build_jobs(ROOT,output,args.scope,args.backlight_seeds,args.rainy_seeds,args.rainy_weights,
                        args.source_controls,args.backlight_image_lr)
        from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot
        snapshot=prepare_snapshot(ROOT)
        output.mkdir(parents=True)
        write(output/'plan.json',{'protocol':'UDA','target_training_labels_used':False,'accuracy_caps_applied':False,
              'evaluation':'same-epoch peak target_val ACC and macro F1; no independent weather test',
              'data_root':str(args.data_root),'snapshot':str(snapshot),'jobs':jobs})
    print(f'UDA stability trials={len(jobs)}; output={output}',flush=True)
    if not args.execute:print('PLAN_ONLY',flush=True);return
    check_environment('backlight' if all(j['target']=='backlight' for j in jobs) else 'both')
    env=dict(os.environ,JMDA_DATA_ROOT=str(args.data_root.resolve()),PYTHONUNBUFFERED='1',
             JMDA_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_HOME=str(ROOT/'pretrained_weights/torch'))
    for target in dict.fromkeys(j['target'] for j in jobs):
        if args.resume:continue
        subprocess.run([sys.executable,'-B','-u',str(snapshot/'tools/prepare_paired_protocol_lists.py'),
            '--data-root',str(args.data_root),'--output-dir',str(output/'lists'/target),
            '--targets',target,'--modality','paired'],env=env,cwd=ROOT,check=True)
    rows=[];probability_hashes={};reused=set()
    def remember_probability(job,result):
        if job['target']!='rainy':return
        value=result['enhancement']['prediction_distillation']['probability_values_sha256'];key=(job['method'],job['seed'])
        if key in probability_hashes and probability_hashes[key]!=value:
            raise ValueError('Distillation-weight controls produced different bootstrap pseudo-probabilities')
        probability_hashes[key]=value
    if args.resume:
        for job in jobs:
            result=completed_result(job)
            if result is not None:
                remember_probability(job,result);rows.append(result_row(job,result));reused.add(job['name'])
                print('REUSED_COMPLETED='+job['name'],flush=True)
        write(output/(attempt+'.json'),{'reused_completed':sorted(reused),'pending':len(jobs)-len(reused),
              'formal_plan_unchanged':True,'student_runtime_snapshot':str(snapshot),
              'queue_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'preflight_encoder_warmup_freeze_epochs':0})
        if rows:summarize(output,rows)
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
                '--max-util-percent',os.environ.get('JMDA_GPU_MAX_UTIL','60'),'--stable-checks','3',
                '--poll-seconds','10','--tag',label,'--',*command]
        print('START='+label,flush=True)
        with (output/(label+'.log')).open('x',encoding='utf-8') as f:
            subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        for job in jobs:
            if job['name'] in reused:continue
            write(output/'queue_status.json',{'status':'RUNNING','active':job['name'],'completed':len(rows),'total':len(jobs)})
            from contextlib import nullcontext
            if args.local:
                from weather.local_capacity import local_training_lease
                lease=local_training_lease(ROOT/'runs/weather_local_screen/.local_training.lock',timeout_seconds=120)
            else:lease=nullcontext()
            with lease:
                probe=preflight_values(job['values'],output/('preflight_'+attempt if attempt else 'preflight')/job['name'])
                worker(job,probe,job['name']+'_preflight'+('_'+attempt if attempt else ''))
                probe_file=Path(probe['save_dir'])/f"{job['method']}_{job['target']}_seed{job['seed']}"/'result.json'
                evidence=runtime_audit(probe_file,job['comparison_role']=='adaptation',job['target']=='rainy')
                write(output/(job['name']+'_preflight_audit.json'),evidence)
                worker(job,job['values'],job['name'])
                path=Path(job['result_file']);r=json.loads(path.read_text())
                c=json.loads((path.parent/'completeness.json').read_text())
                if c['status']!='PASS' or r['smoke'] or r['screening_only'] or not job['minimum_epochs']<=r['completed_epochs']<=job['values']['epochs']:
                    raise ValueError('Incomplete formal trial: '+str(path))
                evidence=runtime_audit(path,job['comparison_role']=='adaptation',job['target']=='rainy')
                write(path.parent/'stability_runtime_audit.json',evidence)
                remember_probability(job,r)
            rows.append(result_row(job,r))
            summarize(output,rows);print('COMPLETED='+job['name'],flush=True)
    except BaseException as error:
        write(output/'queue_status.json',{'status':'FAIL','completed':len(rows),'total':len(jobs),'error':str(error)})
        raise
    write(output/'queue_status.json',{'status':'PASS','completed':len(rows),'total':len(jobs)})


if __name__=='__main__':main()
