"""Independent single-setting server tests of the completed dark CORAL run."""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_label_guided_matrix import preflight_command
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli

DEFAULT_BASELINE=ROOT/'runs/weather_label_guided/server_fewshot_v2_20261118_150154_80507'
DEFAULT_TRIALS=('phase_lr0p3','no_source_crossmodal','ir_prior0p3','coral0p1')
TRIALS={
    'phase_lr0p3':{'image_adaptation_lr_scale':.3},
    'no_source_crossmodal':{'source_crossmodal_weight':0.},
    'ir_prior0p3':{'initial_vis_weight':.3},
    'coral0p1':{'trade_off':.1},
    'phase_lr1':{'image_adaptation_lr_scale':1.},
}


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def build_trials(arguments,output,snapshot,trials=DEFAULT_TRIALS):
    expected={'method':'deepcoral','target':'dark','seed':46,'epochs':40,'target_labels_per_class':3,
        'matched_alignment_weight':.2,'source_anchor_ema_momentum':.9,'image_adaptation_lr_scale':.1,
        'source_crossmodal_weight':.1,'initial_vis_weight':.4,'trade_off':.05,'early_stopping_patience':0,
        'classifier_layout':'reference','prediction_distillation_weight':0.,'prediction_teacher_weight':0.}
    if any(arguments.get(k)!=v for k,v in expected.items()) or arguments.get('holdout_list') is not None:
        raise ValueError('Expected the completed server-v2 dark Deep CORAL seed46 baseline')
    if not trials or len(set(trials))!=len(trials) or set(trials)-set(TRIALS):raise ValueError('Unknown/repeated trial')
    jobs=[]
    for trial in trials:
        values=copy.deepcopy(arguments)
        values.pop('steps_per_epoch',None)  # Recomputed from the same locked cohort by training.
        old=copy.deepcopy(values)
        values.update(TRIALS[trial],configuration_tag='deepcoral_dark_tuning_v1_'+trial,
                      save_dir=str(Path(output)/trial/'seed46'))
        changed={k for k in set(old)|set(values) if old.get(k)!=values.get(k)}
        if changed!=set(TRIALS[trial])|{'save_dir','configuration_tag'}:
            raise ValueError('Unexpected change in a single-setting trial')
        jobs.append({'trial':trial,'values':values,'required_free_MiB':6144,
            'changes':{k:{'previous':old.get(k),'candidate':values.get(k)} for k in sorted(changed)},
            'result_file':str(Path(values['save_dir'])/'deepcoral_dark_seed46/result.json'),
            'command':[sys.executable,'-B','-u',str(Path(snapshot)/'weather/train_vds_core.py'),
                       *recorded_arguments_to_cli(values)]})
    return jobs


def validate_cohort(baseline,result):
    if (result.get('protocol')!='FEWSHOT_TARGET_CLASS_ALIGNMENT' or result.get('test_evaluations',0) or
            result.get('target_true_labels_used_in_classification_loss') is not False):
        raise ValueError('Trial changed supervision or inspected held-out data')
    before=baseline['label_budget'];after=result['label_guided_sampling']
    for key in ('labels_per_class','labelled_target_samples','unlabelled_target_samples',
                'labelled_target_class_counts','label_pool_portable_sha256'):
        if before[key]!=after[key]:raise ValueError('Trial changed the fixed target label pool: '+key)
    if after.get('unselected_truth_used_for_pairing_or_alignment'):
        raise ValueError('Unselected target truth was exposed')


def validate_locked_inputs(baseline,data_dir):
    required=('sunny_train.txt','dark_train.txt','dark_val.txt','dark_train_pairs.tsv')
    hashes={Path(name).name:value for name,value in baseline['code_provenance']['explicit_input_sha256'].items()}
    for name in required:
        if name not in hashes:raise ValueError('Baseline lacks the locked input hash: '+name)
        path=Path(data_dir)/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[name]:
            raise ValueError('Trial input differs from the completed baseline: '+name)


def comparison_row(record,trial):
    peak=record['peak_target_val'];ce=record['CE_selected_target_val']
    return {'trial':trial,'seed':record['seed'],'peak_epoch':peak['epoch'],
        **{name:100*peak['metrics'][key] for name,key in
            [('ACC','accuracy'),('Precision','precision_macro'),('Recall','recall_macro'),('F1','f1_macro')]},
        'CE_selected_epoch':ce['epoch'],'CE_selected_ACC':100*ce['metrics']['accuracy'],
        'last_ACC':100*record['last_epoch_evaluation']['target_val']['metrics']['accuracy']}


def save_comparison(output,rows):
    with (output/'trial_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output/'real_results_inventory.json',{'completed_new_trials':len(rows)-1,'baseline_reused':True,
        'simulated':False,'single_seed_parameter_trials_not_independent_replicates':True,'rows':rows})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    p.add_argument('--data-dir',type=Path,help='Optional relocation of the same locked cohort')
    p.add_argument('--trials',nargs='+',choices=tuple(TRIALS),default=list(DEFAULT_TRIALS))
    p.add_argument('--output',type=Path)
    p.add_argument('--gpus',default=os.environ.get('JMDA_GPU_CANDIDATES','0,1'))
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--check-environment',action='store_true')
    args=p.parse_args()
    records=[json.loads(s) for s in (args.baseline/'run_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
    matches=[r for r in records if r['method']=='deepcoral' and r['target']=='dark' and r['seed']==46]
    if len(matches)!=1:raise ValueError('Baseline must contain exactly one dark Deep CORAL seed46 run')
    baseline=matches[0];original=copy.deepcopy(baseline['arguments'])
    if args.data_dir:original['data_dir']=str(args.data_dir.resolve())
    output=(args.output or ROOT/'runs/weather_label_guided'/('server_fewshot_deepcoral_dark_tuning_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if not output.is_relative_to(ROOT/'runs/weather_label_guided'):p.error('Use the separate weather_label_guided output folder')
    jobs=build_trials(original,output,ROOT,args.trials)
    if args.check_environment or args.execute:
        if not Path(original['data_dir']).is_dir():raise FileNotFoundError('Missing original locked cohort: '+original['data_dir'])
        if not (Path(original['data_dir'])/'holdout_protocol_manifest.json').is_file():raise FileNotFoundError('Missing locked split manifest')
        validate_locked_inputs(baseline,original['data_dir'])
        from tools.run_weather_fewshot_suite import check_environment
        check_environment({'jobs':jobs},Path(os.environ.get('JMDA_DATA_ROOT','/mnt/data/lixiang/data/Data')))
    if args.check_environment:return
    if output.exists():raise FileExistsError(output)
    if args.execute:
        from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot
        snapshot=prepare_snapshot(ROOT);jobs=build_trials(original,output,snapshot,args.trials)
    else:snapshot=ROOT
    output.mkdir(parents=True)
    write(output/'plan.json',{'baseline':str(args.baseline.resolve()),'baseline_peak_ACC':100*baseline['peak_target_val']['metrics']['accuracy'],
        'protocol':baseline['protocol'],'seed':46,'epochs':40,'original_core_loss_formula_preserved':True,
        'no_new_target_labels':True,'heldout_evaluated':False,'jobs':jobs})
    print('Run directory='+str(output),flush=True)
    if not args.execute:print('PLAN_ONLY; no training started',flush=True);return
    env=dict(os.environ,PYTHONUNBUFFERED='1',JMDA_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             TORCH_HOME=str(ROOT/'pretrained_weights/torch'))
    rows=[comparison_row(baseline,'baseline_v2')];save_comparison(output,rows)
    completed=0
    try:
        for job in jobs:
            for smoke in (True,False):
                stage='preflight' if smoke else 'formal'
                write(output/'queue_status.json',{'status':'RUNNING','active':job['trial'],'phase':stage,'completed':completed,'total':len(jobs)})
                command=preflight_command(job['command'],output/'preflight'/job['trial']) if smoke else job['command']
                scheduled=[sys.executable,'-B','-u',str(snapshot/'tools/run_on_available_gpu.py'),
                    '--min-free-mib','6144','--max-util-percent',os.environ.get('JMDA_GPU_MAX_UTIL','60'),
                    '--gpus',args.gpus,'--stable-checks','3','--poll-seconds','10',
                    '--tag','deepcoral_dark_'+job['trial']+'_'+stage,'--',*command]
                started=time.perf_counter();start_time=datetime.now().astimezone().isoformat()
                print('START='+job['trial']+'_'+stage,flush=True)
                with (output/(job['trial']+'_'+stage+'.log')).open('x',encoding='utf-8') as log:
                    code=subprocess.run(scheduled,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if code:raise RuntimeError(job['trial']+' '+stage+' exit='+str(code))
                with (output/'timings.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps({'trial':job['trial'],'phase':stage,'start_time':start_time,
                        'end_time':datetime.now().astimezone().isoformat(),'wall_s_including_gpu_wait':time.perf_counter()-started})+'\n')
                path=(output/'preflight'/job['trial']/'deepcoral_dark_seed46/result.json') if smoke else Path(job['result_file'])
                result=json.loads(path.read_text(encoding='utf-8'));validate_cohort(baseline,result)
                if smoke:
                    from weather.summarize_class_matching import validate_fewshot_updates
                    events=[json.loads(s) for s in (path.parent/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
                    validate_fewshot_updates(result['label_guided_sampling'],events,.9)
                    if not any(e['weighted_core_transfer']>0 for e in events):raise ValueError('CORAL did not run in preflight')
                    if not any(e['weighted_matched_alignment']>0 for e in events):raise ValueError('Few-shot alignment did not run in preflight')
                else:
                    if result['completed_epochs']!=40 or result.get('smoke') or result.get('screening_only'):
                        raise ValueError('Incomplete candidate must not enter the comparison')
                    from weather.summarize_class_matching import summarize
                    summarize([path],path.parent.parent/'summary')
                    from weather.compact_diagnostics import export_run_diagnostics
                    record=export_run_diagnostics(output,path,run_namespace=job['trial'])
                    if any(record['arguments'].get(k)!=v for k,v in job['values'].items()):
                        raise ValueError('Actual training arguments differ from the declared single-setting plan')
                    rows.append(comparison_row(record,job['trial']));save_comparison(output,rows)
                    completed+=1
                    print('RESULT='+json.dumps(rows[-1]),flush=True)
    except BaseException as error:
        write(output/'queue_status.json',{'status':'FAIL','completed':completed,'total':len(jobs),'error':str(error)})
        raise
    write(output/'queue_status.json',{'status':'PASS','completed':completed,'total':len(jobs)})


if __name__=='__main__':main()
