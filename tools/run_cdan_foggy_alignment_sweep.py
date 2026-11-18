"""Change only the few-shot class-alignment coefficient of a completed CDAN run."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_label_guided_matrix import build_jobs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build_variant(baseline,output,seed,weight):
    plan=read(Path(baseline)/'plan.json')
    old=next(j for j in plan['jobs'] if j['name']==f'cdan_foggy_seed{seed}')
    original=old['values']
    if (not plan.get('unified_fewshot') or original['matched_alignment_weight']!=.1 or
            original['target_labels_per_class']!=3 or original['epochs']!=40):
        raise ValueError('Use the completed .1/3-shot/40-epoch CDAN baseline')
    job=build_jobs(Path(output),Path(original['data_dir']),['cdan_foggy'],[seed],ROOT,True,
        overrides={'matched_alignment_weight':weight},evaluation_mode='validation_ce',
        unified_fewshot=True,fewshot_alignment_variant=True)[0]
    changed={k for k in set(original)|set(job['values']) if original.get(k)!=job['values'].get(k)}
    if changed!={'matched_alignment_weight','configuration_tag','save_dir'}:
        raise ValueError(f'Unexpected changes relative to the real baseline: {sorted(changed)}')
    job['baseline_result_file']=old['result_file']
    job['changed_argument_keys']=sorted(changed)
    return job


def comparison_row(path,baseline_sampling):
    path=Path(path);result=read(path);complete=read(path.parent/'completeness.json')
    if (complete['status']!='PASS' or result['completed_epochs']!=40 or result['smoke'] or
            result['screening_only'] or result['test_evaluations']!=0 or result['target_test'] is not None or
            result['method']!='cdan' or result['target']!='foggy' or
            result['protocol']!='FEWSHOT_TARGET_CLASS_ALIGNMENT'):
        raise ValueError(f'Not a complete validation-only training run: {path}')
    sampling=result['label_guided_sampling']
    if (sampling['label_pool_sha256']!=baseline_sampling['label_pool_sha256'] or
            sampling['selected_labelled_samples']!=baseline_sampling['selected_labelled_samples'] or
            sampling['labelled_target_samples']!=27 or sampling['unlabelled_target_samples']!=324):
        raise ValueError('The fixed target label pool changed')
    peak=result['peak_target_val'];epoch=result['peak_target_val_epoch']
    values=[json.loads(s) for s in (path.parent/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    val=[r for r in values if r['partition']=='target_val']
    if len(val)!=40 or peak['accuracy']!=max(r['metrics']['accuracy'] for r in val):
        raise ValueError('Missing full training curve or an invalid peak')
    if next(r['metrics'] for r in val if r['epoch']==epoch)!=peak:
        raise ValueError('The four peak metrics must belong to the same epoch')
    return {'seed':result['seed'],'alignment_weight':sampling['matched_alignment_weight'],
        'protocol':result['protocol'],'peak_epoch':epoch,
        **{name:100*peak[key] for name,key in [('ACC','accuracy'),('Precision','precision_macro'),
            ('Recall','recall_macro'),('F1','f1_macro')]},
        'CE_selected_epoch':result['selected_epoch'],'CE_selected_ACC':100*result['target_val']['accuracy'],
        'last_ACC':100*next(r['metrics']['accuracy'] for r in val if r['epoch']==40),
        'label_pool_portable_sha256':sampling['label_pool_portable_sha256'],'result_file':str(path.resolve())}


def save_comparison(output,rows):
    identities=[(row['alignment_weight'],row['seed']) for row in rows]
    if len(identities)!=len(set(identities)):
        raise ValueError('Repeated coefficient/seed results cannot be aggregated')
    write(output/'comparison.json',rows)
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    expected=sorted({row['seed'] for row in rows})
    summaries=[]
    for weight in sorted({row['alignment_weight'] for row in rows}):
        group=[row for row in rows if row['alignment_weight']==weight]
        record={'alignment_weight':weight,'completed_seeds':len(group),'expected_seeds':len(expected),
            'all_requested_seeds_completed':sorted(row['seed'] for row in group)==expected,
            'std_ddof':0,'protocol':'FEWSHOT_TARGET_CLASS_ALIGNMENT','reporting_partition':'target_val',
            'selection':'highest_ACC_and_same_epoch_macro_metrics'}
        for metric in ('ACC','Precision','Recall','F1'):
            values=[row[metric] for row in group]
            record[metric+'_mean']=statistics.mean(values)
            record[metric+'_std']=statistics.pstdev(values) if len(values)>1 else None
        summaries.append(record)
    write(output/'summary.json',summaries)
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    lines=['# Foggy CDAN coefficient results','',
        'Protocol: FEWSHOT_TARGET_CLASS_ALIGNMENT; 3 labelled targets/class; 40 epochs. '
        'All four metrics come from the same highest-validation-ACC epoch. Units: %. '
        'Population standard deviation (ddof=0); fixed holdout not evaluated.','',
        '|Weight|Completed/expected seeds|ACC|Precision (macro)|Recall (macro)|F1 (macro)|',
        '|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        cells=[f"{row[m+'_mean']:.2f}"+(f" +/- {row[m+'_std']:.2f}" if row[m+'_std'] is not None else ' (single seed)')
               for m in ('ACC','Precision','Recall','F1')]
        lines.append(f"|{row['alignment_weight']}|{row['completed_seeds']}/{row['expected_seeds']}|"+'|'.join(cells)+'|')
    lines+=['','|Weight|Seed|Peak epoch|ACC|Precision (macro)|Recall (macro)|F1 (macro)|',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:(r['alignment_weight'],r['seed'])):
        lines.append(f"|{row['alignment_weight']}|{row['seed']}|{row['peak_epoch']}|"+
                     '|'.join(f'{row[m]:.2f}' for m in ('ACC','Precision','Recall','F1'))+'|')
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def completed_rows(baseline,paths,weights,scheduled_seeds):
    """Reuse a finished seed only after checking its exact original preset."""
    old_plan=read(Path(baseline)/'plan.json')
    rows=[];seen=set()
    for path in paths:
        path=Path(path).resolve();result=read(path);seed=result['seed']
        weight=result['label_guided_sampling']['matched_alignment_weight']
        if weight not in weights or seed in scheduled_seeds or (weight,seed) in seen:
            raise ValueError('Reused results must be unique, requested weights, and outside new training seeds')
        seen.add((weight,seed))
        original=next(j for j in old_plan['jobs'] if j['name']==f'cdan_foggy_seed{seed}')
        original_path=Path(original['result_file']);sampling=read(original_path)['label_guided_sampling']
        old_args=read(original_path.parent/'run_config.json')['arguments']
        new_args=read(path.parent/'run_config.json')['arguments']
        changed={k for k in set(old_args)|set(new_args) if old_args.get(k)!=new_args.get(k)}
        if changed!={'matched_alignment_weight','configuration_tag','save_dir'}:
            raise ValueError(f'Reused run differs from the baseline in other settings: {sorted(changed)}')
        if not any(row['seed']==seed and row['alignment_weight']==.1 for row in rows):
            rows.append(comparison_row(original_path,sampling))
        rows.append(comparison_row(path,sampling))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run',type=Path,default=ROOT/'runs/weather_label_guided/local_fewshot_20261117_175310_163')
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--weights',type=float,nargs='+',choices=(.2,.3),default=[.2,.3])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--include-completed-results',type=Path,nargs='+',default=[],
                        help='Reuse completed coefficient runs without training those seeds again')
    parser.add_argument('--execute',action='store_true',help='Otherwise generate and audit the plan only')
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.weights))!=len(args.weights):
        parser.error('Repeated seeds or weights')
    output=(args.output or ROOT/'runs/weather_label_guided'/('cdan_foggy_alignment_sweep_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if not output.is_relative_to(ROOT/'runs/weather_label_guided'):parser.error('Keep results in runs/weather_label_guided')
    if output.exists():raise FileExistsError(output)
    if read(args.baseline_run/'queue_status.json')['status']!='PASS':raise ValueError('The baseline queue is incomplete')
    jobs=[build_variant(args.baseline_run,output/('weight'+str(w).replace('.','p')),s,w)
          for w in args.weights for s in args.seeds]
    rows=[];baseline_sampling={}
    for seed in args.seeds:
        job=next(j for j in jobs if j['values']['seed']==seed)
        original=read(job['baseline_result_file'])
        baseline_sampling[seed]=original['label_guided_sampling']
        rows.append(comparison_row(job['baseline_result_file'],baseline_sampling[seed]))
    rows.extend(completed_rows(args.baseline_run,args.include_completed_results,args.weights,args.seeds))
    # Verify the label pool before training, using the exact original locked lists.
    if args.execute:
        from weather.train_vds_core import prepare_datasets
        from weather.fewshot import TargetLabelBudget
        datasets,audit=prepare_datasets(Path(jobs[0]['values']['data_dir']),'foggy',True)
        for seed in args.seeds:
            budget=TargetLabelBudget.from_dataset(datasets[1],audit['compact_label_map'],3,seed)
            if sorted(budget.target_lookup.items())!=sorted((x['path'],x['label']) for x in baseline_sampling[seed]['selected_labelled_samples']):
                raise ValueError('Current data no longer reproduces the baseline fixed label pool')
        from tools.run_weather_fewshot_suite import check_environment
        check_environment({'jobs':jobs},Path(os.environ.get('JMDA_DATA_ROOT',r'D:\Code\JMDA-Net\Data')))
    output.mkdir(parents=True)
    write(output/'plan.json',{'baseline_run':str(args.baseline_run.resolve()),'protocol':'FEWSHOT_TARGET_CLASS_ALIGNMENT',
        'weights':args.weights,'seeds':args.seeds,'only_training_parameter_changed':'matched_alignment_weight',
        'heldout_evaluation':False,'reused_result_files':[str(p.resolve()) for p in args.include_completed_results],
        'summary_expected_seeds':sorted({row['seed'] for row in rows}),'jobs':jobs})
    save_comparison(output,rows)
    print('SWEEP_DIRECTORY='+str(output),flush=True)
    if not args.execute:
        write(output/'status.json',{'status':'PLAN_ONLY','completed':0,'total':len(jobs)})
        return
    env=dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1',JMDA_DETERMINISTIC='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_HOME=str(ROOT/'pretrained_weights/torch'))
    completed=0
    try:
        for weight in args.weights:
            selected=[j for j in jobs if j['values']['matched_alignment_weight']==weight]
            variant_output=output/('weight'+str(weight).replace('.','p'))
            write(output/'status.json',{'status':'RUNNING','active_alignment_weight':weight,'completed':completed,'total':len(jobs)})
            command=[sys.executable,'-B','-u','-X','utf8',str(ROOT/'tools/run_label_guided_matrix.py'),
                '--unified-fewshot','--fewshot-alignment-variant','--pairs','cdan_foggy',
                '--seeds',*[str(s) for s in args.seeds],'--data-dir',selected[0]['values']['data_dir'],
                '--matched-alignment-weight',str(weight),'--target-labels-per-class','3',
                '--complete-budget','--parameter-profile','reference','--evaluation-mode','validation_ce',
                '--early-stopping-patience','0','--local','--local-lease-timeout-seconds','0',
                '--output',str(variant_output),'--execute']
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            for job in selected:
                row=comparison_row(job['result_file'],baseline_sampling[job['values']['seed']])
                rows.append(row);completed+=1
                print('WEIGHT_RESULT='+json.dumps(row),flush=True)
            save_comparison(output,rows)
    except BaseException as error:
        write(output/'status.json',{'status':'FAIL','completed':completed,'total':len(jobs),'error':str(error)})
        raise
    write(output/'status.json',{'status':'PASS','completed':completed,'total':len(jobs)})
    print('COMPARISON='+str(output/'comparison.csv'),flush=True)


if __name__=='__main__':main()
