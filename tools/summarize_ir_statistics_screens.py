"""Verify complete screens and the equal-budget IR augmentation comparison."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_weather_goal import read_run


def result_file(folder):
    files=list((ROOT/'runs/weather_enhanced_screen'/folder).glob('*/*_seed42/result.json'))
    if len(files)>1:raise ValueError('Ambiguous trial result')
    return files[0] if files else None


def signal_summary(path):
    config=json.loads((path.parent/'run_config.json').read_text(encoding='utf-8'))['arguments']
    count=0;totals={}
    with (path.parent/'updates.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line).get('input_augmentation',{})
            n=row.get('source_IR_statistics_augmented',0)
            if n:
                count+=n
                for k in ('source_IR_std_before','reference_IR_std','source_IR_std_after','source_IR_clipped_fraction'):
                    totals[k]=totals.get(k,0.)+n*row[k]
    gradients={'VIS':[],'IR':[]}
    with (path.parent/'gradients.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['epoch']<=config['adaptation_warmup_epochs']:continue
            values=row.get('enhancement_components',{}).get('core',row['gradients'])
            for name,alias in (('VIS','RGB'),('IR','NIR')):
                d=values.get(name,values.get(alias,{}))
                if d.get('CE_norm',0)>0:
                    gradients[name].append(d['DA_to_CE_ratio'])
    return {'augmented_source_observations':count,
            'sample_weighted_IR_statistics':{k:v/count for k,v in totals.items()} if count else {},
            'median_core_DA_to_CE_gradient_ratio':{k:float(np.median(v)) if v else None for k,v in gradients.items()}}


def arguments(path):
    return json.loads((path.parent/'run_config.json').read_text(encoding='utf-8'))['arguments']


def training_code_hashes(path):
    plan=json.loads((path.parents[2]/'plan.json').read_text(encoding='utf-8'))
    backend=next(Path(value) for value in plan['jobs'][0]['command'] if value.endswith('train_vds_core.py'))
    files=json.loads((backend.parent.parent/'runtime_snapshot_manifest.json').read_text(encoding='utf-8'))['files']
    return {k:v for k,v in files.items() if ((k.startswith(('weather/','vds2raw/','Transfer-Learning-Library/'))
            and 'summarize' not in k) or k in ('tools/uda_multimodal.py','tools/uda_data.py','tools/data_paths.py','DeepCORAL/models.py'))}


def main():
    names=['ir_statistics_mcc_foggy_seed42_v1','ir_statistics_mcc_foggy_control20_seed42_v1',
           'ir_statistics_cdan_dark_seed42_v1','coral_weight1_deepcoral_dark_seed42_v1',
           'coral_weight1_deepcoral_dark_seed42_v2']
    rows=[];paths={};pending=[]
    for name in names:
        path=result_file(name)
        if path is None:
            pending.append(name);continue
        paths[name]=path
        row=read_run(path,name)
        row['training_signal']=signal_summary(path)
        rows.append(row)
    comparisons=[]
    on,off=names[:2]
    if on in paths and off in paths:
        a,b=arguments(paths[off]),arguments(paths[on])
        diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        if set(diff)!={'save_dir','source_ir_statistics_probability'}:
            raise ValueError('Equal-budget control contains additional changes: '+str(diff))
        code_equal=training_code_hashes(paths[off])==training_code_hashes(paths[on])
        if not code_equal:raise ValueError('Training implementations differ between control and candidate')
        control,candidate=(next(r for r in rows if r['variant']==n) for n in (off,on))
        comparisons.append({'name':'MCC foggy equal-20-epoch IR augmentation comparison',
            'configuration_differences':diff,'training_dependency_code_hashes_equal':code_equal,
            'control_ACC_percent':control['ACC_percent'],
            'candidate_ACC_percent':candidate['ACC_percent'],
            'ACC_change_points':candidate['ACC_percent']-control['ACC_percent']})
    for coral in names[3:]:
        if coral not in paths:continue
        reference=ROOT/'runs/weather_enhanced_screen/negative_free_deepcoral_dark_seed42_v1/deepcoral_dark_source0.1_target0.1/deepcoral_dark_seed42/result.json'
        a,b=arguments(reference),arguments(paths[coral])
        diff={k:[a[k],b.get(k)] for k in a if a[k]!=b.get(k)}
        controlled=coral.endswith('_v2')
        expected={'save_dir','trade_off'} if controlled else {'save_dir','trade_off','crossmodal_stop_gradient'}
        if set(diff)!=expected:raise ValueError('Unexpected reference CORAL changes: '+str(diff))
        old=read_run(reference,'reference_CORAL_weight')
        new=next(r for r in rows if r['variant']==coral)
        comparisons.append({'name':'Deep CORAL dark '+('controlled core weight comparison' if controlled else 'compound weight/gradient-stop candidate'),
            'single_active_parameter_control':controlled,'candidate_variant':coral,
            'shared_reference_argument_changes':diff,'reference_core_weight':a['trade_off'],
            'candidate_core_weight':b['trade_off'],'control_ACC_percent':old['ACC_percent'],
            'candidate_ACC_percent':new['ACC_percent'],'ACC_change_points':new['ACC_percent']-old['ACC_percent'],
            'control_post_adaptation_peak_ACC_percent':old['post_adaptation_peak_ACC_percent'],
            'candidate_post_adaptation_peak_ACC_percent':new['post_adaptation_peak_ACC_percent']})
    data={'updated_UTC':datetime.now(timezone.utc).isoformat(),'target_training_labels_used':False,
          'evaluation':'within-run best target_val; no independent weather test',
          'diagnostic_and_preflight_scores_included':False,'rows':rows,'pending':pending,'comparisons':comparisons}
    (ROOT/'results/ir_statistics_screen_comparison_seed42.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# IR statistics and CORAL-weight screens: verified results','',
           'Source labels only; target_train labels forbidden. Metrics come from the same selected epoch.','',
           '| Trial | ACC% | Precision% | Recall% | F1% | Selected/completed |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['variant']} | {r['ACC_percent']:.2f} | {r['Precision_percent']:.2f} | "
                     f"{r['Recall_percent']:.2f} | {r['F1_percent']:.2f} | {r['selected_epoch']}/{r['completed_epochs']} |")
    lines+=['','Pending: '+(', '.join(pending) if pending else 'none')]
    for row in comparisons:
        lines+=['',f"{row['name']}: {row['ACC_change_points']:+.2f} percentage points."]
    lines+=['','These results do not establish the all-weathers 80% goal. Do not substitute a partial run for a completed result.']
    (ROOT/'results/ir_statistics_screen_comparison_seed42.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'complete':len(rows),'pending':pending,'comparisons':comparisons}))


if __name__=='__main__':main()
