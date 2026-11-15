"""Verify target-anchor and new-layer LR screens against their own references."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_weather_goal import read_run


def main():
    cases=[('initial_target_anchor_deepcoral_dark_seed42_v1','target_representation_anchor_weight'),
           ('initial_target_anchor_mcc_foggy_seed42_v1','target_representation_anchor_weight'),
           ('new_layers_lr_deepcoral_dark_seed42_v1','lr'),
           ('source_weather_deepcoral_dark_seed42_v1','source_weather_probability')]
    rows=[];pending=[]
    for name,factor in cases:
        base=ROOT/'runs/weather_enhanced_screen'/name
        files=list(base.glob('*/*_seed42/result.json'))
        if not files:
            pending.append(name);continue
        if len(files)!=1:raise ValueError('Ambiguous trial result')
        path=files[0];plan=json.loads((base/'plan.json').read_text(encoding='utf-8'))
        reference_path=Path(plan['reference_config']).with_name('result.json')
        changes=plan['jobs'][0]['reference_argument_changes']
        if set(changes)!={'save_dir',factor}:raise ValueError('Unexpected model parameter changes: '+str(changes))
        reference=read_run(reference_path,'reference');candidate=read_run(path,name)
        config=json.loads((path.parent/'run_config.json').read_text(encoding='utf-8'))
        item={'variant':name,'changed_factor':factor,'argument_changes':changes,
              'reference':reference,'candidate':candidate,
              'ACC_change_points':candidate['ACC_percent']-reference['ACC_percent'],
              'effective_initial_optimizer_groups':config['optimizer_groups']}
        if factor=='target_representation_anchor_weight':
            audit=json.loads((path.parent/'target_representation_anchor_audit.json').read_text(encoding='utf-8'))
            if audit['target_training_labels_used'] or audit['validation_used']:raise ValueError('Invalid teacher cohort')
            tail=[]
            with (path.parent/'updates.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    value=json.loads(line)
                    if value['epoch']>candidate['completed_epochs']-3:tail.append(value)
            item['initial_target_cache_audit']=audit
            item['last_three_epoch_mean_cosines']={key:statistics.mean(r['target_representation_cosines'][key] for r in tail)
                for key in ('VIS','IR')}
            item['last_three_epoch_mean_weighted_anchor']=statistics.mean(r['weighted_target_representation_anchor'] for r in tail)
        rows.append(item)
    result={'updated_UTC':datetime.now(timezone.utc).isoformat(),'target_training_labels_used':False,
            'validation_used_for_cache_initialization':False,
            'evaluation':'within-run peak target_val; no independent weather test',
            'initialization_diagnostics_counted_as_training_results':False,'comparisons':rows,'pending':pending}
    (ROOT/'results/initial_target_anchor_comparison_seed42.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Initial target representation / new-layer LR screens','',
           '| Trial | Reference ACC% | Candidate ACC% | Delta | Candidate F1% | Best/completed |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        a,b=row['reference'],row['candidate']
        lines.append(f"| {row['variant']} | {a['ACC_percent']:.2f} | {b['ACC_percent']:.2f} | {row['ACC_change_points']:+.2f} | "
                     f"{b['F1_percent']:.2f} | {b['selected_epoch']}/{b['completed_epochs']} |")
    lines+=['','Pending: '+(', '.join(pending) if pending else 'none'),'',
            'Each completed candidate differs only in the stated factor and output directory. Negative results are not promoted.']
    (ROOT/'results/initial_target_anchor_comparison_seed42.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'completed':len(rows),'pending':pending,
        'ACC_percent':{r['variant']:r['candidate']['ACC_percent'] for r in rows}}))


if __name__=='__main__':main()
