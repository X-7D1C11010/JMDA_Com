"""Audit real label-guided weather results and draw validation learning curves."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weather.class_matching import ALIGNMENT_PROTOCOL

METRICS=('accuracy','precision_macro','recall_macro','f1_macro')
COMPLEXITY=('total_params','trainable_params','inference_GFLOPs','total_train_time_s',
            'train_epoch_mean_s','latency_mean_ms','throughput_samples_s','peak_train_reserved_gb')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def confusion_metrics(matrix):
    n=len(matrix);total=sum(map(sum,matrix))
    precision=[];recall=[];f1=[]
    for i in range(n):
        predicted=sum(row[i] for row in matrix);actual=sum(matrix[i])
        p=matrix[i][i]/predicted if predicted else 0.;r=matrix[i][i]/actual if actual else 0.
        precision.append(p);recall.append(r);f1.append(2*p*r/(p+r) if p+r else 0.)
    return dict(zip(METRICS,[sum(matrix[i][i] for i in range(n))/total,
        sum(precision)/n,sum(recall)/n,sum(f1)/n]))


def inspect(path):
    path=Path(path);folder=path.parent
    result=read(path);complete=read(folder/'completeness.json');config=read(folder/'run_config.json')
    if result.get('selection_metric','accuracy_f1')!='accuracy_f1' or result.get('target_test_available'):
        raise ValueError('Use the heldout/CE summary for the new generalization protocol')
    if (result.get('protocol')!=ALIGNMENT_PROTOCOL or result.get('target_training_labels_used') is not True
            or result.get('target_true_labels_used_in_alignment_loss') is not True
            or result.get('target_true_labels_used_in_classification_loss') is not False
            or result.get('smoke') or result.get('simulated') or complete.get('status')!='PASS'):
        raise ValueError(f'Not a completed real label-guided alignment run: {path}')
    evaluation=list(lines(folder/'evaluation.jsonl'))
    target=[x for x in evaluation if x['partition']=='target_val']
    chosen=next(x for x in target if x['epoch']==result['selected_epoch'])
    scores=confusion_metrics(chosen['metrics']['confusion_matrix'])
    if result['target_val']!=chosen['metrics'] or scores['accuracy']!=max(x['metrics']['accuracy'] for x in target):
        raise ValueError(f'Peak selection differs from epoch log: {path}')
    if any(abs(scores[k]-result['target_val'][k])>1e-10 for k in METRICS):
        raise ValueError(f'Four metrics disagree with the confusion matrix: {path}')
    training=list(lines(folder/'training.jsonl'));audit=read(folder/'dataset_audit.json')
    if len(training)!=result['completed_epochs'] or any(
            x['source_unique_samples']!=audit['counts']['source_train']
            or x['target_unique_samples']!=audit['counts']['target_train']
            or not x.get('all_batch_positions_class_matched') for x in training):
        raise ValueError(f'Missing full training coverage: {path}')
    updates=native=0
    for x in lines(folder/'updates.jsonl'):
        if (x['source_true_classes']!=x['target_true_classes_for_sampling']
                or x.get('target_classification_CE_used') is not False
                or x.get('target_true_labels_used_in_alignment_loss') is not True
                or not x['matched_alignment_details']['source_centroids_detached']
                or not math.isfinite(x['weighted_matched_alignment']) or x['weighted_matched_alignment'] < -1e-6):
            raise ValueError(f'Invalid supervised alignment/update record: {path}')
        updates+=1;native+=int(x['core_adaptation_graph_in_loss'])
    if updates!=result['optimizer_updates'] or updates!=sum(x['updates'] for x in training) or not native:
        raise ValueError(f'Missing optimizer/native DA updates: {path}')
    for key in COMPLEXITY:
        if not math.isfinite(result['complexity'][key]) or result['complexity'][key]<=0:
            raise ValueError(f'Invalid complexity {key}: {path}')
    warmup=config['arguments']['adaptation_warmup_epochs']
    post=[x for x in target if x['epoch']>warmup]
    source=[x for x in evaluation if x['partition']=='source_train']
    return {'protocol':ALIGNMENT_PROTOCOL,'method':result['method'],'target':result['target'],'seed':result['seed'],
        'selected_epoch':result['selected_epoch'],'completed_epochs':result['completed_epochs'],
        'planned_epochs':result['planned_epochs'],'screening_only':result['screening_only'],
        **{key+'_percent':100*value for key,value in scores.items()},
        'last_source_train_ACC_percent':100*source[-1]['metrics']['accuracy'],
        'last_target_val_ACC_percent':100*target[-1]['metrics']['accuracy'],
        'selected_before_native_DA':result['selected_epoch']<=warmup,
        'post_native_DA_peak_ACC_percent':100*max(x['metrics']['accuracy'] for x in post),
        'updates_checked':updates,'native_DA_updates':native,
        'VIS_scale':result['modality_weights_at_selected_epoch'][0],
        'IR_scale':result['modality_weights_at_selected_epoch'][1],
        **{key:result['complexity'][key] for key in COMPLEXITY},
        'result_file':str(path.resolve()),
        'curve':[(x['epoch'],100*x['metrics']['accuracy']) for x in target]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',nargs='+',type=Path,default=[])
    p.add_argument('--plan',type=Path)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--plot',action='store_true')
    args=p.parse_args();paths=list(args.results)
    if args.plan:paths.extend(Path(x['result_file']) for x in read(args.plan)['jobs'])
    missing=[str(x) for x in paths if not x.is_file()]
    if missing and not args.allow_incomplete:raise FileNotFoundError(missing)
    rows=[inspect(path) for path in paths if path.is_file()]
    identities=[(r['method'],r['target'],r['seed']) for r in rows]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate experiment identity')
    if not rows:raise ValueError('No completed experiments')
    payload={'protocol':ALIGNMENT_PROTOCOL,'target_training_labels_used':True,'target_classifier_CE_used':False,
        'simulated':False,'evaluation_partition':'target_val; no separate weather target_test',
        'all_requested_runs_completed':not missing,'aggregation_across_protocols_or_budgets_performed':False,
        'requested_runs':len(paths),'completed_runs':len(rows),'missing':missing,'rows':rows}
    prefix=args.output_prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table=[{k:v for k,v in row.items() if k!='curve'} for row in rows]
    with prefix.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
        for ax,target in zip(axes,['dark','foggy']):
            for row in sorted((r for r in rows if r['target']==target),key=lambda r:r['method']):
                epochs,values=zip(*row['curve'])
                ax.plot(epochs,values,label=row['method'].upper(),linewidth=1.6)
            ax.set(title=target.capitalize(),xlabel='Epoch',ylim=(0,103));ax.grid(alpha=.25);ax.legend(fontsize=8)
        axes[0].set_ylabel('Target validation ACC (%)')
        figure.suptitle('REAL seed42 training | TARGET LABEL GUIDANCE | validation, not held-out test',fontsize=11)
        figure.tight_layout();figure.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(figure)
    print(json.dumps({'completed':len(rows),'requested':len(paths),'missing':len(missing),
        'scores':{r['method']+'_'+r['target']:round(r['accuracy_percent'],2) for r in rows}},ensure_ascii=False))


if __name__=='__main__':main()
