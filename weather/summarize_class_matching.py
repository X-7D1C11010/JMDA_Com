"""Separate summaries for real target-truth-guided sampling experiments."""
import argparse
import csv
import json
from pathlib import Path

from weather.class_matching import PROTOCOL,ALIGNMENT_PROTOCOL


def summarize(paths, output):
    rows=[]
    for path in map(Path, paths):
        result=json.loads(path.read_text(encoding='utf-8'))
        complete=json.loads((path.parent/'completeness.json').read_text(encoding='utf-8'))
        for record in (result,complete):
            if (record.get('protocol') not in (PROTOCOL,ALIGNMENT_PROTOCOL) or record.get('target_training_labels_used') is not True
                    or record.get('target_true_labels_used_for_sampling') is not True
                    or record.get('strict_unsupervised') is not False
                    or record.get('target_true_labels_used_in_classification_loss') is not False):
                raise ValueError(f'Unexpected supervision; do not mix with UDA: {path}')
            if (record.get('protocol')==ALIGNMENT_PROTOCOL)!=(record.get('target_true_labels_used_in_alignment_loss') is True):
                raise ValueError(f'Inconsistent feature alignment supervision: {path}')
        if complete.get('status')!='PASS':raise ValueError(f'Incomplete run: {path}')
        if result.get('smoke') or complete.get('partial_training_smoke'):
            raise ValueError(f'Smoke checks are not experimental scores: {path}')
        if result.get('simulated'):raise ValueError(f'Synthetic result: {path}')
        epochs=[json.loads(line) for line in (path.parent/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
        target=[x for x in epochs if x['partition']=='target_val']
        selected=[x for x in target if x['epoch']==result['selected_epoch']]
        metrics=result['target_val']
        selection=result.get('selection_metric','accuracy_f1')
        eligible=[x for x in target if x['epoch']>=result.get('selection_min_epoch',1)]
        if len(selected)!=1 or selected[0]['metrics']!=metrics:raise ValueError(f'Selected epoch metrics differ: {path}')
        if selection=='cross_entropy':
            if selected[0]['cross_entropy']!=min(x['cross_entropy'] for x in eligible):
                raise ValueError(f'Selected epoch is not minimum validation CE: {path}')
        elif selection!='accuracy_f1' or metrics['accuracy']!=max(x['metrics']['accuracy'] for x in eligible):
            raise ValueError(f'Selected epoch is not maximum ACC: {path}')
        peak=result.get('peak_target_val',metrics)
        if peak['accuracy']!=max(x['metrics']['accuracy'] for x in target):raise ValueError(f'Unrestricted peak missing: {path}')
        test=result.get('target_test')
        if test is not None:
            tests=[x for x in epochs if x['partition']=='target_test']
            if len(tests)!=1 or tests[0]['epoch']!=result['selected_epoch'] or tests[0]['metrics']!=test or result.get('test_evaluations')!=1:
                raise ValueError(f'Heldout must be evaluated once at the selected epoch: {path}')
        training=[json.loads(line) for line in (path.parent/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(training)!=result['completed_epochs'] or not all(
                x.get('all_batch_positions_class_matched') is True for x in training):
            raise ValueError(f'Missing class-matching checks: {path}')
        sampling=result['label_guided_sampling']
        reported=test if test is not None else metrics
        rows.append({'protocol':result['protocol'],'method':result['method'],'target':result['target'],'seed':result['seed'],
            'configuration_tag':result.get('configuration_tag','reference'),
            'target_training_labels_used':True,'target_classification_CE_used':False,
            'selection_metric':selection,'target_test_available':test is not None,
            'reporting_partition':'target_test' if test is not None else 'target_val',
            'peak_validation_ACC_percent':100*peak['accuracy'],
            'selected_validation_ACC_percent':100*metrics['accuracy'],
            'selected_epoch':result['selected_epoch'],'completed_epochs':result['completed_epochs'],
            'screening_only':result.get('screening_only',False),
            **{name:100*reported[key] for name,key in [('ACC_percent','accuracy'),('Precision_macro_percent','precision_macro'),
                ('Recall_macro_percent','recall_macro'),('F1_macro_percent','f1_macro')]},
            **{'heldout_'+name:100*test[key] if test is not None else None for name,key in
                [('ACC_percent','accuracy'),('Precision_macro_percent','precision_macro'),('Recall_macro_percent','recall_macro'),('F1_macro_percent','f1_macro')]},
            'VIS_weight':result['modality_weights_at_selected_epoch'][0],
            'IR_weight':result['modality_weights_at_selected_epoch'][1],
            'steps_per_epoch':sampling['optimizer_steps_per_epoch'],
            'source_exposures_per_epoch':sampling['source_exposures_per_epoch'],
            'target_exposures_per_epoch':sampling['target_exposures_per_epoch'],
            'matched_alignment_weight':sampling.get('matched_alignment_weight',0.),
            'result_file':str(path.resolve())})
    if not rows:raise ValueError('No completed label-guided experiments')
    if len({x['protocol'] for x in rows})!=1:raise ValueError('Keep sampling-only and feature-alignment experiments separate')
    if len({(x['selection_metric'],x['target_test_available'],x['configuration_tag']) for x in rows})!=1:
        raise ValueError('Keep validation-only and CE-selected heldout protocols separate')
    identities=[(x['method'],x['target'],x['seed']) for x in rows]
    if len(set(identities))!=len(identities):raise ValueError('Keep repeated parameter trials in separate summaries')
    if len({x['screening_only'] for x in rows})!=1:raise ValueError('Do not merge prefix screens and complete runs')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'label_guided_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'label_guided_runs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=summarize(args.results,args.output)
    print(f"{rows[0]['protocol']}: {len(rows)} real runs; {args.output}")


if __name__=='__main__':main()
