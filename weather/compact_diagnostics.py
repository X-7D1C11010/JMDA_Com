"""Export recorded evidence for Git sync without copying image caches/raw logs."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def _rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def _load(path,optional=False):
    if optional and not path.exists():return None
    return json.loads(path.read_text(encoding='utf-8'))


def _check_matrix(metrics):
    cm=metrics['confusion_matrix'];size=len(cm)
    if not size or any(len(row)!=size or any(type(n)!=int or n<0 for n in row) for row in cm):
        raise ValueError('Invalid confusion matrix')
    support=[sum(row) for row in cm];total=sum(support)
    if not total:raise ValueError('Empty confusion matrix')
    present=[i for i,n in enumerate(support) if n];precision=[];recall=[];f1=[]
    for i in present:
        column=sum(row[i] for row in cm);p=cm[i][i]/column if column else 0.;r=cm[i][i]/support[i]
        precision.append(p);recall.append(r);f1.append(2*p*r/(p+r) if p+r else 0.)
    expected={'accuracy':sum(cm[i][i] for i in range(size))/total,
        'precision_macro':sum(precision)/len(present),'recall_macro':sum(recall)/len(present),
        'f1_macro':sum(f1)/len(present)}
    if any(not math.isclose(metrics[k],v,abs_tol=1e-10) for k,v in expected.items()):
        raise ValueError('Reported metrics do not match their same-epoch confusion matrix')


def _brief_evaluation(record):
    # Main target-val CM is retained every epoch. Branch CMs are retained at
    # the peak and last epoch in the run record; keep epoch exports smaller.
    result=dict(record)
    if record.get('auxiliary_branches'):
        result['auxiliary_branches']={name:({**branch,'metrics':{k:v for k,v in branch['metrics'].items()
            if k!='confusion_matrix'}} if branch.get('metrics') else dict(branch))
            for name,branch in record['auxiliary_branches'].items()}
    if record['partition']!='target_val' and record.get('metrics'):
        result['metrics']={k:v for k,v in record['metrics'].items() if k!='confusion_matrix'}
    return result


def _replace_run(path,run_id,records):
    previous=[r for r in _rows(path) if r['run_id']!=run_id] if path.exists() else []
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        for row in (*previous,*records):stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def export_run_diagnostics(output,result_file,run_namespace=None):
    output=Path(output);result_file=Path(result_file);folder=result_file.parent
    result=_load(result_file);config=_load(folder/'run_config.json');args=config['arguments']
    if (result.get('smoke') or result.get('screening_only') or result.get('simulated') or
            result.get('target_test_available') or result.get('test_evaluations',0)):
        raise ValueError('Only complete real validation-only runs may be exported')
    if _load(folder/'completeness.json').get('status')!='PASS':raise ValueError('Run is incomplete')
    run_id=f"{result['method']}_{result['target']}_seed{result['seed']}"
    if run_namespace is not None:
        if not run_namespace or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in run_namespace):
            raise ValueError('Diagnostic namespace must be a nonempty lowercase trial identifier')
        run_id=run_namespace+':'+run_id
    identity={k:result[k] for k in ('method','target','seed','protocol','configuration_tag')}
    identity['run_id']=run_id
    if run_namespace is not None:identity['trial']=run_namespace
    evaluations=list(_rows(folder/'evaluation.jsonl'))
    target={row['epoch']:row for row in evaluations if row['partition']=='target_val'}
    if len(target)!=result['completed_epochs'] or set(target)!=set(range(1,result['completed_epochs']+1)):
        raise ValueError('Incomplete epoch validation evidence')
    if any(r['partition']=='target_test' for r in evaluations):raise ValueError('Unexpected held-out evaluation')
    for row in target.values():_check_matrix(row['metrics'])
    peak_epoch=result['peak_target_val_epoch'];ce_epoch=result['selected_epoch']
    if (target[peak_epoch]['metrics']!=result['peak_target_val'] or
            target[ce_epoch]['metrics']!=result['target_val'] or
            target[peak_epoch]['metrics']['accuracy']!=max(r['metrics']['accuracy'] for r in target.values())):
        raise ValueError('Peak and CE-selected epoch evidence disagree with result.json')
    indexed={r['epoch']:r for r in _rows(folder/'training.jsonl')}
    gradients={r['epoch']:r for r in _rows(folder/'gradients.jsonl')}
    parameter_updates={r['epoch']:r for r in _rows(folder/'parameter_updates.jsonl')}
    controls={r['epoch']:r for r in _rows(folder/'validation_control.jsonl')}
    known_fit={r['epoch']:r for r in _rows(folder/'target_train_fit.jsonl')}
    eval_by_epoch=defaultdict(dict)
    for row in evaluations:eval_by_epoch[row['epoch']][row['partition']]=row
    stats={};last_prototypes=None
    scalar_keys=('source_ce','weighted_core_transfer','weighted_matched_alignment','domain_accuracy',
        'gradient_norm_before_clipping','source_ce_unweighted','weighted_auxiliary_source',
        'weighted_source_crossmodal','weighted_prediction_distillation','weighted_domain_confusion')
    for row in _rows(folder/'updates.jsonl'):
        epoch=row['epoch']
        if epoch not in stats:
            stats[epoch]={'updates':0,'known_target_occurrences':0,'aligned_target_occurrences':0,
                'aligned_using_cached_missing_batch_class':0,'clipped_updates':0,'native_active_updates':0,
                'sum':defaultdict(float),'count':defaultdict(int),'first_learning_rates':row['learning_rates']}
        s=stats[epoch];s['updates']+=1;s['last_learning_rates']=row['learning_rates']
        if row.get('target_classification_CE_used') or row['matched_alignment_details'].get('unlabelled_targets_in_loss',0):
            raise ValueError('Unexpected target label usage')
        d=row['matched_alignment_details'];last_prototypes=d
        s['known_target_occurrences']+=sum(y>=0 for y in row['target_known_alignment_labels'])
        s['aligned_target_occurrences']+=d.get('aligned_labelled_targets',0)
        s['aligned_using_cached_missing_batch_class']+=d.get('aligned_from_cache_not_current_batch',0)
        s['native_active_updates']+=bool(row.get('core_adaptation_graph_in_loss'))
        s['clipped_updates']+=bool(args.get('gradient_clip_norm',0)>0 and
            row.get('gradient_norm_before_clipping',0)>args['gradient_clip_norm'])
        for key in scalar_keys:
            if isinstance(row.get(key),(int,float)):
                s['sum'][key]+=row[key];s['count'][key]+=1
    epoch_records=[]
    for epoch in sorted(target):
        s=stats[epoch];s['mean']={k:v/s['count'][k] for k,v in s.pop('sum').items()};s.pop('count')
        s['clipped_fraction']=s['clipped_updates']/s['updates']
        if s['updates']!=indexed[epoch]['updates']:raise ValueError('Missing optimizer-update evidence')
        epoch_records.append({**identity,'epoch':epoch,'training':indexed[epoch],'update_summary':s,
            'evaluation':{p:_brief_evaluation(r) for p,r in eval_by_epoch[epoch].items()},
            'labelled_target_fit':known_fit[epoch],
            'last_batch_gradient_diagnostic':gradients[epoch],
            'last_batch_parameter_update_diagnostic':parameter_updates[epoch],
            'validation_control':controls[epoch]})
    prediction_evidence={}
    for row in _rows(folder/'predictions.jsonl'):
        if row['partition']=='target_val' and row['epoch'] in (peak_epoch,ce_epoch):
            prediction_evidence.setdefault(row['epoch'],[]).append({k:row[k] for k in
                ('sample_index','y_true','y_pred','probabilities')})
    for epoch,rows in prediction_evidence.items():
        rows.sort(key=lambda r:r['sample_index'])
        if (len(rows)!=target[epoch]['metrics']['val_samples'] or
                len({r['sample_index'] for r in rows})!=len(rows)):
            raise ValueError('Incomplete selected predictions')
        cm=target[epoch]['metrics']['confusion_matrix'];reconstructed=[[0]*len(cm) for _ in cm]
        for row in rows:reconstructed[row['y_true']][row['y_pred']]+=1
        if reconstructed!=cm:raise ValueError('Selected predictions disagree with same-epoch confusion matrix')
    if set(prediction_evidence)!={peak_epoch,ce_epoch}:raise ValueError('Missing selected prediction evidence')
    record={**identity,'result_file':str(result_file),'arguments':args,
        'code_provenance':config.get('provenance'),'code_sha256':config.get('code_sha256'),
        'dataset_audit':_load(folder/'dataset_audit.json'),'label_budget':result['label_guided_sampling'],
        'enhancement':result['enhancement'],'peak_target_val':target[peak_epoch],
        'CE_selected_target_val':target[ce_epoch],'last_epoch_evaluation':eval_by_epoch[max(target)],
        'complexity':result['complexity'],'final_source_prototype_diagnostic':last_prototypes,
        'source_branch_probe':_load(folder/'source_branch_probe.json',True),
        'source_branch_probe_final':_load(folder/'source_branch_probe_final.json',True),
        'selected_epoch_modality_ablation':_load(folder/'modality_diagnostics.json',True),
        'prediction_evidence':prediction_evidence,
        'prediction_sha256':{e:hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
                             for e,rows in prediction_evidence.items()},
        'gradient_diagnostic_scope':'last optimizer batch each epoch; combined gradients and native/extra components separate',
        'test_evaluations':result.get('test_evaluations',0),'simulated':False}
    output.mkdir(parents=True,exist_ok=True)
    _replace_run(output/'run_diagnostics.jsonl',run_id,[record])
    _replace_run(output/'epoch_diagnostics.jsonl',run_id,epoch_records)
    return record
