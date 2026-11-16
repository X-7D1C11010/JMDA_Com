"""Posthoc full-target-training fit audit; never modifies trained results."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_label_guided_screens import read,lines,confusion_metrics


def audit_fit(report):
    rows=[]
    for recorded in read(report)['rows']:
        folder=Path(recorded['result_file']).parent
        config=read(folder/'run_config.json');metadata=read(folder/'dataset_audit.json')
        label_map={int(k):int(v) for k,v in metadata['compact_label_map'].items()}
        path=Path(config['arguments']['data_dir'])/f"{recorded['target']}_train.txt"
        truth={}
        for line in path.read_text(encoding='utf-8').splitlines():
            raw,label=line.rsplit(maxsplit=1);truth[str(Path(raw).resolve())]=label_map[int(label)]
        classes=len(label_map);by_epoch={}
        for item in lines(folder/'predictions.jsonl'):
            if item['partition']!='target_train':continue
            table=by_epoch.setdefault(item['epoch'],{'confusion':[[0]*classes for _ in range(classes)],'CE_sum':0.,'ids':set()})
            key=str(Path(item['sample_id']).resolve())
            if key not in truth or key in table['ids']:raise ValueError('Missing/duplicate target training prediction')
            table['ids'].add(key);actual=truth[key]
            table['confusion'][actual][item['y_pred']]+=1
            table['CE_sum']-=math.log(max(float(item['probabilities'][actual]),1e-30))
        if any(t['ids']!=set(truth) for t in by_epoch.values()):raise ValueError('Incomplete target training evaluation')
        evaluation=list(lines(folder/'evaluation.jsonl'))
        epoch=recorded['selected_epoch'];last=recorded['completed_epochs']
        def values(e):
            train=by_epoch[e];val=next(x for x in evaluation if x['partition']=='target_val' and x['epoch']==e)
            return {'target_train_ACC_percent':100*confusion_metrics(train['confusion'])['accuracy'],
                    'target_val_ACC_percent':100*val['metrics']['accuracy'],
                    'target_train_CE':train['CE_sum']/len(truth),'target_val_CE':val['cross_entropy']}
        selected=values(epoch);final=values(last)
        rows.append({'method':recorded['method'],'target':recorded['target'],'selected_epoch':epoch,'completed_epochs':last,
            **{'selected_'+k:v for k,v in selected.items()},**{'last_'+k:v for k,v in final.items()},
            'selected_train_minus_val_pp':selected['target_train_ACC_percent']-selected['target_val_ACC_percent']})
    return {'protocol':read(report)['protocol'],'target_training_labels_used':True,
        'target_train_metrics_joined_posthoc':True,'labels_not_added_to_optimizer_loss_by_this_audit':True,'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit_fit(args.report)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runs':len(result['rows']),'max_selected_train_minus_val_pp':max(x['selected_train_minus_val_pp'] for x in result['rows'])}))


if __name__=='__main__':main()
