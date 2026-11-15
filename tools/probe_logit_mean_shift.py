"""Train-statistic logit-offset diagnostic, never a formal method result."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_weather_goal import read_run
from tools.classification_metrics import classification_metrics

RUNS=[
 'negative_free_deepcoral_dark_seed42_v1/deepcoral_dark_source0.1_target0.1/deepcoral_dark_seed42',
 'initial_target_anchor_deepcoral_dark_seed42_v1/deepcoral_dark_source0.1_target0.1/deepcoral_dark_seed42',
 'domain_confusion_cdan_dark_fullVIS_seed42_v2/cdan_dark_source0_target0/cdan_dark_seed42',
 'ir_statistics_mcc_foggy_seed42_v1/mcc_foggy_source0.1_target0.1/mcc_foggy_seed42',
 'new_layers_lr_deepcoral_dark_seed42_v1/deepcoral_dark_source0.1_target0.1/deepcoral_dark_seed42',
]


def main():
    results=[];pending=[]
    for relative in RUNS:
        run=ROOT/'runs/weather_enhanced_screen'/relative
        if not (run/'result.json').exists():pending.append(relative);continue
        baseline=read_run(run/'result.json',relative)
        epochs=defaultdict(lambda:defaultdict(list))
        with (run/'predictions.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['partition']=='target_train' and (row['labels_used'] or 'y_true' in row):
                    raise ValueError('Target training truth entered logit calibration')
                epochs[row['epoch']][row['partition']].append(row)
        rows=[]
        for epoch,parts in epochs.items():
            source=parts['source_train'];target=parts['target_train'];validation=parts['target_val']
            s=np.asarray([r['logits'] for r in source]);t=np.asarray([r['logits'] for r in target])
            v=np.asarray([r['logits'] for r in validation]);sy=np.asarray([r['y_true'] for r in source])
            labels=[r['y_true'] for r in validation];classes=s.shape[1]
            if set(sy.tolist())!=set(range(classes)):raise ValueError('Source class missing in mean calibration')
            for balanced in (False,True):
                source_mean=(np.stack([s[sy==c].mean(0) for c in range(classes)]).mean(0) if balanced else s.mean(0))
                shift=source_mean-t.mean(0);shift-=shift.mean()
                for alpha in (0.,.25,.5,.75,1.):
                    prediction=(v+alpha*shift).argmax(1)
                    metrics=classification_metrics(labels,prediction.tolist(),classes)
                    rows.append({'epoch':epoch,'balanced_source_class_mean':balanced,'alpha':alpha,
                        'training_fitted_bias':(alpha*shift).tolist(),'metrics':metrics})
        best=max(rows,key=lambda r:(r['metrics']['accuracy'],r['metrics']['f1_macro']))
        results.append({'run':relative,'native_completed_result':baseline,'diagnostic_best':best,
                        'diagnostic_candidates':len(rows)})
        print('LOGIT_MEAN_DIAGNOSTIC',relative,round(100*best['metrics']['accuracy'],2),best['epoch'],best['alpha'],best['balanced_source_class_mean'],flush=True)
    payload={'kind':'source_target_train_mean_logit_shift_diagnostic','formal_method_result':False,
        'target_training_labels_used':False,'validation_used_for_bias_fitting':False,
        'validation_used_for_hyperparameter_scoring':True,'reference':'train-only per-domain logit means; no class permutation',
        'results':results,'pending':pending}
    (ROOT/'results/train_logit_mean_shift_diagnostic_seed42.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


if __name__=='__main__':main()
