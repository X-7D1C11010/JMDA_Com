"""Validation-only model selection and explicitly separated holdout audits."""
import math
from pathlib import Path

from tools.classification_metrics import classification_metrics
from vds2raw.common import better_metrics


def selection_improved(metric,metrics,cross_entropy,best_metrics,best_cross_entropy):
    if metric=='accuracy_f1':return better_metrics(metrics,best_metrics)
    if metric!='cross_entropy':raise ValueError('Unknown model selection metric')
    if not math.isfinite(cross_entropy):raise ValueError('Nonfinite validation CE')
    return best_metrics is None or cross_entropy<best_cross_entropy


def validate_holdout(test_dataset,training_and_validation,label_map):
    """Inspect metadata only: do not decode test images or fit statistics."""
    raw=test_dataset.dataset
    if getattr(raw,'label_free',False):raise ValueError('Heldout evaluation requires labelled metadata')
    if {label for _,label in raw.samples}!=set(label_map.values()):
        raise ValueError('Heldout and training class support must be equal')
    paths=[str(Path(p).resolve()) for row in raw.paired_samples for p in row[:2]]
    if raw.missing_ir or len(paths)!=len(set(paths)) or any(not Path(p).is_file() for p in paths):
        raise ValueError('Heldout VIS/IR pairs are missing, repeated, or invalid')
    others={str(Path(p).resolve()) for ds in training_and_validation for row in ds.dataset.paired_samples for p in row[:2]}
    if set(paths)&others:raise ValueError('Heldout image overlaps a training/validation partition')
    return {'partition':'target_test','samples':len(test_dataset),'path_overlap_with_train_or_val':0,
            'used_for_training_or_statistics_or_selection':False,'images_decoded_before_selection':False,
            'evaluation_rule':'once after validation-selected state restore'}


def labelled_training_diagnostic(records,truth_by_path,num_classes):
    """Join sampler truth for diagnostics only, outside the optimizer loss."""
    ids=[r['sample_id'] for r in records]
    if len(set(ids))!=len(ids) or set(ids)!=set(truth_by_path):raise ValueError('Incomplete target training diagnostics')
    if any(r.get('labels_used') is not False or 'y_true' in r for r in records):
        raise ValueError('Expected predictions from a label-free target training view')
    truth=[truth_by_path[p] for p in ids]
    metrics=classification_metrics(truth,[r['y_pred'] for r in records],num_classes=num_classes)
    losses=[]
    for r,y in zip(records,truth):
        logits=r['logits'];maximum=max(logits)
        losses.append(maximum+math.log(sum(math.exp(v-maximum) for v in logits))-logits[y])
    return {'metrics':metrics,'cross_entropy':sum(losses)/len(losses),
            'target_labels_joined_after_prediction':True,'this_diagnostic_has_no_optimizer_effect':True,
            'target_classifier_CE_in_optimizer':False}
