"""Optional source-only manifold mixup at the classifier input.

Reference: https://proceedings.mlr.press/v97/verma19a.html . This is an
additional regularized variant; it is not part of the five original DA losses.
No target features/labels, extra encoder forwards or inference modules are used.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def source_feature_mixup(head,features,labels,alpha,seed,update,class_weights=None,label_smoothing=0.):
    if not math.isfinite(alpha) or alpha<=0 or len(features)<2 or len(features)!=len(labels):
        raise ValueError("Invalid source feature mixup inputs")
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(update),17019]))
    coefficient=float(rng.beta(alpha,alpha))
    permutation=torch.as_tensor(rng.permutation(len(labels)),device=labels.device)
    mixed=coefficient*features+(1-coefficient)*features[permutation]
    devices=[features.device.index] if features.is_cuda else []
    # An extra MDD/dropout head forward must not change the random streams
    # used by the original method, augmentation or the next optimizer update.
    with torch.random.fork_rng(devices=devices):
        logits=head(mixed)
        loss=(coefficient*F.cross_entropy(logits,labels,weight=class_weights,label_smoothing=label_smoothing)
              +(1-coefficient)*F.cross_entropy(logits,labels[permutation],weight=class_weights,label_smoothing=label_smoothing))
    return loss,{"coefficient":coefficient,"different_label_fraction":float((labels!=labels[permutation]).float().mean()),
                 "source_samples":len(labels),"target_samples":0}
