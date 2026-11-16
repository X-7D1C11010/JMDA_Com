"""Explicit label-guided cross-domain feature alignment; not an UDA loss."""
import torch
import torch.nn.functional as F


def matched_class_alignment(source_features,target_features,matched_classes,margin=.2):
    """Anchor target features to detached source class centroids in this batch.

    The sampler guarantees that target position i has the same true class as
    source position i. That guarantee is supervised information, even though
    this function receives only the source class tensor. No classifier CE or
    target pseudo-labels are used. Class means avoid arbitrary instance-pair
    distances; negative margins discourage all targets choosing one centroid.
    """
    if source_features.shape!=target_features.shape or source_features.ndim!=2:
        raise ValueError('Class alignment requires equal two-dimensional feature batches')
    if len(matched_classes)!=len(source_features):raise ValueError('Class/feature size mismatch')
    if not 0<=margin<2:raise ValueError('Cosine margin must lie in [0,2)')
    classes,inverse=torch.unique(matched_classes,sorted=True,return_inverse=True)
    normalized_source=F.normalize(source_features.detach().float(),dim=1)
    anchors=normalized_source.new_zeros(len(classes),source_features.shape[1])
    anchors.index_add_(0,inverse,normalized_source)
    anchors=F.normalize(anchors/torch.bincount(inverse,minlength=len(classes)).unsqueeze(1),dim=1)
    similarity=F.normalize(target_features.float(),dim=1)@anchors.T
    positive=similarity.gather(1,inverse[:,None]).squeeze(1)
    attraction=(1-positive).mean()
    if len(classes)>1:
        negative=similarity.masked_fill(F.one_hot(inverse,len(classes)).bool(),float('-inf')).max(dim=1).values
        violation=F.relu(margin+negative-positive)
        separation=violation.mean()
        violation_fraction=float((violation.detach()>0).float().mean())
        mean_gap=float((positive-negative).detach().mean())
    else:
        separation=attraction.new_zeros(())
        violation_fraction=0.;mean_gap=None
    return attraction+separation,{'positive_cosine':float(positive.detach().mean()),
        'attraction_loss':float(attraction.detach()),'separation_loss':float(separation.detach()),
        'negative_margin_violation_fraction':violation_fraction,'positive_negative_cosine_gap':mean_gap,
        'classes_in_batch':int(len(classes)),'source_centroids_detached':True,'target_classifier_CE_used':False}
