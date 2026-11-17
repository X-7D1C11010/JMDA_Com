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


def fewshot_class_alignment(source_features,target_features,source_classes,target_known_classes,margin=.2):
    """Unknown targets (-1) receive exactly zero gradient from this extra loss."""
    if source_features.ndim!=2 or target_features.ndim!=2 or source_features.shape[1]!=target_features.shape[1]:
        raise ValueError('Invalid few-shot feature shapes')
    if len(source_classes)!=len(source_features) or len(target_known_classes)!=len(target_features):
        raise ValueError('Few-shot class/feature count mismatch')
    if not 0<=margin<2:raise ValueError('Invalid cosine margin')
    classes,inverse=torch.unique(source_classes,sorted=True,return_inverse=True)
    known=target_known_classes>=0
    available=(target_known_classes[:,None]==classes[None,:]).any(1)&known
    count=int(available.sum())
    if not count:
        return target_features.sum()*0.,{'labelled_targets_in_batch':int(known.sum()),
            'aligned_labelled_targets':0,'unlabelled_targets_in_loss':0,
            'aligned_target_positions':[],
            'source_centroids_detached':True,'target_classifier_CE_used':False}
    normalized=F.normalize(source_features.detach().float(),dim=1)
    anchors=normalized.new_zeros(len(classes),source_features.shape[1])
    anchors.index_add_(0,inverse,normalized)
    anchors=F.normalize(anchors/torch.bincount(inverse,minlength=len(classes))[:,None],dim=1)
    target_labels=target_known_classes[available]
    indices=(target_labels[:,None]==classes[None,:]).long().argmax(1)
    similarity=F.normalize(target_features[available].float(),dim=1)@anchors.T
    positive=similarity.gather(1,indices[:,None]).squeeze(1)
    attraction=(1-positive).mean()
    separation=attraction.new_zeros(())
    if len(classes)>1:
        negative=similarity.masked_fill(F.one_hot(indices,len(classes)).bool(),float('-inf')).max(1).values
        separation=F.relu(margin+negative-positive).mean()
    return attraction+separation,{'labelled_targets_in_batch':int(known.sum()),
        'aligned_labelled_targets':count,'unlabelled_targets_in_loss':0,
        'aligned_target_positions':available.nonzero(as_tuple=True)[0].detach().cpu().tolist(),
        'source_centroids_detached':True,'target_classifier_CE_used':False,
        'attraction_loss':float(attraction.detach()),'separation_loss':float(separation.detach())}
