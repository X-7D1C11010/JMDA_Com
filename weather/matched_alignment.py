"""Explicit label-guided cross-domain feature alignment; not an UDA loss."""
import torch
import torch.nn.functional as F


class SourceClassPrototypeEMA(torch.nn.Module):
    """Detached class anchors updated exclusively from labelled source batches."""
    def __init__(self,num_classes,dimension,momentum=.9):
        super().__init__()
        if num_classes<1 or dimension<1 or not 0<=momentum<1:
            raise ValueError('Invalid source prototype geometry or momentum')
        self.momentum=float(momentum)
        self.register_buffer('centroids',torch.zeros(num_classes,dimension))
        self.register_buffer('sample_counts',torch.zeros(num_classes,dtype=torch.long))
        self.register_buffer('batch_counts',torch.zeros(num_classes,dtype=torch.long))

    @torch.no_grad()
    def update_source(self,features,labels):
        if features.ndim!=2 or features.shape[1]!=self.centroids.shape[1] or len(labels)!=len(features):
            raise ValueError('Invalid source prototype input')
        if bool((labels<0).any()) or bool((labels>=len(self.centroids)).any()):
            raise ValueError('Source labels must use the shared compact class map')
        normalized=F.normalize(features.detach().float(),dim=1)
        counts=torch.bincount(labels,minlength=len(self.centroids))
        sums=torch.zeros_like(self.centroids).index_add_(0,labels,normalized)
        means=F.normalize(sums/counts.clamp_min(1)[:,None],dim=1)
        present=counts>0;existing=present&(self.sample_counts>0);fresh=present&~existing
        old=self.centroids[existing].clone()
        self.centroids[existing]=F.normalize(self.momentum*old+(1-self.momentum)*means[existing],dim=1)
        self.centroids[fresh]=means[fresh]
        self.sample_counts.add_(counts);self.batch_counts.add_(present.long())
        return {'prototype_classes_updated':present.nonzero(as_tuple=True)[0].cpu().tolist(),
            'prototype_delta_L2_mean':float((self.centroids[existing]-old).norm(dim=1).mean()) if len(old) else 0.}


def fewshot_ema_class_alignment(source_features,target_features,source_classes,target_known_classes,memory,margin=.2):
    if not 0<=margin<2 or len(target_known_classes)!=len(target_features):
        raise ValueError('Invalid EMA alignment inputs')
    record=memory.update_source(source_features,source_classes)
    classes=(memory.sample_counts>0).nonzero(as_tuple=True)[0]
    anchors=memory.centroids[classes].detach().clone()
    known=target_known_classes>=0
    available=known&(target_known_classes[:,None]==classes[None,:]).any(1)
    positions=available.nonzero(as_tuple=True)[0]
    in_batch=(target_known_classes[:,None]==source_classes[None,:]).any(1)
    details={'source_anchor_mode':'source_ema','source_anchor_ema_momentum':memory.momentum,
        'source_prototypes_updated_from':'source_train_only','target_samples_in_prototype_updates':0,
        'available_source_prototype_classes':classes.detach().cpu().tolist(),
        'source_prototype_sample_counts':memory.sample_counts.detach().cpu().tolist(),
        'source_prototype_batch_counts':memory.batch_counts.detach().cpu().tolist(),
        'labelled_targets_in_batch':int(known.sum()),'aligned_labelled_targets':int(available.sum()),
        'aligned_from_cache_not_current_batch':int((available&~in_batch).sum()),
        'aligned_target_positions':positions.detach().cpu().tolist(),
        'unlabelled_targets_in_loss':0,'source_centroids_detached':True,'target_classifier_CE_used':False,**record}
    if not len(positions):return target_features.sum()*0.,details
    labels=target_known_classes[available]
    indices=(labels[:,None]==classes[None,:]).long().argmax(1)
    similarity=F.normalize(target_features[available].float(),dim=1)@anchors.T
    positive=similarity.gather(1,indices[:,None]).squeeze(1)
    attraction=(1-positive).mean();separation=attraction.new_zeros(())
    if len(classes)>1:
        negative=similarity.masked_fill(F.one_hot(indices,len(classes)).bool(),float('-inf')).max(1).values
        separation=F.relu(margin+negative-positive).mean()
    details.update(attraction_loss=float(attraction.detach()),separation_loss=float(separation.detach()),
                   positive_cosine=float(positive.detach().mean()))
    return attraction+separation,details


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
