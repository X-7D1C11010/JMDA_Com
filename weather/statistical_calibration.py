"""Opt-in training-only feature calibration and source-mapped initialization.

Neither fitting API accepts target instance labels or validation data. These
components are an explicitly named enhancement, not an official method core.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def unit(values):
    values=np.asarray(values,dtype=np.float64)
    return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)


def fused_features(raw,visible_weight,visible_dimension=None):
    if raw.ndim!=2 or not 0<visible_weight<1:
        raise ValueError("Expected two feature branches and a nonzero weight for each")
    if visible_dimension is None and raw.shape[1]%2:raise ValueError("Specify unequal branch widths")
    width=raw.shape[1]//2 if visible_dimension is None else int(visible_dimension)
    if not 0<width<raw.shape[1]:raise ValueError("Invalid visible feature width")
    return unit(np.concatenate((visible_weight*unit(raw[:,:width]),(1-visible_weight)*unit(raw[:,width:])),1))


class TrainFeatureCalibration(nn.Module):
    def __init__(self,projection,pooled_mean,source_mean,target_mean,source_transform,metadata):
        super().__init__()
        for name,value in (("projection",projection),("pooled_mean",pooled_mean),("source_mean",source_mean),
                           ("target_mean",target_mean),("source_transform",source_transform)):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.metadata=metadata
        self.output_dim=int(self.projection.shape[0])

    @classmethod
    def fit(cls,source_features,unlabelled_target_features,components=32,shrinkage=.1,strategy="coral"):
        a,b=np.asarray(source_features,dtype=np.float64),np.asarray(unlabelled_target_features,dtype=np.float64)
        if (a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or min(len(a),len(b))<2 or
                components<2 or shrinkage<=0 or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("Invalid training feature arrays or calibration settings")
        if strategy not in {"coral","center","pca"}:raise ValueError("Unknown training-statistics strategy")
        count=min(int(components),a.shape[1],len(a)+len(b)-1)
        with threadpool_limits(limits=2):
            pca=PCA(n_components=count,svd_solver="full").fit(np.concatenate((a,b)))
            sa,tb=pca.transform(a),pca.transform(b)
            sm,tm=sa.mean(0),tb.mean(0)
            transform=np.eye(count)
            if strategy=="coral":
                def power(cov,exponent):
                    values,vectors=np.linalg.eigh(cov)
                    ridge=max(float(np.trace(cov))/count*shrinkage,1e-6)
                    return (vectors*(np.maximum(values,0)+ridge)**exponent)@vectors.T
                transform=power(np.cov(sa,rowvar=False),-.5)@power(np.cov(tb,rowvar=False),.5)
            elif strategy=="pca":sm,tm=np.zeros(count),np.zeros(count)
        metadata={"strategy":strategy,"components":count,"shrinkage":shrinkage,
            "fit_partitions":["source_train","target_train"],"fit_samples":{"source_train":len(a),"target_train":len(b)},
            "target_instance_labels_used":False,"validation_statistics_fitted":False,
            "explained_variance_ratio_sum":float(pca.explained_variance_ratio_.sum())}
        result=cls(pca.components_,pca.mean_,sm,tm,transform,metadata)
        digest=hashlib.sha256()
        for value in result.buffers():digest.update(value.numpy().tobytes())
        result.metadata["transform_sha256"]=digest.hexdigest()
        return result

    def forward(self,features,domain="target",source_count=None):
        projected=(features-self.pooled_mean)@self.projection.T
        if source_count is not None:
            if not 0<source_count<len(features):raise ValueError("Invalid mixed-domain split")
            source=(projected[:source_count]-self.source_mean)@self.source_transform
            target=projected[source_count:]-self.target_mean
            values=torch.cat((source,target),0)
        elif domain=="source":values=(projected-self.source_mean)@self.source_transform
        elif domain=="target":values=projected-self.target_mean
        else:raise ValueError("Specify source or target domain")
        return F.normalize(values,dim=1,eps=1e-12)


class ClusterInitialization(nn.Module):
    def __init__(self,centers_by_class,reliability,temperature,metadata):
        super().__init__()
        self.register_buffer("centers",torch.as_tensor(centers_by_class,dtype=torch.float32))
        self.register_buffer("reliability",torch.as_tensor(reliability,dtype=torch.float32))
        self.temperature=float(temperature)
        self.metadata=metadata

    @classmethod
    def fit(cls,source_features,unlabelled_target_features,source_labels,classes,temperature=.1,
            seed=42,matching="source_votes",initialization="kmeans++",neighbors=3,trim_fraction=.25,
            association_source=None,association_target=None):
        a,b=np.asarray(source_features),np.asarray(unlabelled_target_features)
        labels=np.asarray(source_labels,dtype=np.int64)
        if (a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or len(a)!=len(labels) or
                set(labels)!=set(range(classes)) or len(b)<classes or temperature<=0):
            raise ValueError("Invalid source labels, unlabelled target features, or cluster settings")
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Nonfinite training features")
        prototypes=unit(np.stack([a[labels==i].mean(0) for i in range(classes)]))
        if initialization not in {"kmeans++","source_prototypes"}:raise ValueError("Unknown cluster initialization")
        with threadpool_limits(limits=2):
            clusters=KMeans(n_clusters=classes,random_state=seed,n_init=10 if initialization=="kmeans++" else 1,
                init="k-means++" if initialization=="kmeans++" else prototypes.copy()).fit(b)
        association_a,association_centers=a,clusters.cluster_centers_
        if association_source is not None or association_target is not None:
            association_a,association_b=np.asarray(association_source),np.asarray(association_target)
            if (association_a.ndim!=2 or association_b.ndim!=2 or len(association_a)!=len(a) or len(association_b)!=len(b)
                    or association_a.shape[1]!=association_b.shape[1]):
                raise ValueError("Auxiliary paired association features must preserve the training cohort order")
            association_centers=np.stack([association_b[clusters.labels_==i].mean(0) for i in range(classes)])
            source_clusters=((association_a[:,None,:]-association_centers[None,:,:])**2).sum(2).argmin(1)
        else:source_clusters=clusters.predict(a)
        votes=np.zeros((classes,classes),dtype=np.float64)
        np.add.at(votes,(source_clusters,labels),1)
        balanced=votes/np.maximum(votes.sum(0,keepdims=True),1)
        association_prototypes=unit(np.stack([association_a[labels==i].mean(0) for i in range(classes)]))
        similarities=unit(association_centers)@association_prototypes.T
        if matching=="source_votes":score=balanced
        elif matching=="source_prototypes":score=similarities
        elif matching=="source_neighbors":
            if neighbors<1:raise ValueError("Source association needs positive neighbor count")
            distances=((association_centers[:,None,:]-association_a[None,:,:])**2).sum(2)
            score=np.stack([-np.sort(distances[:,labels==c],axis=1)[:,:min(neighbors,int((labels==c).sum()))].mean(1)
                            for c in range(classes)],1)
        elif matching=="source_trimmed":
            if not 0<trim_fraction<=1:raise ValueError("Invalid source-view trimming fraction")
            distances=np.sqrt(((association_a[:,None,:]-association_centers[None,:,:])**2).sum(2))
            nearest=distances.argmin(1)
            local=np.zeros_like(votes)
            for c in range(classes):
                indices=np.flatnonzero(labels==c)
                chosen=indices[np.argsort(distances[indices,nearest[indices]])[:max(1,int(np.ceil(len(indices)*trim_fraction)))]]
                np.add.at(local,(nearest[chosen],labels[chosen]),1/len(chosen))
            score=local
        else:raise ValueError("Unknown source-only cluster association")
        rows,columns=linear_sum_assignment(-score)
        mapping=np.zeros(classes,dtype=np.int64);mapping[rows]=columns
        reverse=np.argsort(mapping)
        purity=balanced/np.maximum(balanced.sum(1,keepdims=True),1e-12)
        reliability=purity[np.arange(classes),mapping][reverse]
        metadata={"fit_partitions":["source_train","target_train"],"cluster_fit_samples":len(b),
            "association_label_source":"source_train labels only","matching":matching,"initialization":initialization,
            "neighbors":neighbors,"source_view_trim_fraction":trim_fraction,
            "auxiliary_paired_modalities_for_source_association":association_source is not None,
            "cluster_to_source_class":mapping.tolist(),"source_cluster_class_votes":votes.astype(int).tolist(),
            "source_association_reliability_by_class":reliability.tolist(),
            "target_cluster_sizes":np.bincount(clusters.labels_,minlength=classes).tolist(),
            "target_instance_labels_used":False,"validation_used_for_fitting":False}
        return cls(clusters.cluster_centers_[reverse],reliability,temperature,metadata)

    def forward(self,features):
        return (2*features@self.centers.T-self.centers.square().sum(1))/self.temperature

    def initialize_head(self,head):
        if not isinstance(head,nn.Linear) or tuple(head.weight.shape)!=tuple(self.centers.shape):
            raise ValueError("Cluster initialization requires the declared linear method classifier")
        with torch.no_grad():
            head.weight.copy_(2*self.centers/self.temperature)
            head.bias.copy_(-self.centers.square().sum(1)/self.temperature)


class ResidualFeatureAdapter(nn.Module):
    def __init__(self,dimension,scale=.1):
        super().__init__()
        self.delta=nn.Linear(dimension,dimension,bias=False)
        nn.init.zeros_(self.delta.weight)
        self.scale=float(scale)

    def forward(self,features):return features+self.scale*self.delta(features)
