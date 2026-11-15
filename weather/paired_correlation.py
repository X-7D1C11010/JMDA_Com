"""Training-only regularized two-view correlation projection.

This is an explicit enhancement, not any of the five released DA objectives.
The covariance-whitening/SVD construction follows ordinary regularized CCA:
https://github.com/gallantlab/pyrcca . The optional balanced source between-
class covariance is our diagnostic variant, not an implementation of Pyrcca.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from weather.statistical_calibration import unit


@dataclass
class PairedCorrelation:
    source_means: tuple
    target_means: tuple
    projections: tuple
    correlations: np.ndarray
    metadata: dict

    @classmethod
    def fit(cls,source_vis,source_ir,target_vis,target_ir,source_labels,
            regularization=.1,target_mixture=.5,source_covariance="paired"):
        sv,si,tv,ti=[np.asarray(x,dtype=np.float64) for x in (source_vis,source_ir,target_vis,target_ir)]
        sy=np.asarray(source_labels,dtype=np.int64)
        if (any(x.ndim!=2 or not np.isfinite(x).all() for x in (sv,si,tv,ti))
                or len(sv)!=len(si) or len(tv)!=len(ti) or len(sv)!=len(sy)
                or sv.shape[1]!=tv.shape[1] or si.shape[1]!=ti.shape[1]
                or min(len(sv),len(tv))<2 or regularization<=0 or not 0<=target_mixture<=1
                or source_covariance not in {"paired","class_between"}):
            raise ValueError("Invalid paired training arrays or covariance settings")
        classes=int(sy.max())+1
        if classes<2 or set(sy)!=set(range(classes)):
            raise ValueError("Expected contiguous supported source classes")
        counts=np.bincount(sy,minlength=classes)
        weights=1/(classes*counts[sy])
        sm=tuple((weights[:,None]*x).sum(0) for x in (sv,si))
        tm=tuple(x.mean(0) for x in (tv,ti))
        sx,sy_view=sv-sm[0],si-sm[1]
        tx,ty=tv-tm[0],ti-tm[1]
        def combine(source,target):return (1-target_mixture)*source+target_mixture*target
        cxx=combine(sx.T@(weights[:,None]*sx),tx.T@tx/len(tx))
        cyy=combine(sy_view.T@(weights[:,None]*sy_view),ty.T@ty/len(ty))
        if source_covariance=="paired":sxy=sx.T@(weights[:,None]*sy_view)
        else:
            a=np.stack([sx[sy==c].mean(0) for c in range(classes)])
            b=np.stack([sy_view[sy==c].mean(0) for c in range(classes)])
            sxy=a.T@b/classes
        cxy=combine(sxy,tx.T@ty/len(tx))
        def whitening(covariance):
            ridge=max(float(np.trace(covariance))/len(covariance)*regularization,1e-8)
            values,vectors=np.linalg.eigh(covariance)
            return (vectors/np.sqrt(np.maximum(values,0)+ridge))@vectors.T
        wx,wy=whitening(cxx),whitening(cyy)
        u,correlation,vt=np.linalg.svd(wx@cxy@wy,full_matrices=False)
        projections=(wx@u,wy@vt.T)
        metadata={"fit_partitions":["source_train","target_train"],
            "fit_samples":{"source_train":len(sv),"target_train":len(tv)},
            "source_covariance":source_covariance,"regularization":regularization,"target_mixture":target_mixture,
            "source_label_usage":"source class balancing and optional between-class covariance",
            "target_training_labels_used":False,"validation_used_for_fit":False,
            "canonical_correlations":correlation.tolist()}
        digest=hashlib.sha256()
        for x in (*sm,*tm,*projections,correlation):digest.update(x.tobytes())
        metadata["fit_sha256"]=digest.hexdigest()
        return cls(sm,tm,projections,correlation,metadata)

    def transform(self,vis,ir,domain,components,visible_weight=.5,correlation_exponent=0.):
        if (domain not in {"source","target"} or not 1<=components<=len(self.correlations)
                or not 0<visible_weight<1 or correlation_exponent<0):
            raise ValueError("Invalid projection domain, width or fusion settings")
        means=self.source_means if domain=="source" else self.target_means
        scale=np.maximum(self.correlations[:components],1e-8)**correlation_exponent
        a=(np.asarray(vis)-means[0])@self.projections[0][:,:components]*scale
        b=(np.asarray(ir)-means[1])@self.projections[1][:,:components]*scale
        return unit(np.concatenate((visible_weight*unit(a),(1-visible_weight)*unit(b)),1))
