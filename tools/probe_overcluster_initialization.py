"""Diagnostic of multiple target prototypes per class, using source labels only."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from weather.statistical_calibration import TrainFeatureCalibration,fused_features,unit
from tools.classification_metrics import classification_metrics


def fit_mapping(source,target,cluster_ids,source_labels,classes,matching):
    clusters=int(cluster_ids.max())+1
    centers=np.stack([target[cluster_ids==i].mean(0) for i in range(clusters)])
    distances=((source[:,None]-centers[None])**2).sum(2)
    votes=np.zeros((clusters,classes));np.add.at(votes,(distances.argmin(1),source_labels),1)
    balanced=votes/np.maximum(votes.sum(0,keepdims=True),1)
    if matching=="source_votes":score=balanced
    else:
        prototypes=unit(np.stack([source[source_labels==c].mean(0) for c in range(classes)]))
        score=unit(centers)@prototypes.T
    if clusters==classes:
        rows,columns=linear_sum_assignment(-score)
        mapping=np.empty(clusters,dtype=np.int64);mapping[rows]=columns
    else:
        # At least one distinct anchor per known class; other clusters may
        # share a class, permitting multiple poses/backgrounds for that class.
        mapping=score.argmax(1)
        rows,columns=linear_sum_assignment(-score.T)
        mapping[columns]=rows
    return mapping,votes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--targets",nargs="+",default=["rainy","foggy","dark"])
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/frozen_feature_probe"
    rows=[]
    for target in args.targets:
        with np.load(base/f"{target}_features.npz",allow_pickle=False) as data:
            source,train,validation=[data[key].copy() for key in ("source","target_train","target_val")]
            sy,vy=data["source_labels"].copy(),data["target_val_labels"].copy()
        classes=int(sy.max())+1
        auxiliary=[]
        for part in (slice(None,512),slice(512,None)):
            a,b=unit(source[:,part]),unit(train[:,part]);cal=TrainFeatureCalibration.fit(a,b,32)
            with torch.no_grad():auxiliary.append((cal(torch.tensor(a,dtype=torch.float32),"source").numpy(),cal(torch.tensor(b,dtype=torch.float32),"target").numpy()))
        ca=unit(np.concatenate((.75*auxiliary[0][0],.25*auxiliary[1][0]),1))
        cb=unit(np.concatenate((.75*auxiliary[0][1],.25*auxiliary[1][1]),1))
        for weight in (.94,.5,.06):
            arrays=[torch.tensor(fused_features(x,weight),dtype=torch.float32) for x in (source,train,validation)]
            for components in (16,32):
                cal=TrainFeatureCalibration.fit(arrays[0].numpy(),arrays[1].numpy(),components)
                with torch.no_grad():a,b,v=[cal(x,d).numpy() for x,d in zip(arrays,("source","target","target"))]
                for multiplier in (1,2,3):
                    with threadpool_limits(limits=2):km=KMeans(n_clusters=classes*multiplier,n_init=10,random_state=42).fit(b)
                    for name,sa,tb in (("primary",a,b),("paired_modalities",ca,cb)):
                        for matching in ("source_votes","source_prototypes"):
                            mapping,votes=fit_mapping(sa,tb,km.labels_,sy,classes,matching)
                            sizes=np.bincount(km.labels_,minlength=len(km.cluster_centers_))
                            candidates={"nearest_subcluster":mapping[km.predict(v)]}
                            for merge in ("sample_weighted","uniform_clusters"):
                                centers=np.stack([np.average(km.cluster_centers_[mapping==c],axis=0,
                                    weights=sizes[mapping==c] if merge=="sample_weighted" else None) for c in range(classes)])
                                candidates[merge]=((v[:,None]-centers[None])**2).sum(2).argmin(1)
                            cluster_logits=-((v[:,None]-km.cluster_centers_[None])**2).sum(2)/.1
                            mixture=np.stack([logsumexp(cluster_logits[:,mapping==c],axis=1)-np.log((mapping==c).sum()) for c in range(classes)],1)
                            candidates["equal_class_mixture"]=mixture.argmax(1)
                            for merge,prediction in candidates.items():
                                rows.append({"target":target,"VIS_weight":weight,"components":components,"cluster_multiplier":multiplier,
                                    "association":name,"matching":matching,"classifier":merge,"source_mapping":mapping.tolist(),
                                    "target_cluster_sizes":sizes.tolist(),"metrics":classification_metrics(vy.tolist(),prediction.tolist(),classes)})
        best=max((r for r in rows if r["target"]==target),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        linear=max((r for r in rows if r["target"]==target and r["classifier"] in {"sample_weighted","uniform_clusters"}),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        print("OVERCLUSTER_DIAGNOSTIC",json.dumps({"best":best,"linear_compatible_best":linear}),flush=True)
        args.output.write_text(json.dumps({"formal_DA_method_result":False,"seed":42,"target_training_labels_used":False,
            "validation_used_for_fit_or_class_assignment":False,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
