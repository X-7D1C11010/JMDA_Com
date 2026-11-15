"""Cross-fitted domain-importance diagnostic; all source instances remain used."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from weather.statistical_calibration import TrainFeatureCalibration,fused_features,unit
from tools.classification_metrics import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--targets",nargs="+",default=["rainy"]);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);base=ROOT/"runs/weather_local_screen";rows=[]
    packs=[("R18",base/"balanced_v4_seed42_20261114_005133/frozen_feature_probe"),
           ("ConvNeXt_UNIPtiny",base/"goal3h_unip_tiny_seed42")]
    for weather in args.targets:
        for backbone,folder in packs:
            with np.load(folder/f"{weather}_features.npz",allow_pickle=False) as cache:
                labels=cache["source_labels"].copy();truth=cache["target_val_labels"].copy()
                if backbone=="R18":raw=[cache[k].copy() for k in ("source","target_train","target_val")];width=512
                else:
                    raw=[np.concatenate((cache[k+"_VIS"],cache[k+"_IR_patch_mean"]),1) for k in ("source","target_train","target_val")];width=raw[0].shape[1]-cache["source_IR_patch_mean"].shape[1]
            classes=int(labels.max())+1
            for visible_weight in (.94,.5,.06):
                source,target,val=[fused_features(x,visible_weight,width) for x in raw]
                pooled=np.concatenate((source,target));domain=np.r_[np.zeros(len(source)),np.ones(len(target))]
                with threadpool_limits(limits=2):
                    posterior=cross_val_predict(LogisticRegression(C=10,class_weight="balanced",max_iter=500,random_state=42),
                        pooled,domain,method="predict_proba",cv=StratifiedKFold(4,shuffle=True,random_state=42),n_jobs=1)[:,1]
                auc=float(roc_auc_score(domain,posterior));ratio=np.clip(posterior[:len(source)]/np.maximum(1-posterior[:len(source)],1e-5),.05,20)
                cal=TrainFeatureCalibration.fit(source,target,16)
                with torch.no_grad():a,b,v=[cal(torch.tensor(x,dtype=torch.float32),d).numpy() for x,d in zip((source,target,val),("source","target","target"))]
                with threadpool_limits(limits=2):km=KMeans(n_clusters=classes,n_init=10,random_state=42).fit(b)
                for power in (0.,.5,1.):
                    weights=ratio**power
                    for c in range(classes):weights[labels==c]/=weights[labels==c].mean()
                    common={"target":weather,"backbone":backbone,"VIS_weight":visible_weight,"importance_power":power,
                        "domain_crossfit_AUC":auc,"source_effective_samples":float(weights.sum()**2/(weights**2).sum()),
                        "source_samples_retained":len(weights),"source_weight_min":float(weights.min()),"source_weight_max":float(weights.max())}
                    with threadpool_limits(limits=2):classifier=LogisticRegression(C=10,class_weight="balanced",max_iter=700,random_state=42).fit(source,labels,sample_weight=weights)
                    rows.append({**common,"classifier":"weighted_source_logistic","metrics":classification_metrics(truth.tolist(),classifier.predict(val).tolist(),classes)})
                    source_clusters=km.predict(a);votes=np.zeros((classes,classes));np.add.at(votes,(source_clusters,labels),weights)
                    balanced=votes/np.maximum(votes.sum(0,keepdims=True),1e-12)
                    prototypes=unit(np.stack([np.average(a[labels==c],axis=0,weights=weights[labels==c]) for c in range(classes)]))
                    for matching,score in (("weighted_source_votes",balanced),("weighted_source_prototypes",unit(km.cluster_centers_)@prototypes.T)):
                        rr,cc=linear_sum_assignment(-score);mapping=np.empty(classes,dtype=np.int64);mapping[rr]=cc
                        prediction=mapping[km.predict(v)]
                        rows.append({**common,"classifier":matching,"source_mapping":mapping.tolist(),"metrics":classification_metrics(truth.tolist(),prediction.tolist(),classes)})
            best=max((r for r in rows if r["target"]==weather and r["backbone"]==backbone),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
            print("SOURCE_DOMAIN_IMPORTANCE_DIAGNOSTIC",json.dumps(best),flush=True)
            args.output.write_text(json.dumps({"formal_DA_method_result":False,"target_training_labels_used":False,
                "validation_used_for_domain_fitting_or_source_weights":False,"source_class_weights_normalized_within_class":True,
                "all_source_samples_have_positive_weight":True,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
