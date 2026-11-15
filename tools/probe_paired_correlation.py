"""Training-only paired CCA screening; never count this as a DA run."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
from tools.classification_metrics import classification_metrics
from weather.paired_correlation import PairedCorrelation
from weather.statistical_calibration import TrainFeatureCalibration,fused_features,unit


BASE=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"


def load_pack(name,target):
    if name=="convnext_unip":
        path=ROOT/"runs/weather_local_screen/goal3h_unip_tiny_seed42"/f"{target}_features.npz"
    else:
        path=BASE/({"resnet18":"frozen_feature_probe","convnext_small":"convnext_small_feature_probe"}[name])/f"{target}_features.npz"
    with np.load(path,allow_pickle=False) as z:
        if "target_train_labels" in z.files:raise ValueError("Target training labels in cache")
        if name=="convnext_unip":
            values=[(z[k+"_VIS"].copy(),z[k+"_IR_patch_mean"].copy()) for k in ("source","target_train","target_val")]
        else:
            arrays=[z[k].copy() for k in ("source","target_train","target_val")]
            half=arrays[0].shape[1]//2
            values=[(x[:,:half],x[:,half:]) for x in arrays]
        return values,z["source_labels"].copy(),z["target_val_labels"].copy()


def associate(source,target,source_labels,training_clusters,matching):
    classes=int(source_labels.max())+1
    centers=np.stack([target[training_clusters==c].mean(0) for c in range(classes)])
    if matching=="source_votes":
        nearest=((source[:,None]-centers[None])**2).sum(2).argmin(1)
        score=np.zeros((classes,classes));np.add.at(score,(nearest,source_labels),1)
        score/=np.maximum(score.sum(0,keepdims=True),1)
    else:
        prototypes=unit(np.stack([source[source_labels==c].mean(0) for c in range(classes)]))
        score=unit(centers)@prototypes.T
    rows,columns=linear_sum_assignment(-score)
    mapping=np.empty(classes,dtype=np.int64);mapping[rows]=columns
    return mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--targets",nargs="+",default=["rainy","foggy","dark"])
    parser.add_argument("--packs",nargs="+",choices=("resnet18","convnext_small","convnext_unip"),
                        default=["resnet18","convnext_small","convnext_unip"])
    parser.add_argument("--precomponents",type=int,default=64)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    rows=[]
    for name in args.packs:
        for target_name in args.targets:
            raw,labels,val_labels=load_pack(name,target_name);classes=int(labels.max())+1
            primary,check_source,check_val=load_pack("resnet18",target_name)
            if not np.array_equal(labels,check_source) or not np.array_equal(val_labels,check_val):
                raise ValueError("Feature cohort order mismatch")
            main=[torch.tensor(fused_features(np.concatenate(x,1),.94),dtype=torch.float32) for x in primary]
            cal=TrainFeatureCalibration.fit(main[0].numpy(),main[1].numpy(),16)
            with torch.no_grad():primary_target=cal(main[1],"target").numpy();primary_val=cal(main[2],"target").numpy()
            with threadpool_limits(limits=2):km=KMeans(n_clusters=classes,n_init=10,random_state=42).fit(primary_target)
            primary_prediction=km.predict(primary_val)
            calibrated=[[],[],[]]
            for modality in (0,1):
                a,b,v=[unit(x[modality]).astype(np.float32) for x in raw]
                c=TrainFeatureCalibration.fit(a,b,args.precomponents)
                with torch.no_grad():parts=[c(torch.tensor(x),domain).numpy() for x,domain in ((a,"source"),(b,"target"),(v,"target"))]
                for partition,features in zip(calibrated,parts):partition.append(features)
            def score(arrays,settings):
                source,train,val=arrays
                with threadpool_limits(limits=2):
                    clf=LogisticRegression(C=1,class_weight="balanced",max_iter=1000).fit(source,labels)
                prediction=clf.predict(val)
                rows.append({**settings,"evaluator":"source_only_C1_logistic",
                    "metrics":classification_metrics(val_labels.tolist(),prediction.tolist(),classes),
                    "seen_source_fit_ACC":float(np.mean(clf.predict(source)==labels))})
                with threadpool_limits(limits=2):
                    lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage=.1,priors=np.full(classes,1/classes)).fit(source,labels)
                for classifier_name,classifier in (("source_C1_logit",clf),("source_LDA_logit",lda)):
                    training_logits=classifier.decision_function(train)
                    source_scale=np.maximum(classifier.decision_function(source).std(0),1e-6)
                    for standardized in (False,True):
                        score_logits=training_logits/source_scale if standardized else training_logits
                        cluster_scores=np.stack([score_logits[km.labels_==c].mean(0) for c in range(classes)])
                        ii,jj=linear_sum_assignment(-cluster_scores)
                        mapping=np.empty(classes,dtype=np.int64);mapping[ii]=jj
                        rows.append({**settings,"evaluator":"fixed_R18_geometry_"+classifier_name,
                            "source_score_standardized":standardized,"mapping":mapping.tolist(),
                            "metrics":classification_metrics(val_labels.tolist(),mapping[primary_prediction].tolist(),classes)})
                for matching in ("source_votes","source_prototypes"):
                    mapping=associate(source,train,labels,km.labels_,matching)
                    rows.append({**settings,"evaluator":"fixed_R18_geometry_"+matching,"mapping":mapping.tolist(),
                        "metrics":classification_metrics(val_labels.tolist(),mapping[primary_prediction].tolist(),classes)})
            for weight in (.75,.5,.25):
                arrays=[unit(np.concatenate((weight*unit(a[0]),(1-weight)*unit(a[1])),1)) for a in calibrated]
                score(arrays,{"pack":name,"target":target_name,"projection":"no_CCA_reference","VIS_weight":weight})
            for regularization in (.01,.1):
                for covariance in ("paired","class_between"):
                    with threadpool_limits(limits=2):
                        cca=PairedCorrelation.fit(*calibrated[0],*calibrated[1],labels,
                            regularization=regularization,source_covariance=covariance)
                    for components in (classes-1,16):
                        for weight in (.75,.5,.25):
                            for exponent in (0.,1.):
                                arrays=[cca.transform(*a,domain,components,weight,exponent)
                                    for a,domain in zip(calibrated,("source","target","target"))]
                                score(arrays,{"pack":name,"target":target_name,"projection":"paired_CCA",
                                    "regularization":regularization,"source_covariance":covariance,"components":components,
                                    "VIS_weight":weight,"correlation_exponent":exponent,"fit_sha256":cca.metadata["fit_sha256"],
                                    "retained_correlations":cca.correlations[:components].tolist()})
            best=max((r for r in rows if r["pack"]==name and r["target"]==target_name),
                     key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
            print(json.dumps({"pack":name,"target":target_name,"best":best}),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps({"kind":"paired_correlation_training_statistics_diagnostic",
                "formal_DA_method_result":False,"target_training_labels_used":False,"validation_used_for_fitting":False,
                "validation_labels_used_for_hyperparameter_scoring":True,"fusion":"two modality features concatenated",
                "precalibration_components":args.precomponents,
                "reference":"https://github.com/gallantlab/pyrcca","results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
