"""Source-supervised projection diagnostic; target/validation labels never fit."""
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
from weather.statistical_calibration import TrainFeatureCalibration,ClusterInitialization,fused_features
from tools.classification_metrics import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    rows=[]
    with threadpool_limits(limits=2):
        for backbone,folder in (("resnet18","frozen_feature_probe"),("convnext_small","convnext_small_feature_probe")):
            for target in ("rainy","foggy","dark"):
                with np.load(base/folder/f"{target}_features.npz",allow_pickle=False) as f:
                    source,train,val=[f[k].copy() for k in ("source","target_train","target_val")]
                    sy,vy=f["source_labels"].copy(),f["target_val_labels"].copy()
                classes=int(sy.max())+1
                for weight in (.94,.75,.5,.25,.06):
                    arrays=[fused_features(x,weight) for x in (source,train,val)]
                    pca=PCA(n_components=64,svd_solver="full").fit(np.concatenate(arrays[:2]))
                    a,b,v=[pca.transform(x) for x in arrays]
                    for shrinkage in (.1,1.):
                        lda=LinearDiscriminantAnalysis(solver="eigen",shrinkage=shrinkage).fit(a,sy)
                        semantic=[lda.transform(x) for x in (a,b,v)]
                        logits=[lda.decision_function(x) for x in (a,b,v)]
                        mean_s,std_s=logits[0].mean(0),np.maximum(logits[0].std(0),1e-6)
                        mean_t,std_t=logits[1].mean(0),np.maximum(logits[1].std(0),1e-6)
                        for strength in (0.,.5,1.):
                            corrected=(logits[2]-strength*mean_t-(1-strength)*mean_s)/(strength*std_t+(1-strength)*std_s)
                            pred=corrected.argmax(1)
                            rows.append({"backbone":backbone,"target":target,"VIS_weight":weight,"source_LDA_shrinkage":shrinkage,
                                "strategy":"class_score_calibration","target_statistics_strength":strength,
                                "metrics":classification_metrics(vy.tolist(),pred.tolist(),classes)})
                        for strategy in ("center","coral"):
                            cal=TrainFeatureCalibration.fit(semantic[0],semantic[1],classes-1,strategy=strategy)
                            with torch.no_grad():s,t,z=[cal(torch.tensor(x,dtype=torch.float32),d) for x,d in zip(semantic,("source","target","target"))]
                            for matching in ("source_votes","source_prototypes"):
                                teacher=ClusterInitialization.fit(s.numpy(),t.numpy(),sy,classes,matching=matching)
                                with torch.no_grad():pred=teacher(z).argmax(1).tolist()
                                rows.append({"backbone":backbone,"target":target,"VIS_weight":weight,"source_LDA_shrinkage":shrinkage,
                                    "strategy":"source_semantic_"+strategy,"matching":matching,
                                    "metrics":classification_metrics(vy.tolist(),pred,classes)})
                best=max((x for x in rows if x["target"]==target and x["backbone"]==backbone),key=lambda x:(x["metrics"]["accuracy"],x["metrics"]["f1_macro"]))
                print("SOURCE_SEMANTIC_PREVIEW",json.dumps(best),flush=True)
                args.output.write_text(json.dumps({"formal_DA_method_result":False,"target_training_labels_used":False,
                    "supervised_projection_label_source":"source_train only","validation_statistics_fitted":False,
                    "results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
