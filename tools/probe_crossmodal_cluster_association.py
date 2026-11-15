"""Pair-aware source association; all mappings fit before validation scoring."""
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
from weather.statistical_calibration import TrainFeatureCalibration,ClusterInitialization,fused_features,unit
from tools.classification_metrics import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--targets",nargs="+",default=["rainy","foggy","dark"])
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/frozen_feature_probe"
    rows=[]
    for target in args.targets:
        with np.load(base/f"{target}_features.npz",allow_pickle=False) as f:
            source,train,val=[f[k].copy() for k in ("source","target_train","target_val")]
            sy,vy=f["source_labels"].copy(),f["target_val_labels"].copy()
        classes=int(sy.max())+1;half=source.shape[1]//2
        associated=[]
        for part in (slice(None,half),slice(half,None)):
            a,b=torch.tensor(unit(source[:,part]),dtype=torch.float32),torch.tensor(unit(train[:,part]),dtype=torch.float32)
            c=TrainFeatureCalibration.fit(a.numpy(),b.numpy(),32)
            with torch.no_grad():associated.append((c(a,"source").numpy(),c(b,"target").numpy()))
        for weight in (.94,.75,.5,.25,.06):
            arrays=[torch.tensor(fused_features(x,weight),dtype=torch.float32) for x in (source,train,val)]
            for components in (16,32,64):
                cal=TrainFeatureCalibration.fit(arrays[0].numpy(),arrays[1].numpy(),components)
                with torch.no_grad():a,b,v=[cal(x,d) for x,d in zip(arrays,("source","target","target"))]
                for association_weight in (0.,.25,.5,.75,1.):
                    sa=unit(np.concatenate((association_weight*associated[0][0],(1-association_weight)*associated[1][0]),1))
                    ta=unit(np.concatenate((association_weight*associated[0][1],(1-association_weight)*associated[1][1]),1))
                    for matching in ("source_votes","source_prototypes","source_neighbors"):
                        teacher=ClusterInitialization.fit(a.numpy(),b.numpy(),sy,classes,matching=matching,
                                                          association_source=sa,association_target=ta)
                        with torch.no_grad():prediction=teacher(v).argmax(1).tolist()
                        rows.append({"target":target,"VIS_weight":weight,"components":components,
                            "association_VIS_weight":association_weight,"matching":matching,
                            "metrics":classification_metrics(vy.tolist(),prediction,classes),
                            "source_mapping":teacher.metadata["cluster_to_source_class"]})
        best=max((r for r in rows if r["target"]==target),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        print("CROSSMODAL_ASSOCIATION_PREVIEW",json.dumps(best),flush=True)
        args.output.write_text(json.dumps({"formal_DA_method_result":False,"target_training_labels_used":False,
            "validation_used_to_fit_mapping":False,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
