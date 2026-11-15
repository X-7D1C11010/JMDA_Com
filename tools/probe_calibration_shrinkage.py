"""Small declared shrinkage grid; source-only class association, no oracle."""
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
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/frozen_feature_probe"
    rows=[]
    for target in ("rainy","foggy","dark"):
        with np.load(base/f"{target}_features.npz",allow_pickle=False) as f:
            source,train,val=[f[k].copy() for k in ("source","target_train","target_val")]
            sy,vy=f["source_labels"].copy(),f["target_val_labels"].copy()
        classes=int(sy.max())+1;half=source.shape[1]//2
        for ridge in (.01,.1,1.):
            parts=[]
            for part in (slice(None,half),slice(half,None)):
                a,b=unit(source[:,part]),unit(train[:,part])
                c=TrainFeatureCalibration.fit(a,b,32,ridge)
                with torch.no_grad():parts.append((c(torch.tensor(a,dtype=torch.float32),"source").numpy(),c(torch.tensor(b,dtype=torch.float32),"target").numpy()))
            sa=unit(np.concatenate((.75*parts[0][0],.25*parts[1][0]),1))
            ta=unit(np.concatenate((.75*parts[0][1],.25*parts[1][1]),1))
            for weight in (.94,.75):
                raw=[torch.tensor(fused_features(x,weight),dtype=torch.float32) for x in (source,train,val)]
                for components in (8,16,32,64):
                    cal=TrainFeatureCalibration.fit(raw[0].numpy(),raw[1].numpy(),components,ridge)
                    with torch.no_grad():a,b,v=[cal(x,d) for x,d in zip(raw,("source","target","target"))]
                    for auxiliary in (False,True):
                        teacher=ClusterInitialization.fit(a.numpy(),b.numpy(),sy,classes,
                            association_source=sa if auxiliary else None,association_target=ta if auxiliary else None)
                        with torch.no_grad():prediction=teacher(v).argmax(1).tolist()
                        rows.append({"target":target,"VIS_weight":weight,"components":components,"shrinkage":ridge,
                            "crossmodal_association":auxiliary,"association_VIS_weight":.75,
                            "metrics":classification_metrics(vy.tolist(),prediction,classes)})
        best=max((r for r in rows if r["target"]==target),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        print("SHRINKAGE_PREVIEW",json.dumps(best),flush=True)
        args.output.write_text(json.dumps({"formal_DA_method_result":False,"target_training_labels_used":False,
            "validation_statistics_fitted":False,"source_only_cluster_label_mapping":True,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
