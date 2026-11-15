"""Source-label-only cluster association diagnostics; no oracle label matching."""
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
from weather.statistical_calibration import TrainFeatureCalibration,ClusterInitialization,fused_features
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
        classes=int(sy.max())+1
        for weight in (.94,.9,.75,.5,.25,.06):
            arrays=[torch.tensor(fused_features(x,weight),dtype=torch.float32) for x in (source,train,val)]
            for components in (8,16,32,64):
                cal=TrainFeatureCalibration.fit(arrays[0].numpy(),arrays[1].numpy(),components)
                with torch.no_grad():a,b,v=[cal(x,domain) for x,domain in zip(arrays,("source","target","target"))]
                for strategy,n in (("source_votes",3),("source_prototypes",3),("source_neighbors",1),
                                   ("source_neighbors",3),("source_neighbors",5),("source_trimmed",3)):
                    teacher=ClusterInitialization.fit(a.numpy(),b.numpy(),sy,classes,matching=strategy,neighbors=n)
                    with torch.no_grad():prediction=teacher(v).argmax(1).tolist()
                    score=classification_metrics(vy.tolist(),prediction,classes)
                    rows.append({"target":target,"VIS_weight":weight,"components":components,"association":strategy,
                                 "neighbors":n,"metrics":score,"source_mapping":teacher.metadata["cluster_to_source_class"]})
        best=max((x for x in rows if x["target"]==target),key=lambda x:(x["metrics"]["accuracy"],x["metrics"]["f1_macro"]))
        print("SOURCE_ASSOCIATION_PREVIEW",json.dumps(best),flush=True)
        args.output.write_text(json.dumps({"formal_DA_method_training_result":False,"target_train_labels_used":False,
            "validation_used_for_fit_or_mapping":False,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
