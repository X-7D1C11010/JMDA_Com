"""Test existing correctly normalized IR/token caches with training-only calibration."""
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
    p.add_argument("--base-only",action="store_true")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    base=ROOT/"runs/weather_local_screen";rows=[]
    packs=[("UNIP-tiny",base/"goal3h_unip_tiny_seed42",("patch_mean","cls","qk_mean","last4")),
           ("UNIP-small",base/"goal3h_unip_small_seed42",("patch_mean","cls","qk_mean","last4")),
           ("DINO-small-correct-IR",base/"goal3h_dino_tokens_seed42",("cls","mean","attention_mean","last4_mean"))]
    if args.base_only:
        packs=[("DINO-base-correct-IR",base/"dinov2_base_tokens_correctIR_seed42",("cls","mean","attention_mean","last4_mean"))]
    for target in ("rainy","foggy","dark"):
        for backbone,folder,modes in packs:
            with np.load(folder/f"{target}_features.npz",allow_pickle=False) as f:
                sy,vy=f["source_labels"].copy(),f["target_val_labels"].copy();classes=int(sy.max())+1
                arrays_by_mode=[]
                for mode in modes:
                    arrays=[]
                    for partition in ("source","target_train","target_val"):
                        if backbone.startswith("UNIP"):
                            vis,ir=f[partition+"_VIS"],f[partition+"_IR_"+mode]
                            width=vis.shape[1];arrays.append(np.concatenate((vis,ir),1))
                        else:
                            arrays.append(f[partition+"_"+mode].copy());width=arrays[-1].shape[1]//2
                    arrays_by_mode.append((mode,width,arrays))
            for mode,width,arrays in arrays_by_mode:
                for weight in (.94,.5,.06):
                    source,train,val=[torch.tensor(fused_features(x,weight,width),dtype=torch.float32) for x in arrays]
                    for components in (16,32):
                        cal=TrainFeatureCalibration.fit(source.numpy(),train.numpy(),components)
                        with torch.no_grad():a,b,v=[cal(x,d) for x,d in zip((source,train,val),("source","target","target"))]
                        for association in ("source_votes","source_prototypes"):
                            teacher=ClusterInitialization.fit(a.numpy(),b.numpy(),sy,classes,matching=association)
                            with torch.no_grad():pred=teacher(v).argmax(1).tolist()
                            rows.append({"target":target,"VIS_backbone":"ConvNeXt-small" if backbone.startswith("UNIP") else backbone,
                                "IR_backbone":backbone,"IR_feature_mode":mode,"VIS_weight":weight,"components":components,
                                "association":association,"metrics":classification_metrics(vy.tolist(),pred,classes)})
            best=max((r for r in rows if r["target"]==target and r["IR_backbone"]==backbone),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
            print("CACHED_IR_CALIBRATION_PREVIEW",json.dumps(best),flush=True)
            args.output.write_text(json.dumps({"formal_DA_method_result":False,"target_training_labels_used":False,
                "validation_used_to_fit_or_associate":False,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
