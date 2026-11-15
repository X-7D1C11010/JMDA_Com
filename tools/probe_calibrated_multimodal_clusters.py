"""Preview the authorized enhancement, without attributing scores to DA methods."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from weather.statistical_calibration import TrainFeatureCalibration,ClusterInitialization,fused_features
from tools.classification_metrics import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--targets",nargs="+",default=["rainy","foggy","dark"])
    p.add_argument("--feature-packs",nargs="+",default=["resnet18","convnext_small"],
        choices=("resnet18","convnext_small","gem3","ibn_resnet50","convnext_letterbox","thermal_inverted"))
    p.add_argument("--strategies",nargs="+",default=["coral"],choices=("coral","center","pca"))
    p.add_argument("--feature-cache",type=Path,default=None)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    rows=[]
    packs={"resnet18":"frozen_feature_probe","convnext_small":"convnext_small_feature_probe",
        "gem3":"gem3_feature_probe","ibn_resnet50":"ibn_resnet50_feature_probe",
        "convnext_letterbox":"convnext_letterbox_feature_probe","thermal_inverted":"thermal_polarity_probe"}
    if args.feature_cache:
        if len(args.targets)!=1:p.error("A selected feature cache must have exactly one declared target")
        metadata=json.loads(args.feature_cache.with_suffix(".json").read_text(encoding="utf-8"))
        parent=json.loads((args.feature_cache.parent/"result.json").read_text(encoding="utf-8"))
        if (parent.get("smoke") or parent["target"]!=args.targets[0] or parent["selected_epoch"]!=metadata["epoch"]
                or hashlib.sha256(args.feature_cache.read_bytes()).hexdigest()!=metadata["sha256"]):
            raise ValueError("Selected feature cache provenance mismatch")
        args.feature_packs=["selected_trained_encoder"]
    for backbone in args.feature_packs:
        folder=packs.get(backbone)
        for target in args.targets:
            file=args.feature_cache or base/folder/f"{target}_features.npz"
            if not file.exists():continue
            with np.load(file,allow_pickle=False) as data:
                if "target_train_labels" in data.files:raise RuntimeError("Target training labels in feature cache")
                source,train,val=[data[k].copy() for k in ("source","target_train","target_val")]
                sy,vy=data["source_labels"].copy(),data["target_val_labels"].copy()
            width=metadata["modality_widths"][0] if args.feature_cache else None
            classes=int(sy.max())+1
            for weight in (.94,.75,.5,.25,.06):
                sa,ta,va=[torch.tensor(fused_features(x,weight,width),dtype=torch.float32) for x in (source,train,val)]
                for components in (16,32,64):
                    for strategy in args.strategies:
                        calibration=TrainFeatureCalibration.fit(sa.numpy(),ta.numpy(),components,strategy=strategy)
                        with torch.no_grad():
                            s=calibration(sa,"source");t=calibration(ta,"target");v=calibration(va,"target")
                        for association in ("source_votes","source_prototypes"):
                            teacher=ClusterInitialization.fit(s.numpy(),t.numpy(),sy,classes,matching=association)
                            with torch.no_grad():pred=teacher(v).argmax(1).tolist();sp=teacher(s).argmax(1)
                            metrics=classification_metrics(vy.tolist(),pred,classes)
                            rows.append({"backbone":backbone,"target":target,"VIS_weight":weight,"components":components,
                                "strategy":strategy,"association":association,"metrics":metrics,"source_initialization_ACC":float((sp==torch.tensor(sy)).float().mean()),
                                "reliability_by_class":teacher.reliability.tolist(),"source_mapping":teacher.metadata["cluster_to_source_class"]})
            best=max((x for x in rows if x["target"]==target and x["backbone"]==backbone),key=lambda x:(x["metrics"]["accuracy"],x["metrics"]["f1_macro"]))
            print("CALIBRATED_CLUSTER_PREVIEW",json.dumps(best),flush=True)
            args.output.write_text(json.dumps({"kind":"authorized_enhancement_preview_not_DA_method_training",
                "formal_UDA_method_results":False,"target_training_labels_used":False,"validation_statistics_fitted":False,
                "validation_labels_used_only_for_scoring":True,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
