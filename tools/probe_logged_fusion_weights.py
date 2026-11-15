"""Exact linear-head fusion diagnostic from selected-epoch ablation logits.

Only applies to unit-normalized concatenation with an identity bottleneck and
linear task head. This is a validation hyperparameter diagnostic, not another
completed training run. No per-instance or per-class weight is selected.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.classification_metrics import classification_metrics


def read_rows(path,epoch):
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sorted((r for r in rows if r["partition"]=="target_val" and r["epoch"]==epoch),key=lambda r:r["sample_index"])


def analyze(path):
    result=json.loads(path.read_text(encoding="utf-8"))
    if (result.get("classifier_layout")!="linear" or result.get("feature_normalization")!="unit" or
            result.get("unit_method_features") or (result.get("enhancement")or{}).get("calibration")):
        return None
    epoch=result["selected_epoch"];base=path.parent/"modality_diagnostics"
    files=[base/name/"predictions.jsonl" for name in ("full","zero_IR","zero_VIS")]
    if not all(p.exists() for p in files):return None
    rows=[read_rows(p,epoch) for p in files]
    if not rows[0]:return None
    keys=[[(r["sample_id"],r["y_true"]) for r in part] for part in rows]
    if keys[0]!=keys[1] or keys[0]!=keys[2]:raise ValueError("Ablation cohort mismatch")
    full,visible,infrared=[np.asarray([r["logits"] for r in part]) for part in rows]
    biases=visible+infrared-full
    deviation=float(np.max(np.abs(biases-biases.mean(0))))
    if deviation>1e-4:raise ValueError(f"Head is not affine as declared: {path}, bias deviation {deviation}")
    bias=biases.mean(0)
    weights=np.asarray(result["modality_weights_at_selected_epoch"])
    original=weights/np.linalg.norm(weights)
    truth=[r["y_true"] for r in rows[0]];classes=full.shape[1]
    variants=[]
    for vis_weight in sorted(set(np.linspace(.05,.95,19).tolist()+[float(weights[0])])):
        proposed=np.asarray([vis_weight,1-vis_weight]);proposed/=np.linalg.norm(proposed)
        logits=proposed[0]/original[0]*(visible-bias)+proposed[1]/original[1]*(infrared-bias)+bias
        predicted=logits.argmax(1).tolist()
        metrics=classification_metrics(truth,predicted,classes)
        confusion=np.zeros((classes,classes),dtype=np.int64);np.add.at(confusion,(truth,predicted),1)
        metrics["confusion_matrix"]=confusion.tolist()
        variants.append({"VIS_weight":vis_weight,"metrics":metrics})
    restored=min(variants,key=lambda r:abs(r["VIS_weight"]-float(weights[0])))
    if restored["metrics"]!=result["target_val"]:raise ValueError("Original global-weight reconstruction changed selected metrics")
    best=max(variants,key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
    return {"method":result["method"],"target":result["target"],"selected_epoch":epoch,
            "result_file":str(path.relative_to(ROOT)),"head_bias_deviation":deviation,
            "original_VIS_weight":float(weights[0]),"original_metrics":result["target_val"],"best":best,"variants":variants}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paths=list((ROOT/"runs/weather_enhanced_screen").rglob("result.json"))
    base=json.loads((ROOT/"results/goal3h_results.json").read_text())
    paths.extend(ROOT/cell["best"]["result_file"] for cell in base["coverage"] if cell.get("best"))
    results=[]
    for path in sorted(set(paths)):
        row=analyze(path)
        if row:
            results.append(row)
            print("LOGGED_GLOBAL_FUSION",row["method"],row["target"],round(100*row["original_metrics"]["accuracy"],2),
                  "->",round(100*row["best"]["metrics"]["accuracy"],2),"VIS=",round(row["best"]["VIS_weight"],3),flush=True)
    args.output.write_text(json.dumps({"formal_DA_training_result":False,"validation_used_for_hyperparameter_diagnostic":True,
        "target_training_labels_used":False,"global_weight_per_method_weather_only":True,"results":results},indent=2),encoding="utf-8")


if __name__=="__main__":main()
