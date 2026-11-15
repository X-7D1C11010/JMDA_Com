"""Fixed-window/EMA prediction diagnostic; not trained-model EMA results."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.classification_metrics import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    results=[]
    for path in sorted((ROOT/"runs/weather_enhanced_screen").rglob("result.json")):
        result=json.loads(path.read_text(encoding="utf-8"))
        if result.get("smoke"):continue
        by_epoch={}
        with (path.parent/"predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line)
                if row["partition"]=="target_val":by_epoch.setdefault(row["epoch"],[]).append(row)
        ordered=[sorted(rows,key=lambda r:r["sample_index"]) for _,rows in sorted(by_epoch.items())]
        identities=[(r["sample_id"],r["y_true"]) for r in ordered[0]]
        if any([(r["sample_id"],r["y_true"]) for r in rows]!=identities for rows in ordered):raise ValueError("Temporal cohort changed")
        truth=[r["y_true"] for r in ordered[0]];candidate=[]
        for key in ("logits","probabilities"):
            data=np.asarray([[r[key] for r in rows] for rows in ordered]);classes=data.shape[2]
            ema={factor:data[0].copy() for factor in (.5,.9)}
            for epoch in range(len(data)):
                variants={"prefix_mean":data[:epoch+1].mean(0)}
                variants.update({f"window{window}":data[max(0,epoch+1-window):epoch+1].mean(0) for window in (3,5,10)})
                for factor in ema:
                    if epoch:ema[factor]=factor*ema[factor]+(1-factor)*data[epoch]
                    variants[f"prediction_EMA{factor}"]=ema[factor].copy()
                for mode,scores in variants.items():
                    candidate.append({"space":key,"rule":mode,"epoch":epoch+1,
                        "metrics":classification_metrics(truth,scores.argmax(1).tolist(),classes)})
        best=max(candidate,key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        record={"method":result["method"],"target":result["target"],"original_metrics":result["target_val"],
            "best_diagnostic":best,"result_file":str(path.relative_to(ROOT))}
        results.append(record)
        print("TEMPORAL_DIAGNOSTIC",path.parents[2].name,result["method"],result["target"],
            round(100*result["target_val"]["accuracy"],2),"->",round(100*best["metrics"]["accuracy"],2),best["rule"],flush=True)
    args.output.write_text(json.dumps({"formal_DA_training_result":False,"parameter_EMA_implemented":False,
        "validation_used_to_score_fixed_averages":True,"per_instance_or_per_class_ensemble_weights_fitted":False,
        "target_training_labels_used":False,"results":results},indent=2),encoding="utf-8")


if __name__=="__main__":main()
