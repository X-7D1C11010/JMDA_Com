"""Evaluation-only partition audit. Never returns a usable label remapping."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score


def grouping_metrics(confusion):
    matrix=np.asarray(confusion,dtype=np.int64)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or matrix.sum()<1:
        raise ValueError("Expected a nonempty square evaluation confusion matrix")
    truth=[];prediction=[]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            truth.extend([i]*int(matrix[i,j]));prediction.extend([j]*int(matrix[i,j]))
    rows,columns=linear_sum_assignment(-matrix)
    return {"actual_ACC":float(np.trace(matrix)/matrix.sum()),
        "permutation_only_partition_score_not_classifier_ACC":float(matrix[rows,columns].sum()/matrix.sum()),
        "NMI":float(normalized_mutual_info_score(truth,prediction)),
        "ARI":float(adjusted_rand_score(truth,prediction)),
        "predicted_nonempty_classes":int((matrix.sum(0)>0).sum()),"classes":len(matrix)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();rows=[]
    for root in (ROOT/"runs/weather_local_screen",ROOT/"runs/weather_enhanced_screen"):
        for path in root.rglob("result.json"):
            result=json.loads(path.read_text(encoding="utf-8"))
            if (result.get("core")!="vds2raw.adaptation.AdaptationModel" or result.get("smoke")
                    or result.get("target") not in {"rainy","foggy","dark"}):continue
            complete=path.with_name("completeness.json")
            if not complete.exists() or json.loads(complete.read_text())["status"]!="PASS":continue
            matrix=result["target_val"].get("confusion_matrix")
            if matrix is None:continue
            metrics=grouping_metrics(matrix)
            rows.append({"method":result["method"],"target":result["target"],"selected_epoch":result["selected_epoch"],
                "completed_epochs":result["completed_epochs"],"result":str(path.relative_to(ROOT)),**metrics})
    best=[]
    for target in ("rainy","foggy","dark"):
        for method in ("dann","deepcoral","cdan","mdd","mcc"):
            candidates=[r for r in rows if r["target"]==target and r["method"]==method]
            if candidates:best.append(max(candidates,key=lambda r:r["actual_ACC"]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"kind":"evaluation_only_partition_geometry_audit",
        "no_labels_or_model_predictions_reassigned":True,"no_usable_permutation_saved":True,
        "target_training_labels_used":False,"formal_method_scores_changed":False,
        "best_observed_classifier_runs":best,"all_runs":rows},indent=2),encoding="utf-8")
    for r in best:
        print(r["method"],r["target"],round(100*r["actual_ACC"],2),
              round(100*r["permutation_only_partition_score_not_classifier_ACC"],2),round(r["ARI"],3))


if __name__=="__main__":main()
