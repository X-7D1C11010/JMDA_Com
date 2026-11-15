"""Training-statistic label-prior diagnostic; scores are not DA training runs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from tools.classification_metrics import classification_metrics


def fit_target_bias(unlabelled_logits,prior,bound):
    """Convex fit from unlabelled target logits and an explicit class prior."""
    logits=np.asarray(unlabelled_logits,dtype=np.float64)
    prior=np.asarray(prior,dtype=np.float64)
    if (logits.ndim!=2 or logits.shape[1]!=len(prior) or np.any(prior<=0)
            or not np.isclose(prior.sum(),1) or bound<=0 or not np.isfinite(logits).all()):
        raise ValueError("Invalid training logits, prior, or bias bound")
    def objective(bias):
        z=logits+bias
        return float(logsumexp(z,axis=1).mean()-prior@bias),softmax(z,axis=1).mean(0)-prior
    result=minimize(objective,np.zeros(len(prior)),jac=True,method="L-BFGS-B",
                    bounds=[(-bound,bound)]*len(prior),options={"maxiter":500,"ftol":1e-12,"gtol":1e-9})
    return result.x,{"converged":bool(result.success),"iterations":int(result.nit),
                     "mean_probabilities_after":softmax(logits+result.x,axis=1).mean(0).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[]
    for directory in args.runs:
        result=json.loads((directory/"result.json").read_text(encoding="utf-8"))
        config=json.loads((directory/"run_config.json").read_text(encoding="utf-8"))
        if result["smoke"]:raise ValueError("Do not inspect a smoke run as a performance candidate")
        train=[]
        with (directory/"predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record=json.loads(line)
                if record["epoch"]==result["selected_epoch"] and record["partition"]=="target_train":
                    if record["labels_used"] or "y_true" in record:raise ValueError("Target labels in prior calibration")
                    train.append(record["logits"])
        val=json.loads((directory/"selected_target_val_predictions.json").read_text(encoding="utf-8"))
        logits=np.asarray([r["logits"] for r in val]);truth=[r["y_true"] for r in val]
        baseline=classification_metrics(truth,logits.argmax(1).tolist(),logits.shape[1])
        if not np.isclose(baseline["accuracy"],result["target_val"]["accuracy"]):raise ValueError("Selected-state mismatch")
        train=np.asarray(train);counts=np.asarray(config["source_classification"]["class_counts"],dtype=np.float64)
        observed=softmax(train,axis=1).mean(0)
        for exponent in (0.,.5,1.):
            source_prior=counts**exponent;source_prior/=source_prior.sum()
            for mixing in (.25,.5,1.):
                prior=(1-mixing)*observed+mixing*source_prior
                for bound in (1.,2.,4.):
                    bias,fit=fit_target_bias(train,prior,bound)
                    metrics=classification_metrics(truth,(logits+bias).argmax(1).tolist(),len(counts))
                    rows.append({"run":str(directory),"method":result["method"],"target":result["target"],
                        "source_prior_exponent":exponent,"prior_mixing":mixing,"bias_bound":bound,
                        "baseline_metrics":baseline,"metrics":metrics,"bias":bias.tolist(),
                        "train_fit":fit,"desired_prior":prior.tolist(),"target_train_samples":len(train)})
        best=max((r for r in rows if r["run"]==str(directory)),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        print(json.dumps({"method":result["method"],"target":result["target"],
            "baseline_ACC":baseline["accuracy"],"diagnostic_peak_ACC":best["metrics"]["accuracy"]}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"kind":"training_only_marginal_prior_bias_diagnostic",
        "formal_DA_training_result":False,"target_training_labels_used":False,"validation_used_for_bias_fit":False,
        "validation_used_for_hyperparameter_scoring":True,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
