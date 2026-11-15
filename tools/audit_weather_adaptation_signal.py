"""Summarize measured adaptation gradients from completed method runs."""
import argparse
import json
from pathlib import Path
import statistics


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(directory):
    rows = []
    for path in sorted(directory.glob("*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent/"run_config.json").read_text(encoding="utf-8"))
        warmup = config["arguments"]["adaptation_warmup_epochs"]
        updates = [x for x in read_lines(path.parent/"updates.jsonl") if x["epoch"] > warmup]
        gradients = [x for x in read_lines(path.parent/"gradients.jsonl") if x["epoch"] > warmup]
        if not updates or not gradients:
            continue
        tail = updates[-min(len(updates), 3*config["arguments"]["steps_per_epoch"]):]
        late_gradients = gradients[-min(len(gradients), 3):]
        domain_accuracy = [x["domain_accuracy"] for x in tail if x["domain_accuracy"] is not None]
        row = {"method":result["method"], "target":result["target"],
               "ACC_percent":100*result["target_val"]["accuracy"],
               "selected_epoch":result["selected_epoch"],
               "completed_epochs":result["completed_epochs"],
               "median_late_transfer_loss":statistics.median(x["transfer"] for x in tail),
               "median_late_domain_accuracy":statistics.median(domain_accuracy) if domain_accuracy else None,
               "late_gradient_epochs":[x["epoch"] for x in late_gradients],
               "branch_gradient_ratios":{}, "result_file":str(path)}
        for branch in ("VIS", "IR"):
            ratios=[x["gradients"][branch]["DA_to_CE_ratio"] for x in late_gradients]
            row["branch_gradient_ratios"][branch]={"median_DA_to_CE":statistics.median(ratios),
                "weighted_DA_norms":[x["gradients"][branch]["weighted_DA_norm"] for x in late_gradients]}
        row["saturated_domain_discriminator"] = bool(domain_accuracy and
            statistics.median(domain_accuracy)>.99 and
            all(x["median_DA_to_CE"]<1e-3 for x in row["branch_gradient_ratios"].values()))
        rows.append(row)
    return {"kind":"measured_training_signal_audit", "not_a_new_accuracy_experiment":True,
            "source_directory":str(directory), "results":rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    data=audit(args.runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2),encoding="utf-8")
    for r in data["results"]:
        print(json.dumps({k:r[k] for k in ("method","target","ACC_percent",
              "median_late_domain_accuracy","saturated_domain_discriminator")},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
