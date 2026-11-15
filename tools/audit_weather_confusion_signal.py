"""Read-only audit of the optional confusion term and its reference run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def lines(path):
    if not path.is_file():
        return []
    # A live writer may not have finished its last line yet.
    values=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        try:values.append(json.loads(line))
        except json.JSONDecodeError:continue
    return values


def audit(root):
    plan=json.loads((root/"plan.json").read_text(encoding="utf-8"))
    reference_path=Path(plan["reference_config"])
    reference=json.loads(reference_path.with_name("result.json").read_text(encoding="utf-8"))
    records=[]
    for config_path in sorted(root.glob("*/*/run_config.json")):
        directory=config_path.parent
        config=json.loads(config_path.read_text(encoding="utf-8"))
        if config["method"]!=reference["method"] or config["target"]!=reference["target"]:
            raise ValueError("Reference method/domain mismatch")
        options=config["arguments"]
        if not options["domain_confusion_weight"]:raise ValueError("Not a confusion experiment")
        evaluation=lines(directory/"evaluation.jsonl")
        val=[r for r in evaluation if r["partition"]=="target_val"]
        source=[r for r in evaluation if r["partition"]=="source_train"]
        if not val:continue
        best=max(val,key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
        gradients=lines(directory/"gradients.jsonl")
        active=[r for r in gradients if r["epoch"]>options["adaptation_warmup_epochs"]]
        signal={}
        for branch in ("RGB","NIR","domain_discriminator","head"):
            signal[branch]={}
            for term in ("core","domain_confusion"):
                rows=[r["enhancement_components"][term][branch] for r in active]
                signal[branch][term]={
                    "recorded_epochs":len(rows),
                    "median_weighted_gradient_norm":statistics.median(r["weighted_DA_norm"] for r in rows) if rows else None,
                    "max_weighted_gradient_norm":max((r["weighted_DA_norm"] for r in rows),default=None),
                    "median_CE_gradient_cosine":statistics.median(r["cosine"] for r in rows) if rows else None,
                }
        for branch in ("domain_discriminator","head"):
            if (signal[branch]["domain_confusion"]["max_weighted_gradient_norm"] or 0)!=0:
                raise AssertionError(f"Unexpected confusion gradient ownership: {branch}")
        complete=(directory/"result.json").exists()
        records.append({"method":config["method"],"target":config["target"],"seed":options["seed"],
            "training_complete":complete,"running_prefix_is_not_final_result":not complete,
            "observed_epochs":len(val),"observed_peak_epoch":best["epoch"],
            "observed_peak_metrics":best["metrics"],"last_source_metrics":source[-1]["metrics"] if source else None,
            "reference_peak_metrics":reference["target_val"],"reference_planned_epochs":reference["planned_epochs"],
            "candidate_planned_epochs":options["epochs"],
            "identical_horizon_control":reference["planned_epochs"]==options["epochs"],
            "gradient_signal":signal,"result_directory":str(directory.resolve())})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    records=[r for root in args.roots for r in audit(root)]
    payload={"kind":"gradient_signal_and_accuracy_audit_not_a_training_result",
        "updated_UTC":datetime.now(timezone.utc).isoformat(),"records":records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"records":len(records),"complete":sum(r["training_complete"] for r in records)}))


if __name__=="__main__":main()
