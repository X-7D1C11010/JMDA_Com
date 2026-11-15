"""Source-only view sensitivity; not a DA score or held-out source accuracy."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from weather.train_vds_core import prepare_datasets
from weather.statistical_calibration import unit
from tools.classification_metrics import classification_metrics


def main():
    output=ROOT/"results/source_object_view_sensitivity_seed42.json"
    if output.exists():raise FileExistsError(output)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    inspection=ROOT/"runs/weather_local_screen/source_context_association_seed42"
    records=json.loads((inspection/"source_view_inspection.json").read_text(encoding="utf-8"))
    with np.load(inspection/"source_extra_views.npz",allow_pickle=False) as z:
        bank=dict(zip(z["image_sha256"].tolist(),z["features"].copy()))
    results=[]
    for name in ("rainy","foggy","dark"):
        source=prepare_datasets(base/"prepared_exact_pairs",name,False)[0][2]
        with np.load(base/"frozen_feature_probe"/f"{name}_features.npz",allow_pickle=False) as z:
            a=unit(z["source"][:,:512]);labels=z["source_labels"].copy()
        if not np.array_equal(labels,[label for _,label in source.dataset.samples]):
            raise ValueError("Source feature cohort mismatch")
        chosen=[];extra=[]
        for i,path in enumerate(source.paths):
            sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if sha in bank:chosen.append(i);extra.append(bank[sha])
        with threadpool_limits(limits=2):
            classifier=LogisticRegression(C=1,class_weight="balanced",max_iter=1000).fit(a,labels)
        y=labels[chosen];original=classifier.predict(a[chosen]);crop=classifier.predict(unit(np.stack(extra)))
        classes=int(labels.max())+1
        results.append({"cohort":name,"viewed_source_samples":len(chosen),
            "full_view_seen_source_metrics":classification_metrics(y.tolist(),original.tolist(),classes),
            "object_view_same_source_metrics":classification_metrics(y.tolist(),crop.tolist(),classes),
            "prediction_changes":int((original!=crop).sum()),
            "source_classes_in_extra_views":sorted(set(y.tolist()))})
    output.write_text(json.dumps({"kind":"source_same_image_view_sensitivity_diagnostic",
        "formal_DA_method_result":False,"held_out_source_accuracy":False,
        "classifier_fit_inputs":"original full-frame source training features and source labels only",
        "target_images_or_labels_used":False,"source_object_images_used_to_fit_classifier":False,
        "encoder_sha256":records["encoder_sha256"],"results":results},indent=2),encoding="utf-8")
    print(json.dumps(results),flush=True)


if __name__=="__main__":main()
