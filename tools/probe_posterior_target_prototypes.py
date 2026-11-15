"""Train-fitted prototype diagnostic; never a formal DA training result."""
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weather.target_prototypes import PosteriorTargetPrototypes
from tools.classification_metrics import classification_metrics


RUNS = {
    "foggy_mcc": "runs/weather_enhanced_screen/negative_free_mcc_foggy_seed42_v1/mcc_foggy_source0.1_target0.1/mcc_foggy_seed42",
    "dark_cdan": "runs/weather_enhanced_screen/domain_confusion_cdan_dark_fullVIS_seed42_v2/cdan_dark_source0_target0/cdan_dark_seed42",
    "dark_deepcoral": "runs/weather_enhanced_screen/negative_free_deepcoral_dark_seed42_v1/deepcoral_dark_source0.1_target0.1/deepcoral_dark_seed42",
}


def load_aligned_probabilities(run, epoch, paths):
    records = {}
    with (run / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["epoch"] == epoch and row["partition"] == "target_train":
                if row.get("labels_used") or "y_true" in row:
                    raise RuntimeError("Target truth entered prediction records")
                key = row["sample_id"]
                if key in records:
                    raise ValueError("Duplicate target prediction path")
                records[key] = row["probabilities"]
    if set(records) != set(paths.tolist()):
        raise ValueError("Feature and probability target cohorts differ")
    return np.asarray([records[path] for path in paths], dtype=np.float64)


def geometry(values, width, vis_weight):
    def unit(a): return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return np.concatenate((vis_weight * unit(values[:, :width]),
                           (1-vis_weight) * unit(values[:, width:])), 1)


def main():
    rows = []
    for name, relative in RUNS.items():
        run = ROOT / relative
        meta = json.loads((run / "selected_feature_cache.json").read_text(encoding="utf-8"))
        with np.load(run / "selected_feature_cache.npz", allow_pickle=False) as packed:
            if "target_train_labels" in packed.files:
                raise RuntimeError("Target training truth is forbidden")
            train = packed["target_train"].copy()
            validation = packed["target_val"].copy()
            paths = packed["target_train_paths"].copy()
            # Fit parameters and predictions below do not accept this scoring-only array.
            validation_labels = packed["target_val_labels"].copy()
        probabilities = load_aligned_probabilities(run, meta["epoch"], paths)
        width = meta["modality_widths"][0]
        logged = meta["modality_weights"][0]
        for vis_weight in (0., .25, logged, .75, 1.):
            t = geometry(train, width, vis_weight)
            v = geometry(validation, width, vis_weight)
            for center in (False, True):
                prototype = PosteriorTargetPrototypes(center=center).fit(t, probabilities)
                prediction = prototype.predict(v)
                metrics = classification_metrics(validation_labels.tolist(), prediction.tolist(), probabilities.shape[1])
                rows.append({"run": relative, "name": name, "selected_epoch": meta["epoch"],
                             "VIS_weight": vis_weight, "target_centering": center,
                             "trace": prototype.trace, "metrics": metrics})
        best = max((r for r in rows if r["name"] == name), key=lambda r:r["metrics"]["accuracy"])
        print("POSTERIOR_PROTOTYPE_DIAGNOSTIC", json.dumps(best), flush=True)
    result = {"kind": "target_posterior_guided_prototype_diagnostic", "formal_method_result": False,
              "target_training_labels_used": False, "validation_fitted": False,
              "validation_labels_used_for_scoring_only": True,
              "feature_scope": "raw encoder embeddings before optional classifier bottleneck",
              "reference": "https://github.com/tim-learn/SHOT/blob/master/object/image_target.py",
              "results": rows}
    (ROOT / "results/posterior_target_prototypes_seed42.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
