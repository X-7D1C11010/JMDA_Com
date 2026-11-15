"""Train-only target density diagnostic for the calibrated initialization."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from weather.statistical_calibration import TrainFeatureCalibration, fused_features, unit
from tools.classification_metrics import classification_metrics


def source_mapping(source, target, source_labels, responsibilities):
    classes = responsibilities.shape[1]
    centers = responsibilities.T @ target / np.maximum(responsibilities.sum(0)[:, None], 1e-8)
    nearest = ((source[:, None] - centers[None]) ** 2).sum(2).argmin(1)
    votes = np.zeros((classes, classes))
    np.add.at(votes, (nearest, source_labels), 1)
    votes /= np.maximum(votes.sum(0, keepdims=True), 1)
    rows, cols = linear_sum_assignment(-votes)
    mapping = np.empty(classes, dtype=np.int64)
    mapping[rows] = cols
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", default="rainy", choices=("rainy", "foggy", "dark"))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    base = ROOT / "runs/weather_local_screen/balanced_v4_seed42_20261114_005133/frozen_feature_probe"
    with np.load(base / f"{args.target}_features.npz", allow_pickle=False) as z:
        assert "target_train_labels" not in z.files
        raw = [z[k].copy() for k in ("source", "target_train", "target_val")]
        sy, vy = z["source_labels"].copy(), z["target_val_labels"].copy()
    classes = int(sy.max()) + 1
    parts = []
    for part in (slice(None, 512), slice(512, None)):
        a, b = [unit(x[:, part]).astype(np.float32) for x in raw[:2]]
        cal = TrainFeatureCalibration.fit(a, b, 32)
        with torch.no_grad():
            parts.append([cal(torch.tensor(x), domain).numpy() for x, domain in ((a, "source"), (b, "target"))])
    assoc_source, assoc_target = [unit(np.concatenate((.75 * parts[0][i], .25 * parts[1][i]), 1)) for i in (0, 1)]
    results = []
    for weight, dimension in ((.94, 16), (.75, 32), (.5, 16)):
        a, b, v = [fused_features(x, weight).astype(np.float32) for x in raw]
        cal = TrainFeatureCalibration.fit(a, b, dimension)
        with torch.no_grad():
            a, b, v = [cal(torch.tensor(x), domain).numpy() for x, domain in ((a, "source"), (b, "target"), (v, "target"))]
        with threadpool_limits(limits=2):
            km = KMeans(n_clusters=classes, n_init=10, random_state=42).fit(b)
        original = source_mapping(assoc_source, assoc_target, sy, np.eye(classes)[km.labels_])
        for covariance in ("spherical", "tied", "diag", "full"):
            for ridge in (.0001, .001, .01, .05):
                with threadpool_limits(limits=2):
                    gm = GaussianMixture(n_components=classes, covariance_type=covariance,
                        random_state=42, means_init=km.cluster_centers_, n_init=1,
                        reg_covar=ridge, max_iter=200).fit(b)
                if not gm.converged_:
                    continue
                rr, cc = linear_sum_assignment(((gm.means_[:, None] - km.cluster_centers_[None]) ** 2).sum(2))
                fixed_mapping = np.empty(classes, dtype=np.int64)
                fixed_mapping[rr] = original[cc]
                source_fit_mapping = source_mapping(assoc_source, assoc_target, sy, gm.predict_proba(b))
                # All covariance, mixture weights and maps are fit without val.
                train_description = {"weights": gm.weights_.tolist(), "mapping_fixed": fixed_mapping.tolist(),
                                     "mapping_source_refit": source_fit_mapping.tolist(), "training_lower_bound": float(gm.lower_bound_)}
                weighted = gm._estimate_weighted_log_prob(v)
                for prior_scale in (0., .5, 1.):
                    logits = weighted + (prior_scale - 1) * np.log(gm.weights_)[None]
                    component = logits.argmax(1)
                    for matching, mapping in (("preserved_source_association", fixed_mapping), ("refitted_source_association", source_fit_mapping)):
                        metrics = classification_metrics(vy.tolist(), mapping[component].tolist(), classes)
                        results.append({"target": args.target, "VIS_weight": weight, "components": dimension,
                            "covariance": covariance, "ridge": ridge, "prior_scale": prior_scale,
                            "matching": matching, "metrics": metrics, "training_fit": train_description})
    best = max(results, key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["f1_macro"]))
    print(json.dumps({"best": best}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"kind": "unlabelled_target_density_diagnostic",
        "formal_DA_method_result": False, "target_training_labels_used": False,
        "validation_used_for_distribution_or_mapping_fit": False, "validation_scored_for_hyperparameters": True,
        "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
