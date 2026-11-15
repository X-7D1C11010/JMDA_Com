"""Train-only class-relationship matching diagnostic; never a DA result."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from weather.statistical_calibration import TrainFeatureCalibration, fused_features, unit
from tools.classification_metrics import classification_metrics


def relationship_matching(source, target, source_labels, target_clusters, classes, coefficients):
    """Enumerate a small class bijection; API excludes all validation arrays."""
    if classes > 9 or set(source_labels) != set(range(classes)):
        raise ValueError("Expected at most nine supported source classes")
    sp = unit(np.stack([source[source_labels == c].mean(0) for c in range(classes)]))
    raw_tp = np.stack([target[target_clusters == c].mean(0) for c in range(classes)])
    tp = unit(raw_tp)
    sd, td = np.maximum(1 - sp @ sp.T, 0), np.maximum(1 - tp @ tp.T, 0)
    mask = ~np.eye(classes, dtype=bool)
    sd /= max(float(sd[mask].mean()), 1e-12)
    td /= max(float(td[mask].mean()), 1e-12)
    votes = np.zeros((classes, classes), dtype=np.float64)
    nearest = ((source[:, None] - raw_tp[None]) ** 2).sum(2).argmin(1)
    np.add.at(votes, (nearest, source_labels), 1)
    votes /= np.maximum(votes.sum(0, keepdims=True), 1)
    output = {}
    best = {c: (float("inf"), None) for c in coefficients}
    pairs = np.triu_indices(classes, 1)
    permutations = itertools.permutations(range(classes))
    while True:
        batch = np.asarray(list(itertools.islice(permutations, 4096)), dtype=np.int64)
        if not len(batch):
            break
        structural = ((sd[batch[:, pairs[0]], batch[:, pairs[1]]] - td[pairs]) ** 2).mean(1)
        unary = votes[np.arange(classes)[None], batch].mean(1)
        for coefficient in coefficients:
            costs = structural - coefficient * unary
            index = int(costs.argmin())
            if costs[index] < best[coefficient][0]:
                best[coefficient] = (float(costs[index]), batch[index].copy())
    for coefficient, (cost, mapping) in best.items():
        output[str(coefficient)] = {"mapping": mapping, "cost": cost}
    rows, cols = linear_sum_assignment(-votes)
    mapping = np.empty(classes, dtype=np.int64)
    mapping[rows] = cols
    output["source_votes_only"] = {"mapping": mapping, "cost": None}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", nargs="+", default=["rainy", "foggy", "dark"])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    base = ROOT / "runs/weather_local_screen/balanced_v4_seed42_20261114_005133/frozen_feature_probe"
    results = []
    for name in args.targets:
        with np.load(base / f"{name}_features.npz", allow_pickle=False) as z:
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
        for weight, dimension in ((.94, 16), (.75, 32), (.5, 32)):
            a, b, v = [fused_features(x, weight).astype(np.float32) for x in raw]
            cal = TrainFeatureCalibration.fit(a, b, dimension)
            with torch.no_grad():
                a, b, v = [cal(torch.tensor(x), domain).numpy() for x, domain in ((a, "source"), (b, "target"), (v, "target"))]
            with threadpool_limits(limits=2):
                km = KMeans(n_clusters=classes, n_init=10, random_state=42).fit(b)
            nearest_val = km.predict(v)
            for association_weight in (.75, .25):
                aa, bb = [unit(np.concatenate((association_weight * parts[0][i],
                                              (1 - association_weight) * parts[1][i]), 1)) for i in (0, 1)]
                mappings = relationship_matching(aa, bb, sy, km.labels_, classes, (0., .05, .2, 1., 5.))
                for coefficient, item in mappings.items():
                    mapping = item["mapping"]
                    metrics = classification_metrics(vy.tolist(), mapping[nearest_val].tolist(), classes)
                    results.append({"target": name, "VIS_weight": weight, "components": dimension,
                                    "association_VIS_weight": association_weight, "unary_coefficient": coefficient,
                                    "metrics": metrics, "mapping": mapping.tolist(), "train_objective": item["cost"]})
        best = max((r for r in results if r["target"] == name), key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["f1_macro"]))
        print(json.dumps({"target": name, "best": best}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"kind": "source_class_target_cluster_relationship_diagnostic",
            "formal_DA_method_result": False, "target_training_labels_used": False,
            "validation_used_for_mapping_fit": False, "validation_scored_for_hyperparameters": True,
            "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
