"""Diagnostic only: train-only cluster matching using fixed class-name text.

No validation features/labels enter fit_association. The cached text vectors
are from the earlier fixed three-prompt CLIP class.txt diagnostic, not learned
from any target annotations. This is not a result of any DA method.
"""
import argparse
import hashlib
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
from weather.train_vds_core import prepare_datasets
from tools.classification_metrics import classification_metrics


def fit_association(source, target, source_labels, clip_target, text, text_weight,
                    association_source=None, association_target=None):
    """Fit geometry and matching from training data only; no validation API."""
    classes = len(text)
    with threadpool_limits(limits=2):
        km = KMeans(n_clusters=classes, random_state=42, n_init=10).fit(target)
    votes = np.zeros((classes, classes), dtype=np.float64)
    if association_source is None:
        source_clusters = km.predict(source)
    else:
        associated_centers = np.stack([association_target[km.labels_ == c].mean(0) for c in range(classes)])
        source_clusters = ((association_source[:, None] - associated_centers[None]) ** 2).sum(2).argmin(1)
    np.add.at(votes, (source_clusters, source_labels), 1)
    votes /= np.maximum(votes.sum(0, keepdims=True), 1)
    image_text = unit(clip_target) @ unit(text).T
    score = np.stack([image_text[km.labels_ == c].mean(0) for c in range(classes)])

    def standardize(x):
        # Removing class bias is independent of a cluster-to-class mapping.
        x = x - x.mean(0, keepdims=True)
        return x / max(float(x.std()), 1e-12)

    blended = (1 - text_weight) * standardize(votes) + text_weight * standardize(score)
    rows, columns = linear_sum_assignment(-blended)
    mapping = np.empty(classes, dtype=np.int64)
    mapping[rows] = columns
    return km.cluster_centers_, mapping, {
        "source_vote_scores": votes.tolist(), "fixed_text_scores": score.tolist(),
        "cluster_to_source_class": mapping.tolist(),
        "target_cluster_sizes": np.bincount(km.labels_, minlength=classes).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", nargs="+", default=["rainy", "foggy", "dark"])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    base = ROOT / "runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    text_path = ROOT / "results/goal3h_clip_class_text_vectors.npy"
    all_text = np.load(text_path, allow_pickle=False)
    if all_text.shape != (14, 512) or not np.isfinite(all_text).all():
        raise ValueError("Expected the existing 14 class-name text vectors")
    rows = []
    for target_name in args.targets:
        datasets, audit = prepare_datasets(str(base / "prepared_exact_pairs"), target_name, augment=False)
        original = audit["original_class_ids"]
        with np.load(base / "frozen_feature_probe" / f"{target_name}_features.npz") as z:
            source, target, val = [z[k].copy() for k in ("source", "target_train", "target_val")]
            sy, vy = z["source_labels"].copy(), z["target_val_labels"].copy()
            assert "target_train_labels" not in z.files
        with np.load(base / "clip_feature_probe_cached" / f"{target_name}_features.npz") as z:
            clip_target = z["target_train"].copy()
            assert np.array_equal(sy, z["source_labels"])
            assert np.array_equal(vy, z["target_val_labels"])
            assert "target_train_labels" not in z.files
        _, _, source_view, target_view, val_view = datasets
        assert not target_view.labelled
        assert np.array_equal(sy, [label for _, label in source_view.dataset.samples])
        assert np.array_equal(vy, [label for _, label in val_view.dataset.samples])
        assert len(target) == len(target_view) == len(clip_target)
        parts = []
        for part in (slice(None, 512), slice(512, None)):
            aa, bb = [unit(x[:, part]).astype(np.float32) for x in (source, target)]
            c = TrainFeatureCalibration.fit(aa, bb, 32)
            with torch.no_grad():
                parts.append([c(torch.tensor(x), d).numpy() for x, d in ((aa, "source"), (bb, "target"))])
        associated_source, associated_target = [unit(np.concatenate((.75 * parts[0][i], .25 * parts[1][i]), 1))
                                                for i in (0, 1)]
        for weight in (.94, .5, .06):
            a, b, v = [fused_features(x, weight) for x in (source, target, val)]
            for dimension in (16, 32):
                cal = TrainFeatureCalibration.fit(a, b, dimension)
                with torch.no_grad():
                    aa, bb, vv = [cal(torch.tensor(x, dtype=torch.float32), domain).numpy()
                                  for x, domain in ((a, "source"), (b, "target"), (v, "target"))]
                for clip_weight in (1., .5, 0.):
                    ct = clip_weight * unit(clip_target[:, :512]) + (1 - clip_weight) * unit(clip_target[:, 512:])
                    for text_weight in (0., .1, .25, .5, .75, 1.):
                        centers, mapping, metadata = fit_association(aa, bb, sy, ct, all_text[original], text_weight,
                            association_source=associated_source, association_target=associated_target)
                        # Validation is evaluated only after a train-only fit.
                        nearest = ((vv[:, None] - centers[None]) ** 2).sum(2).argmin(1)
                        metrics = classification_metrics(vy.tolist(), mapping[nearest].tolist(), len(original))
                        rows.append({"target": target_name, "VIS_weight": weight, "components": dimension,
                                     "CLIP_VIS_weight": clip_weight, "text_association_weight": text_weight,
                                     "metrics": metrics, "association": metadata})
        best = max((r for r in rows if r["target"] == target_name),
                   key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["f1_macro"]))
        print(json.dumps({"target": target_name, "best": best}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "kind": "fixed_text_train_cluster_association_diagnostic", "formal_DA_method_result": False,
            "target_training_labels_used": False, "validation_used_to_fit_association": False,
            "validation_labels_used_for_hyperparameter_scoring": True,
            "text_vectors_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
            "results": rows,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
