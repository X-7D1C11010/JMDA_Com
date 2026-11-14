"""Train-only feature-geometry diagnostics, not named DA method results.

Consumes existing frozen-encoder caches. No target-training labels, image
augmentation changes, or neural-network training. Validation labels are used
only after predictions to score the classifier/cluster diagnostics.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.classification_metrics import classification_metrics


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class TrainingGeometry:
    """Fit a projection/alignment from the two training arrays only."""

    def __init__(self, strategy, components=32):
        if strategy not in {"pooled_pca", "domain_center", "coral"}:
            raise ValueError("Unknown geometry strategy")
        self.strategy, self.components = strategy, components

    def fit(self, source, target):
        source, target = np.asarray(source), np.asarray(target)
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError("Training arrays must have matching feature widths")
        if min(len(source), len(target)) < 2:
            raise ValueError("Each training domain needs at least two samples")
        n = min(self.components, source.shape[1], len(source)+len(target)-1)
        self.pca = PCA(n_components=n, svd_solver="full").fit(np.concatenate((source, target)))
        a, b = self.pca.transform(source), self.pca.transform(target)
        self.source_mean, self.target_mean = a.mean(0), b.mean(0)
        self.source_transform = np.eye(n)
        if self.strategy == "coral":
            # Regularized linear covariance alignment. The target mean and
            # covariance below contain training features only, never val.
            def power(covariance, exponent):
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                regularizer = max(float(np.trace(covariance))/n*.1, 1e-6)
                return (eigenvectors * (np.maximum(eigenvalues, 0)+regularizer)**exponent) @ eigenvectors.T
            self.source_transform = power(np.cov(a, rowvar=False), -.5) @ power(np.cov(b, rowvar=False), .5)
        self.fit_samples = {"source_train": len(source), "target_train": len(target)}
        return self

    def transform(self, values, domain):
        projected = self.pca.transform(values)
        if domain not in {"source", "target"}:
            raise ValueError("Specify source or target domain")
        if self.strategy != "pooled_pca":
            projected = projected - (self.source_mean if domain == "source" else self.target_mean)
            if domain == "source":
                projected = projected @ self.source_transform
        return unit_rows(projected)


def source_cluster_mapping(source_clusters, source_labels, classes):
    """Assign target-fitted cluster IDs using source labels only."""
    counts = np.zeros((classes, classes), dtype=np.float64)
    for cluster, label in zip(source_clusters, source_labels):
        counts[int(cluster), int(label)] += 1
    counts /= np.maximum(counts.sum(0, keepdims=True), 1)
    cluster_ids, label_ids = linear_sum_assignment(-counts)
    mapping = np.zeros(classes, dtype=np.int64)
    mapping[cluster_ids] = label_ids
    return mapping


def prototype_cluster_mapping(source_features, source_labels, centers, classes):
    """Resolve assignment ties with continuous source prototype similarity."""
    prototypes = unit_rows(np.stack([source_features[source_labels == label].mean(0)
                                     for label in range(classes)]))
    scores = unit_rows(centers) @ prototypes.T
    clusters, labels = linear_sum_assignment(-scores)
    mapping = np.zeros(classes, dtype=np.int64)
    mapping[clusters] = labels
    return mapping


def clustering_score_diagnostic(cluster_ids, labels, classes):
    """Post-prediction diagnostic only; this mapping is NEVER used to predict."""
    counts = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(counts, (cluster_ids, labels), 1)
    rows, cols = linear_sum_assignment(-counts)
    return {"kind": "validation_label_association_oracle_diagnostic_only",
            "ACC_with_oracle_cluster_association": float(counts[rows, cols].sum()/len(labels)),
            "validation_labels_used_to_generate_predictions": False,
            "not_an_upper_bound_for_other_models": True}


def probe(feature_file, components=32):
    with np.load(feature_file, allow_pickle=False) as cache:
        # No target-training labels exist in this cache or this API.
        source, target, validation = [cache[k].copy() for k in ("source", "target_train", "target_val")]
        source_labels, validation_labels = [cache[k].copy() for k in ("source_labels", "target_val_labels")]
    classes = int(source_labels.max())+1
    half = source.shape[1]//2
    results = []
    for mode, part in (("VIS", slice(0, half)), ("IR", slice(half, None)), ("concat", slice(None))):
        arrays = []
        for values in (source, target, validation):
            v = values[:, part].copy()
            if mode == "concat":
                v = np.concatenate((unit_rows(v[:, :half]), unit_rows(v[:, half:])), 1)
            arrays.append(unit_rows(v))
        for strategy in ("pooled_pca", "domain_center", "coral"):
            geometry = TrainingGeometry(strategy, components).fit(arrays[0], arrays[1])
            a, b, v = [geometry.transform(values, domain) for values, domain in
                       zip(arrays, ("source", "target", "target"))]
            for regularization in (1., 10.):
                classifier = LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000).fit(a, source_labels)
                prediction = classifier.predict(v)
                results.append({"mode": mode, "strategy": strategy, "classifier": "source_logistic",
                                "C": regularization, "fit_samples": geometry.fit_samples,
                                "metrics": classification_metrics(validation_labels.tolist(), prediction.tolist(), classes)})
            clusters = KMeans(n_clusters=classes, n_init=10, random_state=42).fit(b)
            cluster_ids = clusters.predict(v)
            mappings = {
                "source_mapping": source_cluster_mapping(clusters.predict(a), source_labels, classes),
                "source_prototypes": prototype_cluster_mapping(a, source_labels, clusters.cluster_centers_, classes),
            }
            for name, mapping in mappings.items():
                prediction = mapping[cluster_ids]
                results.append({"mode": mode, "strategy": strategy, "classifier": "unlabelled_target_KMeans_"+name,
                                "fit_samples": geometry.fit_samples,
                                "metrics": classification_metrics(validation_labels.tolist(), prediction.tolist(), classes),
                                "cluster_oracle_diagnostic": clustering_score_diagnostic(cluster_ids, validation_labels, classes)})
    return {"target": Path(feature_file).stem.removesuffix("_features"), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    results = []
    with threadpool_limits(limits=2):
        for path in sorted(Path(args.features_dir).glob("*_features.npz")):
            record = probe(path, args.components)
            results.append(record)
            best = max(record["results"], key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["f1_macro"]))
            oracle = max(row["cluster_oracle_diagnostic"]["ACC_with_oracle_cluster_association"]
                         for row in record["results"] if "cluster_oracle_diagnostic" in row)
            print(json.dumps({"target": record["target"], "best_non_oracle": best, "cluster_association_oracle_diagnostic": oracle}), flush=True)
    if not results:
        raise FileNotFoundError("No feature cache files")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"kind": "training_only_feature_geometry_diagnostic", "formal_UDA_result": False,
        "target_training_labels_used": False, "statistics_fit_partitions": ["source_train", "target_train"],
        "validation_labels_used_for_fit_or_prediction": False, "seed": 42,
        "feature_dir": str(Path(args.features_dir).resolve()), "components": args.components, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
