"""Evaluation-only target geometry; training clusters never see class labels.

Partition agreement is not classification accuracy and cannot relabel a model.
No usable label permutation or validation-fitted transform is returned/saved.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from tools.audit_prediction_grouping import grouping_metrics
from tools.probe_paired_correlation import load_pack
from weather.statistical_calibration import unit


def cluster_training_features(training, validation, classes, components=None):
    """This fitting API accepts no source/target class labels."""
    training, validation = np.asarray(training), np.asarray(validation)
    if (training.ndim != 2 or validation.ndim != 2 or training.shape[1] != validation.shape[1]
            or len(training) < classes or classes < 2
            or not np.isfinite(training).all() or not np.isfinite(validation).all()):
        raise ValueError("Invalid target feature arrays")
    with threadpool_limits(limits=2):
        if components is not None:
            projection = PCA(n_components=min(components, training.shape[1], len(training)-1),
                             svd_solver="full").fit(training)
            training, validation = projection.transform(training), projection.transform(validation)
        clusters = KMeans(n_clusters=classes, n_init=20, random_state=42).fit(training)
        prediction = clusters.predict(validation)
    return prediction, np.bincount(clusters.labels_, minlength=classes).tolist()


def evaluate_partition(labels, prediction, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, prediction), 1)
    metrics = grouping_metrics(matrix)
    # Cluster integer IDs are arbitrary; their raw diagonal accuracy is meaningless.
    metrics.pop("actual_ACC")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for weather in ("rainy", "foggy", "dark"):
        for pack in ("resnet18", "convnext_small", "convnext_unip"):
            raw, source_labels, validation_labels = load_pack(pack, weather)
            classes = len(np.unique(source_labels))
            for modality in ("VIS", "IR", "concat"):
                def features(pair):
                    vis, ir = [unit(x) for x in pair]
                    return {"VIS": vis, "IR": ir, "concat": unit(np.concatenate((vis, ir), 1))}[modality]
                training, validation = features(raw[1]), features(raw[2])
                for components in (None, 16, 32):
                    prediction, counts = cluster_training_features(training, validation, classes, components)
                    row = {"target": weather, "pack": pack, "modality": modality,
                           "target_train_samples": len(training), "target_val_samples": len(validation),
                           "target_only_PCA_components": components, "training_cluster_counts": counts,
                           **evaluate_partition(validation_labels, prediction, classes)}
                    rows.append(row)
        best = max((r for r in rows if r["target"] == weather),
                   key=lambda r: r["permutation_only_partition_score_not_classifier_ACC"])
        print(json.dumps(best), flush=True)
    payload = {"updated_UTC": datetime.now(timezone.utc).isoformat(),
               "kind": "target_train_only_clustering_geometry_diagnostic", "formal_method_result": False,
               "target_training_labels_used": False, "validation_used_for_fit": False,
               "source_labels_used_only_for_known_number_of_classes": True,
               "no_usable_permutation_saved": True, "classifier_results_modified": False, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
