"""Read-only source-label associations across cached pretrained views.

Target train labels are neither loaded nor accepted. Validation enters scoring
only after all source-derived cluster mappings have been fitted.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from weather.statistical_calibration import TrainFeatureCalibration, fused_features, unit
from tools.classification_metrics import classification_metrics


def calibrated(source, target, components=32, strategy="coral"):
    cal = TrainFeatureCalibration.fit(source, target, components=components, strategy=strategy)
    with torch.no_grad():
        return cal(torch.tensor(source, dtype=torch.float32), "source").numpy(), cal(torch.tensor(target, dtype=torch.float32), "target").numpy()


def source_score(source, target, cluster_ids, labels, classes, soft_temperature=None):
    centers = np.stack([target[cluster_ids == i].mean(0) for i in range(classes)])
    distance = ((source[:, None] - centers[None]) ** 2).sum(2)
    if soft_temperature is None:
        membership = np.eye(classes)[distance.argmin(1)]
    else:
        logits = -distance / soft_temperature
        membership = np.exp(logits - logits.max(1, keepdims=True))
        membership /= membership.sum(1, keepdims=True)
    return np.stack([membership[labels == i].mean(0) for i in range(classes)], 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=["rainy", "foggy", "dark"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    base = ROOT / "runs/weather_local_screen"
    prepared = base / "balanced_v4_seed42_20261114_005133"
    rows = []
    for weather in args.targets:
        with np.load(prepared / f"frozen_feature_probe/{weather}_features.npz", allow_pickle=False) as cache:
            raw = [cache[k].copy() for k in ("source", "target_train", "target_val")]
            labels, validation_labels = cache["source_labels"].copy(), cache["target_val_labels"].copy()
        classes = len(np.unique(labels))
        raw_views = {}
        for name, folder, key in (
            ("r18", prepared / "frozen_feature_probe", ""),
            ("convnext", prepared / "convnext_small_feature_probe", ""),
            ("dino_small_cls", base / "goal3h_dino_tokens_seed42", "_cls"),
            ("dino_base_cls", base / "dinov2_base_tokens_correctIR_seed42", "_cls"),
            ("dino_base_mean", base / "dinov2_base_tokens_correctIR_seed42", "_mean"),
        ):
            with np.load(folder / f"{weather}_features.npz", allow_pickle=False) as cache:
                if not np.array_equal(cache["source_labels"], labels):
                    raise ValueError("Source cohort order mismatch")
                source, target = cache["source" + key].copy(), cache["target_train" + key].copy()
                if len(target) != len(raw[1]):
                    raise ValueError("Target cohort size mismatch")
            width = source.shape[1] // 2
            for modality, part in (("VIS", slice(None, width)), ("IR", slice(width, None))):
                a, b = unit(source[:, part]), unit(target[:, part])
                for strategy in ("coral", "center"):
                    raw_views[f"{name}_{modality}_{strategy}"] = calibrated(a, b, strategy=strategy)
        for visible_weight, components in itertools.product((.94, .5, .06), (16, 32)):
            arrays = [torch.tensor(fused_features(x, visible_weight), dtype=torch.float32) for x in raw]
            calibration = TrainFeatureCalibration.fit(arrays[0].numpy(), arrays[1].numpy(), components)
            with torch.no_grad():
                a, b, val = [calibration(x, domain).numpy() for x, domain in zip(arrays, ("source", "target", "target"))]
            with threadpool_limits(limits=2):
                km = KMeans(n_clusters=classes, n_init=10, random_state=42).fit(b)
            validation_cluster = km.predict(val)
            views = {"primary": (a, b), **raw_views}
            candidates = []
            for temperature in (None, .2):
                scores = {name: source_score(s, t, km.labels_, labels, classes, temperature) for name, (s, t) in views.items()}
                selections = [(name,) for name in scores]
                selections += [("primary", name) for name in scores if name != "primary"]
                selections += [tuple(name for name in scores if name.endswith(f"_{modality}_coral")) for modality in ("VIS", "IR")]
                selections += [tuple(name for name in scores if name.endswith("_coral"))]
                for names in selections:
                    score = np.stack([scores[name] for name in names]).mean(0)
                    cluster_rows, class_ids = linear_sum_assignment(-score)
                    mapping = np.empty(classes, dtype=np.int64)
                    mapping[cluster_rows] = class_ids
                    # Mapping is fitted before any validation scoring.
                    candidates.append((names, temperature, mapping))
            for names, temperature, mapping in candidates:
                prediction = mapping[validation_cluster]
                rows.append({"target": weather, "VIS_weight": visible_weight, "components": components,
                             "association_views": names, "soft_temperature": temperature,
                             "source_mapping": mapping.tolist(), "metrics": classification_metrics(validation_labels.tolist(), prediction.tolist(), classes)})
        best = max((r for r in rows if r["target"] == weather), key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["f1_macro"]))
        print("SOURCE_CONSENSUS_PREVIEW", json.dumps(best), flush=True)
        args.output.write_text(json.dumps({"kind": "source-only pretrained-view association diagnostic", "formal_DA_method_result": False,
            "target_training_labels_used": False, "validation_used_to_fit_mapping": False,
            "cached_cohort_order": "prepare_datasets deterministic source_eval/target_eval order; source label vector and target counts checked",
            "results": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
