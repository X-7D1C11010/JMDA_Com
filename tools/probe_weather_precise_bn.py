"""Train-only population BN diagnostic; never report as five DA methods."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from weather.precise_bn import refit_population_bn,refit_population_bn_layerwise
from weather.train_vds_core import prepare_datasets
from weather.statistical_calibration import TrainFeatureCalibration, ClusterInitialization, fused_features
from tools.classification_metrics import classification_metrics


def score_features(packed, weather, iteration):
    source, train, validation = [packed[k] for k in ("source", "target_train", "target_val")]
    sy, vy = packed["source_labels"], packed["target_val_labels"]
    classes, results = int(sy.max()) + 1, []
    for vis_weight in (.94, .5, .06):
        arrays = [torch.tensor(fused_features(x, vis_weight), dtype=torch.float32) for x in (source, train, validation)]
        with threadpool_limits(limits=2):
            for cost in (1., 10.):
                classifier = LogisticRegression(C=cost, class_weight="balanced", max_iter=700, random_state=42).fit(arrays[0], sy)
                prediction = classifier.predict(arrays[2])
                results.append({"target": weather, "BN_round": iteration, "VIS_weight": vis_weight, "classifier": "source-only logistic",
                    "C": cost, "source_fit_ACC": float((classifier.predict(arrays[0]) == sy).mean()),
                    "metrics": classification_metrics(vy.tolist(), prediction.tolist(), classes)})
        for components in (16, 32):
            calibration = TrainFeatureCalibration.fit(arrays[0].numpy(), arrays[1].numpy(), components)
            with torch.no_grad():
                s, t, v = [calibration(x, domain) for x, domain in zip(arrays, ("source", "target", "target"))]
            for association in ("source_votes", "source_prototypes"):
                teacher = ClusterInitialization.fit(s.numpy(), t.numpy(), sy, classes, matching=association)
                with torch.no_grad(): prediction = teacher(v).argmax(1).tolist()
                results.append({"target": weather, "BN_round": iteration, "VIS_weight": vis_weight, "components": components,
                    "classifier": "source-associated unlabelled clusters", "association": association,
                    "metrics": classification_metrics(vy.tolist(), prediction, classes)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=["rainy", "foggy", "dark"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--layerwise",action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.rounds < 1 or not 0 < args.strength <= 1:
        parser.error("Invalid round count or blend strength")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_per_process_memory_fraction(.35)
    weight = ROOT / "pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"): raise ValueError("Unexpected pretrained weights")
    original = resnet18(weights=None)
    original.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    original.fc = torch.nn.Identity()
    original.eval().requires_grad_(False)
    base = ROOT / "runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    results, audits = [], []
    started = time.perf_counter()
    for weather in args.targets:
        _, _, source, target, validation = prepare_datasets(base / "prepared_exact_pairs", weather, False)[0]
        if target.labelled: raise RuntimeError("Labelled target training view")
        for view in (source, target, validation):
            view.cache_evaluation_inputs(0)
        with np.load(base / f"frozen_feature_probe/{weather}_features.npz", allow_pickle=False) as cache:
            baseline = {key: cache[key].copy() for key in cache.files}
        sy = np.asarray([label for _, label in source.dataset.samples])
        vy = np.asarray([label for _, label in validation.dataset.samples])
        if not np.array_equal(sy, baseline["source_labels"]): raise ValueError("Source cohort mismatch")
        results.extend(score_features(baseline, weather, 0))
        encoders = {domain: {key: copy.deepcopy(original).to(device) for key in ("rgb", "nir")} for domain in ("source", "target")}

        def collect(view, domain):
            arrays = []
            with torch.no_grad():
                for batch in DataLoader(view, batch_size=16, shuffle=False, num_workers=0,
                                        generator=torch.Generator().manual_seed(42)):
                    arrays.append(torch.cat([encoders[domain][key](batch[key].to(device)) for key in ("rgb", "nir")], 1).cpu().numpy())
            return np.concatenate(arrays)

        for iteration in range(1, args.rounds + 1):
            for domain, view in (("source", source), ("target", target)):
                for key in ("rgb", "nir"):
                    fit=refit_population_bn_layerwise if args.layerwise else refit_population_bn
                    record = fit(encoders[domain][key], view, key, device, strength=args.strength)
                    record.update({"target": weather, "round": iteration,
                        "fit_cohort_sha256": hashlib.sha256("\n".join(view.paths).encode()).hexdigest()})
                    audits.append(record)
            packed = {"source": collect(source, "source"), "target_train": collect(target, "target"),
                      "target_val": collect(validation, "target"), "source_labels": sy, "target_val_labels": vy}
            np.savez_compressed(args.output / f"{weather}_BNround{iteration}_features.npz", **packed)
            results.extend(score_features(packed, weather, iteration))
            best = max((row for row in results if row["target"] == weather and row["BN_round"] == iteration),
                       key=lambda row: (row["metrics"]["accuracy"], row["metrics"]["f1_macro"]))
            print("POPULATION_BN_DIAGNOSTIC", json.dumps(best), flush=True)
            (args.output / "diagnostic.json").write_text(json.dumps({"formal_DA_method_result": False,
                "target_training_labels_used": False, "validation_statistics_fitted": False,
                "source_or_target_preprocessing_changed": False, "convolution_and_affine_weights_changed": False,
                "weight_sha256": digest, "statistics_audit": audits, "results": results,
                "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
        del encoders, source, target, validation
        if device.type == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__": main()
