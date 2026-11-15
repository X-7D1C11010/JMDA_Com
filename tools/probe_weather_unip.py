"""Real-data IR representation diagnostic; source labels only for fitting."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from weather.unip_encoder import UNIPEncoder
from weather.train_vds_core import prepare_datasets
from tools.classification_metrics import classification_metrics
from tools.reproducibility import seed_everything


def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", choices=("tiny", "small"), required=True)
    p.add_argument("--targets", nargs="+", default=["dark", "rainy", "foggy", "backlight"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    seed_everything(42); torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    weight = ROOT/f"pretrained_weights/unip/unip_vit_{args.scale}_by_ibot_large.pth"
    model = UNIPEncoder(weight, args.scale).cuda().eval().requires_grad_(False)
    base = ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    cache, rows = {}, []

    def features(view):
        missing = [i for i, path in enumerate(view.paths) if path not in cache]
        with torch.no_grad():
            for step, batch in enumerate(DataLoader(Subset(view, missing), batch_size=16, num_workers=0)):
                values = {k: v.cpu().numpy() for k, v in model.extract_modes(batch["nir"].cuda()).items()}
                for j, path in enumerate(batch["path"]): cache[path] = {k: v[j] for k, v in values.items()}
                if step % 20 == 0: print("IR_FEATURES", view.partition, step, "cached", len(cache), flush=True)
        return {key: np.stack([cache[path][key] for path in view.paths]) for key in next(iter(cache.values()))}

    start = time.perf_counter()
    for target_name in args.targets:
        _, _, source, target, validation = prepare_datasets(base/"prepared_exact_pairs", target_name, False)[0]
        if target.labelled: raise RuntimeError("Target instance labels are forbidden")
        ys = np.asarray([y for _, y in source.dataset.samples]); yv = np.asarray([y for _, y in validation.dataset.samples])
        ir = [features(ds) for ds in (source, target, validation)]
        with np.load(base/f"convnext_small_feature_probe/{target_name}_features.npz") as old:
            if not np.array_equal(old["source_labels"], ys) or not np.array_equal(old["target_val_labels"], yv):
                raise ValueError("Existing feature cohort labels differ")
            vis = [old[k][:, :old[k].shape[1]//2].copy() for k in ("source", "target_train", "target_val")]
        packed = {"source_labels": ys, "target_val_labels": yv}
        for feature_mode in ir[0]:
            for name, d, a in zip(("source", "target_train", "target_val"), ir, vis):
                packed[name+"_IR_"+feature_mode] = d[feature_mode]
                packed[name+"_VIS"] = a
            for gate in (.1, .5, .9):
                a = unit(np.concatenate((gate*unit(vis[0]), (1-gate)*unit(ir[0][feature_mode])), 1))
                b = unit(np.concatenate((gate*unit(vis[2]), (1-gate)*unit(ir[2][feature_mode])), 1))
                with threadpool_limits(limits=2):
                    for c in (.1, 1., 10.):
                        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=700).fit(a, ys)
                        score = classification_metrics(yv.tolist(), clf.predict(b).tolist(), len(set(ys)))
                        rows.append({"target": target_name, "feature_mode": feature_mode, "VIS_weight": gate,
                            "C": c, "classifier": "source_logistic", "metrics": score})
        np.savez_compressed(args.output/(target_name+"_features.npz"), **packed)
        best = max((r for r in rows if r["target"] == target_name), key=lambda r:(r["metrics"]["accuracy"], r["metrics"]["f1_macro"]))
        print("BEST_DIAGNOSTIC", json.dumps(best), flush=True)
        (args.output/"diagnostic.json").write_text(json.dumps({"formal_UDA_result": False,
            "target_training_labels_used": False, "validation_used_for_fit": False,
            "UNIP": model.provenance, "VIS": "unchanged cached ConvNeXt-Small ImageNet",
            "elapsed_s": time.perf_counter()-start, "results": rows}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
