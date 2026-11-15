"""Inspect source-only extra views and test their cluster association evidence.

Original source features remain available. Target/validation preprocessing is
unchanged. This tool never trains a DA method or uses target training labels.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from weather.statistical_calibration import TrainFeatureCalibration, ClusterInitialization, fused_features, unit
from tools.classification_metrics import classification_metrics


def extra_box(boxes, size):
    width, height = size
    # Low-resolution examples already cropped around a boat are excluded.
    if min(size) < 512:
        return None, "low_resolution_full_context"
    visible = [(box, confidence) for box, confidence in boxes
               if confidence >= .15 and (box[2] - box[0]) * (box[3] - box[1]) / (width * height) >= .005]
    if len(visible) != 1:
        return None, "ambiguous_or_no_boat"
    box, confidence = visible[0]
    area = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
    if confidence < .5 or not .005 <= area <= .08:
        return None, "confidence_or_object_scale_guard"
    dx, dy = box[2] - box[0], box[3] - box[1]
    return [max(0, math.floor(box[0] - dx)), max(0, math.floor(box[1] - dy)),
            min(width, math.ceil(box[2] + dx)), min(height, math.ceil(box[3] + dy))], "extra_view_only"


def prepare_bank(prepared, output):
    from torchvision.models import resnet18
    from ultralytics import YOLO, settings
    from tools.uda_multimodal import jmda_val_transform
    weight = ROOT / "pretrained_weights/yolo11s.pt"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5":
        raise ValueError("Unexpected detector checkpoint")
    settings.update({"sync": False})
    detector = YOLO(str(weight), task="detect")
    paths = [row.rsplit(" ", 1)[0] for row in (prepared / "sunny_train.txt").read_text(encoding="utf-8-sig").splitlines() if row.strip()]
    records, features, keys = [], [], []
    encoder = resnet18(weights=None)
    encoder_weight = ROOT / "pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
    encoder_sha = hashlib.sha256(encoder_weight.read_bytes()).hexdigest()
    if not encoder_sha.startswith("f37072fd"):
        raise ValueError("Unexpected encoder checkpoint")
    encoder.load_state_dict(torch.load(encoder_weight, weights_only=True, map_location="cpu"))
    encoder.fc = torch.nn.Identity()
    encoder.eval().requires_grad_(False)
    transform = jmda_val_transform()
    sheets = []
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            size = opened.size
            if min(size) < 512:
                records.append({"path": path, "reason": "low_resolution_full_context"})
                continue
            original = opened.convert("RGB")
        prediction = detector.predict(original, device="cpu", imgsz=640, conf=.15, classes=[8], max_det=20,
                                      verbose=False, save=False, show=False)[0]
        boxes = list(zip(prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.conf.cpu().tolist()))
        box, reason = extra_box(boxes, size)
        row = {"path": path, "size": size, "box": box, "reason": reason}
        if box:
            extra = original.crop(box)
            with torch.no_grad():
                feature = encoder(transform(extra).unsqueeze(0))[0].numpy()
            image_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            features.append(feature)
            keys.append(image_sha)
            row["image_sha256"] = image_sha
            if len(sheets) < 48:
                sheets.append((path, original.copy(), extra.copy()))
        records.append(row)
        if index % 50 == 0:
            print(f"SOURCE_CONTEXT_INSPECTION {index + 1}/{len(paths)}; proposed extra views={len(features)}", flush=True)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source image bytes in extra-view bank")
    packed = np.stack(features) if features else np.empty((0, 512), dtype=np.float32)
    np.savez_compressed(output / "source_extra_views.npz", image_sha256=np.asarray(keys, dtype="U64"), features=packed)
    for page in range(math.ceil(len(sheets) / 12)):
        sheet = Image.new("RGB", (1200, 6 * 225), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (_, original, extra) in enumerate(sheets[page * 12:(page + 1) * 12]):
            x, y = index % 2 * 600, index // 2 * 225
            for offset, image in ((0, original), (300, extra)):
                image = ImageOps.contain(image, (290, 190))
                sheet.paste(image, (x + offset, y + 28))
            draw.text((x, y + 5), "Original source retained", fill="black")
            draw.text((x + 300, y + 5), "Auxiliary view; not replacement", fill="black")
        sheet.save(output / f"source_extra_views_{page}.jpg")
    (output / "source_view_inspection.json").write_text(json.dumps({
        "kind": "source-only additional-view inspection and frozen-feature diagnostic",
        "source_samples": len(paths), "proposed_views": len(features), "reasons": dict(Counter(r["reason"] for r in records)),
        "detector_sha256": digest, "encoder_sha256": encoder_sha, "student_training_changed": False,
        "target_or_validation_preprocessing_changed": False, "target_images_used_for_view_construction": False,
        "original_source_context_retained": True, "records": records}, indent=2), encoding="utf-8")
    return dict(zip(keys, packed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=["rainy", "foggy", "dark"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    base = ROOT / "runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    bank = prepare_bank(base / "prepared_exact_pairs", args.output)
    from weather.train_vds_core import prepare_datasets
    results = []
    for weather in args.targets:
        _, _, source_view, target_view, _ = prepare_datasets(base / "prepared_exact_pairs", weather, False)[0]
        if target_view.labelled:
            raise ValueError("Target training labels must be absent")
        with np.load(base / f"frozen_feature_probe/{weather}_features.npz", allow_pickle=False) as cache:
            source, target, validation = [cache[k].copy() for k in ("source", "target_train", "target_val")]
            labels, val_labels = cache["source_labels"].copy(), cache["target_val_labels"].copy()
        if not np.array_equal(labels, [label for _, label in source_view.dataset.samples]):
            raise ValueError("Source cache cohort mismatch")
        source_extra = source.copy()
        used = 0
        for i, path in enumerate(source_view.paths):
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest in bank:
                source_extra[i, :512] = bank[digest]
                used += 1
        classes = int(labels.max()) + 1
        for extra_weight in (.25, .5, 1.):
            associated = source.copy()
            associated[:, :512] = (1 - extra_weight) * unit(source[:, :512]) + extra_weight * unit(source_extra[:, :512])
            for association_vis in (.5, .94):
                aa, ab = [torch.tensor(fused_features(x, association_vis), dtype=torch.float32) for x in (associated, target)]
                auxiliary = TrainFeatureCalibration.fit(aa.numpy(), ab.numpy(), 32)
                with torch.no_grad():
                    association_source, association_target = auxiliary(aa, "source").numpy(), auxiliary(ab, "target").numpy()
                for primary_vis in (.94, .5, .06):
                    sa, ta, va = [torch.tensor(fused_features(x, primary_vis), dtype=torch.float32) for x in (source, target, validation)]
                    for components in (16, 32):
                        calibration = TrainFeatureCalibration.fit(sa.numpy(), ta.numpy(), components)
                        with torch.no_grad():
                            s, t, v = [calibration(x, domain) for x, domain in zip((sa, ta, va), ("source", "target", "target"))]
                        for matching in ("source_votes", "source_prototypes"):
                            teacher = ClusterInitialization.fit(s.numpy(), t.numpy(), labels, classes, matching=matching,
                                association_source=association_source, association_target=association_target)
                            with torch.no_grad():
                                predicted = teacher(v).argmax(1).tolist()
                            results.append({"target": weather, "source_extra_views_used": used, "source_extra_view_weight": extra_weight,
                                "association_VIS_weight": association_vis, "primary_VIS_weight": primary_vis, "components": components,
                                "matching": matching, "source_mapping": teacher.metadata["cluster_to_source_class"],
                                "metrics": classification_metrics(val_labels.tolist(), predicted, classes)})
        best = max((r for r in results if r["target"] == weather), key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["f1_macro"]))
        print("SOURCE_CONTEXT_ASSOCIATION_DIAGNOSTIC", json.dumps(best), flush=True)
        (args.output / "association_diagnostic.json").write_text(json.dumps({"formal_DA_method_result": False,
            "target_training_labels_used": False, "validation_used_for_fitting": False,
            "source_or_target_student_preprocessing_changed": False, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
