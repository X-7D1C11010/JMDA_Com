"""Inspect label-independent boat ROIs on a small, fixed weather sample."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.data_paths import exact_ir_partner


def choose_crop(boxes, size, padding=.2):
    width, height = size
    candidates = []
    for box, confidence in boxes:
        x1, y1, x2, y2 = box
        area = max(x2-x1, 0)*max(y2-y1, 0)/(width*height)
        if confidence >= .25 and area >= .005 and x2-x1 >= 4 and y2-y1 >= 4:
            candidates.append((confidence*math.sqrt(area), box, confidence, area))
    candidates.sort(reverse=True)
    if not candidates:
        return None, "no_reliable_boat", None
    score, box, confidence, area = candidates[0]
    if len(candidates) > 1 and candidates[1][0] >= .7*score:
        return None, "ambiguous_multiple_boats", confidence
    if area >= .9:
        return None, "boat_already_fills_image", confidence
    x1, y1, x2, y2 = box
    dx, dy = padding*(x2-x1), padding*(y2-y1)
    crop = [max(0, int(x1-dx)), max(0, int(y1-dy)), min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy))]
    if crop == [0, 0, width, height]:
        return None, "boat_already_fills_image", confidence
    return crop, "cropped", confidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=ROOT/"pretrained_weights/yolo11s.pt")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = ROOT/".cache/ultralytics"
    (config/"Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from PIL import Image, ImageDraw, ImageOps
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    model = YOLO(str(args.weights), task="detect")
    if model.names[8] != "boat":
        raise ValueError("The detector must use the COCO boat class, without fine-grained target classes")
    rng, groups = random.Random(42), {}
    source = [row.rsplit(" ", 1)[0] for row in (args.data_dir/"sunny_train.txt").read_text(encoding="utf-8-sig").splitlines() if row.strip()]
    groups["sunny"] = [("source_train", path, exact_ir_partner(path)) for path in rng.sample(source, 4)]
    for domain in ("rainy", "foggy", "dark"):
        pairs = [row.split("\t") for row in (args.data_dir/f"{domain}_train_pairs.tsv").read_text(encoding="utf-8-sig").splitlines() if row.strip()]
        val = [row.rsplit(" ", 1)[0] for row in (args.data_dir/f"{domain}_val.txt").read_text(encoding="utf-8-sig").splitlines() if row.strip()]
        groups[domain] = [("target_train", a, b) for a, b in rng.sample(pairs, 3)]
        groups[domain] += [("target_val", path, exact_ir_partner(path)) for path in rng.sample(val, 3)]
    records = []
    for domain, rows in groups.items():
        sheet = Image.new("RGB", (960, 254*len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (partition, visible, infrared) in enumerate(rows):
            for modality, path, offset in (("VIS", visible, 0), ("IR", infrared, 480)):
                with Image.open(path) as opened:
                    original = opened.convert("RGB") if modality == "VIS" else opened.convert("L").convert("RGB")
                result = model.predict(original, device="cpu", imgsz=640, conf=.25, classes=[8], max_det=10,
                                       verbose=False, save=False, show=False)[0]
                boxes = list(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()))
                crop, reason, confidence = choose_crop(boxes, original.size)
                raw = original.copy()
                if crop is not None:
                    ImageDraw.Draw(raw).rectangle(crop, outline="lime", width=max(1, min(raw.size)//100))
                thumbnail = ImageOps.contain(raw, (230, 220))
                sheet.paste(thumbnail, (offset+(240-thumbnail.width)//2, index*254+28+(220-thumbnail.height)//2))
                processed = original.crop(crop) if crop is not None else original
                processed = processed.resize((220, 220), Image.Resampling.BILINEAR)
                sheet.paste(processed, (offset+250, index*254+28))
                draw.text((offset+4, index*254+4), f"{partition} #{index} {modality} raw", fill="black")
                draw.text((offset+244, index*254+4), reason, fill="black")
                records.append({"domain": domain, "partition": partition, "modality": modality,
                                "path": path, "size": list(original.size), "detections": boxes, "crop": crop,
                                "decision": reason, "confidence": confidence, "class_labels_used": False})
        sheet.save(args.output/f"{domain}_inspection.png")
        print(domain, dict(Counter(r["decision"] for r in records if r["domain"] == domain)), flush=True)
    report = {"kind": "authorized_small_sample_boat_crop_inspection", "classifier_training_started": False,
              "detector": "official YOLO11s COCO; boat class8 only", "device": "CPU", "seed": 42,
              "weights_SHA256_recorded": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "target_class_labels_used": False, "same_rule_for_source_target_and_both_modalities": True,
              "undetected_or_ambiguous_images_kept_in_full": True, "sample_images": len(records), "records": records}
    (args.output/"inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
