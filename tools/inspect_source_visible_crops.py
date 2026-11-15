"""Inspect a conservative source-VIS crop proposal; does not train anything."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]


def conservative_crop(boxes,size):
    width,height=size
    visible=[(box,score) for box,score in boxes
             if score>=.15 and (box[2]-box[0])*(box[3]-box[1])/(width*height)>=.005]
    if len(visible)!=1:return None,"multiple_boats" if visible else "no_boat"
    box,score=visible[0]
    if score<.5:return None,"low_confidence"
    x1,y1,x2,y2=box
    area=(x2-x1)*(y2-y1)/(width*height)
    if not .02<=area<=.8:return None,"area_guard"
    dx,dy=.25*(x2-x1),.25*(y2-y1)
    crop=[max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(width,math.ceil(x2+dx)),min(height,math.ceil(y2+dy))]
    if (crop[2]-crop[0])*(crop[3]-crop[1])>=.95*width*height:return None,"already_fills_image"
    return crop,"proposal_needs_review"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (ROOT/".cache/ultralytics/Ultralytics").mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"]=str(ROOT/".cache/ultralytics")
    os.environ["YOLO_AUTOINSTALL"]="false"
    from PIL import Image,ImageDraw,ImageOps
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO,settings
    settings.update({"sync":False})
    weight=ROOT/"pretrained_weights/yolo11s.pt"
    digest=hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest!="85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5":raise ValueError("Wrong detector checkpoint")
    model=YOLO(str(weight),task="detect")
    groups=defaultdict(list)
    for row in (args.data_dir/"sunny_train.txt").read_text(encoding="utf-8-sig").splitlines():
        if row.strip():
            path,label=row.rsplit(" ",1);groups[label].append(path)
    rng=random.Random(42)
    sample=[rng.choice(paths) for _,paths in sorted(groups.items(),key=lambda row:int(row[0]))]
    unused=[p for paths in groups.values() for p in paths if p not in sample]
    sample+=rng.sample(unused,24-len(sample))
    rows=[];sheet=None
    for i,path in enumerate(sample):
        if i%12==0:sheet=Image.new("RGB",(960,6*254),"white")
        with Image.open(path) as opened:original=opened.convert("RGB")
        predicted=model.predict(original,device="cpu",imgsz=640,conf=.15,classes=[8],max_det=20,verbose=False,save=False,show=False)[0]
        boxes=list(zip(predicted.boxes.xyxy.cpu().tolist(),predicted.boxes.conf.cpu().tolist()))
        crop,reason=conservative_crop(boxes,original.size)
        x,y=(i%2)*480,((i%12)//2)*254
        raw=original.copy()
        if crop:ImageDraw.Draw(raw).rectangle(crop,outline="lime",width=max(1,min(raw.size)//80))
        raw=ImageOps.contain(raw,(230,220))
        sheet.paste(raw,(x+(240-raw.width)//2,y+28+(220-raw.height)//2))
        processed=(original.crop(crop) if crop else original).resize((220,220),Image.Resampling.BILINEAR)
        sheet.paste(processed,(x+245,y+28))
        draw=ImageDraw.Draw(sheet)
        draw.text((x+3,y+5),f"source VIS #{i} raw",fill="black")
        draw.text((x+240,y+5),reason,fill="black")
        rows.append({"path":path,"size":list(original.size),"boxes":boxes,"crop":crop,"decision":reason})
        if i%12==11:sheet.save(args.output/f"source_visible_{i//12}.png")
    report={"kind":"source_only_crop_inspection_not_training","samples":24,"seed":42,
        "target_images_or_labels_used":False,"IR_or_validation_preprocessing_changed":False,
        "training_adoption_authorized":False,"detector_sha256":digest,
        "decisions":dict(Counter(x["decision"] for x in rows)),"records":rows}
    (args.output/"inspection.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report["decisions"]),flush=True)


if __name__=="__main__":main()
