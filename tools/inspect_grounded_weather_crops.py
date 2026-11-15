"""Inspect a fixed source/label-free-target sample; never starts classifier training."""
import argparse
from collections import Counter,defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.data_paths import exact_ir_partner
from weather.local_capacity import local_training_lease,commit_free_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    weights=ROOT/'pretrained_weights/grounding_dino_tiny'
    if not (weights/'model.safetensors').is_file():raise FileNotFoundError('Download the verified official local detector first')
    data=ROOT/'runs/weather_local_screen/balanced_v4_seed42_20261114_005133/prepared_exact_pairs'
    rng=random.Random(42);source=defaultdict(list);items=[]
    for line in (data/'sunny_train.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            path,label=line.rsplit(' ',1);source[int(label)].append(path)
    for label,paths in sorted(source.items()):
        vis=rng.choice(paths);items.append(('sunny','source_train',f'source_class{label}',vis,exact_ir_partner(vis)))
    for weather in ['foggy','dark']:
        pairs=[x.split('\t') for x in (data/f'{weather}_train_pairs.tsv').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        for index,(vis,ir) in enumerate(rng.sample(pairs,6)):
            items.append((weather,'target_train',f'{weather}_random{index}',vis,ir))
    from PIL import Image,ImageDraw,ImageOps
    import torch
    from torchvision.ops import nms
    from transformers import AutoConfig,AutoProcessor,AutoModelForZeroShotObjectDetection
    import psutil
    torch.set_num_threads(2);records=[];panels=[]
    with local_training_lease(ROOT/'runs/weather_local_screen/.local_training.lock'):
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            commit=commit_free_bytes()
            if free>=3500 and psutil.virtual_memory().available>=6*1024**3 and (commit is None or commit>=6*1024**3):break
            print('WAITING_DETECTOR_HEADROOM',free,flush=True);time.sleep(10)
        torch.cuda.set_per_process_memory_fraction(.45)
        config=AutoConfig.from_pretrained(weights,local_files_only=True,trust_remote_code=False)
        config.disable_custom_kernels=True
        processor=AutoProcessor.from_pretrained(weights,local_files_only=True,trust_remote_code=False)
        model=AutoModelForZeroShotObjectDetection.from_pretrained(weights,config=config,local_files_only=True,
            trust_remote_code=False).eval().to('cuda')
        post=processor.post_process_grounded_object_detection
        threshold_key='threshold' if 'threshold' in inspect.signature(post).parameters else 'box_threshold'
        for weather,partition,identity,visible,infrared in items:
            for modality,path in [('VIS',visible),('IR',infrared)]:
                with Image.open(path) as stream:
                    image=stream.convert('RGB') if modality=='VIS' else stream.convert('L').convert('RGB')
                inputs=processor(images=image,text='a boat. a ship.',return_tensors='pt').to('cuda')
                with torch.inference_mode():outputs=model(**inputs)
                detected=post(outputs,inputs.input_ids,**{threshold_key:.35,'text_threshold':.25,'target_sizes':[image.size[::-1]]})[0]
                boxes,scores=detected['boxes'].cpu(),detected['scores'].cpu()
                keep=nms(boxes,scores,.5) if len(boxes) else []
                candidates=[];width,height=image.size
                for index in keep:
                    box=boxes[index].tolist();confidence=float(scores[index]);x1,y1,x2,y2=box
                    if (x2-x1)*(y2-y1)/(width*height)>=.0005 and x2-x1>=8 and y2-y1>=4:
                        candidates.append((box,confidence))
                crop=None;reason='multiple_or_no_vessels_keep_full'
                if len(candidates)==1:
                    x1,y1,x2,y2=candidates[0][0];dx=(x2-x1)*.5;dy=(y2-y1)*.5
                    proposed=[max(0,int(x1-dx)),max(0,int(y1-dy)),min(width,int(x2+dx+.999)),min(height,int(y2+dy+.999))]
                    if (proposed[2]-proposed[0])*(proposed[3]-proposed[1])<.9*width*height:
                        crop=proposed;reason='single_vessel_padded_crop'
                    else:reason='near_full_image_keep_full'
                raw=image.copy();draw=ImageDraw.Draw(raw)
                for box,confidence in candidates:draw.rectangle(box,outline='orange',width=max(2,min(image.size)//100))
                if crop:draw.rectangle(crop,outline='lime',width=max(2,min(image.size)//100))
                panel=Image.new('RGB',(640,300),'white');drawing=ImageDraw.Draw(panel)
                drawing.text((4,3),f'{identity} {modality}: {len(candidates)} vessels',fill='black')
                drawing.text((325,3),reason,fill='black')
                for offset,view in [(0,raw),(320,image.crop(crop) if crop else image)]:
                    thumb=ImageOps.contain(view,(312,270));panel.paste(thumb,(offset+(320-thumb.width)//2,28+(270-thumb.height)//2))
                panels.append(panel)
                records.append({'weather':weather,'partition':partition,'identity':identity,'modality':modality,'path':path,
                    'native_size':list(image.size),'detections':candidates,'crop':crop,'decision':reason,'target_class_labels_used':False})
                print('INSPECTED',identity,modality,reason,flush=True)
    for page in range((len(panels)+7)//8):
        chunk=panels[page*8:(page+1)*8];sheet=Image.new('RGB',(1280,300*((len(chunk)+1)//2)),'white')
        for index,panel in enumerate(chunk):sheet.paste(panel,((index%2)*640,(index//2)*300))
        sheet.save(args.output/f'page_{page}.jpg',quality=94)
    result={'kind':'authorized_grounded_detector_crop_inspection','classifier_training_started':False,
        'target_class_labels_used':False,'target_validation_images_used':False,'text_prompt':'a boat. a ship.',
        'detector_repository':'IDEA-Research/grounding-dino-tiny','remote_code_used':False,
        'source_sampling':'one image per known source class, seed42','target_sampling':'six uniform label-free training pairs per weather, seed42',
        'single_vessel_rule':'crop only one remaining vessel after confidence .35, NMS .5; 2x padded box; otherwise full image',
        'images':len(records),'decisions':dict(Counter(x['decision'] for x in records)),'records':records}
    (args.output/'inspection.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('DETECTOR_INSPECTION_COMPLETED',len(records),flush=True)


if __name__=='__main__':main()
