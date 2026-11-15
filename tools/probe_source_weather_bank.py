"""Test whether approved source augmentation improves frozen representations.

This is a source-fitted classifier diagnostic, never a five-method result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torchvision.models import convnext_small
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from weather.train_vds_core import prepare_datasets
from weather.weather_augmentation import augment_source_weather
from weather.source_aspect import augment_source_aspect
from tools.classification_metrics import classification_metrics


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--views",type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.15)
    weight=Path.home()/".cache/torch/hub/checkpoints/convnext_small-0c510722.pth"
    if not hashlib.sha256(weight.read_bytes()).hexdigest().startswith("0c510722"):raise ValueError("Checkpoint mismatch")
    model=convnext_small(weights=None)
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True));model.classifier[-1]=torch.nn.Identity()
    model.cuda().eval().requires_grad_(False)
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    rows=[];start=time.perf_counter()
    for weather in ("dark","rainy","foggy"):
        source=prepare_datasets(base/"prepared_exact_pairs",weather,True)[0][0]
        features=[];labels=[]
        photometric=torch.Generator(device="cuda").manual_seed(42)
        geometry=torch.Generator(device="cuda").manual_seed(43)
        loader=DataLoader(source,batch_size=4,num_workers=0,generator=torch.Generator().manual_seed(44))
        with torch.no_grad():
            for view in range(args.views):
                for step,b in enumerate(loader):
                    batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
                    batch,_=augment_source_weather(batch,weather,.75,photometric)
                    batch,_=augment_source_aspect(batch,.35 if weather=="dark" else .5,2.85 if weather=="dark" else 2.,.75,geometry)
                    features.append(torch.cat((model(batch["rgb"]),model(batch["nir"])),1).cpu().numpy())
                    labels.extend(b["label"].tolist())
                print("BANK",weather,"view",view+1,"images",len(labels),flush=True)
        source_values=np.concatenate(features);ys=np.asarray(labels)
        with np.load(base/f"convnext_small_feature_probe/{weather}_features.npz") as cache:
            validation=cache["target_val"].copy();yv=cache["target_val_labels"].copy()
        np.savez_compressed(args.output/(weather+"_features.npz"),source=source_values,source_labels=ys,
                            target_val=validation,target_val_labels=yv)
        width=source_values.shape[1]//2
        with threadpool_limits(limits=2):
            for w in (.1,.5,.9):
                a,b=[unit(np.concatenate((w*unit(x[:,:width]),(1-w)*unit(x[:,width:])),1)) for x in (source_values,validation)]
                for c in (1.,10.,100.):
                    clf=LogisticRegression(C=c,class_weight="balanced",max_iter=700).fit(a,ys)
                    metrics=classification_metrics(yv.tolist(),clf.predict(b).tolist(),int(ys.max()+1))
                    rows.append({"target":weather,"VIS_weight":w,"C":c,"metrics":metrics})
        print("BEST_BANK_DIAGNOSTIC",json.dumps(max((r for r in rows if r["target"]==weather),key=lambda r:r["metrics"]["accuracy"])),flush=True)
        (args.output/"diagnostic.json").write_text(json.dumps({"formal_UDA_result":False,
            "target_training_labels_used":False,"validation_used_for_fit":False,"source_views":args.views,
            "elapsed_s":time.perf_counter()-start,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
