"""Frozen DINOv2 token pooling diagnostic; not a named-method score."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from weather.train_vds_core import prepare_datasets
from tools.classification_metrics import classification_metrics


def unit(x): return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--scale",choices=("small","base"),default="small")
    p.add_argument("--targets",nargs="+",default=["dark","rainy","foggy","backlight"])
    p.add_argument("--features-only",action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(42)
    weight=ROOT/("pretrained_weights/dinov2_vits14_pretrain.pth" if args.scale=="small" else "pretrained_weights/dinov2_vitb14_pretrain.pth")
    expected="b938bf1b" if args.scale=="small" else "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
    if not hashlib.sha256(weight.read_bytes()).hexdigest().startswith(expected):
        raise ValueError("DINOv2 checksum mismatch")
    if args.scale=="base":
        from tools.reproducibility import seed_everything
        seed_everything(42)
    model=timm.create_model("vit_small_patch14_dinov2" if args.scale=="small" else "vit_base_patch14_dinov2", pretrained=False, num_classes=0, img_size=224)
    model.load_state_dict(checkpoint_filter_fn(torch.load(weight,map_location="cpu",weights_only=True),model),strict=True)
    model=model.to(args.device).eval().requires_grad_(False)
    capture={}
    def attention_input(module, values): capture["x"]=values[0]
    model.blocks[-1].attn.register_forward_pre_hook(attention_input)
    def encode(x):
        inter=model.forward_intermediates(x,indices=4,return_prefix_tokens=True,norm=True,
            output_fmt="NLC",intermediates_only=True)
        patches,cls=inter[-1]; attn=model.blocks[-1].attn
        z=capture["x"];B,N,C=z.shape
        q,k,_=attn.qkv(z).reshape(B,N,3,attn.num_heads,C//attn.num_heads).permute(2,0,3,1,4)
        q,k=attn.q_norm(q),attn.k_norm(k)
        weights=((q[:,:,0:1]@k.transpose(-1,-2))*attn.scale).softmax(-1)[:,:,0,1:].mean(1)
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        return {"cls":cls[:,0],"mean":patches.mean(1),"cls_mean":torch.cat((cls[:,0],patches.mean(1)),1),
            "attention_mean":(patches*weights.unsqueeze(-1)).sum(1),
            "last4_mean":torch.cat([v.mean(1) for v,_ in inter],1)}
    base=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133"
    cache={};rows=[];started=time.perf_counter()
    def features(ds):
        missing=[i for i,path in enumerate(ds.paths) if path not in cache]
        with torch.no_grad():
            for j,b in enumerate(DataLoader(Subset(ds,missing),batch_size=8,num_workers=0)):
                vis=encode(b["rgb"].to(args.device));nir=b["nir"].to(args.device)
                mean=nir.new_tensor([.485,.456,.406]).view(1,3,1,1);std=nir.new_tensor([.229,.224,.225]).view(1,3,1,1)
                ir=encode((nir*.5+.5-mean)/std)
                packed={k:torch.cat((vis[k],ir[k]),1).cpu().numpy() for k in vis}
                for i,path in enumerate(b["path"]):cache[path]={k:v[i] for k,v in packed.items()}
                if j%25==0:print("TOKENS",ds.partition,"cached",len(cache),flush=True)
        return {k:np.stack([cache[path][k] for path in ds.paths]) for k in next(iter(cache.values()))}
    for target in args.targets:
        _,_,s,t,v=prepare_datasets(base/"prepared_exact_pairs",target,False)[0]
        if t.labelled:raise ValueError("Target train labels forbidden")
        arrays=[features(ds) for ds in (s,t,v)]
        ys=np.asarray([y for _,y in s.dataset.samples]);yv=np.asarray([y for _,y in v.dataset.samples])
        packed={"source_labels":ys,"target_val_labels":yv}
        for key in arrays[0]:
            for name,a in zip(("source","target_train","target_val"),arrays):packed[name+"_"+key]=a[key]
            width=arrays[0][key].shape[1]//2
            for w in (() if args.features_only else (.1,.5,.9)):
                values=[unit(np.concatenate((w*unit(a[key][:,:width]),(1-w)*unit(a[key][:,width:])),1)) for a in arrays]
                with threadpool_limits(limits=2):
                    for c in (1.,10.):
                        clf=LogisticRegression(C=c,class_weight="balanced",max_iter=700).fit(values[0],ys)
                        score=classification_metrics(yv.tolist(),clf.predict(values[2]).tolist(),len(set(ys)))
                        rows.append({"target":target,"pooling":key,"VIS_weight":w,"C":c,"metrics":score})
        np.savez_compressed(args.output/(target+"_features.npz"),**packed)
        if not args.features_only:
            best=max((r for r in rows if r["target"]==target),key=lambda r:(r["metrics"]["accuracy"],r["metrics"]["f1_macro"]))
            print("BEST",json.dumps(best),flush=True)
        else:print("FEATURE_CACHE_COMPLETED",target,flush=True)
        (args.output/"diagnostic.json").write_text(json.dumps({"formal_UDA_result":False,"target_training_labels_used":False,
            "statistics_fitted_on_val":False,"IR_normalization":"restore half normalization then ImageNet stats",
            "scale":args.scale,"weights_sha256":hashlib.sha256(weight.read_bytes()).hexdigest(),
            "features_only":args.features_only,
            "elapsed_s":time.perf_counter()-started,"results":rows},indent=2),encoding="utf-8")


if __name__=="__main__":main()
