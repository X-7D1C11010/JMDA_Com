"""Fast frozen-feature screening of source category-center regularization.

This omits image augmentation and is explicitly diagnostic, not a completed
image-training result. Only labelled source and unlabelled target fit models.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT","2")
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from vds2raw.adaptation import AdaptationModel,source_classification_weights
from vds2raw.crossmodal_loss import SourceCategoryMemory
from vds2raw.train_adaptation import make_optimizer
from weather.train_vds_core import parser as weather_parser,paired_epoch_samplers
from weather.prepare_calibrated_initialization import prepare_calibrated_initialization
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.classification_metrics import classification_metrics


class Features(Dataset):
    def __init__(self,values,partition,labels=None):
        self.values=torch.tensor(values,dtype=torch.float32);self.labels=labels
        self.partition=partition;self.labelled=labels is not None
    def __len__(self):return len(self.values)
    def __getitem__(self,index):
        item={"rgb":self.values[index,:512],"nir":self.values[index,512:],"path":f"{self.partition}_{index}"}
        if self.labelled:item["label"]=int(self.labels[index])
        return item


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--components",nargs="+",type=int,default=[16,32]);p.add_argument("--weights",nargs="+",type=float,default=[0.,.05,.2,.5])
    p.add_argument("--epochs",type=int,default=24);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    folder=ROOT/"runs/weather_enhanced_screen/calibrated_rainy_soft_teacher_seed42_v3/mcc_rainy_teacher0.1/mcc_rainy_seed42"
    reference=json.loads((folder/"run_config.json").read_text())["arguments"]
    data=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/frozen_feature_probe/rainy_features.npz"
    with np.load(data,allow_pickle=False) as cache:packed={k:cache[k].copy() for k in cache.files}
    source=Features(packed["source"],"source_train",packed["source_labels"])
    target=Features(packed["target_train"],"target_train")
    validation=Features(packed["target_val"],"target_val",packed["target_val_labels"])
    classes=int(packed["source_labels"].max())+1;results=[]
    for width in args.components:
        for strength in args.weights:
            torch.manual_seed(42)
            a=weather_parser().parse_args(recorded_arguments_to_cli(reference));a.epochs=args.epochs;a.calibration_components=width
            model=AdaptationModel("mcc",pretrained=False,num_classes=classes,use_ais=False,ir_channels=3,
                feature_normalization="unit",freeze_image_encoders=True,classifier_layout="linear",calibration_feature_dim=width)
            model.rgb_encoder=model.nir_encoder=nn.Identity();model.encoder_precision="fp32";model.mcc.temperature=a.mcc_temperature
            desired=a.initial_vis_weight
            with torch.no_grad():model.modality_logits.copy_(torch.tensor([(desired-.05)/.9,(.95-desired)/.9]).log())
            prepare_calibrated_initialization(model,source,target,"cpu",a)
            _,weight=source_classification_weights([(str(i),int(y)) for i,y in enumerate(source.labels)],classes)
            model.source_class_weights=weight
            a.steps_per_epoch,samplers=paired_epoch_samplers(source,target,a.batch_size,a.target_batch_size,a.seed,"mcc")
            loaders=[DataLoader(view,batch_sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(20042+i))
                     for i,(view,sampler) in enumerate(zip((source,target),samplers))]
            optimizer,scheduler,per_update=make_optimizer(model,a);memory=SourceCategoryMemory(classes,width)
            captured={};handle=model.head.register_forward_pre_hook(lambda _m,inputs:captured.update(features=inputs[0]))
            update=0;best=None;history=[]
            for epoch in range(1,a.epochs+1):
                model.train()
                for sampler in samplers:sampler.epoch=epoch-1
                for sb,tb in zip(*loaders):
                    optimizer.zero_grad(set_to_none=True)
                    loss,detail=model.objective(sb,tb,update,a.epochs*a.steps_per_epoch,a)
                    if strength:
                        values=captured["features"][:len(sb["label"])]
                        extra,_=memory(values,values,sb["label"],.2,model.source_class_weights)
                        loss=loss+strength*extra
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();update+=1
                    if scheduler is not None and per_update:scheduler.step()
                if scheduler is not None and not per_update:scheduler.step()
                model.eval();model.set_bn_domain("target")
                with torch.no_grad():
                    batch={"rgb":validation.values[:,:512],"nir":validation.values[:,512:]}
                    predicted=model.head(model.encode(batch)).argmax(1).tolist()
                metrics=classification_metrics(validation.labels.tolist(),predicted,classes)
                history.append({"epoch":epoch,"metrics":metrics})
                if best is None or (metrics["accuracy"],metrics["f1_macro"])>(best["metrics"]["accuracy"],best["metrics"]["f1_macro"]):best={"epoch":epoch,"metrics":metrics}
            handle.remove()
            row={"components":width,"source_center_weight":strength,"best":best,"history":history,"source_optimizer_updates":update}
            results.append(row);print("FROZEN_CENTER_DIAGNOSTIC",json.dumps({k:v for k,v in row.items() if k!="history"}),flush=True)
            args.output.write_text(json.dumps({"formal_image_DA_training_result":False,"image_augmentation_omitted":True,
                "target_training_labels_used":False,"validation_used_for_gradient_or_statistics":False,"results":results},indent=2),encoding="utf-8")


if __name__=="__main__":main()
