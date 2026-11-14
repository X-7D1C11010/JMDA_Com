"""Experimental paired representation pretraining, not formal UDA scores.

Uses labelled source images and unlabelled target VIS/IR pairs. Keeps the
JMDA image augmentation unchanged. Does not save model checkpoints.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tools.reproducibility import seed_everything
from tools.classification_metrics import classification_metrics
from vds2raw.adaptation import AdaptationModel, PairedEpochSampler
from weather.train_vds_core import prepare_datasets


def paired_contrastive_loss(visible, infrared, source_labels, temperature=.1):
    if visible.shape != infrared.shape or len(source_labels) > len(visible):
        raise ValueError("Invalid paired representation batch")
    similarity = F.normalize(visible, dim=1) @ F.normalize(infrared, dim=1).T / temperature
    positives = torch.eye(len(visible), device=visible.device, dtype=torch.bool)
    n = len(source_labels)
    positives[:n, :n] |= source_labels[:, None] == source_labels[None, :]
    def direction(scores, mask):
        return -(scores.log_softmax(1)*mask).sum(1).div(mask.sum(1)).mean()
    return (direction(similarity, positives)+direction(similarity.T, positives.T))/2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", default="dark")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--contrastive-weight", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", default=str(ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"))
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(args.seed)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    sets, audit = prepare_datasets(args.data_dir, args.target, True)
    source, target, source_eval, _, validation = sets
    for ds in (source_eval, validation):
        ds.cache_evaluation_inputs(workers=2)
    classes = len(audit["original_class_ids"])
    model = AdaptationModel("deepcoral", args.weights,
                            True, num_classes=classes, use_ais=False, ir_channels=3).cuda()
    projectors = nn.ModuleList(nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128)) for _ in range(2)).cuda()
    branches = nn.ModuleList(nn.Linear(512, classes) for _ in range(2)).cuda()
    encoders = [*model.rgb_encoder.parameters(), *model.nir_encoder.parameters()]
    optimizer = torch.optim.AdamW([{"params": encoders, "lr": 1e-4},
        {"params": [*model.head.parameters(), *projectors.parameters(), *branches.parameters()], "lr": 1e-3}], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    steps = math.ceil(max(len(source), len(target))/args.batch_size)
    samplers = [PairedEpochSampler(ds, args.batch_size, steps, args.seed+offset) for ds, offset in [(source, 0), (target, 10000)]]
    loaders = [DataLoader(ds, batch_sampler=s, num_workers=2, persistent_workers=True,
                         generator=torch.Generator().manual_seed(args.seed+20000+i))
               for i, (ds, s) in enumerate(zip((source, target), samplers))]
    best, history = None, []

    def evaluate_representations():
        model.eval()
        collected = []
        for ds in (source_eval, validation):
            vis, ir, logits, truth = [], [], [], []
            with torch.no_grad():
                for b in DataLoader(ds, batch_size=32, num_workers=0, generator=torch.Generator().manual_seed(0)):
                    a, c = model.rgb_encoder(b["rgb"].cuda()), model.nir_encoder(b["nir"].cuda())
                    vis.append(a.cpu());ir.append(c.cpu());logits.append(model.head(torch.cat((a, c),1)).cpu());truth.extend(b["label"].tolist())
            collected.append((torch.cat(vis),torch.cat(ir),torch.cat(logits),torch.tensor(truth)))
        sv, si, _, sy = collected[0];vv, vi, logits, vy = collected[1]
        scores = {}
        for name, s, v in [("VIS",sv,vv),("IR",si,vi),("concat",torch.cat((sv,si),1),torch.cat((vv,vi),1))]:
            s, v = F.normalize(s,dim=1), F.normalize(v,dim=1)
            prototypes = F.normalize(torch.stack([s[sy==i].mean(0) for i in range(classes)]),dim=1)
            scores[name] = float(((v@prototypes.T).argmax(1)==vy).float().mean())
        return classification_metrics(vy.tolist(),logits.argmax(1).tolist(),classes),scores

    for epoch in range(1, args.epochs+1):
        model.train();projectors.train();branches.train()
        for s in samplers:s.epoch=epoch-1
        totals = np.zeros(2)
        for sb, tb in zip(*loaders):
            if "label" in tb:raise RuntimeError("Target labels entered pretraining")
            n=len(sb["label"]);labels=sb["label"].cuda()
            rgb=torch.cat((sb["rgb"],tb["rgb"])).cuda();ir=torch.cat((sb["nir"],tb["nir"])).cuda()
            optimizer.zero_grad(set_to_none=True)
            fv,fi=model.rgb_encoder(rgb),model.nir_encoder(ir)
            ce=.5*F.cross_entropy(model.head(torch.cat((fv[:n],fi[:n]),1)),labels,label_smoothing=.05)
            ce=ce+.25*F.cross_entropy(branches[0](fv[:n]),labels,label_smoothing=.05)+.25*F.cross_entropy(branches[1](fi[:n]),labels,label_smoothing=.05)
            contrastive=paired_contrastive_loss(projectors[0](fv),projectors[1](fi),labels)
            loss=ce+args.contrastive_weight*contrastive
            if not torch.isfinite(loss):raise FloatingPointError("Pretraining loss is not finite")
            loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*projectors.parameters(),*branches.parameters()],5.,error_if_nonfinite=True);optimizer.step()
            totals += [float(ce.detach()),float(contrastive.detach())]
        scheduler.step()
        metrics,prototypes=evaluate_representations()
        row={"epoch":epoch,"source_CE":totals[0]/steps,"paired_contrastive":totals[1]/steps,"target_val":metrics,"prototype_accuracy":prototypes,"target_training_labels_used":False}
        history.append(row)
        with (root/"history.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        if best is None or (metrics["accuracy"],metrics["f1_macro"])>(best["target_val"]["accuracy"],best["target_val"]["f1_macro"]):
            best=row
        print(json.dumps(row),flush=True)
    (root/"summary.json").write_text(json.dumps({"kind":"paired_self_supervised_initialization_probe","formal_UDA_result":False,
        "arguments":vars(args),"dataset_audit":audit,"best":best,"history":history,"target_training_labels_used":False,
        "model_checkpoints_saved":False},indent=2)+"\n")


if __name__ == "__main__":main()
