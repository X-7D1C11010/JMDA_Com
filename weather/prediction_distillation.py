"""Train-only pseudo-probabilities from the fixed source-mapped ensemble."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class TargetProbabilityCache(nn.Module):
    def __init__(self,paths,classes):
        super().__init__()
        if len(set(paths))!=len(paths):raise ValueError("Duplicate target pseudo-probability IDs")
        self.paths=tuple(paths);self.indices={p:i for i,p in enumerate(paths)}
        self.register_buffer("probabilities",torch.full((len(paths),classes),1/classes))
        self.register_buffer("ready",torch.tensor(False))

    def assign(self,probabilities):
        p=torch.as_tensor(probabilities,device=self.probabilities.device,dtype=torch.float32)
        if p.shape!=self.probabilities.shape or not torch.isfinite(p).all() or (p<0).any():
            raise ValueError("Invalid target pseudo-probabilities")
        if not torch.allclose(p.sum(1),torch.ones(len(p),device=p.device),atol=1e-5):
            raise ValueError("Target pseudo-probabilities must sum to one")
        with torch.no_grad():self.probabilities.copy_(p);self.ready.fill_(True)

    def forward(self,logits,paths):
        if not bool(self.ready):raise RuntimeError("Target pseudo-probabilities not prepared")
        if len(paths)!=len(logits):raise ValueError("Pseudo-probability batch identity mismatch")
        try:indices=[self.indices[p] for p in paths]
        except KeyError as error:raise ValueError("Unknown target pseudo-probability ID") from error
        reference=self.probabilities[indices].detach().to(logits.device)
        return F.kl_div(logits.log_softmax(1),reference,reduction="batchmean")


def prepare_cached_target_probabilities(model,target_eval,device,directory,epoch,sharpening=.5,seed=42):
    started=time.perf_counter()
    if target_eval.partition!="target_train" or target_eval.labelled:
        raise ValueError("Pseudo-probabilities may only use label-free target_train")
    if not hasattr(model,"prediction_teacher") or not hasattr(model,"target_probability_cache"):
        raise ValueError("Install the frozen prediction teacher and empty cache first")
    model.eval();model.set_bn_domain("target");paths=[];rows=[]
    with torch.no_grad():
        for batch in DataLoader(target_eval,batch_size=32,shuffle=False,num_workers=0,
                generator=torch.Generator().manual_seed(seed)):
            if "label" in batch:raise ValueError("Target labels entered pseudo-probability preparation")
            inputs={k:batch[k].to(device) for k in ("rgb","nir")}
            logits=model.head(model.encode(inputs))
            blended=model.prediction_teacher(logits,inputs,"target")
            rows.append((blended/sharpening).softmax(1).cpu())
            paths.extend(batch["path"])
    if tuple(paths)!=model.target_probability_cache.paths:raise ValueError("Target pseudo-probability cohort order changed")
    probabilities=torch.cat(rows);model.target_probability_cache.assign(probabilities)
    output=Path(directory)/"target_pseudo_probabilities.npz"
    np.savez_compressed(output,paths=np.asarray(paths),probabilities=probabilities.numpy())
    report={"prepared_before_epoch":epoch,"samples":len(paths),"target_training_labels_used":False,
        "validation_used":False,"teacher_training_source":"source_train labels and label-free target_train statistics",
        "current_source_supervised_student_epoch":epoch-1,"sharpening_temperature":sharpening,
        "mean_confidence":float(probabilities.max(1).values.mean()),
        "pseudo_class_counts":probabilities.argmax(1).bincount(minlength=probabilities.shape[1]).tolist(),
        "cohort_sha256":hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "probability_file_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "probability_values_sha256":hashlib.sha256(probabilities.numpy().tobytes()).hexdigest(),
        "inference_includes_teacher":False,"pseudo_predictions_are_ground_truth":False}
    # Stable keys/shapes also allow restoration of a pre-distillation RAM best.
    model.prediction_teacher.to("cpu")
    report["preparation_time_s"]=time.perf_counter()-started
    (Path(directory)/"prediction_distillation_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
