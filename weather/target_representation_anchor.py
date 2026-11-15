"""Training-only anchoring to initial target-train encoder representations.

Targets are feature vectors computed before the first optimizer update, not
class labels. No additional image encoder is used during training/inference.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class TargetRepresentationCache(nn.Module):
    def __init__(self, paths, features, names=("VIS","IR")):
        super().__init__()
        self.paths=tuple(str(p) for p in paths)
        if not self.paths or len(set(self.paths))!=len(self.paths):
            raise ValueError("Initial target representations need unique nonempty training paths")
        self.lookup={p:i for i,p in enumerate(self.paths)}
        self.names=tuple(names)
        if len(features)!=len(names):raise ValueError("Representation modality count mismatch")
        for i,values in enumerate(features):
            values=values.detach().float().clone()
            if (values.ndim!=2 or len(values)!=len(paths) or not torch.isfinite(values).all()
                    or (values.norm(dim=1)<1e-8).any()):
                raise ValueError("Invalid initial target representations")
            self.register_buffer(f"reference_{i}",F.normalize(values,dim=1))

    def forward(self, features, paths):
        if len(features)!=len(self.names):raise ValueError("Live encoder modality count mismatch")
        try:indices=[self.lookup[str(p)] for p in paths]
        except KeyError as exc:raise ValueError("Unknown target training path in representation loss") from exc
        losses=[];cosines=[]
        for i,live in enumerate(features):
            reference=getattr(self,f"reference_{i}")
            if live.shape!=(len(paths),reference.shape[1]):raise ValueError("Live/initial representation shape mismatch")
            index=torch.tensor(indices,device=reference.device,dtype=torch.long)
            cosine=(F.normalize(live.float(),dim=1)*reference[index]).sum(1).clamp(-1,1)
            losses.append((1-cosine).mean());cosines.append(cosine.detach().mean())
        return torch.stack(losses).mean(),dict(zip(self.names,torch.stack(cosines).cpu().tolist()))


def prepare_target_representation_cache(model, view, device, output, batch_size=32):
    if view.partition!="target_train" or view.labelled:
        raise ValueError("Representation anchoring may only read unlabelled target_train")
    started=time.perf_counter();was_training=model.training
    collected=None;paths=[]
    try:
        model.eval();model.set_bn_domain("target")
        with torch.no_grad():
            for batch in DataLoader(view,batch_size=batch_size,shuffle=False,num_workers=0,
                                    generator=torch.Generator().manual_seed(0)):
                if {"label","labels","y_true"}&batch.keys():raise RuntimeError("Target labels entered representation initialization")
                model.encode({key:batch[key].to(device) for key in ("rgb","nir")})
                values=model.last_features
                if collected is None:collected=[[] for _ in values]
                for rows,values in zip(collected,values):rows.append(values.cpu())
                paths.extend(batch["path"])
    finally:
        model.train(was_training)
    if paths!=list(view.paths):raise RuntimeError("Initial target representation cohort order changed")
    features=tuple(torch.cat(rows) for rows in collected)
    cache=TargetRepresentationCache(paths,features).to(device)
    destination=Path(output)/"target_initial_representations.npz"
    arrays={"paths":np.asarray(paths)}
    arrays.update({name:values.numpy() for name,values in zip(cache.names,features)})
    with destination.open("wb") as stream:np.savez_compressed(stream,**arrays)
    metadata={"kind":"initial-encoder target representation anchor; not class pseudo-labels",
        "fit_partition":"target_train","target_training_labels_used":False,"validation_used":False,
        "initialization_before_optimizer_updates":True,"samples":len(paths),
        "feature_widths":[v.shape[1] for v in features],"extra_trainable_parameters":0,
        "extra_inference_image_encoders":0,"inference_changed":False,
        "loss":"mean of per-modality cosine distances to cached initial target representation",
        "initial_target_mean_norms":[float(v.norm(dim=1).mean()) for v in features],
        "target_cohort_sha256":hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        "cache_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
        "cache_file":str(destination),"preparation_time_s":time.perf_counter()-started}
    (Path(output)/"target_representation_anchor_audit.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return cache,metadata
