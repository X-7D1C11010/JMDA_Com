"""Optional selected-state embeddings for diagnostics, never student weights."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def cache_selected_features(model,views,device,output,epoch,batch_size=32):
    if [view.partition for view in views]!=["source_train","target_train","target_val"]:
        raise ValueError("Feature cache needs the declared training and validation views in order")
    if views[1].labelled:raise ValueError("Target training feature cache must be label-free")
    packed={};audits=[];attention_audits={}
    model.eval()
    for key,view in zip(("source","target_train","target_val"),views):
        model.set_bn_domain("source" if key=="source" else "target")
        arrays=[];labels=[];paths=[];attentions=[]
        with torch.no_grad():
            for batch in DataLoader(view,batch_size=batch_size,num_workers=0,shuffle=False,
                                    generator=torch.Generator().manual_seed(42)):
                if key=="target_train" and "label" in batch:raise RuntimeError("Target labels entered the diagnostic feature cache")
                model.encode({name:batch[name].to(device) for name in ("rgb","nir")})
                arrays.append(torch.cat(model.last_features,1).cpu().numpy())
                attention=getattr(model.nir_encoder,"last_pooling_attention",None)
                if attention is not None:attentions.append(attention.detach().float().cpu().numpy())
                paths.extend(batch["path"])
                if view.labelled:labels.extend(batch["label"].tolist())
        if len(paths)!=len(view):raise RuntimeError("Incomplete feature cache")
        packed[key]=np.concatenate(arrays)
        packed[key+"_paths"]=np.asarray(paths)
        if view.labelled:packed[key+"_labels"]=np.asarray(labels,dtype=np.int64)
        if attentions:
            values=np.concatenate(attentions)
            if (len(values)!=len(paths) or not np.isfinite(values).all() or (values<0).any()
                    or not np.allclose(values.sum(1),1.,atol=1e-5)):
                raise RuntimeError("Incomplete or invalid IR attention diagnostic")
            packed[key+"_IR_attention"]=values
            attention_audits[key]={"samples":len(values),"patches":values.shape[1],
                "kind":"network attention weights; paths align with corresponding feature rows"}
        audits.append({"partition":view.partition,"samples":len(paths),"labels_in_cache":view.labelled,
                       "cohort_sha256":hashlib.sha256("\n".join(paths).encode()).hexdigest()})
    path=Path(output)/"selected_feature_cache.npz"
    temporary=path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:np.savez_compressed(stream,**packed)
    temporary.replace(path)
    record={"epoch":epoch,"kind":"selected-state image embeddings only; not a model checkpoint",
        "file":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "modality_widths":[model.image_feature_dim,model.ir_feature_dim],"modality_weights":model.modality_weights().detach().cpu().tolist(),
        "target_training_labels_used":False,"statistics_or_models_fitted":False,"partitions":audits}
    if attention_audits:record["infrared_attention_maps"]=attention_audits
    (path.parent/"selected_feature_cache.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    return record
