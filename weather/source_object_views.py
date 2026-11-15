"""Extra source-only views and supervised context-invariance loss.

The main VIS/IR pair and all evaluation transforms remain unchanged. Reviewed
boxes create an extra VIS view; other source samples use an independent full
image view. The extra view never receives target images or target labels.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tools.uda_multimodal import jmda_train_transform


class SourceObjectViews:
    def __init__(self,source_view,manifest,compact_map,seed):
        if source_view.partition!="source_train" or not source_view.labelled:
            raise ValueError("Object views may only be built from labelled source_train")
        started=time.perf_counter();path=Path(manifest)
        payload=json.loads(path.read_text(encoding="utf-8"))
        if (payload["kind"]!="reviewed_source_only_object_views" or payload["target_images_used"]
                or payload["source_domain"]!="sunny"):
            raise ValueError("Only the reviewed source-only object manifest is accepted")
        records={r["image_sha256"]:r for r in payload["records"]}
        mapping={int(key):int(value) for key,value in compact_map.items()}
        if len(records)!=len(payload["records"]):raise ValueError("Duplicate source object records")
        self.transform=jmda_train_transform();self.pictures=[];self.roi=[];self.seed=int(seed);self.generator=None
        cached=getattr(source_view.dataset,"_resized_pair_cache",None)
        source_labels=[int(label) for _,label in source_view.dataset.samples]
        roi_classes={}
        for i,source_path in enumerate(source_view.paths):
            digest=hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
            record=records.get(digest)
            if record is not None:
                expected=mapping.get(int(record["original_source_class_id"]))
                if expected!=source_labels[i]:raise ValueError("Object/source class mapping mismatch")
                with Image.open(source_path) as image:
                    if list(image.size)!=record["size"]:raise ValueError("Object image size mismatch")
                    x0,y0,x1,y1=record["box"]
                    if not 0<=x0<x1<=image.width or not 0<=y0<y1<=image.height:
                        raise ValueError("Invalid reviewed source box")
                    picture=self.transform.transforms[0](image.convert("RGB").crop(record["box"]))
                roi_classes[str(source_labels[i])]=roi_classes.get(str(source_labels[i]),0)+1
            elif cached is not None:picture=cached[i][0]
            else:
                with Image.open(source_path) as image:picture=self.transform.transforms[0](image.convert("RGB"))
            self.pictures.append(picture);self.roi.append(record is not None)
        self.metadata={"manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_samples":len(self.pictures),"reviewed_roi_samples":sum(self.roi),
            "independent_full_view_samples":len(self.roi)-sum(self.roi),"roi_compact_class_counts":roi_classes,
            "main_source_pair_changed":False,"target_or_validation_views_changed":False,
            "view_transform":"existing JMDA Resize256/RandomCrop224/HFlip/VIS normalization",
            "view_rng_independent":True,"build_time_s":time.perf_counter()-started}
        if not sum(self.roi):raise ValueError("No reviewed source object matched the current cohort")

    def sample(self,index):
        if self.generator is None:
            worker=torch.utils.data.get_worker_info()
            self.generator=torch.Generator().manual_seed(self.seed+47000+(worker.id if worker else 0))
        seed=int(torch.randint(0,2**31-1,(),generator=self.generator))
        py_state=random.getstate();np_state=np.random.get_state();cpu_state=torch.get_rng_state()
        try:
            random.seed(seed);np.random.seed(seed)
            # Do not use torch.manual_seed: it would also reseed CUDA in a
            # num_workers=0 loader. Only the CPU transform RNG is substituted.
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            picture=self.transform(self.pictures[index].copy())
        finally:
            random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(cpu_state)
        return picture,self.roi[index]


@contextmanager
def frozen_running_statistics(module):
    states=[(m,m.training) for m in module.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    for m,_ in states:m.training=False
    try:yield
    finally:
        for m,mode in states:m.training=mode


def source_object_loss(model,source,source_branches,label_smoothing=0.):
    if source_branches is None or "source_object_rgb" not in source:
        raise ValueError("Live source features and extra source views are required")
    images=source["source_object_rgb"]
    if len(images)!=len(source["label"]):raise ValueError("Source object cohort mismatch")
    if getattr(model,"encoder_precision","fp32")=="bf16":
        with torch.autocast("cuda",dtype=torch.bfloat16):visible=model.encode_visible(images)
        visible=visible.float()
    else:visible=model.encode_visible(images)
    branches=(visible,source_branches[1])
    normalization=model.feature_normalization
    if normalization in {"l2","unit"}:
        scale=math.sqrt(model.image_feature_dim) if normalization=="l2" else 1.
        branches=tuple(F.normalize(x,dim=1,eps=1e-8)*scale for x in branches)
    weights=model.modality_weights()
    fused=torch.cat([2*weights[i]*branch for i,branch in enumerate(branches)],1)
    if normalization=="unit":fused=F.normalize(fused,dim=1,eps=1e-8)
    # Auxiliary views do not overwrite the main joint-domain BN statistics.
    with frozen_running_statistics(model.bottleneck):
        features=model.bottleneck(fused)
    if model.unit_method_features:features=F.normalize(features,dim=1,eps=1e-8)
    logits=model.head(features)
    ce=F.cross_entropy(logits,source["label"],weight=getattr(model,"source_class_weights",None),label_smoothing=label_smoothing)
    consistency=(1-F.cosine_similarity(source_branches[0],visible.detach(),dim=1)).mean()
    return ce,consistency,int((logits.argmax(1)==source["label"]).sum().detach())
