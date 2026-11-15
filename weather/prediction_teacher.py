"""Optional frozen source-mapped target clusters for probability averaging.

This is an additional inference ensemble, not the original method's score.
Fitting accepts source_train labels and label-free target_train views only.
"""
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from weather.statistical_calibration import TrainFeatureCalibration,ClusterInitialization,unit


class FrozenClusterPrediction(nn.Module):
    def __init__(self,encoders,weight,teacher_temperature,student_temperature):
        super().__init__()
        if (not 0<weight<1 or not all(math.isfinite(x) and x>0 for x in
                (teacher_temperature,student_temperature))):raise ValueError("Invalid prediction blend settings")
        self.encoders=nn.ModuleList(encoders).requires_grad_(False)
        self.weight=weight
        self.teacher_temperature=teacher_temperature
        self.student_temperature=student_temperature
        self.metadata={}
        self.eval()

    def train(self,mode=True):
        # Parent model.train() must never update the frozen teacher's BN.
        return super().train(False)

    @torch.no_grad()
    def image_features(self,batch):
        return tuple(encoder(batch[key]).float() for encoder,key in zip(self.encoders,("rgb","nir")))

    @staticmethod
    def fuse(branches,disabled=()):
        normal=[F.normalize(x.float(),dim=1) for x in branches]
        original=torch.cat((.94*normal[0],.06*normal[1]),1)
        masked=torch.cat((.94*(torch.zeros_like(normal[0]) if 0 in disabled else normal[0]),
                          .06*(torch.zeros_like(normal[1]) if 1 in disabled else normal[1])),1)
        return masked/original.norm(dim=1,keepdim=True).clamp_min(1e-12)

    def fit(self,source_branches,target_branches,source_labels,classes,seed=42):
        source=self.fuse(source_branches).cpu().numpy()
        target=self.fuse(target_branches).cpu().numpy()
        calibrator=TrainFeatureCalibration.fit(source,target,16,.1)
        with torch.no_grad():
            sa=calibrator(torch.tensor(source),"source").numpy()
            tb=calibrator(torch.tensor(target),"target").numpy()
        parts=[];association_audits=[]
        half=source.shape[1]//2
        for sl in (slice(None,half),slice(half,None)):
            a,b=unit(source[:,sl]),unit(target[:,sl])
            c=TrainFeatureCalibration.fit(a,b,32,.1)
            with torch.no_grad():parts.append([c(torch.tensor(x,dtype=torch.float32),d).numpy()
                for x,d in ((a,"source"),(b,"target"))])
            association_audits.append(c.metadata)
        association=[unit(np.concatenate((.75*parts[0][i],.25*parts[1][i]),1)) for i in range(2)]
        self.calibrator=calibrator
        self.cluster=ClusterInitialization.fit(sa,tb,source_labels,classes,.1,seed,
            association_source=association[0],association_target=association[1])
        self.metadata.update(calibration=calibrator.metadata,cluster_initialization=self.cluster.metadata,
            association_calibrations=association_audits,teacher_VIS_weight=.94,association_VIS_weight=.75,
            fit_partitions=["source_train","target_train"],validation_used_for_fitting=False,
            target_training_labels_used=False,probability_weight=self.weight,
            teacher_temperature=self.teacher_temperature,student_temperature=self.student_temperature,
            student_objective_changed=False,teacher_frozen=True,
            inference="constant weighted average of two softmax probability vectors",
            extra_inference_image_encoders=2)
        return self

    def combine(self,student_logits,teacher_logits):
        probabilities=(self.weight*(teacher_logits/self.teacher_temperature).softmax(1)
            +(1-self.weight)*(student_logits/self.student_temperature).softmax(1))
        return probabilities.clamp_min(1e-12).log()

    def forward(self,student_logits,batch,domain="target",disabled=()):
        with torch.no_grad():
            raw=self.fuse(self.image_features(batch),disabled)
            teacher_logits=self.cluster(self.calibrator(raw,domain))
        return self.combine(student_logits,teacher_logits)


def prepare_prediction_teacher(model,source_eval,target_eval,device,weights,weight,
                               teacher_temperature=1.,student_temperature=1.,seed=42):
    started=time.perf_counter()
    if (source_eval.partition!="source_train" or not source_eval.labelled
            or target_eval.partition!="target_train" or target_eval.labelled):
        raise ValueError("Prediction teacher accepts labelled source_train and label-free target_train only")
    if not weights:raise ValueError("Provide offline ResNet-18 prediction teacher weights")
    path=Path(weights)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):raise ValueError("Prediction teacher requires the official ResNet-18 V1 weights")
    from torchvision.models import resnet18
    # Separate frozen ImageNet encoders retain the same fixed teacher for all
    # five student methods, including students with different native backbones.
    with torch.random.fork_rng(devices=[]):
        state=torch.load(path,map_location="cpu",weights_only=True)
        encoders=[]
        for _ in range(2):
            encoder=resnet18(weights=None);encoder.load_state_dict(state,strict=True)
            encoder.fc=nn.Identity();encoders.append(encoder)
        teacher=FrozenClusterPrediction(encoders,weight,teacher_temperature,student_temperature).to(device)
    def collect(view):
        values=[[],[]];labels=[];paths=[]
        with torch.no_grad():
            for batch in DataLoader(view,batch_size=32,num_workers=0,shuffle=False,
                    generator=torch.Generator().manual_seed(seed)):
                if not view.labelled and "label" in batch:raise ValueError("Target labels entered teacher fitting")
                inputs={key:batch[key].to(device) for key in ("rgb","nir")}
                for rows,features in zip(values,teacher.image_features(inputs)):rows.append(features.cpu())
                paths.extend(batch["path"])
                if view.labelled:labels.extend(batch["label"].tolist())
        return tuple(torch.cat(x) for x in values),np.asarray(labels,dtype=np.int64),paths
    a,labels,source_paths=collect(source_eval);b,absent,target_paths=collect(target_eval)
    if len(absent):raise ValueError("Target labels entered teacher fitting")
    teacher.fit(a,b,labels,model.num_classes,seed).to(device)
    teacher.metadata.update(weights_sha256=digest,source_samples=len(source_paths),target_samples=len(target_paths),
        source_cohort_sha256=hashlib.sha256("\n".join(source_paths).encode()).hexdigest(),
        target_cohort_sha256=hashlib.sha256("\n".join(target_paths).encode()).hexdigest())
    teacher.metadata["preparation_time_s"]=time.perf_counter()-started
    model.prediction_teacher=teacher
    return teacher.metadata
