"""Explicit optional paired-modality losses; target instances need no labels."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def install_crossmodal_projectors(model,width,seed=42):
    """Train-only nonlinear projection; classifier fusion stays unchanged.

    Separate projection for the contrastive task follows the general rationale
    studied in https://proceedings.mlr.press/v119/chen20j.html . This is an
    additional multimodal loss variant, not a reproduction of SimCLR.
    """
    if width<2:raise ValueError("Projection width must be at least two")
    if hasattr(model,"crossmodal_projectors"):raise ValueError("Projection already installed")
    with torch.random.fork_rng(devices=[]):
        projectors=nn.ModuleList(nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Linear(256,width))
            for dim in (model.image_feature_dim,model.ir_feature_dim))
        generator=torch.Generator().manual_seed(seed+39000)
        for layer in projectors.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight,generator=generator)
                nn.init.zeros_(layer.bias)
    model.crossmodal_projectors=projectors.to(next(model.parameters()).device)
    return {"type":"separate Linear-ReLU-Linear projections; training only","width":width,
            "input_widths":[model.image_feature_dim,model.ir_feature_dim],
            "trainable_parameters":sum(p.numel() for p in projectors.parameters()),
            "classification_feature_fusion_changed":False,"training_RNG_preserved":True}


def paired_crossmodal_loss(visible, infrared, temperature=.2, source_labels=None):
    if (visible.ndim != 2 or visible.shape != infrared.shape or len(visible) < 2
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Crossmodal alignment requires equal feature widths, paired rows, and a positive temperature")
    visible, infrared = F.normalize(visible, dim=1), F.normalize(infrared, dim=1)
    logits = visible @ infrared.T / temperature
    if source_labels is None:
        # Exact VIS/IR pairing supplies the positive index, never a class label.
        labels = torch.arange(len(visible), device=visible.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    else:
        if tuple(source_labels.shape) != (len(visible),): raise ValueError("Source label shape mismatch")
        positives = (source_labels[:, None] == source_labels[None, :]).to(logits.dtype)
        loss = (-(logits.log_softmax(1) * positives).sum(1) / positives.sum(1)).mean()
        loss = (loss + (-(logits.T.log_softmax(1) * positives.T).sum(1) / positives.T.sum(1)).mean()) / 2
    return loss, (visible * infrared).sum(1).mean().detach()


def paired_variance_covariance_loss(visible, infrared):
    """Negative-free paired target regularizer, with VICReg's 25:25:1 ratio.

    Formula reference: https://github.com/facebookresearch/vicreg . Divide the
    complete objective by 25 so its external weight is explicit. This is an
    opt-in crossmodal adaptation variant, not the original VICReg experiment.
    Inputs are unnormalized training-projection outputs; no class labels enter.
    """
    if (visible.ndim != 2 or visible.shape != infrared.shape or len(visible) < 2
            or visible.shape[1] < 2):
        raise ValueError("Paired variance/covariance needs at least two rows and equal feature widths")
    visible, infrared = visible.float(), infrared.float()
    invariance = F.mse_loss(visible, infrared)
    stds = [torch.sqrt(x.var(0, unbiased=True) + 1e-4) for x in (visible, infrared)]
    variance = sum(F.relu(1-std).mean() for std in stds)/2
    covariances = []
    width = visible.shape[1]
    for values in (visible, infrared):
        centered = values-values.mean(0)
        covariance = centered.T @ centered/(len(values)-1)
        off_diagonal = covariance-torch.diag_embed(covariance.diagonal())
        covariances.append(off_diagonal.square().sum()/width)
    covariance_loss = sum(covariances)
    loss = invariance+variance+covariance_loss/25
    details = {"invariance": invariance, "variance": variance, "covariance": covariance_loss,
               "VIS_mean_std": stds[0].mean(), "IR_mean_std": stds[1].mean()}
    cosine = F.cosine_similarity(visible, infrared, dim=1).mean().detach()
    return loss, cosine, details


class SourceCategoryMemory(nn.Module):
    """Source-GT class prototypes provide positives beyond a small batch."""
    def __init__(self,classes,width,momentum=.9):
        super().__init__()
        if not 0<=momentum<1:raise ValueError("Invalid source prototype momentum")
        self.momentum=momentum
        self.register_buffer("visible",torch.zeros(classes,width))
        self.register_buffer("infrared",torch.zeros(classes,width))
        self.register_buffer("observations",torch.zeros(classes,dtype=torch.long))

    def forward(self,visible,infrared,source_labels,temperature=.2,class_weights=None):
        if visible.shape!=infrared.shape or visible.shape[1]!=self.visible.shape[1]:raise ValueError("Prototype feature shape mismatch")
        if source_labels.ndim!=1 or len(source_labels)!=len(visible) or int(source_labels.min())<0 or int(source_labels.max())>=len(self.visible):
            raise ValueError("Invalid source class labels")
        visible,infrared=F.normalize(visible,dim=1),F.normalize(infrared,dim=1)
        with torch.no_grad():
            for label in source_labels.unique().tolist():
                selected=source_labels==label
                for bank,values in ((self.visible,visible),(self.infrared,infrared)):
                    center=F.normalize(values[selected].detach().mean(0),dim=0)
                    if self.observations[label]:center=F.normalize(self.momentum*bank[label]+(1-self.momentum)*center,dim=0)
                    bank[label].copy_(center)
                self.observations[label]+=selected.sum()
        seen=self.observations>0
        v=visible@self.infrared.detach().clone().T/temperature
        i=infrared@self.visible.detach().clone().T/temperature
        v=v.masked_fill(~seen[None],-1e4);i=i.masked_fill(~seen[None],-1e4)
        loss=(F.cross_entropy(v,source_labels,weight=class_weights)+F.cross_entropy(i,source_labels,weight=class_weights))/2
        return loss,int(seen.sum())
