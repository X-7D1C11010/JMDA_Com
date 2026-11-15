"""Optional source-only feature-statistic augmentation using unlabelled targets.

Reference idea: Zhou et al., MixStyle, https://github.com/KaiyangZhou/mixstyle-release .
This implementation is an explicitly different variant: only source features
change, the two modalities share target indices/mixing coefficients, and the
source proportion is uniform in [0, source_max]. All moment tensors are
detached. It is an extra training augmentation, not a named DA method core.
"""
import torch
from torch import nn


class TargetFeatureStyle(nn.Module):
    def __init__(self,probability=.5,source_max=.5,seed=42):
        super().__init__()
        if not 0<=probability<=1 or not 0<=source_max<=1:
            raise ValueError("Invalid feature-style probability or source proportion")
        self.probability=float(probability);self.source_max=float(source_max);self.seed=int(seed)
        self.context=None;self.records={};self.handles=[]

    def begin_step(self,update,source_count,target_count):
        if min(source_count,target_count)<1:raise ValueError("Both training domains are required")
        self.records={}
        generator=torch.Generator().manual_seed(self.seed+41000+int(update))
        chosen=torch.rand(source_count,generator=generator)<self.probability
        source_fraction=torch.rand(source_count,generator=generator)*self.source_max
        target_indices=torch.randperm(target_count,generator=generator).repeat(
            (source_count+target_count-1)//target_count)[:source_count]
        self.context={"source_count":source_count,"target_count":target_count,"chosen":chosen,
                      "source_fraction":source_fraction,"target_indices":target_indices,
                      "selected_count":int(chosen.sum()),"update":int(update)}

    def spatial(self,features,name):
        if not self.training or not self.probability:return features
        if self.context is None:raise RuntimeError("Prepare a joint training step before feature mixing")
        ns,nt=self.context["source_count"],self.context["target_count"]
        if features.ndim!=4 or len(features)!=ns+nt:
            raise ValueError("Feature style requires a joint NCHW source/target forward")
        if not self.context["selected_count"]:return features
        dtype=features.dtype
        # The transformed source feature derivative never passes through a
        # target statistic, nor through a source statistic's normalization.
        values=features.float()
        means=values.mean((-2,-1),keepdim=True).detach()
        scales=(values.var((-2,-1),unbiased=False,keepdim=True)+1e-6).sqrt().detach()
        indices=self.context["target_indices"].to(features.device)
        fraction=self.context["source_fraction"].to(features.device).view(ns,1,1,1)
        chosen=self.context["chosen"].to(features.device).view(ns,1,1,1)
        reference_mean=means[ns:][indices]
        reference_scale=scales[ns:][indices]
        # Bound gain to prevent near-constant target patches causing an
        # arbitrarily large/small source feature scale.
        reference_scale=scales[:ns]*(reference_scale/scales[:ns]).clamp(.25,4.)
        mixed_mean=fraction*means[:ns]+(1-fraction)*reference_mean
        mixed_scale=fraction*scales[:ns]+(1-fraction)*reference_scale
        styled=(values[:ns]-means[:ns])/scales[:ns]*mixed_scale+mixed_mean
        source=torch.where(chosen,styled,values[:ns]).to(dtype)
        before=(means[:ns]-reference_mean).square().mean().sqrt()
        after=(torch.where(chosen,mixed_mean,means[:ns])-reference_mean).square().mean().sqrt()
        self.records[name]=(before,after)
        # The target portion remains bitwise equal and in its original order.
        return torch.cat((source,features[ns:]),0)

    def tokens(self,features,name):
        if not self.training or not self.probability:return features
        if features.ndim!=3 or features.shape[1]<2:raise ValueError("Expected CLS plus patch tokens")
        patches=self.spatial(features[:,1:].transpose(1,2).unsqueeze(-1),name)
        return torch.cat((features[:,:1],patches.squeeze(-1).transpose(1,2)),1)

    def diagnostics(self):
        if self.context is None:return {"active":False}
        values=(torch.stack([x for pair in self.records.values() for x in pair]).detach().cpu().tolist()
                if self.records else [])
        layers={name:{"moment_mean_gap_before":values[2*i],"moment_mean_gap_after":values[2*i+1]}
                for i,name in enumerate(self.records)}
        selected=self.context["chosen"]
        return {"active":bool(self.records),"update":self.context["update"],
                "source_count":self.context["source_count"],"target_count":self.context["target_count"],
                "styled_source_samples":self.context["selected_count"],
                "source_fraction_mean":float(self.context["source_fraction"][selected].mean()) if bool(selected.any()) else None,
                "target_indices":self.context["target_indices"].tolist(),"layers":layers,
                "target_features_changed":False,"target_statistics_detached":True}


def install_target_feature_style(model,probability,source_max=.5,seed=42):
    if hasattr(model,"target_feature_style"):raise ValueError("Feature style is already installed")
    if getattr(model,"image_backbone",None)!="convnext_small" or getattr(model,"use_ais",False):
        raise ValueError("The feature-style screen currently supports paired weather ConvNeXt/UNIP only")
    mixer=TargetFeatureStyle(probability,source_max,seed)
    layers=[]
    for modality,encoder in (("VIS",model.rgb_encoder),("IR",model.nir_encoder)):
        if hasattr(encoder,"trunk"):
            selections=(("patch_projection",encoder.trunk.patch_embed.proj,False),
                        ("early_tokens",encoder.trunk.blocks[1],True))
        else:
            selections=(("stage1",encoder.features[1][-1],False),("stage2",encoder.features[3][-1],False))
        for stage,module,tokens in selections:
            name=modality+"."+stage
            def hook(unused_module,unused_inputs,output,name=name,tokens=tokens):
                return mixer.tokens(output,name) if tokens else mixer.spatial(output,name)
            mixer.handles.append(module.register_forward_hook(hook));layers.append(name)
    model.target_feature_style=mixer
    return {"probability":probability,"source_max":source_max,"layers":layers,
        "fit_or_style_data":"source_train features and label-free target_train moments",
        "applies_to":"source features during joint training forward only",
        "shared_style_reference_across_modalities":True,"target_features_changed":False,
        "target_statistics_detached":True,"extra_trainable_parameters":0,
        "inference_active":False,"reference":"https://github.com/KaiyangZhou/mixstyle-release",
        "variant_differences":"source-only, paired modality references, uniform source fraction, bounded std ratio"}
