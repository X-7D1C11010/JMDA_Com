"""Optional encoder warm-up freeze and sampled optimizer-update diagnostics."""
import math
import torch
from torch import nn


def scale_image_lr_for_cached_distillation(optimizer,scheduler,factor=1.):
    """Change encoder LRs after pseudo-cache creation, keeping that cache fixed."""
    if not math.isfinite(factor) or factor<=0:raise ValueError("Invalid distillation encoder LR scale")
    changes=[]
    if factor!=1:
        for i,group in enumerate(optimizer.param_groups):
            if group.get("role") in {"pretrained_RGB_NIR","pretrained_RGB","pretrained_NIR"}:
                before=group["lr"];group["lr"]*=factor
                if "initial_lr" in group:group["initial_lr"]*=factor
                if scheduler is not None:scheduler.base_lrs[i]*=factor
                changes.append({"role":group["role"],"before":before,"after":group["lr"]})
    return {"encoder_LR_scale_after_cache":factor,"changes":changes,
            "source_bootstrap_and_pseudo_probabilities_changed":False,"head_learning_rates_changed":False}


def lower_image_lr_after_source_warmup(optimizer, scheduler, epoch, warmup_epochs, factor=1.):
    """One explicit phase transition, preserving head LR and scheduler clocks."""
    if not 0 < factor <= 1:
        raise ValueError("Post-warmup image LR factor must lie in (0,1]")
    changed = []
    if factor != 1 and warmup_epochs > 0 and epoch == warmup_epochs+1:
        for i, group in enumerate(optimizer.param_groups):
            if group.get("role") in {"pretrained_RGB_NIR", "pretrained_RGB", "pretrained_NIR"}:
                group["lr"] *= factor
                if "initial_lr" in group:
                    group["initial_lr"] *= factor
                if scheduler is not None:
                    scheduler.base_lrs[i] *= factor
                changed.append(i)
    return {"image_LR_phase_factor": factor, "changed_optimizer_groups": changed}


def apply_encoder_warmup(model, epoch, freeze_epochs):
    if freeze_epochs < 0:
        raise ValueError("Encoder warm-up freeze must be nonnegative")
    if not freeze_epochs:
        return {"warmup_frozen": False, "freeze_epochs": 0}
    if not hasattr(model, "_encoder_warmup_trainability"):
        flags = [(p, p.requires_grad) for encoder in (model.rgb_encoder, model.nir_encoder) for p in encoder.parameters()]
        if not any(enabled for _, enabled in flags):
            raise ValueError("Encoder warm-up requires an encoder that will become trainable")
        model._encoder_warmup_trainability = flags
    frozen = epoch <= freeze_epochs
    for parameter, enabled in model._encoder_warmup_trainability:
        parameter.requires_grad_(enabled and not frozen)
    return {"warmup_frozen": frozen, "freeze_epochs": freeze_epochs,
            "encoder_trainable_parameters": sum(p.numel() for p, _ in model._encoder_warmup_trainability if p.requires_grad)}


def sampled_parameters(model):
    result = {}
    for name, encoder in (("VIS", model.rgb_encoder), ("IR", model.nir_encoder)):
        conv = [module for module in encoder.modules() if isinstance(module, nn.Conv2d)]
        if conv:
            result[name+"_first_conv"] = conv[0].weight
            result[name+"_last_conv"] = conv[-1].weight
        matrices = [p for p in encoder.parameters() if p.requires_grad and p.ndim > 1]
        if matrices:
            result[name+"_last_trainable_matrix"] = matrices[-1]
    layers = [module for module in model.head.modules() if isinstance(module, nn.Linear)]
    if layers:
        result["classifier_last_weight"] = layers[-1].weight
    result["modality_gate"] = model.modality_logits
    return result


@torch.no_grad()
def capture_parameter_sample(model):
    return {name: (parameter, parameter.detach().clone()) for name, parameter in sampled_parameters(model).items()}


@torch.no_grad()
def parameter_update_record(before, optimizer):
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    result = {}
    for name, (parameter, previous) in before.items():
        norm, delta = float(previous.norm()), float((parameter-previous).norm())
        group = groups.get(id(parameter), {})
        result[name] = {"weight_L2_before": norm, "update_L2": delta,
                        "relative_update_L2": delta/norm if norm > 1e-12 else None,
                        "requires_grad": parameter.requires_grad, "gradient_present": parameter.grad is not None,
                        "lr": group.get("lr"), "optimizer_role": group.get("role"),
                        "momentum_buffer_present": "momentum_buffer" in optimizer.state.get(parameter, {})}
    return result
