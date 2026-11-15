"""Opt-in IR-only source intensity-statistics augmentation using target_train.

This affine pixel transform preserves ship geometry and source labels. It
does not alter VIS or target images, and never runs during evaluation.
"""
import math
import torch


@torch.no_grad()
def augment_source_ir_statistics(source, target, probability=0., strength=1., generator=None):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("IR-statistics probability must lie in [0,1]")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("IR-statistics strength must lie in [0,1]")
    record = {"source_IR_statistics_probability": probability, "source_IR_statistics_strength": strength,
              "source_IR_statistics_augmented": 0, "IR_statistics_fit_partition": "target_train",
              "IR_statistics_target_labels_used": False, "target_IR_changed": False}
    if not probability or not strength:
        return source, record
    if generator is None:
        raise ValueError("IR augmentation requires a private generator")
    image, reference = source["nir"], target["nir"]
    if (image.ndim != 4 or reference.ndim != 4 or not len(image) or not len(reference)
            or image.shape[1] != reference.shape[1] or image.device != reference.device):
        raise ValueError("Expected nonempty compatible source/target NCHW IR tensors")
    n = len(image)
    active = torch.rand((n, 1, 1, 1), device=image.device, generator=generator) < probability
    index = torch.randint(len(reference), (n,), device=image.device, generator=generator)
    # Replicated gray channels use shared statistics, preserving their equality.
    raw = (image * .5 + .5).clamp(0, 1)
    style = (reference.detach()[index] * .5 + .5).clamp(0, 1)
    dims = (1, 2, 3)
    source_mean = raw.mean(dims, keepdim=True)
    source_std = raw.std(dims, unbiased=False, keepdim=True)
    target_mean = style.mean(dims, keepdim=True)
    target_std = style.std(dims, unbiased=False, keepdim=True)
    ratio = (target_std / source_std.clamp_min(1e-5)).clamp(.05, 4.)
    # Source content has a positive affine coefficient even at full strength.
    transformed = (raw-source_mean) * ((1-strength) + strength*ratio)
    transformed = transformed + (1-strength)*source_mean + strength*target_mean
    selected = active.flatten()
    record["source_IR_statistics_augmented"] = int(selected.sum())
    if bool(selected.any()):
        before, after = raw[selected], transformed[selected]
        record.update(source_IR_std_before=float(source_std[selected].mean()),
                      reference_IR_std=float(target_std[selected].mean()),
                      source_IR_std_after=float(after.clamp(0,1).std(dims,unbiased=False).mean()),
                      source_IR_gain_mean=float(ratio[selected].mean()),
                      source_IR_clipped_fraction=float(((after<0)|(after>1)).float().mean()))
    result = dict(source, nir=torch.where(active, transformed.clamp(0,1)*2-1, image))
    return result, record
