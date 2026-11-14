"""Opt-in paired FDA input experiment; disabled in every default profile.

Low-frequency amplitude replacement follows the authors' NumPy formulation:
https://github.com/YanchaoYang/FDA/blob/master/utils/__init__.py
This is an additional input-augmentation variant, not a named UDA core loss.
"""
import math
import torch


def replace_low_frequency_amplitude(content, style, beta):
    if content.shape != style.shape or content.ndim != 4:
        raise ValueError("FDA requires equal NCHW image shapes")
    if not 0 <= beta <= .25:
        raise ValueError("FDA beta must be in [0,.25]")
    if beta == 0:
        return content
    spectrum = torch.fft.fft2(content)
    amplitude = torch.fft.fftshift(spectrum.abs(), dim=(-2, -1))
    reference = torch.fft.fftshift(torch.fft.fft2(style).abs(), dim=(-2, -1))
    height, width = content.shape[-2:]
    radius = math.floor(min(height, width)*beta)
    rows = slice(height//2-radius, height//2+radius+1)
    columns = slice(width//2-radius, width//2+radius+1)
    amplitude[..., rows, columns] = reference[..., rows, columns]
    spectrum = torch.polar(torch.fft.ifftshift(amplitude, dim=(-2, -1)), spectrum.angle())
    return torch.fft.ifft2(spectrum).real


@torch.no_grad()
def augment_source_style(source, target, beta=0., probability=.5, generator=None):
    record = {"FDA_beta": beta, "FDA_probability": probability, "FDA_applied_samples": 0,
              "style_partition": "target_train", "style_labels_used": False, "FDA_pixel_statistics": {}}
    if not 0 <= probability <= 1:
        raise ValueError("FDA probability must be in [0,1]")
    if beta == 0 or probability == 0:
        return source, record
    if generator is None:
        raise ValueError("Use a dedicated augmentation generator to preserve training RNG")
    count = len(source["rgb"])
    mask = torch.rand(count, 1, 1, 1, device=source["rgb"].device, generator=generator) < probability
    record["FDA_applied_samples"] = int(mask.sum())
    result = dict(source)
    for key in ("rgb", "nir"):
        value = source[key]
        if key == "rgb":
            mean = value.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
            std = value.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        else:
            mean, std = .5, .5
        raw = value*std+mean
        index = torch.arange(count, device=value.device) % len(target[key])
        style = target[key][index]*std+mean
        unclipped = replace_low_frequency_amplitude(raw, style, beta)
        transformed = unclipped.clamp(0., 1.)
        if record["FDA_applied_samples"]:
            selected = mask.flatten()
            content_values, reference_values = raw[selected], style[selected]
            before, after = unclipped[selected], transformed[selected]
            record["FDA_pixel_statistics"][key] = {
                "source_mean": float(content_values.mean()), "source_std": float(content_values.std(unbiased=False)),
                "style_mean": float(reference_values.mean()), "style_std": float(reference_values.std(unbiased=False)),
                "styled_mean": float(after.mean()), "styled_std": float(after.std(unbiased=False)),
                "clipped_fraction": float(((before < 0) | (before > 1)).float().mean()),
            }
        result[key] = (torch.where(mask, transformed, raw)-mean)/std
    return result, record
