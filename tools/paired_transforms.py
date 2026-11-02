"""Synchronized VIS/IR preprocessing for multimodal weather baselines."""

from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class PairedNaturalImageTransform:
    """Apply identical geometry to VIS and IR while normalizing each modality."""

    def __init__(
        self,
        *,
        size=224,
        resize=256,
        rgb_mean=(0.5, 0.5, 0.5),
        rgb_std=(0.5, 0.5, 0.5),
        random_crop=False,
        horizontal_flip=False,
        vertical_flip=False,
        color_jitter=0.0,
    ):
        self.size = int(size)
        self.resize = int(resize)
        self.rgb_mean = tuple(rgb_mean)
        self.rgb_std = tuple(rgb_std)
        self.random_crop = bool(random_crop)
        self.horizontal_flip = bool(horizontal_flip)
        self.vertical_flip = bool(vertical_flip)
        self.color_jitter = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(float(color_jitter) / 2.0, 0.1),
            )
            if color_jitter
            else None
        )

    def __call__(self, vis, ir):
        vis = TF.resize(vis, self.resize, antialias=True)
        ir = TF.resize(ir, self.resize, antialias=True)
        if self.random_crop:
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                vis, scale=(0.8, 1.0), ratio=(0.75, 4.0 / 3.0)
            )
            vis = TF.resized_crop(
                vis, top, left, height, width, (self.size, self.size), antialias=True
            )
            ir = TF.resized_crop(
                ir, top, left, height, width, (self.size, self.size), antialias=True
            )
        else:
            vis = TF.center_crop(vis, (self.size, self.size))
            ir = TF.center_crop(ir, (self.size, self.size))
        if self.horizontal_flip and bool(torch.rand(()) < 0.5):
            vis, ir = TF.hflip(vis), TF.hflip(ir)
        if self.vertical_flip and bool(torch.rand(()) < 0.5):
            vis, ir = TF.vflip(vis), TF.vflip(ir)
        if self.color_jitter is not None:
            vis = self.color_jitter(vis)
        vis = TF.normalize(TF.to_tensor(vis), self.rgb_mean, self.rgb_std)
        ir = TF.normalize(TF.to_tensor(ir), (0.5,), (0.5,))
        return {"rgb": vis, "nir": ir}


__all__ = ["PairedNaturalImageTransform"]
