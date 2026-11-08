"""Shared helpers for protocol-faithful VIS and IR single-modal baselines.

VIS runs keep the transforms and backbones published by each adaptation
method.  IR runs are the only allowed architectural extension: an ImageNet
pretrained RGB stem is converted to one input channel by averaging its three
kernel channels.  Everything after the input stem is left untouched.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


MODALITIES = ("vis", "ir")
IR_IMAGE_MODE = "L"
VIS_IMAGE_MODE = "RGB"

# Channel-wise averages of the ImageNet RGB normalization used by the
# pretrained ResNet family.  Keeping the normalization on the same scale as
# pretraining is preferable to estimating statistics from target validation.
IR_MEAN = (sum((0.485, 0.456, 0.406)) / 3.0,)
IR_STD = (sum((0.229, 0.224, 0.225)) / 3.0,)
VIS_MEAN = (0.485, 0.456, 0.406)
VIS_STD = (0.229, 0.224, 0.225)


def image_mode(modality: str) -> str:
    if modality not in MODALITIES:
        raise ValueError(f"unsupported modality: {modality!r}")
    return IR_IMAGE_MODE if modality == "ir" else VIS_IMAGE_MODE


def normalization(modality: str):
    if modality == "ir":
        return IR_MEAN, IR_STD
    if modality == "vis":
        return VIS_MEAN, VIS_STD
    raise ValueError(f"unsupported modality: {modality!r}")


def _replace_references(module: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """Replace every registered reference to ``old`` inside ``module``.

    CDAN's released ``ResNetFc`` registers conv1 both as ``self.conv1`` and as
    item zero of ``self.feature_layers``.  Replacing all references prevents a
    stale three-channel convolution from surviving in the Sequential wrapper.
    """

    for name, child in list(module.named_children()):
        if child is old:
            module._modules[name] = new
        else:
            _replace_references(child, old, new)


def adapt_resnet_stem_to_ir(module: nn.Module) -> nn.Module:
    """Convert a ResNet-like 3-channel stem to a pretrained 1-channel stem."""

    old = getattr(module, "conv1", None)
    if not isinstance(old, nn.Conv2d):
        raise TypeError("IR backbone must expose a Conv2d attribute named conv1")
    if old.in_channels == 1:
        return module
    if old.in_channels != 3:
        raise ValueError(
            f"expected a 3-channel pretrained stem, got {old.in_channels} channels"
        )
    new = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    _replace_references(module, old, new)
    # ``conv1`` can be a plain attribute in third-party wrappers, so assign it
    # explicitly after replacing registered aliases.
    module.conv1 = new
    return module


def validate_single_modality_args(modality: str, multimodal: bool) -> None:
    if modality not in MODALITIES:
        raise ValueError(f"unsupported modality: {modality!r}")
    if multimodal and modality != "vis":
        raise ValueError("--multimodal and --modality ir are mutually exclusive")


def modality_label(modality: str, multimodal: bool = False) -> str:
    return "VIS+IR" if multimodal else modality.upper()


class MinDomainLoader:
    """Zip independent source/target loaders using the released CORAL budget."""

    def __init__(self, source_loader: DataLoader, target_loader: DataLoader):
        if len(source_loader) < 1 or len(target_loader) < 1:
            raise ValueError(
                "official domain loaders require at least one full batch per domain"
            )
        self.source_loader = source_loader
        self.target_loader = target_loader

    def __len__(self):
        return min(len(self.source_loader), len(self.target_loader))

    def __iter__(self):
        return zip(iter(self.source_loader), iter(self.target_loader))


def official_min_domain_loader(source_dataset, target_dataset,
                               source_batch_size: int, target_batch_size: int,
                               seed: int, **loader_kwargs) -> MinDomainLoader:
    """Build independent shuffled/drop-last loaders as released Deep CORAL."""

    source_generator = torch.Generator().manual_seed(int(seed))
    target_generator = torch.Generator().manual_seed(int(seed) + 1)
    source_loader = DataLoader(
        source_dataset,
        batch_size=source_batch_size,
        shuffle=True,
        drop_last=True,
        generator=source_generator,
        **loader_kwargs,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=target_batch_size,
        shuffle=True,
        drop_last=True,
        generator=target_generator,
        **loader_kwargs,
    )
    return MinDomainLoader(source_loader, target_loader)


class ForeverDataIterator:
    """Small local equivalent of tllib.utils.data.ForeverDataIterator."""

    def __init__(self, data_loader: DataLoader):
        if len(data_loader) < 1:
            raise ValueError("forever iterator requires at least one full batch")
        self.data_loader = data_loader
        self.iterator = iter(data_loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data_loader)
            return next(self.iterator)


class IndependentDomainIterator:
    """Return independently shuffled source and target batches forever."""

    def __init__(self, source_loader: DataLoader, target_loader: DataLoader):
        self.source = ForeverDataIterator(source_loader)
        self.target = ForeverDataIterator(target_loader)

    def __next__(self):
        return next(self.source), next(self.target)


def official_forever_domain_iterator(source_dataset, target_dataset,
                                     batch_size: int, seed: int,
                                     **loader_kwargs) -> IndependentDomainIterator:
    """Build the two independent loaders used by CDAN/MDD/MCC examples."""

    source_generator = torch.Generator().manual_seed(int(seed))
    target_generator = torch.Generator().manual_seed(int(seed) + 1)
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=source_generator,
        **loader_kwargs,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=target_generator,
        **loader_kwargs,
    )
    return IndependentDomainIterator(source_loader, target_loader)


__all__ = [
    "IR_MEAN",
    "IR_STD",
    "MODALITIES",
    "VIS_MEAN",
    "VIS_STD",
    "adapt_resnet_stem_to_ir",
    "image_mode",
    "modality_label",
    "normalization",
    "official_min_domain_loader",
    "official_forever_domain_iterator",
    "validate_single_modality_args",
]
