"""Keep half-normalized IR input compatible with an ImageNet patch stem."""
import torch
from torch import nn


@torch.no_grad()
def calibrate_half_normalized_stem(stem):
    """Absorb (g-ImageNet_mean)/std into a no-padding RGB patch convolution.

    The loader still supplies (g-.5)/.5. The stem then has exactly the same
    activations as its original pretrained weights given ImageNet-normalized
    g. Padding would invalidate the constant bias correction at the boundary.
    """
    if (not isinstance(stem, nn.Conv2d) or stem.in_channels != 3 or stem.groups != 1
            or stem.padding != (0, 0) or stem.bias is None):
        raise ValueError("Calibration requires a biased unpadded three-channel convolution")
    weight = stem.weight.detach().clone()
    mean = weight.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = weight.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    stem.weight.copy_(weight/(2*std))
    stem.bias.add_((weight*((.5-mean)/std)).sum((1, 2, 3)))
    return stem
