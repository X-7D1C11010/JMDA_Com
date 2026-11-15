"""Memory-saving forwards that preserve the original torchvision ConvNeXt.

Uses non-reentrant checkpoints because the training audit calls autograd.grad.
Reference: https://pytorch.org/docs/2.6/checkpoint.html
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.convnext import ConvNeXt


def checkpointed_convnext(encoder,images):
    if not isinstance(encoder,ConvNeXt):
        raise TypeError("Block checkpointing requires the original torchvision ConvNeXt")
    if not torch.is_grad_enabled() or not any(p.requires_grad for p in encoder.parameters()):
        return encoder(images)
    x=images
    for stage in encoder.features:
        # Checkpoint each residual block, rather than retaining an entire
        # 27-block stage's activations during its backward recomputation.
        operations=stage if isinstance(stage,nn.Sequential) else (stage,)
        for operation in operations:
            x=checkpoint(operation,x,use_reentrant=False,preserve_rng_state=True)
    return encoder.classifier(encoder.avgpool(x))
