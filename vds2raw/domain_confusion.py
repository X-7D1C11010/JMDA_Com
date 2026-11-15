"""Explicit optional encoder-only domain confusion, additional to the core.

The original discriminator BCE remains responsible for discriminator updates.
This separate loss uses detached discriminator parameters and detached class
probabilities, so it changes the feature encoder objective only. In particular
its feature derivative remains nonzero when a correct domain logit saturates.
"""
import torch
from torch.nn import functional as F
from torch.func import functional_call


def encoder_domain_confusion(features, class_logits, discriminator, source_count, method):
    if method not in {"dann", "cdan"} or not 0 < source_count < len(features):
        raise ValueError("Encoder confusion requires DANN/CDAN and both domains")
    if method == "cdan":
        probabilities = class_logits.detach().softmax(1)
        joint = torch.bmm(probabilities.unsqueeze(2), features.unsqueeze(1)).flatten(1)
    else:
        probabilities = None
        joint = features
    parameters = {key: value.detach() for key, value in discriminator.named_parameters()}
    buffers = {key: value.detach().clone() for key, value in discriminator.named_buffers()}
    logits = functional_call(discriminator, (parameters, buffers), (joint,), strict=True).flatten()
    loss = F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, .5), reduction="none")
    if probabilities is None:
        return loss[:source_count].mean() + loss[source_count:].mean()
    entropy = -(probabilities * (probabilities + 1e-5).log()).sum(1)
    weight = 1 + (-entropy).exp()
    # Equal total mass per domain, matching the CDAN+E reduction convention.
    return .5 * ((loss[:source_count] * weight[:source_count]).sum() / weight[:source_count].sum()
                 + (loss[source_count:] * weight[source_count:]).sum() / weight[source_count:].sum())
