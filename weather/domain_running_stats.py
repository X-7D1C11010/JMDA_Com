"""Separate source/target BN running statistics with shared affine parameters.

This is an explicit normalization variant, not the full DSBN paper's
two-stage pseudo-label algorithm. It never fits validation/test statistics.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DomainRunningStats(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not isinstance(original, nn.modules.batchnorm._BatchNorm) or not original.track_running_stats:
            raise ValueError("Domain statistics require a BN layer with running buffers")
        self.bn = original
        self.register_buffer("target_mean", original.running_mean.clone())
        self.register_buffer("target_var", original.running_var.clone())
        self.register_buffer("target_batches", original.num_batches_tracked.clone())
        self.register_buffer("pooled_mean", original.running_mean.clone())
        self.register_buffer("pooled_var", original.running_var.clone())
        self.register_buffer("pooled_batches", original.num_batches_tracked.clone())
        self.domain = "source"

    def forward(self, value):
        if self.bn.training:
            self.pooled_batches.add_(1)
            pooled_momentum = self.bn.momentum if self.bn.momentum is not None else 1./float(self.pooled_batches)
            with torch.no_grad():
                F.batch_norm(value.detach(), self.pooled_mean, self.pooled_var, None, None, True, pooled_momentum, self.bn.eps)
        if self.domain == "pooled":
            if self.bn.training:
                raise ValueError("Pooled statistics are a matched-weight evaluation diagnostic only")
            return F.batch_norm(value, self.pooled_mean, self.pooled_var, self.bn.weight, self.bn.bias, False, 0., self.bn.eps)
        if self.domain == "source":
            return self.bn(value)
        if self.domain != "target":
            raise ValueError("Unknown BN domain")
        self.bn._check_input_dim(value)
        momentum = self.bn.momentum
        if self.bn.training:
            self.target_batches.add_(1)
            if momentum is None:
                momentum = 1./float(self.target_batches)
        return F.batch_norm(value, self.target_mean, self.target_var, self.bn.weight, self.bn.bias,
                            self.bn.training, momentum if momentum is not None else 0., self.bn.eps)


def install_domain_statistics(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, DomainRunningStats):
            raise ValueError("Domain statistics already installed")
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            setattr(module, name, DomainRunningStats(child))
            count += 1
        else:
            count += install_domain_statistics(child)
    return count


def set_domain(module, domain):
    if domain not in {"source", "target", "pooled"}:
        raise ValueError("Unknown BN domain")
    for child in module.modules():
        if isinstance(child, DomainRunningStats):
            child.domain = domain


def statistics_summary(module):
    layers = [m for m in module.modules() if isinstance(m, DomainRunningStats)]
    if not layers:
        return None
    source = [int(m.bn.num_batches_tracked) for m in layers]
    target = [int(m.target_batches) for m in layers]
    gaps = [float((m.bn.running_mean-m.target_mean).norm()) for m in layers]
    return {"layers": len(layers), "source_batches_min_max": [min(source), max(source)],
            "target_batches_min_max": [min(target), max(target)],
            "running_mean_gap_L2_mean": sum(gaps)/len(gaps), "fit_partitions": ["source_train", "target_train"]}
