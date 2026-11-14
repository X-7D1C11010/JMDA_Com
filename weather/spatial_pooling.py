"""Optional spatial-aggregation probe; the training default stays average.

GeM formula: authors' cirtorch/layers/functional.py in
https://github.com/filipradenovic/cnnimageretrieval-pytorch
This probe fixes p rather than learning it, and is not a full reproduction of
that image-retrieval training algorithm.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FixedGeM(nn.Module):
    def __init__(self, p=3., eps=1e-6):
        super().__init__()
        if p < 1 or eps <= 0:
            raise ValueError("Use p >= 1 and positive epsilon")
        self.p, self.eps = float(p), float(eps)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("Spatial pooling expects an NCHW feature map")
        if self.p == 1.:
            return F.adaptive_avg_pool2d(features, 1)
        return F.adaptive_avg_pool2d(features.clamp_min(self.eps).pow(self.p), 1).pow(1./self.p)

    def extra_repr(self):
        return f"p={self.p}, eps={self.eps}, fixed=True"
