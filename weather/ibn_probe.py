"""Optional IBN-a feature probe using the authors' pretrained state layout.

Source: https://github.com/XingangPan/IBN-Net (ibnnet/modules.py,
ibnnet/resnet_ibn.py). Image preprocessing stays outside this encoder.

MIT License
Copyright (c) 2017 Wei Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torchvision.models import resnet50


class IBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.half = channels//2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(channels-self.half)

    def forward(self, features):
        return torch.cat((self.IN(features[:, :self.half].contiguous()),
                          self.BN(features[:, self.half:].contiguous())), dim=1)


def resnet50_ibn_a_probe():
    model = resnet50(weights=None)
    for stage in (model.layer1, model.layer2, model.layer3):
        for block in stage:
            block.bn1 = IBN(block.bn1.num_features)
    # Authors' 224px model uses fixed 7x7 pooling; retain it for parity.
    model.avgpool = nn.AvgPool2d(7)
    return model
