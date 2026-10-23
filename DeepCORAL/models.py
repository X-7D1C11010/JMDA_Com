import torch
import torch.nn as nn
from torch.autograd import Function, Variable

CUDA = True if torch.cuda.is_available() else False


'''
MODELS
'''


def CORAL(source, target):
    """Deep CORAL loss from Sun & Saenko, ECCV 2016, Eq. (1)--(3).

    ``source`` and ``target`` contain one mini-batch of activations.  The
    paper aligns the *unbiased covariance matrices*, not the unnormalised
    scatter matrices.  It then applies the squared Frobenius norm (a sum,
    not an element-wise mean) and the ``1 / (4 d^2)`` scale.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError('CORAL expects 2-D tensors shaped [batch, features]')
    if source.size(1) != target.size(1):
        raise ValueError('source and target feature dimensions must match')
    if source.size(0) < 2 or target.size(0) < 2:
        raise ValueError('CORAL covariance requires at least two samples per domain')

    d = source.size(1)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.t().matmul(source_centered) / (source.size(0) - 1)
    target_cov = target_centered.t().matmul(target_centered) / (target.size(0) - 1)
    return (source_cov - target_cov).pow(2).sum() / (4.0 * d * d)


class DeepCORAL(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.sharedNet = AlexNet()
        self.fc = nn.Linear(4096, num_classes)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.sharedNet(source)
        source = self.fc(source)

        target = self.sharedNet(target)
        target = self.fc(target)
        return source, target


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
