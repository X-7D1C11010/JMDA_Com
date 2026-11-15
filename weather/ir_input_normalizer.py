"""Opt-in, sample-independent IR normalization inside the feature encoder."""
import torch
from torch import nn


class InfraredInputNormalizer(nn.Module):
    def __init__(self,mode='none',imagenet=False,strength=.85,min_std=.02):
        super().__init__()
        if mode not in {'none','zscore','percentile'} or not 0<=strength<1 or min_std<=0:
            raise ValueError('Invalid IR input normalization')
        self.mode,self.imagenet,self.strength,self.min_std=mode,bool(imagenet),strength,min_std
        if imagenet:
            self.register_buffer('imagenet_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
            self.register_buffer('imagenet_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self,half_normalized):
        if self.mode=='none' and not self.imagenet:return half_normalized
        raw=(half_normalized.float()*.5+.5).clamp(0,1)
        if self.mode=='zscore':
            mean=raw.mean((-2,-1),keepdim=True)
            std=raw.std((-2,-1),unbiased=False,keepdim=True).clamp_min(self.min_std)
            canonical=((raw-mean).div(1.5*std).tanh()+1)*.5
            raw=(1-self.strength)*raw+self.strength*canonical
        elif self.mode=='percentile':
            bounds=torch.quantile(raw.flatten(2),raw.new_tensor([.01,.99]),dim=-1)
            low,high=bounds[0][...,None,None],bounds[1][...,None,None]
            canonical=((raw-low)/(high-low).clamp_min(2*self.min_std)).clamp(0,1)
            raw=(1-self.strength)*raw+self.strength*canonical
        if self.imagenet:
            if raw.shape[1]!=3:raise ValueError('ImageNet normalization requires three IR channels')
            return (raw-self.imagenet_mean)/self.imagenet_std
        return (raw-.5)/.5
