"""Shared VIS+IR input plumbing for multimodal UDA baselines.

The JMDA protocol represents one observation by an exact-stem visible/infrared
pair.  Geometric and photometric random choices are replayed for the infrared
image so the two registered views remain aligned.  The returned tensor is
``[VIS(3), IR(3)]`` along the channel axis; model adapters split it back into
two independent feature branches before feature fusion.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from data_paths import exact_ir_partner, remap_data_path
from uda_data import read_image_list


class UdaPairedImageListDataset(torch.utils.data.Dataset):
    """Target-driven exact VIS/IR pairs with synchronized augmentation."""

    def __init__(self, list_path, transform, label_map, ir_transform=None):
        self.list_path = list_path
        self.transform = transform
        self.ir_transform = ir_transform or transform
        self.label_map = dict(label_map)
        self.samples = []
        self.paired_samples = []
        self.skipped = 0
        self.missing_ir = 0
        for stored_path, original_label in read_image_list(list_path):
            if original_label not in self.label_map:
                self.skipped += 1
                continue
            vis_path = remap_data_path(stored_path)
            ir_path = exact_ir_partner(vis_path)
            if ir_path is None:
                self.missing_ir += 1
                continue
            label = self.label_map[original_label]
            self.samples.append((vis_path, label))
            self.paired_samples.append((vis_path, ir_path, label))
        if not self.samples:
            raise ValueError(f'{list_path} has no exact VIS/IR pair in target label space')

    def __len__(self):
        return len(self.paired_samples)

    @staticmethod
    def _rng_state():
        return random.getstate(), np.random.get_state(), torch.get_rng_state()

    @staticmethod
    def _set_rng_state(state):
        python_state, numpy_state, torch_state = state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)

    def __getitem__(self, index):
        vis_path, ir_path, label = self.paired_samples[index]
        vis = Image.open(vis_path).convert('RGB')
        ir = Image.open(ir_path).convert('L').convert('RGB')

        start_state = self._rng_state()
        vis = self.transform(vis)
        end_state = self._rng_state()
        self._set_rng_state(start_state)
        ir = self.ir_transform(ir)
        # Applying the IR transform must not consume a second set of random
        # choices; advance the worker RNG exactly as one transform invocation.
        self._set_rng_state(end_state)
        if vis.shape != ir.shape or vis.ndim != 3 or vis.shape[0] != 3:
            raise ValueError(
                f'VIS/IR transform shape mismatch: vis={tuple(vis.shape)}, '
                f'ir={tuple(ir.shape)}, sample={vis_path!r}'
            )
        return torch.cat((vis, ir), dim=0), label


def ir_transform_from_vis(vis_transform):
    """Clone a torchvision Compose and use JMDA's IR normalization."""
    if not isinstance(vis_transform, transforms.Compose):
        raise TypeError('multimodal transform must be torchvision.transforms.Compose')
    operations = []
    replaced = False
    for operation in vis_transform.transforms:
        if isinstance(operation, transforms.Normalize):
            operations.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
            replaced = True
        else:
            operations.append(copy.deepcopy(operation))
    if not replaced:
        raise ValueError('VIS transform has no Normalize operation to replace for IR')
    return transforms.Compose(operations)


def split_modalities(images):
    if images.ndim != 4 or images.shape[1] != 6:
        raise ValueError(f'expected [B,6,H,W] VIS+IR tensor, got {tuple(images.shape)}')
    return images[:, :3], images[:, 3:]


class DualBackbone(nn.Module):
    """Run two independent same-family backbones and concatenate features."""

    def __init__(self, vis_backbone, ir_backbone):
        super().__init__()
        self.vis_backbone = vis_backbone
        self.ir_backbone = ir_backbone
        self.out_features = vis_backbone.out_features + ir_backbone.out_features

    def forward(self, images):
        vis, ir = split_modalities(images)
        vis_features = self.vis_backbone(vis)
        ir_features = self.ir_backbone(ir)
        if vis_features.shape[2:] != ir_features.shape[2:]:
            raise ValueError('VIS/IR backbone feature maps have different spatial shapes')
        return torch.cat((vis_features, ir_features), dim=1)


def paired_dataset(list_path, transform, label_map):
    return UdaPairedImageListDataset(
        list_path, transform, label_map,
        ir_transform=ir_transform_from_vis(transform),
    )


def describe_pairing(name, dataset):
    return (f'{name}: exact VIS/IR pairs={len(dataset)}, '
            f'dropped_label={dataset.skipped}, missing_ir={dataset.missing_ir}')


__all__ = [
    'DualBackbone', 'UdaPairedImageListDataset', 'describe_pairing',
    'ir_transform_from_vis', 'paired_dataset', 'split_modalities',
]
