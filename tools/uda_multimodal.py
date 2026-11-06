"""Shared VIS+IR plumbing and JMDA-style encoders for UDA baselines.

The JMDA protocol represents one observation by an exact-stem visible/infrared
pair.  Geometric and photometric random choices are replayed for the infrared
image so the two registered views remain aligned.  The returned tensor is
``[VIS(3), IR(3)]`` along the channel axis.  Multimodal baselines use the
non-tensor part of JMDA's released architecture: an ImageNet ResNet-18 VIS
encoder, a four-stage U-Net-like IR encoder, 512-D projections per modality,
and direct feature concatenation.  JMDA's tensor alignment, OT generator and
domain discriminator are deliberately not imported here.
"""
from __future__ import annotations

import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision import models

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


class UdaLabelFreePairedDataset(torch.utils.data.Dataset):
    """Exact VIS/IR target-training pairs without per-sample target labels.

    JMDA's strict UDA path consumes ``*_train_pairs.tsv`` rather than the
    labelled target list.  Returning a constant sentinel keeps the existing
    baseline training-loop interface while proving that target truth cannot be
    consumed by an adaptation loss accidentally.
    """

    label_free = True

    def __init__(self, pair_list_path, transform, label_list_path, label_map,
                 ir_transform=None):
        self.pair_list_path = pair_list_path
        self.label_list_path = label_list_path
        self.transform = transform
        self.ir_transform = ir_transform or transform
        self.label_map = dict(label_map)
        self.samples = []
        self.paired_samples = []
        self.skipped = 0
        self.missing_ir = 0
        with open(pair_list_path, 'r', encoding='utf-8') as handle:
            for line_number, raw_line in enumerate(handle, 1):
                fields = raw_line.rstrip('\r\n').split('\t')
                if fields == ['']:
                    continue
                if len(fields) != 2 or not all(fields):
                    raise ValueError(
                        f'invalid label-free VIS/IR pair at '
                        f'{pair_list_path}:{line_number}'
                    )
                vis_path, ir_path = map(remap_data_path, fields)
                if (os.path.splitext(os.path.basename(vis_path))[0] !=
                        os.path.splitext(os.path.basename(ir_path))[0]):
                    raise ValueError(
                        f'mismatched VIS/IR filename stems at '
                        f'{pair_list_path}:{line_number}'
                    )
                self.samples.append((vis_path, -1))
                self.paired_samples.append((vis_path, ir_path))
        if not self.paired_samples:
            raise ValueError(f'label-free target pair list is empty: {pair_list_path}')

    def __len__(self):
        return len(self.paired_samples)

    def __getitem__(self, index):
        vis_path, ir_path = self.paired_samples[index]
        vis = Image.open(vis_path).convert('RGB')
        ir = Image.open(ir_path).convert('L').convert('RGB')
        start_state = UdaPairedImageListDataset._rng_state()
        vis = self.transform(vis)
        end_state = UdaPairedImageListDataset._rng_state()
        UdaPairedImageListDataset._set_rng_state(start_state)
        ir = self.ir_transform(ir)
        UdaPairedImageListDataset._set_rng_state(end_state)
        if vis.shape != ir.shape or vis.ndim != 3 or vis.shape[0] != 3:
            raise ValueError(
                f'VIS/IR transform shape mismatch: vis={tuple(vis.shape)}, '
                f'ir={tuple(ir.shape)}, sample={vis_path!r}'
            )
        return torch.cat((vis, ir), dim=0), -1


def jmda_train_transform():
    """Exact paired-image augmentation used by JMDA's strict UDA loader."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def jmda_val_transform():
    """Deterministic target validation transform used by JMDA."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


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


class JmdaVisualEncoder(nn.Module):
    """JMDA's non-tensor VIS feature extractor (ResNet-18 + projection)."""

    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        except AttributeError:  # torchvision < 0.13
            resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(resnet.fc.in_features, output_dim)
        self.out_features = output_dim

    def forward(self, images):
        features = self.features(images)
        return self.proj(torch.flatten(features, 1))


class JmdaInfraredEncoder(nn.Module):
    """JMDA's non-tensor four-stage U-Net-style infrared encoder."""

    def __init__(self, input_channels=3, output_dim=512):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.enc1 = conv_block(input_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, output_dim)
        self.out_features = output_dim

    def forward(self, images):
        features = self.enc1(images)
        features = self.enc2(features)
        features = self.enc3(features)
        features = self.enc4(features)
        features = torch.flatten(self.avgpool(features), 1)
        return self.proj(features)


class JmdaMultimodalEncoder(nn.Module):
    """Directly concatenate JMDA VIS and IR features, without tensor fusion."""

    def __init__(self, modality_dim=512, pretrained_vis=True):
        super().__init__()
        self.vis_encoder = JmdaVisualEncoder(modality_dim, pretrained_vis)
        self.ir_encoder = JmdaInfraredEncoder(3, modality_dim)
        self.out_features = modality_dim * 2

    def forward_components(self, images):
        vis, ir = split_modalities(images)
        return self.vis_encoder(vis), self.ir_encoder(ir)

    def forward(self, images):
        vis_features, ir_features = self.forward_components(images)
        return torch.cat((vis_features, ir_features), dim=1)


class JmdaTllibBackbone(JmdaMultimodalEncoder):
    """JMDA encoder adapter for tllib image classifiers expecting 4-D maps."""

    def forward(self, images):
        return super().forward(images).unsqueeze(-1).unsqueeze(-1)


def configure_jmda_visual_finetuning(encoder, mode='legacy_supervised'):
    """Match JMDA's declared VIS fine-tuning policy.

    The controlled JMDA UDA protocol uses the actual supervised-reference
    behavior: freeze the ImageNet ResNet feature trunk and update only its
    512-D projection.  ``layer2_4`` remains available for explicit ablations.
    The random IR encoder is always trainable.
    """
    if mode not in {'legacy_supervised', 'layer2_4'}:
        raise ValueError(f'unsupported JMDA VIS fine-tuning mode: {mode}')
    if not hasattr(encoder, 'vis_encoder') or not hasattr(encoder, 'ir_encoder'):
        raise TypeError('expected a JMDA multimodal encoder')
    vis_encoder = encoder.vis_encoder
    for parameter in vis_encoder.parameters():
        parameter.requires_grad = False
    if mode == 'layer2_4':
        if not isinstance(vis_encoder.features, nn.Sequential) or len(vis_encoder.features) < 8:
            raise ValueError('unexpected ResNet-18 feature layout')
        for stage_index in (5, 6, 7):
            for parameter in vis_encoder.features[stage_index].parameters():
                parameter.requires_grad = True
    for parameter in vis_encoder.proj.parameters():
        parameter.requires_grad = True
    for parameter in encoder.ir_encoder.parameters():
        parameter.requires_grad = True
    return {
        'mode': mode,
        'vis_trainable': sum(
            parameter.numel() for parameter in vis_encoder.parameters()
            if parameter.requires_grad
        ),
        'vis_total': sum(parameter.numel() for parameter in vis_encoder.parameters()),
        'ir_trainable': sum(
            parameter.numel() for parameter in encoder.ir_encoder.parameters()
            if parameter.requires_grad
        ),
    }


class CyclingDataIterator:
    """Cycle a shorter loader without truncating the longer domain loader."""

    def __init__(self, loader):
        if len(loader) < 1:
            raise ValueError('cannot cycle an empty data loader')
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class _UnpairedDomainDataset(torch.utils.data.Dataset):
    """Index-addressable source/target wrapper for strict label-free UDA."""

    def __init__(self, source_dataset, target_dataset):
        if len(source_dataset) < 1 or len(target_dataset) < 1:
            raise ValueError('source and target datasets must both be non-empty')
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return max(len(self.source_dataset), len(self.target_dataset))

    def __getitem__(self, index_pair):
        source_index, target_index = index_pair
        return (self.source_dataset[source_index],
                self.target_dataset[target_index])


class UnpairedDomainIndexBatchSampler(torch.utils.data.Sampler):
    """JMDA strict-UDA domain sampler.

    An epoch covers the larger domain once, while the smaller domain is
    reshuffled and cycled.  Every optimizer update therefore receives a full,
    equally sized source and target batch without reading target labels.
    """

    def __init__(self, source_count, target_count, batch_size, seed):
        if source_count < 1 or target_count < 1:
            raise ValueError('source_count and target_count must be positive')
        if batch_size < 2:
            raise ValueError('strict UDA domain batches require batch_size >= 2')
        self.source_count = int(source_count)
        self.target_count = int(target_count)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.n_batches = math.ceil(
            max(self.source_count, self.target_count) / self.batch_size)

    @staticmethod
    def _take(indices, position, count, rng):
        selected = []
        while len(selected) < count:
            if position >= len(indices):
                rng.shuffle(indices)
                position = 0
            take = min(count - len(selected), len(indices) - position)
            selected.extend(indices[position:position + take])
            position += take
        return selected, position

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        source_order = list(range(self.source_count))
        target_order = list(range(self.target_count))
        rng.shuffle(source_order)
        rng.shuffle(target_order)
        source_position = 0
        target_position = 0
        for _ in range(self.n_batches):
            source_indices, source_position = self._take(
                source_order, source_position, self.batch_size, rng)
            target_indices, target_position = self._take(
                target_order, target_position, self.batch_size, rng)
            yield list(zip(source_indices, target_indices))

    def __len__(self):
        return self.n_batches


def unpaired_domain_loader(source_dataset, target_dataset, batch_size, seed,
                           **loader_kwargs):
    """Build the same larger-domain epoch used by JMDA strict UDA."""
    dataset = _UnpairedDomainDataset(source_dataset, target_dataset)
    batch_sampler = UnpairedDomainIndexBatchSampler(
        len(source_dataset), len(target_dataset), batch_size, seed)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        generator=generator,
        **loader_kwargs,
    )


def gradient_l2_norm(parameters):
    """Return the global L2 norm of currently accumulated gradients."""
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    if squared is None:
        return 0.0
    return float(squared.sqrt().item())


class FullCoverageBatchSampler(torch.utils.data.Sampler):
    """Shuffle all indices while preventing a final singleton covariance batch."""

    def __init__(self, data_source, batch_size, shuffle=True):
        if batch_size < 2:
            raise ValueError('full-coverage covariance batches require batch_size >= 2')
        if len(data_source) < 2:
            raise ValueError('full-coverage covariance batches require at least 2 samples')
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.data_source)).tolist()
        else:
            indices = list(range(len(self.data_source)))
        batches = [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-1].insert(0, batches[-2].pop())
        yield from batches

    def __len__(self):
        return math.ceil(len(self.data_source) / self.batch_size)


def paired_dataset(list_path, transform, label_map):
    return UdaPairedImageListDataset(
        list_path, transform, label_map,
        ir_transform=ir_transform_from_vis(transform),
    )


def label_free_target_paired_dataset(list_path, transform, label_map):
    if not list_path.endswith('_train.txt'):
        raise ValueError('label-free target dataset requires a *_train.txt path')
    pair_list_path = list_path[:-len('_train.txt')] + '_train_pairs.tsv'
    if not os.path.isfile(pair_list_path):
        raise FileNotFoundError(f'missing label-free target pair list: {pair_list_path}')
    return UdaLabelFreePairedDataset(
        pair_list_path,
        transform,
        label_list_path=list_path,
        label_map=label_map,
        ir_transform=ir_transform_from_vis(transform),
    )


def describe_pairing(name, dataset):
    return (f'{name}: exact VIS/IR pairs={len(dataset)}, '
            f'dropped_label={dataset.skipped}, missing_ir={dataset.missing_ir}')


__all__ = [
    'CyclingDataIterator', 'FullCoverageBatchSampler',
    'UnpairedDomainIndexBatchSampler',
    'UdaLabelFreePairedDataset',
    'JmdaInfraredEncoder', 'JmdaMultimodalEncoder',
    'JmdaTllibBackbone', 'JmdaVisualEncoder', 'UdaPairedImageListDataset',
    'configure_jmda_visual_finetuning', 'describe_pairing',
    'gradient_l2_norm',
    'ir_transform_from_vis', 'jmda_train_transform', 'jmda_val_transform',
    'label_free_target_paired_dataset', 'paired_dataset', 'split_modalities',
    'unpaired_domain_loader',
]
