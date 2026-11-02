"""Method models used by the VDS2Raw cross-day protocol."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    VGG16_Weights,
    vgg16,
)

from MCFT.mcft_model import MViT


CONTINUOUS_AIS_INDICES = (0, 1, 2, 3)


def _split_modalities(inputs):
    """Return RGB, aligned NIR/IR, AIS values, and the AIS validity mask."""
    if torch.is_tensor(inputs):
        return inputs, None, None, None
    if not isinstance(inputs, dict):
        raise TypeError(
            "model input must be an image tensor or a multimodal dictionary"
        )
    if "rgb" not in inputs:
        raise KeyError("multimodal model input is missing 'rgb'")
    return (
        inputs["rgb"],
        inputs.get("nir", inputs.get("ir")),
        inputs.get("ais"),
        inputs.get("ais_mask"),
    )


def _single_channel(image):
    if image is None:
        return None
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError(
            f"NIR/IR input must have shape [B,1,H,W] or [B,3,H,W], got {tuple(image.shape)}"
        )
    return image if image.shape[1] == 1 else image.mean(dim=1, keepdim=True)


def _auxiliary_image_encoder(output_dim):
    """Memory-bounded NIR/IR residual branch for ImageNet RGB baselines."""
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, output_dim),
    )


def _modality_dropout(features, probability, training):
    if not training or not probability:
        return features
    keep = (
        torch.rand(features.shape[0], 1, device=features.device) >= probability
    ).to(features.dtype)
    return features * keep / (1.0 - probability)


def _load_backbone_state(module, checkpoint, prefix=""):
    """Load a portable backbone state dict, optionally from a full model."""
    source = Path(checkpoint).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload.get("model_state", payload))
    if not isinstance(state, dict):
        raise TypeError(f"backbone checkpoint does not contain a state dict: {source}")
    if prefix:
        prefixed = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if prefixed:
            state = prefixed
    module.load_state_dict(state, strict=True)
    print(f"Loaded pretrained backbone from {source}")


def euclidean_distance(first, second, eps=1e-8):
    return torch.sqrt(
        torch.clamp(((first - second) ** 2).sum(dim=1, keepdim=True), min=eps)
    )


def ccsa_contrastive_loss(pair_labels, distances, margin=1.0):
    pair_labels = pair_labels.float().view(-1, 1)
    return (
        pair_labels * distances.square()
        + (1.0 - pair_labels) * torch.clamp(margin - distances, min=0.0).square()
    ).mean()


class CCSA96(nn.Module):
    """Released CCSA CNN/MLP topology adapted from 16x16 gray to 96x96 RGB."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * 46 * 46, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(84, num_classes))

    def embed(self, images):
        return self.encoder(images)

    def forward(self, first, second=None):
        first_features = self.embed(first)
        first_logits = self.classifier(first_features)
        if second is None:
            return first_logits
        second_features = self.embed(second)
        second_logits = self.classifier(second_features)
        return first_logits, second_logits, first_features, second_features


class CCSAOfficeModel(nn.Module):
    """Natural-image CCSA architecture reported for Office-31.

    The paper uses ImageNet-pretrained VGG-16 convolutional layers followed by
    1024- and 128-dimensional fully connected embedding layers and a softmax
    classifier.  Softmax is intentionally omitted here because
    ``CrossEntropyLoss`` consumes logits.
    """

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        pretrained_backbone_checkpoint="",
        multimodal=False,
        use_ais=False,
        ais_dropout_probability=0.5,
    ):
        super().__init__()
        if use_ais and not multimodal:
            raise ValueError("AIS fusion requires multimodal=True")
        if not 0.0 <= ais_dropout_probability < 1.0:
            raise ValueError("ais_dropout_probability must be in [0, 1)")
        self.multimodal = bool(multimodal)
        self.use_ais = bool(use_ais)
        self.ais_dropout_probability = float(ais_dropout_probability)
        weights = (
            VGG16_Weights.IMAGENET1K_V1
            if pretrained and not pretrained_backbone_checkpoint
            else None
        )
        backbone = vgg16(weights=weights)
        self.features = backbone.features
        if pretrained_backbone_checkpoint:
            _load_backbone_state(
                self.features, pretrained_backbone_checkpoint, prefix="features."
            )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, num_classes)
        if self.multimodal:
            self.nir_encoder = _auxiliary_image_encoder(128)
            self.nir_residual_scale = nn.Parameter(torch.tensor(0.1))
        if self.use_ais:
            self.register_buffer(
                "ais_feature_indices",
                torch.tensor(CONTINUOUS_AIS_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.ais_encoder = nn.Sequential(
                nn.LayerNorm(len(CONTINUOUS_AIS_INDICES)),
                nn.Linear(len(CONTINUOUS_AIS_INDICES), 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 128),
            )
            self.ais_residual_scale = nn.Parameter(torch.tensor(0.1))
        for module in self.embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def embed(self, images):
        rgb, nir, ais, ais_mask = _split_modalities(images)
        embedding = self.embedding(self.avgpool(self.features(rgb)))
        if self.multimodal:
            if nir is None:
                raise ValueError("multimodal CCSA requires aligned NIR/IR input")
            embedding = embedding + self.nir_residual_scale * self.nir_encoder(
                _single_channel(nir)
            )
        if self.use_ais:
            if ais is None or ais_mask is None:
                raise ValueError("three-modal CCSA requires AIS values and ais_mask")
            indices = self.ais_feature_indices
            ais_features = self.ais_encoder(
                ais.index_select(1, indices) * ais_mask.index_select(1, indices)
            )
            ais_features = _modality_dropout(
                ais_features, self.ais_dropout_probability, self.training
            )
            embedding = embedding + self.ais_residual_scale * ais_features
        return embedding

    def forward(self, first, second=None):
        first_features = self.embed(first)
        first_logits = self.classifier(first_features)
        if second is None:
            return first_logits
        second_features = self.embed(second)
        second_logits = self.classifier(second_features)
        return first_logits, second_logits, first_features, second_features

    def set_source_initialization_stage(self, enabled):
        """Freeze VGG convolutions while initializing the new FC layers."""
        for parameter in self.features.parameters():
            parameter.requires_grad = not enabled

    def source_initialization_parameters(self):
        parameters = list(self.embedding.parameters()) + list(self.classifier.parameters())
        if self.multimodal:
            parameters.extend(self.nir_encoder.parameters())
            parameters.append(self.nir_residual_scale)
        if self.use_ais:
            parameters.extend(self.ais_encoder.parameters())
            parameters.append(self.ais_residual_scale)
        return parameters


def dsne_loss(
    source_features,
    source_labels,
    target_features,
    target_labels,
    margin=1.0,
    feature_normalization=False,
):
    """PyTorch equivalent of the released d-SNE loss."""
    if feature_normalization:
        source_features = F.normalize(source_features, p=2, dim=1)
        target_features = F.normalize(target_features, p=2, dim=1)
    distances = (
        (target_features.unsqueeze(1) - source_features.unsqueeze(0)) ** 2
    ).sum(dim=2)
    same = target_labels.view(-1, 1).eq(source_labels.view(1, -1))
    intra = distances * same.float()
    inter = distances * (~same).float()
    row_max = distances.max(dim=1, keepdim=True).values.expand_as(distances)
    revised_inter = torch.where(same, row_max, inter)
    max_intra = intra.max(dim=1).values
    min_inter = revised_inter.min(dim=1).values
    return F.relu(max_intra - min_inter + margin).mean()


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, inputs):
        activated = F.relu(self.bn1(inputs), inplace=True)
        residual = self.projection(activated) if self.projection is not None else inputs
        output = self.conv1(activated)
        output = self.conv2(F.relu(self.bn2(output), inplace=True))
        return output + residual


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        hidden = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, inputs):
        activated = F.relu(self.bn1(inputs), inplace=True)
        residual = self.projection(activated) if self.projection is not None else inputs
        output = self.conv1(activated)
        output = self.conv2(F.relu(self.bn2(output), inplace=True))
        output = self.conv3(F.relu(self.bn3(output), inplace=True))
        return output + residual


class ResNetV2Features(nn.Module):
    SPECS: ClassVar = {
        18: (BasicBlockV2, [2, 2, 2, 2], [64, 64, 128, 256, 512]),
        50: (BottleneckV2, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
        101: (BottleneckV2, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
        152: (BottleneckV2, [3, 8, 36, 3], [64, 256, 512, 1024, 2048]),
    }

    def __init__(self, nlayers):
        super().__init__()
        if nlayers not in self.SPECS:
            raise ValueError(f"unsupported ResNet depth {nlayers}")
        block, layer_counts, channels = self.SPECS[nlayers]
        self.input_norm = nn.BatchNorm2d(3, affine=False)
        self.conv1 = nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        stages = []
        in_channels = channels[0]
        for stage_index, (count, out_channels) in enumerate(
            zip(layer_counts, channels[1:])
        ):
            stride = 1 if stage_index == 0 else 2
            blocks = [block(in_channels, out_channels, stride=stride)]
            blocks.extend(block(out_channels, out_channels) for _ in range(count - 1))
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.final_bn = nn.BatchNorm2d(in_channels)
        self.output_channels = in_channels
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d) and module.affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images):
        features = self.input_norm(images)
        features = self.pool(F.relu(self.bn1(self.conv1(features)), inplace=True))
        features = self.stages(features)
        features = F.relu(self.final_bn(features), inplace=True)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)


class DSNE96(nn.Module):
    def __init__(
        self,
        num_classes=3,
        embed_size=512,
        nlayers=101,
        dropout=False,
        l2_normalize=False,
    ):
        super().__init__()
        self.features = ResNetV2Features(nlayers)
        layers = [nn.Linear(self.features.output_channels, 4096), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend([nn.Linear(4096, embed_size), nn.ReLU(inplace=True)])
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.embeds = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize
        self.output = nn.Linear(embed_size, num_classes)
        for module in self.embeds.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, images, return_embedding=False):
        features = self.embeds(self.features(images))
        if self.l2_normalize:
            features = F.normalize(features, p=2, dim=1)
        logits = self.output(features)
        return (logits, features) if return_embedding else logits

    def load_backbone_checkpoint(self, path):
        _load_backbone_state(self.features, path, prefix="features.")


class DSNEOfficeModel(nn.Module):
    """PyTorch port of the official natural-image d-SNE model.

    The released model is a ResNetV2-101 backbone followed by a 4096-unit dense
    layer and a 512-dimensional latent layer.  The ImageNet-initialized timm
    ResNetV2 keeps that pre-activation architecture in current PyTorch; the
    d-SNE-specific dense layers retain the release's normal/zero initialization.
    """

    def __init__(
        self,
        num_classes=3,
        embed_size=512,
        pretrained=True,
        pretrained_backbone_checkpoint="",
        dropout=False,
        l2_normalize=False,
        multimodal=False,
        use_ais=False,
        ais_dropout_probability=0.5,
    ):
        super().__init__()
        if use_ais and not multimodal:
            raise ValueError("AIS fusion requires multimodal=True")
        if not 0.0 <= ais_dropout_probability < 1.0:
            raise ValueError("ais_dropout_probability must be in [0, 1)")
        self.multimodal = bool(multimodal)
        self.use_ais = bool(use_ais)
        self.ais_dropout_probability = float(ais_dropout_probability)
        self.features = timm.create_model(
            "resnetv2_101.a1h_in1k",
            pretrained=pretrained and not pretrained_backbone_checkpoint,
            num_classes=0,
            global_pool="avg",
        )
        if pretrained_backbone_checkpoint:
            _load_backbone_state(
                self.features, pretrained_backbone_checkpoint, prefix="features."
            )
        layers = [nn.Flatten(), nn.Linear(2048, 4096), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.extend([nn.Linear(4096, embed_size), nn.ReLU(inplace=True)])
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.embeds = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize
        self.output = nn.Linear(embed_size, num_classes)
        if self.multimodal:
            self.nir_encoder = _auxiliary_image_encoder(embed_size)
            self.nir_residual_scale = nn.Parameter(torch.tensor(0.1))
        if self.use_ais:
            self.register_buffer(
                "ais_feature_indices",
                torch.tensor(CONTINUOUS_AIS_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.ais_encoder = nn.Sequential(
                nn.LayerNorm(len(CONTINUOUS_AIS_INDICES)),
                nn.Linear(len(CONTINUOUS_AIS_INDICES), embed_size),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(embed_size, embed_size),
            )
            self.ais_residual_scale = nn.Parameter(torch.tensor(0.1))
        for module in self.embeds.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, images, return_embedding=False):
        rgb, nir, ais, ais_mask = _split_modalities(images)
        features = self.embeds(self.features(rgb))
        if self.multimodal:
            if nir is None:
                raise ValueError("multimodal d-SNE requires aligned NIR/IR input")
            features = features + self.nir_residual_scale * self.nir_encoder(
                _single_channel(nir)
            )
        if self.use_ais:
            if ais is None or ais_mask is None:
                raise ValueError("three-modal d-SNE requires AIS values and ais_mask")
            indices = self.ais_feature_indices
            ais_features = self.ais_encoder(
                ais.index_select(1, indices) * ais_mask.index_select(1, indices)
            )
            ais_features = _modality_dropout(
                ais_features, self.ais_dropout_probability, self.training
            )
            features = features + self.ais_residual_scale * ais_features
        if self.l2_normalize:
            features = F.normalize(features, p=2, dim=1)
        logits = self.output(features)
        return (logits, features) if return_embedding else logits

    def source_head_parameters(self):
        parameters = list(self.embeds.parameters()) + list(self.output.parameters())
        if self.multimodal:
            parameters.extend(self.nir_encoder.parameters())
            parameters.append(self.nir_residual_scale)
        if self.use_ais:
            parameters.extend(self.ais_encoder.parameters())
            parameters.append(self.ais_residual_scale)
        return parameters

    def load_source_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload.get("state_dict", payload))
        self.load_state_dict(state)


class MCFTWithAIS(nn.Module):
    """Official RGB+NIR MCFT trunk plus a regularized AIS late-fusion branch.

    ``ais_mask`` is a missing-value validity indicator, not a fourth modality.
    It gates invalid AIS values but is deliberately not concatenated to the
    classifier input: missingness is correlated with vessel class in VDS2Raw
    and otherwise becomes an unintended shortcut.  The default ``continuous``
    profile also excludes navigation-status one-hot fields such as "engaged in
    fishing" and "under way sailing", which are near-direct label proxies.
    """

    AIS_FEATURE_PROFILES: ClassVar[dict[str, tuple[int, ...]]] = {
        "continuous": (0, 1, 2, 3),
        "full": tuple(range(13)),
    }

    def __init__(
        self,
        num_classes=3,
        patch_size=15,
        ais_feature_profile="continuous",
        ais_dropout_probability=0.5,
    ):
        super().__init__()
        if ais_feature_profile not in self.AIS_FEATURE_PROFILES:
            raise ValueError(
                f"unknown AIS feature profile {ais_feature_profile!r}; expected "
                f"one of {sorted(self.AIS_FEATURE_PROFILES)}"
            )
        if not 0.0 <= ais_dropout_probability < 1.0:
            raise ValueError("ais_dropout_probability must be in [0, 1)")
        feature_indices = self.AIS_FEATURE_PROFILES[ais_feature_profile]
        self.ais_feature_profile = ais_feature_profile
        self.ais_dropout_probability = float(ais_dropout_probability)
        self.register_buffer(
            "ais_feature_indices",
            torch.tensor(feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.visual = MViT(
            patch_size=patch_size,
            num_patches=[3, 1],
            num_classes=num_classes,
            dim=64,
            depth=6,
            heads=4,
            mlp_dim=32,
            dropout=0.1,
            emb_dropout=0.1,
            mode="MViT",
        )
        self.ais_encoder = nn.Sequential(
            nn.LayerNorm(len(feature_indices)),
            nn.Linear(len(feature_indices), 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.fusion_delta = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def encode_ais(self, ais, ais_mask):
        """Encode valid AIS values without exposing mask bits as features."""
        indices = self.ais_feature_indices
        values = ais.index_select(1, indices)
        validity = ais_mask.index_select(1, indices)
        features = self.ais_encoder(values * validity)
        if self.training and self.ais_dropout_probability:
            keep = (
                torch.rand(features.shape[0], 1, device=features.device)
                >= self.ais_dropout_probability
            ).to(features.dtype)
            features = features * keep / (1.0 - self.ais_dropout_probability)
        return features

    def forward(self, rgb, nir, ais, ais_mask, labels=None, compute_auxiliary=True):
        visual_logits, loss_ita, loss_itm, visual_features = self.visual(
            rgb,
            nir,
            labels,
            return_features=True,
            compute_auxiliary=compute_auxiliary,
        )
        ais_features = self.encode_ais(ais, ais_mask)
        logits = visual_logits + self.fusion_delta(
            torch.cat((visual_features, ais_features), dim=1)
        )
        return logits, loss_ita, loss_itm


__all__ = [
    "CCSA96",
    "DSNE96",
    "CCSAOfficeModel",
    "DSNEOfficeModel",
    "MCFTWithAIS",
    "ccsa_contrastive_loss",
    "dsne_loss",
    "euclidean_distance",
]
