"""Model adapters retained for the VDS2Raw MCFT experiment."""

from typing import ClassVar

import torch
from torch import nn

from MCFT.mcft_model import MViT


class MCFTWithAIS(nn.Module):
    """Official RGB+NIR MCFT trunk plus a regularized AIS late-fusion branch.

    ``ais_mask`` is a missing-value validity indicator, not a fourth modality.
    The default JMDA-aligned profile encodes all 13 gated AIS values together
    with their 13 validity bits. The MCFT raster trunk convolves the native
    96x96 inputs before pooling its feature maps to the released 15x15
    transformer token grid.
    """

    AIS_FEATURE_PROFILES: ClassVar[dict[str, tuple[int, ...]]] = {
        "continuous": (0, 1, 2, 3),
        "full": tuple(range(13)),
    }

    def __init__(
        self,
        num_classes=3,
        patch_size=15,
        ais_feature_profile="full",
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
        ais_input_dim = 2 * len(feature_indices)
        self.ais_encoder = nn.Sequential(
            nn.LayerNorm(ais_input_dim),
            nn.Linear(ais_input_dim, 64),
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
        """Encode gated AIS values and validity bits using the JMDA convention."""
        indices = self.ais_feature_indices
        values = ais.index_select(1, indices)
        validity = ais_mask.index_select(1, indices)
        features = self.ais_encoder(torch.cat((values * validity, validity), dim=1))
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


__all__ = ["MCFTWithAIS"]
