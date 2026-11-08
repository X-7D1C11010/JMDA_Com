"""Faithful TIR core with an explicit VDS2Raw multimodal adaptation.

The visual/text backbone, CoOp prompt construction and LoRA implementation are
from the vendored official repository.  VDS2Raw-specific changes are isolated
here:

* RGB follows the official CLIP ViT-B/16 visual/TIR path. NIR uses an
  independent, directly supervised convolutional classifier.
* only proxy-safe AIS heading, length and width values are encoded by a
  directly supervised MLP; validity bits gate missing values but are not
  classifier inputs.
* NIR and AIS class logits are fused with the visual logits through trainable
  softplus-positive gates, and are also constrained by auxiliary CE losses.
* the paper's simplified K-way ordering is used for three-class VDS2Raw, with
  bounded weights to avoid complete token deletion or 2x amplification. The
  released five-way path remains unchanged.

These adaptations preserve the TIR mechanism but are not claimed to be the
paper's original single-image, source-free 5-way protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from TIR.clip import clip as tir_clip
from TIR.loralib.utils import apply_lora


OFFICIAL_COMMIT = "188ff2fcb7373fd5882319212bf712a676e90773"
OFFICIAL_BACKBONE = "ViT-B/16"
DEFAULT_CLASS_NAMES = ("cargo ship", "fishing vessel", "sailing and pleasure vessel")


def load_clip_checkpoint_strict(checkpoint: str | Path) -> nn.Module:
    """Load the official CLIP archive and reject every structural mismatch.

    The upstream ``build_model`` copies matching tensors without reporting
    missing keys.  Re-loading the canonical state with ``strict=True`` here
    prevents an incomplete or wrong CLIP file from being accepted silently.
    """

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing CLIP ViT-B/16 checkpoint: {path}")
    try:
        archive = torch.jit.load(str(path), map_location="cpu").eval()
        source_state = archive.state_dict()
    except RuntimeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source_state = payload.get("state_dict", payload)
    if not isinstance(source_state, dict):
        raise TypeError(f"unsupported CLIP checkpoint payload in {path}")

    source_state = dict(source_state)
    model = tir_clip.build_model(dict(source_state)).float()
    loadable = {
        key: value
        for key, value in source_state.items()
        if key not in {"input_resolution", "context_length", "vocab_size"}
    }
    expected = model.state_dict()
    missing = sorted(set(expected) - set(loadable))
    unexpected = sorted(set(loadable) - set(expected))
    shape_mismatch = sorted(
        key
        for key in set(expected).intersection(loadable)
        if tuple(expected[key].shape) != tuple(loadable[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "CLIP checkpoint is structurally incompatible: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatch={shape_mismatch[:8]}"
        )
    model.load_state_dict(loadable, strict=True)
    return model


class TokenImportanceRecalibration(nn.Module):
    """Device-safe implementation of the paper/release TIR token weighting."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        tokens: torch.Tensor,
        text_features: torch.Tensor,
        topk_ratio: float = 0.3,
        temper_weight: float = 0.5,
        params=None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"expected [tokens,batch,width], got {tokens.shape}")
        if text_features.ndim != 2:
            raise ValueError(
                f"expected [classes,text_dim] text features, got {text_features.shape}"
            )
        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        if temper_weight < 0:
            raise ValueError("temper_weight must be non-negative")

        batch_first = tokens.permute(1, 0, 2)
        cls_token, patches = batch_first[:, :1], batch_first[:, 1:]
        if params is None:
            raise ValueError("TIR runtime parameters are required")
        projected = (
            params.aux_container["visual_ln_post"](batch_first)
            @ params.aux_container["visual_projection"]
        )
        projected = projected[:, 1:]
        projected = F.normalize(projected, dim=-1)
        text_features = F.normalize(text_features.to(projected.dtype), dim=-1)
        similarities = torch.matmul(
            text_features.unsqueeze(0), projected.transpose(1, 2)
        )
        patch_count = patches.shape[1]
        keep = max(1, min(patch_count, int(patch_count * topk_ratio)))
        topk_indices = similarities.topk(keep, dim=-1).indices
        binary = torch.zeros_like(similarities)
        binary.scatter_(2, topk_indices, 1)
        sum_scores = binary.sum(dim=1)

        class_count = int(text_features.shape[0])
        if class_count == 5:
            # Exact device-safe equivalent of the released 5-way rule.
            weights = torch.where(
                sum_scores.eq(0),
                torch.ones_like(sum_scores),
                1.0 - temper_weight * (sum_scores - 3.0),
            )
            rule = "official_linear_5way"
        else:
            # The paper appendix explicitly generalises TIR to K-way tasks by
            # assigning Sum=K weight 0 and Sum=1 a weight greater than 1.
            # VDS2Raw has only three classes. Keep the paper's simplified
            # K-way extremes by default, while exposing explicit bounds for
            # controlled ablations.
            weights = torch.ones_like(sum_scores)
            weights = torch.where(sum_scores.eq(1), weights.new_tensor(2.0), weights)
            weights = torch.where(
                sum_scores.eq(class_count), weights.new_tensor(0.0), weights
            )
            min_weight = float(getattr(params, "tir_min_weight", 0.0))
            max_weight = float(getattr(params, "tir_max_weight", 2.0))
            if not 0.0 <= min_weight <= max_weight:
                raise ValueError(
                    "TIR K-way weight bounds must satisfy "
                    f"0 <= min <= max; got {min_weight}, {max_weight}"
                )
            raw_weights = weights
            weights = weights.clamp(min=min_weight, max=max_weight)
            rule = "paper_simplified_kway_extremes_bounded"
        if params is not None:
            params.aux_container["tir_last_sum_scores"] = sum_scores.detach()
            params.aux_container["tir_last_weights"] = weights.detach()
            params.aux_container["tir_weight_rule"] = rule
            params.aux_container.setdefault("tir_stats", []).append(
                {
                    "class_count": class_count,
                    "tokens": int(sum_scores.numel()),
                    # Keep these as detached device scalars. Converting with
                    # .item() inside forward would add a GPU synchronization
                    # to every inference/complexity measurement.
                    "discriminative": sum_scores.eq(1).sum().detach(),
                    "sink": sum_scores.eq(class_count).sum().detach(),
                    "suppressed": (
                        raw_weights.eq(0).sum().detach()
                        if class_count != 5
                        else weights.eq(0).sum().detach()
                    ),
                    "clipped": (
                        raw_weights.ne(weights).sum().detach()
                        if class_count != 5
                        else weights.new_zeros((), dtype=torch.long)
                    ),
                }
            )
        recalibrated = patches * weights.unsqueeze(-1)
        return torch.cat((cls_token, recalibrated), dim=1).permute(1, 0, 2)


class PromptLearner(nn.Module):
    """CoOp prompt learner matching the official class-token-at-end path."""

    def __init__(
        self,
        class_names: Sequence[str],
        clip_model: nn.Module,
        context_init: str = "a photo of a",
    ):
        super().__init__()
        context_init = context_init.replace("_", " ").strip()
        if not context_init:
            raise ValueError("context_init cannot be empty")
        context_length = len(context_init.split())
        prefix_tokens = tir_clip.tokenize(context_init)
        prompts = [f"{context_init} {name.replace('_', ' ')}." for name in class_names]
        tokenized = torch.cat([tir_clip.tokenize(prompt) for prompt in prompts])
        with torch.no_grad():
            prefix_embedding = clip_model.token_embedding(prefix_tokens).float()
            prompt_embedding = clip_model.token_embedding(tokenized).float()
        self.context = nn.Parameter(prefix_embedding[0, 1 : 1 + context_length].clone())
        self.register_buffer("token_prefix", prompt_embedding[:, :1])
        self.register_buffer("token_suffix", prompt_embedding[:, 1 + context_length :])
        self.register_buffer("tokenized_prompts", tokenized)
        self.class_count = len(class_names)

    def forward(self) -> torch.Tensor:
        context = self.context.unsqueeze(0).expand(self.class_count, -1, -1)
        return torch.cat((self.token_prefix, context, self.token_suffix), dim=1)


@dataclass
class TIRRuntime:
    topk_ratio: float = 0.3
    temper_rate: float = 0.5
    text_layer: int = 10
    tir_min_weight: float = 0.0
    tir_max_weight: float = 2.0
    aux_container: dict = field(default_factory=dict)

    def as_namespace(self):
        return SimpleNamespace(
            topk_ratio=self.topk_ratio,
            temper_rate=self.temper_rate,
            txtlayer=self.text_layer,
            tir_min_weight=self.tir_min_weight,
            tir_max_weight=self.tir_max_weight,
            aux_container=self.aux_container,
        )


class TIRVDS2Raw(nn.Module):
    """CLIP-LoRA + CoOp + TIR with explicit NIR/AIS VDS2Raw adapters."""

    def __init__(
        self,
        clip_checkpoint: str | Path,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        topk_ratio: float = 0.3,
        temper_rate: float = 0.5,
        text_layer: int = 10,
        tir_min_weight: float = 0.0,
        tir_max_weight: float = 2.0,
        lora_rank: int = 16,
        lora_alpha: int = 8,
        lora_dropout: float = 0.25,
        lora_encoder: str = "both",
        lora_position: str = "all",
        lora_params: Sequence[str] = ("q", "k", "v"),
        ais_dropout_probability: float = 0.1,
        nir_initial_scale: float = 0.10,
        ais_initial_scale: float = 2.0,
        logit_scale: float = 4.0,
    ):
        super().__init__()
        if len(class_names) < 2:
            raise ValueError("TIR requires at least two classes")
        if not 0.0 <= ais_dropout_probability < 1.0:
            raise ValueError("ais_dropout_probability must be in [0, 1)")
        if not 0.0 <= tir_min_weight <= tir_max_weight:
            raise ValueError("TIR weight bounds must satisfy 0 <= min <= max")
        if not 0.0 < nir_initial_scale:
            raise ValueError("nir_initial_scale must be positive")
        if not 0.0 < ais_initial_scale:
            raise ValueError("ais_initial_scale must be positive")
        if not lora_params or any(name not in {"q", "k", "v"} for name in lora_params):
            raise ValueError("lora_params must contain one or more of: q, k, v")
        self.class_names = tuple(class_names)
        self.runtime = TIRRuntime(
            topk_ratio,
            temper_rate,
            text_layer,
            tir_min_weight,
            tir_max_weight,
        )
        self.ais_dropout_probability = float(ais_dropout_probability)
        self.fixed_logit_scale = float(logit_scale)
        self.clip_model = load_clip_checkpoint_strict(clip_checkpoint)

        lora_args = SimpleNamespace(
            encoder=lora_encoder,
            position=lora_position,
            backbone=OFFICIAL_BACKBONE,
            params=list(lora_params),
            r=int(lora_rank),
            alpha=int(lora_alpha),
            dropout_rate=float(lora_dropout),
        )
        self.lora_layers = apply_lora(lora_args, self.clip_model)
        self.clip_model.visual.transformer.tir = TokenImportanceRecalibration()
        self.prompt_learner = PromptLearner(self.class_names, self.clip_model)

        class_count = len(self.class_names)
        # VDS2Raw NIR is not semantically interchangeable with RGB.  Learn a
        # directly supervised NIR classifier instead of injecting a tiny,
        # randomly initialised residual into the frozen CLIP RGB input.
        self.nir_classifier = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, class_count),
        )
        # The AIS fields are already compact tabular measurements.  A direct
        # supervised classifier is substantially better conditioned than
        # forcing four values to imitate a 512-D CLIP text embedding.
        self.ais_classifier = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(self.ais_dropout_probability),
            nn.Linear(64, class_count),
        )
        self.nir_scale = nn.Parameter(
            torch.tensor(self._inverse_softplus(nir_initial_scale))
        )
        self.ais_scale = nn.Parameter(
            torch.tensor(self._inverse_softplus(ais_initial_scale))
        )

        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.clip_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)

    def _encode_text(self, params) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = self.prompt_learner()
        tokenized = self.prompt_learner.tokenized_prompts
        model = self.clip_model
        x = prompts + model.positional_embedding.float()
        x = model.transformer(x.permute(1, 0, 2), params=params).permute(1, 0, 2)
        eot = tokenized.argmax(dim=-1)
        rows = torch.arange(x.shape[0], device=x.device)
        final = model.ln_final(x).float()[rows, eot] @ model.text_projection.float()

        block_key = f"text_block_output_{self.runtime.text_layer}"
        if block_key not in params.aux_container:
            raise RuntimeError(f"official TIR text feature {block_key} was not produced")
        intermediate = params.aux_container[block_key].permute(1, 0, 2)
        intermediate = (
            model.ln_final(intermediate).float()[rows, eot]
            @ model.text_projection.float()
        )
        return final, intermediate

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        lora = [
            parameter
            for name, parameter in self.clip_model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        ]
        prompt = list(self.prompt_learner.parameters())
        adapters = list(self.nir_classifier.parameters()) + list(
            self.ais_classifier.parameters()
        )
        return {
            "lora": lora,
            "prompt": prompt,
            "adapters": adapters,
            "gates": [self.nir_scale, self.ais_scale],
        }

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        """Stable inverse for a finite, strictly positive softplus value."""

        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("fusion scale must be finite and positive")
        return value if value > 20.0 else math.log(math.expm1(value))

    def effective_nir_scale(self) -> torch.Tensor:
        """Return the trainable NIR contribution constrained to be positive."""

        return F.softplus(self.nir_scale)

    def effective_ais_scale(self) -> torch.Tensor:
        """Return the trainable AIS contribution constrained to be positive."""

        return F.softplus(self.ais_scale)

    def forward(
        self,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        ais: torch.Tensor,
        ais_mask: torch.Tensor,
        labels=None,
        compute_auxiliary=True,
    ):
        del labels, compute_auxiliary
        params = self.runtime.as_namespace()
        params.aux_container.clear()
        text_features, tir_text_features = self._encode_text(params)
        params.aux_container["txt_feat"] = tir_text_features
        params.aux_container["visual_ln_post"] = self.clip_model.visual.ln_post
        params.aux_container["visual_projection"] = self.clip_model.visual.proj

        image_features = self.clip_model.visual(rgb, params=params).float()
        self.last_tir_stats = tuple(params.aux_container.get("tir_stats", ()))
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        visual_logits = self.fixed_logit_scale * image_features @ text_features.t()

        if ais.ndim != 2 or ais.shape[1] != 4 or ais_mask.shape != ais.shape:
            raise ValueError(
                "AIS and AIS mask must both have shape [batch, 4] for "
                "heading_sin, heading_cos, length_z, width_z"
            )
        nir_logits = self.nir_classifier(nir)
        ais_logits = self.ais_classifier(ais * ais_mask)
        logits = (
            visual_logits
            + self.effective_nir_scale() * nir_logits
            + self.effective_ais_scale() * ais_logits
        )
        return logits, nir_logits, ais_logits


__all__ = [
    "DEFAULT_CLASS_NAMES",
    "OFFICIAL_BACKBONE",
    "OFFICIAL_COMMIT",
    "PromptLearner",
    "TIRVDS2Raw",
    "TokenImportanceRecalibration",
    "load_clip_checkpoint_strict",
]
