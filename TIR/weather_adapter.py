"""TIR core adapted to the paired VIS+IR weather protocol.

The CLIP/CoOp/LoRA/TIR path is identical to the reviewed VDS2Raw adapter.  The
only task-specific extension is an independently supervised grayscale IR
classifier whose probability is fused with the VIS prediction through a
strictly-positive, confidence-calibrated gate.  No AIS or synthetic modality
is used by this weather model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from TIR.loralib.utils import apply_lora
from TIR.vds2raw_adapter import (
    OFFICIAL_BACKBONE,
    PromptLearner,
    TIRRuntime,
    TokenImportanceRecalibration,
    load_clip_checkpoint_strict,
)


class TIRWeather(nn.Module):
    """Official TIR VIS path plus a supervised, positive-fusion IR branch."""

    def __init__(
        self,
        clip_checkpoint: str | Path,
        class_names: Sequence[str],
        topk_ratio: float = 0.3,
        temper_rate: float = 0.5,
        text_layer: int = 10,
        tir_min_weight: float = 0.5,
        tir_max_weight: float = 1.5,
        lora_rank: int = 16,
        lora_alpha: int = 8,
        lora_dropout: float = 0.25,
        lora_encoder: str = "both",
        lora_position: str = "all",
        lora_params: Sequence[str] = ("q", "k", "v"),
        ir_initial_scale: float = 0.1,
        ir_dropout: float = 0.1,
        logit_scale: float = 4.0,
    ):
        super().__init__()
        if len(class_names) < 2:
            raise ValueError("TIR requires at least two target-domain classes")
        if not 0.0 <= tir_min_weight <= tir_max_weight:
            raise ValueError("TIR weight bounds must satisfy 0 <= min <= max")
        if not 0.0 < ir_initial_scale:
            raise ValueError("ir_initial_scale must be positive")
        if not 0.0 <= ir_dropout < 1.0:
            raise ValueError("ir_dropout must be in [0, 1)")
        if not lora_params or any(name not in {"q", "k", "v"} for name in lora_params):
            raise ValueError("lora_params must contain one or more of: q, k, v")

        self.class_names = tuple(class_names)
        self.fixed_logit_scale = float(logit_scale)
        self.runtime = TIRRuntime(
            topk_ratio=topk_ratio,
            temper_rate=temper_rate,
            text_layer=text_layer,
            tir_min_weight=tir_min_weight,
            tir_max_weight=tir_max_weight,
        )
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
        self.ir_classifier = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(ir_dropout),
            nn.Linear(64, class_count),
        )
        self.ir_scale = nn.Parameter(
            torch.tensor(self._inverse_softplus(ir_initial_scale))
        )

        # Freeze the pretrained CLIP model except for injected LoRA tensors.
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.clip_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("fusion scale must be finite and positive")
        return value if value > 20.0 else math.log(math.expm1(value))

    @staticmethod
    def _probabilities_and_confidence(logits: torch.Tensor):
        probabilities = logits.softmax(dim=-1)
        class_count = logits.shape[-1]
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=-1)
        confidence = (1.0 - entropy / math.log(class_count)).clamp(0.0, 1.0)
        return probabilities, confidence.detach()

    def effective_ir_scale(self) -> torch.Tensor:
        """The IR gate is trainable but cannot become negative."""

        return F.softplus(self.ir_scale)

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
            raise RuntimeError(
                f"official TIR text feature {block_key} was not produced"
            )
        intermediate = params.aux_container[block_key].permute(1, 0, 2)
        intermediate = (
            model.ln_final(intermediate).float()[rows, eot]
            @ model.text_projection.float()
        )
        return final, intermediate

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        lora_named = [
            (name, parameter)
            for name, parameter in self.clip_model.named_parameters()
            if "lora_" in name and parameter.requires_grad
        ]
        lora_vision = [
            parameter for name, parameter in lora_named if name.startswith("visual.")
        ]
        lora_text = [
            parameter
            for name, parameter in lora_named
            if not name.startswith("visual.")
        ]
        return {
            "lora": [parameter for _, parameter in lora_named],
            "lora_vision": lora_vision,
            "lora_text": lora_text,
            "prompt": list(self.prompt_learner.parameters()),
            "adapters": list(self.ir_classifier.parameters()),
            "gates": [self.ir_scale],
        }

    def forward(self, vis: torch.Tensor, ir: torch.Tensor):
        if vis.ndim != 4 or vis.shape[1] != 3:
            raise ValueError("VIS input must have shape [batch, 3, height, width]")
        if ir.ndim != 4 or ir.shape[1] != 1:
            raise ValueError("IR input must have shape [batch, 1, height, width]")
        params = self.runtime.as_namespace()
        params.aux_container.clear()
        text_features, tir_text_features = self._encode_text(params)
        params.aux_container["txt_feat"] = tir_text_features
        params.aux_container["visual_ln_post"] = self.clip_model.visual.ln_post
        params.aux_container["visual_projection"] = self.clip_model.visual.proj

        image_features = self.clip_model.visual(vis, params=params).float()
        self.last_tir_stats = tuple(params.aux_container.get("tir_stats", ()))
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        visual_logits = self.fixed_logit_scale * image_features @ text_features.t()
        ir_logits = self.ir_classifier(ir)

        visual_probabilities = visual_logits.softmax(dim=-1)
        ir_probabilities, ir_confidence = self._probabilities_and_confidence(ir_logits)
        ir_weight = self.effective_ir_scale() * ir_confidence
        denominator = 1.0 + ir_weight
        fused_probabilities = (
            visual_probabilities + ir_weight.unsqueeze(1) * ir_probabilities
        ) / denominator.unsqueeze(1)
        fused_logits = fused_probabilities.clamp_min(1e-8).log()
        self.last_fusion_stats = {
            "visual_fraction_sum": (1.0 / denominator).sum().detach(),
            "ir_fraction_sum": (ir_weight / denominator).sum().detach(),
            "ir_confidence_sum": ir_confidence.sum().detach(),
            "samples": int(fused_logits.shape[0]),
        }
        return fused_logits, visual_logits, ir_logits


__all__ = ["TIRWeather"]
