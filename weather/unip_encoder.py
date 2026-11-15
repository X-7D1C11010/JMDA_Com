"""Opt-in UNIP infrared feature extraction, retaining the released architecture.

Reference: https://github.com/casiatao/UNIP (ICLR 2025).
UNIP distils Q/K attention: the last block's value/MLP output is not the
distillation target. We explicitly probe penultimate tokens and last Q/K,
rather than silently treating an ImageNet ViT as the released encoder.
"""
import argparse
from functools import partial
import hashlib
from pathlib import Path

import torch
from torch import nn


CHECKSUMS = {
    "tiny": "d5ebe70319a1d5a5fd7c2ed4fdfc6518f34c1c2020c25a57175fcf4191805f59",
    "small": "54a8aa79e92c3e1a7f1936c02faf411ec34651fd5aa5cc8eeeb7e6c09ca14af4",
}


class UNIPEncoder(nn.Module):
    def __init__(self, weights, scale="small", feature_mode="patch_mean"):
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        if scale not in CHECKSUMS or feature_mode not in {"patch_mean", "cls", "qk_mean", "last4", "cls_attention"}:
            raise ValueError("Unsupported UNIP checkpoint or feature mode")
        path = Path(weights)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != CHECKSUMS[scale]:
            raise ValueError("Official UNIP checkpoint checksum mismatch")
        # Explicit safe allowlist for the official training metadata only.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        args = checkpoint["args"]
        dimension, heads = (192, 3) if scale == "tiny" else (384, 6)
        self.trunk = VisionTransformer(img_size=224, patch_size=16, embed_dim=dimension,
            depth=12, num_heads=heads, mlp_ratio=4., qkv_bias=True, num_classes=0,
            global_pool="", norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.)
        last_heads = int(args.last_heads)
        if dimension % last_heads:
            raise ValueError("Released final attention head shape is incompatible")
        last = self.trunk.blocks[-1].attn
        last.num_heads, last.head_dim = last_heads, dimension//last_heads
        last.scale = last.head_dim**-.5
        self.trunk.load_state_dict(checkpoint["model"], strict=True)
        self.trunk.pos_embed.requires_grad_(False)
        self.feature_mode, self.dimension = feature_mode, dimension
        self.last_pooling_attention=None
        self.output_dim = dimension*({"qk_mean": 2, "last4": 4}.get(feature_mode, 1))
        self.provenance = {"source": "official UNIP iBOT-L attention-distilled encoder",
            "scale": scale, "sha256": digest, "last_attention_heads": last_heads,
            "normalization": {"mean": .3801, "std": .1871},
            "feature_mode": feature_mode, "output_dim": self.output_dim,
            "classification_benefit_verified": False}
        if feature_mode=="cls_attention":
            self.provenance["pooling"]={"kind":"experimental soft pooling using released final CLS Q/K attention",
                "global_mean_fraction":.5,"attention_fraction":.5,"hard_crop_or_mask":False,
                "extra_trainable_pooling_parameters":0,"official_classification_head":False}

    def extract_modes(self, half_normalized_ir):
        self.last_pooling_attention=None
        if half_normalized_ir.shape[1:] != (3, 224, 224):
            raise ValueError("UNIP expects the shared three-channel 224x224 IR view")
        # Preserve the loader's geometric view, restore its documented .5/.5
        # normalization, then apply the released infrared pretraining scale.
        image = (half_normalized_ir*.5+.5-.3801)/.1871
        tokens = self.trunk.patch_embed(image)
        tokens = torch.cat((self.trunk.cls_token.expand(len(image), -1, -1), tokens), 1)
        tokens = tokens + self.trunk.pos_embed
        levels = []
        for i, block in enumerate(self.trunk.blocks[:-1]):
            tokens = block(tokens)
            if i >= 7:
                levels.append(self.trunk.norm(tokens)[:, 1:].mean(1))
        normalized = self.trunk.norm(tokens)
        qkv = self.trunk.blocks[-1].attn.qkv(self.trunk.blocks[-1].norm1(tokens))
        q, k, _ = qkv.chunk(3, dim=-1)
        modes={"patch_mean": normalized[:, 1:].mean(1), "cls": normalized[:, 0],
                "qk_mean": torch.cat((q[:, 1:].mean(1), k[:, 1:].mean(1)), 1),
                "last4": torch.cat(levels, 1)}
        if self.feature_mode=="cls_attention":
            batch,count,width=q.shape
            attention=self.trunk.blocks[-1].attn
            query=q.reshape(batch,count,attention.num_heads,attention.head_dim).permute(0,2,1,3)[:,:,0]
            keys=k.reshape(batch,count,attention.num_heads,attention.head_dim).permute(0,2,1,3)[:,:,1:]
            scores=(query.unsqueeze(2)*keys).sum(-1)*attention.scale
            # Keep normalized diagnostic/attention weights in FP32 even when
            # the encoder runs in BF16; avoid quantized row-sum drift.
            weights=scores.float().softmax(-1).mean(1)
            weighted=torch.bmm(weights.unsqueeze(1),normalized[:,1:]).squeeze(1)
            modes["cls_attention"]=(modes["patch_mean"]+weighted)*.5
            self.last_pooling_attention=weights.detach()
        return modes

    def forward(self, images):
        return self.extract_modes(images)[self.feature_mode]

    def unfreeze_last_stage(self):
        self.requires_grad_(False)
        for block in self.trunk.blocks[-4:-1]:
            block.requires_grad_(True)
        self.trunk.norm.requires_grad_(True)
        if self.feature_mode in {"qk_mean","cls_attention"}:
            self.trunk.blocks[-1].norm1.requires_grad_(True)
            self.trunk.blocks[-1].attn.qkv.requires_grad_(True)

    def unfreeze_all_used_layers(self):
        self.requires_grad_(False)
        self.trunk.patch_embed.requires_grad_(True)
        self.trunk.cls_token.requires_grad_(True)
        for block in self.trunk.blocks[:-1]:block.requires_grad_(True)
        self.trunk.norm.requires_grad_(True)
        if self.feature_mode in {"qk_mean","cls_attention"}:
            self.trunk.blocks[-1].norm1.requires_grad_(True)
            self.trunk.blocks[-1].attn.qkv.requires_grad_(True)
