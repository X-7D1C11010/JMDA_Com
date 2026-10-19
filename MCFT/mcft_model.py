"""
MCFT (Multimodal Contrastive Fusion Transformer) model, ported to
multimodal (VIS + IR) weather image classification.

Official implementation used as the reference:
    https://github.com/SYFYN0317/MCFT   ->  ``MViT_pytorch_upload.py``

Reference architecture (``MViT`` in the release, driven by
``MViT_demo_h2013.py`` with patch_size=15, dim=64, depth=6, heads=4,
mlp_dim=32, dropout=0.1, emb_dropout=0.1 and loss
``CE + 0.1 * loss_ita + loss_itm``):

    separable1(x1) -> patch embedding -> relative-pixel position embedding
    separable2(x2) -> patch embedding -> relative-pixel position embedding
    TransformerC(x1)   (intra-modal, residual + attention)
    TransformerC(x2)   (intra-modal)
    ITA: image/text projection + soft-target contrastive alignment (momentum
         alpha = 0.7 over the batch similarity matrix)
    ITM: cross-modal TransformerCC over the concatenated token sequence with
         ALBEF-style hard-negative mining, binary (matched / unmatched) head
    classification head: learned per-token softmax weights
         (mlp_head0 -> einsum -> mlp_head1)

Faithfulness notes -- deviations are listed explicitly:

  * ``einops.rearrange`` / ``timm`` / ``scipy.stats`` are gone: the port uses
    plain ``reshape``/``permute`` and ``torch.nn.init.trunc_normal_`` so that no
    new dependency is required.  The tensor algebra itself is unchanged.
  * The release hardcodes the 15x15 reshape in ``get_position_embedding`` and
    the centre index ``(7, 7)`` at its call site.  Both are rewritten in terms
    of the actual token count / patch size, which for patch_size = 15 gives the
    identical (15, 15) / (7, 7) values.
  * ``Attentionc`` splits the concatenated sequence at a hardcoded index 169
    (``x[:, :169]`` / ``x[:, 169:]``), a constant inherited from the
    Agent-Attention 13x13 default rather than from this model's own 15x15 patch
    grid: the sequence holds 225 tokens per modality, so the boundary lands
    *inside* the first modality (169 of 225) and the "cross-modal" attention is
    in fact misaligned.  This is reproduced **verbatim** because the released
    code is the official implementation; the constant lives in the single
    module-level ``AGENT_TOKEN_SPLIT`` so a geometry-correct variant
    (225 at patch_size = 15) is one edit away.  Any claim about this model's
    cross-modal semantics has to account for it.
  * Never-used parameters of the release (``pos_embedding``, ``vision_pos_m``,
    ``text_neg_m``, commented-out ``cls_token``) are omitted because they never
    participate in the forward pass; parameter counts therefore differ slightly
    from a literal transcription of the dead code.
  * Negative mining.  The release calls ``torch.multinomial(weights[b], 1)``
    per row; rows whose weights are entirely zero (duplicate labels inside a
    batch -- always the case at batch size 1) raise a RuntimeError.  Such rows
    fall back to a uniform draw so that the benchmark's batch-1 inference path
    and small-batch training both run.  Row weights are otherwise untouched.
  * ``mlp_head0`` squeeze is applied on the last axis only, so batch size 1
    (used by the efficiency benchmark) does not collapse the token axis.  For
    all batch sizes > 1 this is identical to the release's ``torch.squeeze``.
  * The ITM head keeps the release's shape: the sequence is flattened, not
    pooled, so ``itm_head`` maps ``2 * patch_size^2 * nout * 4`` (28800 at
    patch_size = 15) to 2 binary logits.
"""

import math

import numpy as np
import torch
import torch.nn as nn

AGENT_TOKEN_SPLIT = 169  # 13x13, the release's hardcoded agent-token boundary


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Standard multi-head self attention (the release's ``Attention``)."""

    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        heads = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, heads, -1).permute(0, 2, 1, 3) for t in qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class Attentionc(nn.Module):
    """The release's cross-branch attention: agent tokens vs. the remaining
    tokens of the same (already concatenated) sequence."""

    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, heads = x.shape[0], x.shape[1], self.heads
        x1 = x[:, :AGENT_TOKEN_SPLIT, :]
        x2 = x[:, AGENT_TOKEN_SPLIT:, :]

        def split(t):
            return (v.reshape(b, t.shape[1], heads, -1).permute(0, 2, 1, 3)
                    for v in self.to_qkv(t).chunk(3, dim=-1))

        q1, k1, v1 = split(x1)
        q2, k2, v2 = split(x2)
        dots1 = torch.einsum('bhid,bhjd->bhij', q1, k2) * self.scale
        dots2 = torch.einsum('bhid,bhjd->bhij', q2, k1) * self.scale
        attn1 = dots1.softmax(dim=-1)
        attn2 = dots2.softmax(dim=-1)
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v2)
        out2 = torch.einsum('bhij,bhjd->bhid', attn2, v1)
        out1 = out1.permute(0, 2, 1, 3).reshape(b, x1.shape[1], -1)
        out2 = out2.permute(0, 2, 1, 3).reshape(b, x2.shape[1], -1)
        out = torch.cat((x1 + out1, x2 + out2), dim=1)
        return self.to_out(out)


class TransformerC(nn.Module):
    """Intra-modal transformer stack (``TransformerC`` in the release)."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ])
            for _ in range(depth)
        ])
        self.mode = mode

    def forward(self, x, mask=None):
        if self.mode == 'MViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        return x


class TransformerCC(nn.Module):
    """Cross-modal transformer stack (``TransformerCC`` in the release)."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attentionc(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ])
            for _ in range(depth)
        ])
        self.mode = mode

    def forward(self, x, mask=None):
        if self.mode == 'MViT':
            for attnc, ff in self.layers:
                x = attnc(x, mask=mask)
                x = ff(x)
        return x


def _separable_stack(num_patches, nout):
    """The release's ``separable1`` / ``separable2`` depthwise-separable stack."""
    return nn.Sequential(
        nn.Conv2d(num_patches, num_patches, kernel_size=3, padding=1, groups=num_patches),
        nn.Conv2d(num_patches, nout, kernel_size=1),
        nn.BatchNorm2d(nout),
        nn.GELU(),
        nn.Conv2d(nout, nout, kernel_size=3, padding=1, groups=nout),
        nn.Conv2d(nout, nout * 2, kernel_size=1),
        nn.BatchNorm2d(nout * 2),
        nn.GELU(),
        nn.Conv2d(nout * 2, nout * 2, kernel_size=3, padding=1, groups=nout * 2),
        nn.Conv2d(nout * 2, nout * 4, kernel_size=1),
        nn.BatchNorm2d(nout * 4),
        nn.GELU(),
    )


class MViT(nn.Module):
    def __init__(self, patch_size, num_patches, num_classes, dim, depth, heads,
                 mlp_dim, pool='cls', channels=1, dim_head=16, dropout=0.0,
                 emb_dropout=0.0, mode='MViT'):
        super().__init__()
        nout = 16
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.separable1 = _separable_stack(num_patches[0], nout)
        self.separable2 = _separable_stack(num_patches[1], nout)

        self.to_patch_embedding2 = nn.Linear(nout * 4, dim)
        self.to_patch_embedding2c = nn.Linear(nout * 4, dim)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerCC(dim, max(depth - 4, 1), heads, dim_head, mlp_dim, dropout, mode)
        self.transformer1 = TransformerC(dim, max(depth - 2, 1), heads, dim_head, mlp_dim, dropout, mode)
        self.transformer2 = TransformerC(dim, max(depth - 2, 1), heads, dim_head, mlp_dim, dropout, mode)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.vision_proj_m = nn.Linear(patch_size * patch_size * nout * 4, dim)
        self.text_proj_m = nn.Linear(patch_size * patch_size * nout * 4, dim)
        # Release: Linear(patch_size*patch_size*nout*8, 2); with nout*4 == dim the
        # flattened 2-modality sequence is patch_size^2*2*dim wide -- same number.
        self.itm_head = nn.Linear(patch_size * patch_size * nout * 8, 2)

        self.mlp_head0 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1),
        )
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        self.pixel_pos_embedding_relative = nn.Parameter(torch.randn(patch_size * patch_size, dim))
        nn.init.trunc_normal_(self.pixel_pos_embedding_relative, std=.02)

        self.alpha = 0.7

    def get_position_embedding(self, x, cls_token=False):
        """Chebyshev-distance relative pixel position embedding (release logic)."""
        b, n, c = x.shape
        side = int(round(n ** 0.5))
        x = x.reshape(b, side, side, c).permute(0, 3, 1, 2)
        center = side // 2
        pos_index = [[max(abs(i - center), abs(j - center)) for j in range(side)]
                     for i in range(side)]
        pos_index = np.asarray(pos_index, dtype=np.int64).reshape(-1) + 1
        if cls_token:
            pos_index = np.asarray([-1] + list(pos_index))
        return self.pixel_pos_embedding_relative[pos_index, :]

    @staticmethod
    def _sample_negative(weights, fallback_size):
        """Pick one index per row; rows whose weights collapsed to zero fall
        back to a uniform draw (the release calls ``torch.multinomial`` directly
        and crashes on such rows)."""
        idx = []
        for row in weights:
            if float(row.sum()) > 0:
                idx.append(int(torch.multinomial(row, 1).item()))
            else:
                idx.append(int(torch.randint(0, max(fallback_size, 1), (1,), device=row.device).item()))
        return torch.tensor(idx, device=weights.device, dtype=torch.long)

    def forward(self, x1, x2, label, mask=None):
        # ── multimodal tokenisation ─────────────────────────────────────────
        x1 = self.separable1(x1).flatten(2).transpose(1, 2)
        x1 = self.to_patch_embedding2(x1)
        pos_emb = torch.unsqueeze(self.get_position_embedding(x1, cls_token=False), dim=0)
        x1 = self.dropout(x1 + pos_emb)

        x2 = self.separable2(x2).flatten(2).transpose(1, 2)
        x2 = self.to_patch_embedding2c(x2)
        x2 = self.dropout(x2 + pos_emb)

        # ── intra-modal transformers ────────────────────────────────────────
        x1 = self.transformer1(x1)
        x2 = self.transformer2(x2)

        # ── ITA: multimodal contrastive alignment ───────────────────────────
        image_features = torch.nn.functional.normalize(self.vision_proj_m(x1.flatten(1)), dim=-1)
        text_features = torch.nn.functional.normalize(self.text_proj_m(x2.flatten(1)), dim=-1)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        sim_targets = torch.zeros(logits_per_image.size(), device=x1.device)
        sim_targets.fill_diagonal_(1)
        sim_i2t_targets = (self.alpha * torch.nn.functional.softmax(logits_per_image, dim=1)
                           + (1 - self.alpha) * sim_targets)
        sim_t2i_targets = (self.alpha * torch.nn.functional.softmax(logits_per_text, dim=1)
                           + (1 - self.alpha) * sim_targets)
        loss_i2t = -torch.sum(torch.nn.functional.log_softmax(logits_per_image, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(torch.nn.functional.log_softmax(logits_per_text, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2

        # ── cross-modal fusion transformer + classification ─────────────────
        x = self.transformer(torch.cat((x1, x2), dim=1))
        output_pos = x

        xs = self.mlp_head0(x).squeeze(-1)          # released: torch.squeeze(...)
        xs = torch.einsum('bn,bnd->bd', xs, x)
        logits = self.mlp_head1(xs)

        # ── ITM with ALBEF-style hard negative mining ───────────────────────
        bs = x1.size(0)
        weights_i2t = torch.nn.functional.softmax(logits_per_image[:, :bs], dim=1)
        weights_t2i = torch.nn.functional.softmax(logits_per_text[:, :bs], dim=1)

        # matrix[i][j] = label[j]; same-class entries zeroed, then such entries
        # (and the diagonal) are removed from the negative pool.
        matrix = label.view(1, -1).expand(bs, bs).clone()
        same = matrix.eq(label.view(-1, 1))
        matrix = torch.where(same, torch.zeros_like(matrix), matrix)
        weights_i2t = weights_i2t.clone()
        weights_t2i = weights_t2i.clone()
        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)
        weights_i2t = weights_i2t.masked_fill(matrix == 0, 0)
        weights_t2i = weights_t2i.masked_fill(matrix == 0, 0)

        neg_idx_i = self._sample_negative(weights_t2i, bs)
        neg_idx_t = self._sample_negative(weights_i2t, bs)
        image_embeds_neg = x1[neg_idx_i]
        text_embeds_neg = x2[neg_idx_t]

        text_embeds_all = torch.cat([x2, text_embeds_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, x1], dim=0)
        x4 = torch.cat((text_embeds_all, image_embeds_all), dim=1)
        output_neg = self.transformer(x4)

        # Release flattens the token sequence (``b n d -> b (n d)``), it does not pool.
        vl_embeddings = torch.cat([output_pos.flatten(1), output_neg.flatten(1)], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([
            torch.ones(bs, dtype=torch.long, device=x1.device),
            torch.zeros(2 * bs, dtype=torch.long, device=x1.device),
        ], dim=0)
        loss_itm = torch.nn.functional.cross_entropy(vl_output, itm_labels)

        return logits, loss_ita, loss_itm


__all__ = ['MViT']
