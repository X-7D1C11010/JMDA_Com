"""
PDA (Prompt-based Distribution Alignment, AAAI 2024) ported to the JMDA-Net
weather dataset as an unsupervised domain adaptation (UDA) method.

Official implementation used as the reference:
    https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment
    Ported pieces: ``trainers/da/pda.py`` (PromptLearner / CustomCLIP / PDA),
    ``utils/clip_part.py`` (TextEncoder, Simple_TextEncoder, ImageEncoder_Trans,
    load_clip_to_cpu), ``utils/ift_block.py`` (IFT_Module),
    ``trainers/baseda.py`` (Base_PromptLearner, run_epoch) and
    ``configs/trainers/PDA/b32_ep20_office31.yaml`` +
    ``scripts/pda/main_pda.sh`` (hyper-parameters).

PDA in one paragraph: CLIP's image and text encoders are frozen; learnable
text context vectors, visual context vectors and per-layer deep prompts are
trained.  A source and a target feature bank (K = 5 slots per class) are filled
from high-confidence CLIP predictions; the ``IFT_Module`` lets the image
feature attend to both banks, producing a second set of logits that are added
to the base logits.  The target branch is trained with confidence-thresholded
pseudo labels.

Faithfulness notes -- every deviation is listed:

  * ``dassl`` is not vendored: the trainer loop is re-implemented with the same
    budget semantics (``COUNT_ITER="train_x"`` -> one epoch = ``len(src_loader)``
    steps, the unlabeled iterator is recycled when exhausted), the same
    optimizer (SGD) and the same cosine-with-warmup schedule, stepped once per
    epoch exactly like ``BaseDA.run_epoch`` -> ``update_lr()``.
  * ``einops.rearrange`` in ``IFT_Module`` is replaced by ``chunk(3, dim=-1)``,
    which is exactly the same tensor split (a (d k) reshape with k = 3 followed
    by a stack over k).
  * Only the ViT-B/16 path is wired up (``ImageEncoder_Trans``); the released
    ``ImageEncoder_Conv`` alternative cannot be prompt-tuned anyway (its own
    comment: "Maybe furtherly there will be some prompt tuning methods for
    convolutional networks"), and the released ``PromptLearner`` only supports
    ``SHARE_LAYER == [0, 5]`` + ``DEEP_LAYERS is None`` for the visual deep
    prompt, so other combinations raise a clear error instead of silently
    doing something else.
  * ``torch.jit`` loading is bypassed: ``clip.build_model(state_dict)`` is used
    in place of the released ``torch.jit.load(...).eval()`` fallback pair, which
    yields the identical module tree (the JIT graph is only needed for speed).
  * Confidence threshold / warm-up / epoch-switch hyper-parameters are the
    Office-31 values used by the released script (CONFI = 0.9, WARM_UP = 1,
    EPOCH = 0); they are exposed as flags.
  * CLIP weight download: the released helper calls ``urllib`` with the default
    Windows certificate store, which raises ``ssl.SSLError`` on this machine, so
    the downloader falls back to ``certifi``'s CA bundle.  The SHA-256 check is
    kept.
  * Class names: this project's dataset only exposes numeric class folders, so
    prompts use numbered placeholder names (``class_1`` ... ``class_{|T|}``)
    unless a real mapping is supplied via ``--class-names-file`` /
    ``JMDA_CLASS_NAMES_FILE``.  The efficiency record stores which of the two was
    used.  2026-10-20: the count follows the target-driven label space
    (``tools/uda_data.py``), i.e. the number of classes that actually exist in
    the target domain, not the dataset-wide 14.
  * Precision: the release defaults to fp16 (``TRAINER.PDA.PREC = "fp16"``);
    fp16 is kept, and ``fp32`` is forced when CUDA is unavailable.

Protocol: source = sunny (labelled, used for training), target = <weather>
unlabelled training split (pseudo-labels), target val split for reporting only.

Usage:
    python PDA/train_pda.py --target backlight --epochs 20 --seed 42 \
        --efficiency_output_dir efficiency_results/uda_sunny
"""

import os
import sys
import time
import copy
import hashlib
import argparse
import ssl
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
# UDA 共用数据加载：以目标域类别集为准裁剪源域/目标域并重映射标签
from uda_data import UdaImageListDataset, uda_label_space, format_class_space, format_dropped

import clip
from clip.clip import _MODELS, _download
from clip.model import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TEMPLATE = 'a photo of a {}.'


# ── CLIP loading (release ``load_clip_to_cpu`` + certifi SSL fallback) ──────
def _open_url(url):
    """urlopen with a certifi CA bundle when the OS trust store is unusable."""
    try:
        context = ssl.create_default_context()
    except Exception:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.urlopen(url, context=context)


def load_clip_to_cpu(backbone_name='ViT-B/16', cache_dir=None):
    url = _MODELS[backbone_name]
    cache_dir = cache_dir or os.path.expanduser('~/.cache/clip')
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, os.path.basename(url))
    expected_sha256 = url.split('/')[-2]

    have_valid = (os.path.isfile(target) and
                  hashlib.sha256(open(target, 'rb').read()).hexdigest() == expected_sha256)
    if not have_valid:
        print(f'Downloading CLIP {backbone_name} -> {target}')
        with _open_url(url) as source, open(target, 'wb') as output:
            while True:
                chunk = source.read(1 << 16)
                if not chunk:
                    break
                output.write(chunk)
        digest = hashlib.sha256(open(target, 'rb').read()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError('CLIP checkpoint SHA-256 mismatch after download')

    try:
        model = torch.jit.load(target, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(target, map_location='cpu')
    return build_model(state_dict or model.state_dict())


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ── data ────────────────────────────────────────────────────────────────────
def build_transforms():
    """INPUT.TRANSFORMS from the released Office-31 config: random_resized_crop,
    random_flip, normalize (CLIP statistics)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return train_tf, test_tf


def load_classnames(args, class_ids):
    """Real names from --class-names-file / JMDA_CLASS_NAMES_FILE, else numbered
    placeholders (recorded so results can never be mistaken for a real run).

    The mapping file always covers the full 14-class source label space; the
    prompts use only the subset of classes present in the target domain
    (``class_ids`` are original label ids, in compact-label order)."""
    path = args.class_names_file or os.environ.get('JMDA_CLASS_NAMES_FILE', '')
    if path:
        from class_names import load_class_names
        all_names = load_class_names(path, expected_count=14)
        return [all_names[cid] for cid in class_ids], 'file'
    print('WARNING: no class-name mapping supplied; using numbered placeholder '
          'prompts (not scientifically valid, recorded as prompt_source=placeholder).')
    return [f'class_{i + 1}' for i in range(len(class_ids))], 'placeholder'


# ── CLIP sub-modules (ported) ───────────────────────────────────────────────
class Base_PromptLearner(nn.Module):
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        return torch.cat([prefix, ctx.half(), suffix], dim=1)


class Simple_TextEncoder(nn.Module):
    """Released ``Simple_TextEncoder``: plain zero-shot text features."""

    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection


class TextEncoder(nn.Module):
    def __init__(self, clip_model, num_layer, num_tokens, location):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.num_layer = num_layer
        self.num_tokens = num_tokens
        self.location = location
        self.prompt_dropout = nn.Dropout(0.0)

    def forward(self, prompts, tokenized_prompts, deep_prompts=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_deep(x, deep_prompts) if deep_prompts is not None else self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

    def transformer_deep(self, x, deep_x):
        for i in range(self.num_layer):
            if i == 0:
                x = self.transformer.resblocks[i](x)
            else:
                if i <= deep_x.shape[0]:
                    deep_ctx_i = self.prompt_dropout(deep_x[i - 1])
                    deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                    if self.location == 'middle':
                        x = torch.cat((x[:1, :, :], deep_ctx, x[(1 + self.num_tokens):, :, :]), dim=0)
                    else:
                        x = torch.cat([x[0: x.shape[0] - self.num_tokens, :, :], deep_ctx], dim=0)
                x = self.transformer.resblocks[i](x)
        return x


class ImageEncoder_Trans(nn.Module):
    """Released ``ImageEncoder_Trans`` (ViT visual encoder + visual deep prompts)."""

    def __init__(self, clip_model, num_layer, num_tokens, location):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        self.num_layer = num_layer
        self.num_tokens = num_tokens
        self.location = location
        self.prompt_dropout = nn.Dropout(0.0)
        self.dim = clip_model.text_projection.shape[1]

    def forward(self, x, vctx=None, deep_vctx=None, return_feat=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        if vctx is not None:
            x = self.incorporate_prompt(x, vctx)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_deep(x, deep_vctx) if deep_vctx is not None else self.transformer(x)
        x = x.permute(1, 0, 2)

        feats = None
        if return_feat:
            feats = F.adaptive_avg_pool1d(x, self.dim)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return (x, feats) if return_feat else x

    def transformer_deep(self, x, deep_x):
        for i in range(self.num_layer):
            if i == 0:
                x = self.transformer.resblocks[i](x)
            else:
                if i <= deep_x.shape[0]:
                    deep_ctx_i = self.prompt_dropout(deep_x[i - 1])
                    deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                    if self.location == 'middle':
                        x = torch.cat((x[:1, :, :], deep_ctx, x[(1 + self.num_tokens):, :, :]), dim=0)
                    else:
                        x = torch.cat([x[0: x.shape[0] - self.num_tokens, :, :], deep_ctx], dim=0)
                x = self.transformer.resblocks[i](x)
        return x

    def incorporate_prompt(self, x, vctx):
        if self.location == 'middle':
            return torch.cat((
                x[:, :1, :],
                self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half(),
                x[:, 1:, :],
            ), dim=1)
        return torch.cat([x, self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half()], dim=1)


class IFT_Module(nn.Module):
    """Released ``IFT_Module``: image-guided feature tuning over both banks."""

    def __init__(self, clip_model, beta_s=0.1, beta_t=0.1):
        super().__init__()
        self.softmax = nn.Softmax(-1)
        input_dim = clip_model.text_projection.shape[1]
        pre_dim1 = input_dim // 8
        pre_dim2 = input_dim // 8
        self.beta_s = beta_s
        self.beta_t = beta_t
        self.scale = 0.1
        self.pre_project = nn.Sequential(
            nn.Linear(input_dim, pre_dim1),
            nn.BatchNorm1d(pre_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(pre_dim1, pre_dim2),
            nn.BatchNorm1d(pre_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(pre_dim2, input_dim * 3),
        ).half()
        self.post_project = nn.Linear(input_dim, input_dim).half()
        self.logit_scale = clip_model.logit_scale

    def forward(self, Ft, Fv, Fvs_bank, Fvt_bank):
        out_fv = self.pre_project(Fv)
        out_fvs = self.pre_project(Fvs_bank)
        out_fvt = self.pre_project(Fvt_bank)
        # 官方实现为 rearrange(out, 'b (d k) -> k b d', k=3)：按 d 外层、k 内层
        # 交错拆分（q/k/v 各取每隔 3 个元素的一段），与 chunk(3) 的连续三段
        # 划分不同；这里按官方索引逐位复现。
        q_fv, k_fv, v_fv = out_fv.view(out_fv.size(0), -1, 3).permute(2, 0, 1).unbind(0)
        q_fvs, k_fvs, v_fvs = out_fvs.view(out_fvs.size(0), -1, 3).permute(2, 0, 1).unbind(0)
        q_fvt, k_fvt, v_fvt = out_fvt.view(out_fvt.size(0), -1, 3).permute(2, 0, 1).unbind(0)

        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))
        At = self.softmax(self.scale * q_fv @ k_fvt.permute(1, 0))
        Fsa = Fv + self.post_project(As @ v_fvs)
        Fta = Fv + self.post_project(At @ v_fvt)
        Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)
        Fta = Fta / Fta.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_s = self.beta_s * logit_scale * Fsa @ Ft.t()
        logits_t = self.beta_t * logit_scale * Fta @ Ft.t()
        return logits_s + logits_t


class PromptLearner(Base_PromptLearner):
    """Released ``PromptLearner`` limited to the shipped PDA configuration
    (TP + VP with SHARE_LAYER=[0,5] + T_DEEP + V_DEEP with DEEP_SHARED)."""

    def __init__(self, classnames, clip_model, n_ctx, num_tokens, ctx_init,
                 num_layer, hidden_size, location, share_layer, deep_layer):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dim = clip_model.text_projection.shape[1]
        assert clip_model.visual.input_resolution == 224, 'CLIP ViT-B/16 expects 224px inputs'

        self.num_tokens = num_tokens
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.location = location
        self.deep_layer = deep_layer
        self.share_layer = share_layer
        self.prompt_dropout = nn.Dropout(0.0)

        # Text context.  NOTE: the release assigns self.ctx from the tokenised
        # init words and then immediately overwrites it with nn.init.normal_
        # through the shared tensor, so the effective initialisation is a
        # normal draw with std 0.02 -- reproduced here directly.
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        if ctx_init and n_ctx <= 4:
            prompt_prefix = ctx_init.replace('_', ' ')
        else:
            prompt_prefix = ' '.join(['X'] * n_ctx)

        # Visual context: SHARE_LAYER[0] == 0 -> projected from the text context.
        if share_layer is None or share_layer[0] != 0:
            raise NotImplementedError('Only SHARE_LAYER == [0, 5] is supported by this port.')
        self.proj = nn.Linear(ctx_dim, hidden_size).half()

        # Deep text prompts: DEEP_LAYERS is None -> every layer above the first.
        deep_ctx_vectors = torch.empty(num_layer - 1, num_tokens, ctx_dim)
        nn.init.normal_(deep_ctx_vectors, std=0.02)
        self.deep_ctx = nn.Parameter(deep_ctx_vectors)

        # Deep visual prompts (DEEP_SHARED): projected from deep_ctx, plus the
        # shared-layers block that is appended after the projections.
        single_layer = nn.Linear(ctx_dim, hidden_size)
        deep_vctx_vectors = torch.empty(num_layer - share_layer[1] - 1, num_tokens, hidden_size)
        self.deep_prompt_proj = get_clones(single_layer, share_layer[1] - share_layer[0] + 1)
        nn.init.normal_(deep_vctx_vectors, std=0.02)
        self.deep_vctx = nn.Parameter(deep_vctx_vectors)

        print('Prompt design: Prompt-based Distribution Alignment for UDA')
        print(f'Initial context: "{prompt_prefix}"')
        print(f'Number of PDA context words (tokens): {n_ctx}')

        names = [name.replace('_', ' ') for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in names]
        # The class name is placed after the n_ctx context tokens; with
        # n_ctx == 2 the prompt is "a photo <class name>."
        prompts = [prompt_prefix + ' ' + name + '.' for name in names]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        self.attn_block = IFT_Module(clip_model, beta_s=0.1, beta_t=0.1)
        self.K = 5
        source_feat_bank = torch.zeros((n_cls * self.K, self.dim)).half()
        target_feat_bank = torch.zeros((n_cls * self.K, self.dim)).half()
        self.source_feat_bank = nn.Parameter(source_feat_bank)
        self.target_feat_bank = nn.Parameter(target_feat_bank)

    def forward(self):
        vctx = self.proj(self.ctx)
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        deep_vctx = []
        for index, layer in enumerate(self.deep_prompt_proj):
            deep_vctx.append(layer(self.deep_ctx[index]))
        deep_vctx = torch.cat((torch.stack(deep_vctx), self.deep_vctx), dim=0)
        return prompts, self.deep_ctx, vctx, deep_vctx


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx, num_tokens, ctx_init,
                 num_layer, hidden_size, location, share_layer, deep_layer):
        super().__init__()
        self.prompt_learner = PromptLearner(
            classnames, clip_model, n_ctx, num_tokens, ctx_init, num_layer,
            hidden_size, location, share_layer, deep_layer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.K = self.prompt_learner.K
        self.dim = clip_model.text_projection.shape[1]
        self.text_encoder = TextEncoder(clip_model, num_layer, num_tokens, location)
        self.image_encoder = ImageEncoder_Trans(clip_model, num_layer, num_tokens, location)
        self.n_cls = len(classnames)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.text_encoder_u = Simple_TextEncoder(clip_model)
        prompts_u = [TEMPLATE.format(c.replace('_', ' ')) for c in classnames]
        self.tokenized_prompts_u = clip.tokenize(prompts_u)

        self.source_key_dict = {i: i for i in range(self.n_cls * self.K)}
        self.target_key_dict = {i: i for i in range(self.n_cls * self.K)}
        self.source_max_probs_list = [0.0 for _ in range(self.n_cls * self.K)]
        self.target_max_probs_list = [0.0 for _ in range(self.n_cls * self.K)]

        self.confi = 0.9
        self.epoch_switch = 0
        self.warm_up = 1

    def forward(self, image, label=None, epoch=None, train=False, construct=False, source=True):
        prompts, deep_ctx, vctx, deep_vctx = self.prompt_learner()

        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        text_features_u = self.text_encoder_u(self.tokenized_prompts_u.to(self.logit_scale.device))
        text_features_u = text_features_u / text_features_u.norm(dim=-1, keepdim=True)
        F_u = self.image_encoder(image.type(self.dtype), None, None)
        F_u = F_u / F_u.norm(dim=-1, keepdim=True)

        if construct:
            logits_u = logit_scale * F_u @ text_features_u.t()
            pseudo_label = torch.softmax(logits_u, dim=-1)
            max_probs, label_p = torch.max(pseudo_label, dim=-1)
            if source:
                for i, l in enumerate(label):
                    if l == label_p[i]:
                        index = l.item() * self.K
                        l_list = self.source_max_probs_list[index: index + self.K]
                        if max_probs[i] > min(l_list):
                            min_index = l_list.index(min(l_list))
                            self.source_max_probs_list[index + min_index] = max_probs[i]
                            self.prompt_learner.source_feat_bank[index + min_index] = F_u[i]
                            self.source_key_dict[index + min_index] = label_p[i]
            else:
                for i, l in enumerate(label_p):
                    index = l.item() * self.K
                    l_list = self.target_max_probs_list[index: index + self.K]
                    if max_probs[i] > min(l_list):
                        min_index = l_list.index(min(l_list))
                        self.target_max_probs_list[index + min_index] = max_probs[i]
                        self.prompt_learner.target_feat_bank[index + min_index] = F_u[i]
                        self.target_key_dict[index + min_index] = label_p[i]
            return

        if train:
            source_bank = torch.mean(
                self.prompt_learner.source_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            target_bank = torch.mean(
                self.prompt_learner.target_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            logits_c = self.prompt_learner.attn_block(text_features, image_features,
                                                      source_bank, target_bank)
            if source:
                loss_c = F.cross_entropy(logits_c, label)
                loss_x = F.cross_entropy(logits, label)
                return logits, loss_x + loss_c

            if epoch is None or epoch <= self.epoch_switch:
                logits_u = logit_scale * F_u @ text_features_u.t()
                pseudo_label = torch.softmax(logits_u, dim=-1)
            else:
                pseudo_label = torch.softmax(logits, dim=-1)
            max_probs, label_p = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.confi).float()
            if mask.sum() == 0 or self.warm_up > epoch:
                loss_c = torch.zeros((), device=image.device)
                loss_x = torch.zeros((), device=image.device)
            else:
                loss_c = (F.cross_entropy(logits_c, label_p, reduction='none') * mask).sum() / mask.sum()
                loss_x = (F.cross_entropy(logits, label_p, reduction='none') * mask).sum() / mask.sum()
            return loss_x + loss_c

        source_bank = torch.mean(
            self.prompt_learner.source_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
        target_bank = torch.mean(
            self.prompt_learner.target_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
        logits_c = self.prompt_learner.attn_block(text_features, image_features,
                                                  source_bank, target_bank)
        return logits + 0.5 * logits_c


# ── cosine schedule with warm-up (release OPTIM: cosine + 1 constant epoch) ──
class CosineWarmupScheduler:
    """Stepped once per epoch, mirroring BaseDA.run_epoch -> update_lr()."""

    def __init__(self, optimizer, max_epochs, warmup_epochs, base_lr, warmup_lr):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.epoch = 0
        for group in optimizer.param_groups:
            group['lr'] = warmup_lr if warmup_epochs > 0 else base_lr

    def step(self):
        self.epoch += 1
        if self.epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            progress = (self.epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))) * self.base_lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--warmup-lr', type=float, default=1e-5)
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--n-ctx', type=int, default=2)
    parser.add_argument('--num-tokens', type=int, default=2)
    parser.add_argument('--ctx-init', type=str, default='a_photo_of_a')
    parser.add_argument('--location', type=str, default='middle', choices=['middle', 'end'])
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'amp', 'fp32'])
    parser.add_argument('--confi', type=float, default=0.9,
                        help='Pseudo-label confidence threshold (Office-31 default 0.9)')
    parser.add_argument('--warm-up', type=int, default=1,
                        help='Epoch index before which the target branch is disabled (Office-31 default 1)')
    parser.add_argument('--epoch-switch', type=int, default=0,
                        help='Epoch index after which pseudo labels come from the adapted logits (Office-31 default 0)')
    parser.add_argument('--clip-cache-dir', type=str, default=None)
    parser.add_argument('--class-names-file', type=str, default=None)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists_vis'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.seed = seed_everything(args.seed)
    print(f'Controlled random seed: {args.seed}')

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and args.precision != 'fp32':
        print('CUDA unavailable -> forcing fp32 (fp16 CLIP needs the GPU kernels).')
        args.precision = 'fp32'
    print(f'Device: {device} | precision: {args.precision}')

    train_tf, test_tf = build_transforms()
    # 标签空间以目标域为准：只保留目标域出现的类别，并把标签压到 0..|T|-1。
    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    src_ds = UdaImageListDataset(os.path.join(args.data_dir, 'sunny_train.txt'),
                                 train_tf, label_map)
    tgt_u_ds = UdaImageListDataset(os.path.join(args.data_dir, f'{args.target}_train.txt'),
                                   train_tf, label_map)
    tgt_val_ds = UdaImageListDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'),
                                     test_tf, label_map)
    print(format_class_space(class_ids, num_classes))
    print(f'Source train (sunny):            {len(src_ds)} images')
    print(f'Target unlabeled ({args.target}): {len(tgt_u_ds)} images')
    print(f'Target val ({args.target}):       {len(tgt_val_ds)} images')
    print(format_dropped('source train', src_ds))

    # 类名文件始终为完整 14 类，按目标域类别子集取提示词（修复 14 类文件与
    # 9/6/8 类目标域的 expected_count 冲突）。
    classnames, prompt_source = load_classnames(args, class_ids)
    print(f'Class names ({prompt_source}): {classnames}')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    tgt_u_loader = DataLoader(tgt_u_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    print(f'Loading CLIP (backbone: {args.backbone})')
    clip_model = load_clip_to_cpu(args.backbone, args.clip_cache_dir)
    if args.precision in ('fp32', 'amp'):
        clip_model.float()
    model = CustomCLIP(classnames, clip_model, args.n_ctx, args.num_tokens, args.ctx_init,
                       num_layer=len(clip_model.transformer.resblocks),
                       hidden_size=clip_model.visual.conv1.out_channels,
                       location=args.location, share_layer=[0, 5], deep_layer=None)
    model.confi = args.confi
    model.warm_up = args.warm_up
    model.epoch_switch = args.epoch_switch

    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if 'prompt_learner' in name:
            param.requires_grad_(True)
        if 'bank' in name:
            param.requires_grad_(False)
    enabled = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    print(f'Trainable tensor count: {len(enabled)}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params:,}')
    model.to(device)

    optimizer = torch.optim.SGD([p for p in model.prompt_learner.parameters() if p.requires_grad],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, args.epochs, args.warmup_epochs,
                                      args.lr, args.warmup_lr)
    scaler = torch.cuda.amp.GradScaler() if args.precision == 'amp' else None
    print(f'Initial lr: {optimizer.param_groups[0]["lr"]:g}')

    eff = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'PDA',
                               seed=args.seed, device=str(device), modalities='VIS',
                               protocol='UDA', target_label_count=0, target_labels_per_class=0,
                               backbone=f'CLIP {args.backbone} + PDA prompts (n_ctx={args.n_ctx}, '
                                        f'num_tokens={args.num_tokens})',
                               pretraining=f'CLIP zero-shot ({args.backbone}, {args.precision})')
        print(f'Efficiency logging -> {eff.run_dir}')

    # ── feature banks (release ``construct_bank``) ─────────────────────────
    print('Constructing source feature bank...')
    model.eval()
    with torch.no_grad():
        for batch in src_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            model(images, label=labels, construct=True)
            if min(model.source_max_probs_list) > 0.99:
                break
    print('Constructing target feature bank...')
    with torch.no_grad():
        for batch in tgt_u_loader:
            images = batch[0].to(device)
            model(images, construct=True, source=False)
            if min(model.target_max_probs_list) > 0.99:
                break
    filled_s = sum(1 for v in model.source_max_probs_list if v > 0.0)
    filled_t = sum(1 for v in model.target_max_probs_list if v > 0.0)
    print(f'Feature banks completed. Filled slots: source {filled_s}/{model.n_cls * model.K}, '
          f'target {filled_t}/{model.n_cls * model.K} (unfilled slots stay zero, as released)')

    n_batches = len(src_loader)  # COUNT_ITER = "train_x"
    final_metrics = None
    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        src_iter, tgt_iter = iter(src_loader), iter(tgt_u_loader)
        for i in range(n_batches):
            if eff:
                eff.step_start()
            try:
                image_x, label = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                image_x, label = next(src_iter)
            try:
                image_u, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_u_loader)
                image_u, _ = next(tgt_iter)
            image_x, label = image_x.to(device), label.to(device)
            image_u = image_u.to(device)
            if eff:
                eff.dataloader_done()

            optimizer.zero_grad()
            if args.precision == 'amp':
                with torch.cuda.amp.autocast():
                    output_x, loss_x = model(image_x, label, epoch=epoch, train=True)
                    loss_u = model(image_u, epoch=epoch, train=True, source=False)
                    loss = loss_x + loss_u
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output_x, loss_x = model(image_x, label, epoch=epoch, train=True)
                loss_u = model(image_u, epoch=epoch, train=True, source=False)
                loss = loss_x + loss_u
                loss.backward()
                optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                acc_x = (output_x.argmax(1) == label).float().mean().item()
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'loss={loss.item():.4f} loss_x={loss_x.item():.4f} '
                      f'loss_u={loss_u.item():.4f} acc_x={acc_x:.4f} '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e}')
        scheduler.step()

        model.eval()
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for images, labels in tgt_val_loader:
                logits = model(images.to(device))
                all_labels.extend(labels.tolist())
                all_predictions.extend(logits.argmax(1).cpu().tolist())
        final_metrics = classification_metrics(all_labels, all_predictions, num_classes=num_classes)
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {final_metrics["accuracy"]:.4f}')
        print(f'  {format_classification_metrics(final_metrics)}')
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.prompt_learner.state_dict(),
               os.path.join(args.save_dir, f'pda_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={
            'best_acc': final_metrics['accuracy'] if final_metrics else None,
            'final_acc': final_metrics['accuracy'] if final_metrics else None,
            'target': args.target, 'protocol': 'UDA',
            'prompt_source': prompt_source,
            'class_names_used': prompt_source == 'file',
            'trainable_params': trainable_params,
            'precision': args.precision,
        })
    if final_metrics:
        print(f'\nPDA finished. Target val accuracy ({args.target}): '
              f'{final_metrics["accuracy"]:.4f} (prompt source: {prompt_source})')


if __name__ == '__main__':
    main()
