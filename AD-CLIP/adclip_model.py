"""
Standalone AD-CLIP model components (no dassl dependency).
Adapted from AD-CLIP trainers/adclip_rn50.py
"""
import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

# Use the AD-CLIP clip package loaded by the training script
clip = sys.modules.get('adclip_clip_clip')
if clip is None:
    # Fallback: import AD-CLIP's clip directly
    ADCLIP_DIR = os.path.dirname(__file__)
    sys.path.insert(0, ADCLIP_DIR)
    from clip import clip

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
device_cuda = "cuda"


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.mean(x, dim=(2, 3))

    def sigma(self, x):
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        squared_diff = (x - mean) ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=(2, 3))
        epsilon = 1e-8
        std_dev = torch.sqrt((sum_squared_diff + epsilon) / (x.shape[2] * x.shape[3]))
        return std_dev


class domain_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = []
        self.linear1.append(nn.Linear(256, 256).to(device_cuda))
        self.linear1.append(nn.Linear(512, 256).to(device_cuda))
        self.linear1.append(nn.Linear(1024, 256).to(device_cuda))
        self.linear1.append(nn.Linear(2048, 256).to(device_cuda))
        self.adain = AdaIN()
        self.gap = []
        self.gap.append(nn.AdaptiveAvgPool2d((1, 256)))
        self.gap.append(nn.AdaptiveAvgPool2d((1, 512)))
        self.gap.append(nn.AdaptiveAvgPool2d((1, 1024)))
        self.gap.append(nn.AdaptiveAvgPool2d((1, 2048)))
        self.linear2 = nn.ModuleList(nn.Linear(256, 512) for _ in range(4))

    def forward(self, data):
        data_prompt = []
        for i in range(len(data)):
            x_mu = self.adain.mu(data[i]).unsqueeze(1).to(torch.float32)
            x_sigma = self.adain.sigma(data[i]).unsqueeze(1).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma), 1)
            x_cat = self.gap[i](x_cat).squeeze(1)
            x_out = self.linear1[i](x_cat)
            x_final = self.linear2[i](x_out)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output


class image_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = []
        self.linear.append(nn.Linear(256, 512).to(device_cuda))
        self.linear.append(nn.Linear(512, 512).to(device_cuda))
        self.linear.append(nn.Linear(1024, 512).to(device_cuda))
        self.linear.append(nn.Linear(2048, 512).to(device_cuda))
        self.adain = AdaIN()
        self.lin = nn.Linear(4, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, data, n_imgctx):
        data_prompt = []
        for i in range(len(data)):
            x_gap = self.gap(data[i]).squeeze(3).squeeze(2)
            x_lin = self.linear[i](x_gap)
            data_prompt.append(x_lin)
        feat = torch.stack(data_prompt, dim=1)
        output = []
        for i in range(n_imgctx):
            x = self.lin(feat.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            output.append(x)
        feat_tokens = torch.stack(output, dim=1).squeeze(2)
        return feat_tokens


class style_mapping_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = []
        self.linear1.append(nn.Linear(256, 384).to(device_cuda))
        self.linear1.append(nn.Linear(512, 384).to(device_cuda))
        self.linear1.append(nn.Linear(1024, 384).to(device_cuda))
        self.linear1.append(nn.Linear(2048, 384).to(device_cuda))
        self.adain = AdaIN()
        self.relu = nn.ReLU()
        self.gap = []
        self.gap.append(nn.AdaptiveAvgPool1d((256)))
        self.gap.append(nn.AdaptiveAvgPool1d((512)))
        self.gap.append(nn.AdaptiveAvgPool1d((1024)))
        self.gap.append(nn.AdaptiveAvgPool1d((2048)))
        self.linear2 = nn.ModuleList(nn.Linear(384, 512) for _ in range(4))

    def forward(self, data):
        data_prompt = []
        for i in range(len(data)):
            x_mu = self.adain.mu(data[i]).to(torch.float32)
            x_sigma = self.adain.sigma(data[i]).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma), 1)
            x_gap = self.gap[i](x_cat)
            x_out = self.linear1[i](x_gap)
            x_relu = self.relu(x_out)
            x_final = self.linear2[i](x_relu)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_imgctx = 4
        n_ctx = 8 + n_imgctx

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        self.domain_tokens = domain_projector()
        self.image_tokens = image_projector()
        self.style_mapping_tokens = style_mapping_projector()

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_imgctx = n_imgctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    @autocast()
    def forward(self, source_data, target_data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_imgctx = self.n_imgctx

        source_domaintokens = self.domain_tokens(source_data)
        source_imagetokens = self.image_tokens(source_data, n_imgctx)
        source_style_mappingtokens = self.style_mapping_tokens(source_data)

        target_domaintokens = self.domain_tokens(target_data)
        target_imagetokens = self.image_tokens(target_data, n_imgctx)

        source_tokens = torch.cat((source_domaintokens, target_domaintokens, source_imagetokens), dim=1)
        target_tokens = torch.cat((source_domaintokens, target_domaintokens, target_imagetokens), dim=1)

        source_prompts = []
        for tokens_i in source_tokens:
            ctx_i = tokens_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            source_prompts.append(pts_i)
        source_prompts = torch.stack(source_prompts)

        target_prompts = []
        for tokens_i in target_tokens:
            ctx_i = tokens_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            target_prompts.append(pts_i)
        target_prompts = torch.stack(target_prompts)

        return source_prompts, target_prompts, source_domaintokens, source_style_mappingtokens


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, s_image, t_image):
        source_image_features, source_data = self.image_encoder(s_image.type(self.dtype))
        target_image_features, target_data = self.image_encoder(t_image.type(self.dtype))

        source_prompts, target_prompts, source_domaintokens, source_style_mappingtokens = \
            self.prompt_learner(source_data, target_data)
        tokenized_prompts = self.tokenized_prompts

        source_image_features = source_image_features / source_image_features.norm(dim=-1, keepdim=True)
        target_image_features = target_image_features / target_image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        source_text_features = []
        for pts_i in source_prompts:
            tf = self.text_encoder(pts_i, tokenized_prompts)
            source_text_features.append(tf)
        source_text_features = torch.stack(source_text_features)
        source_text_features = source_text_features / source_text_features.norm(dim=-1, keepdim=True)

        target_text_features = []
        for pts_i in target_prompts:
            tf = self.text_encoder(pts_i, tokenized_prompts)
            target_text_features.append(tf)
        target_text_features = torch.stack(target_text_features)
        target_text_features = target_text_features / target_text_features.norm(dim=-1, keepdim=True)

        source_logits = []
        for txt, im in zip(source_text_features, source_image_features):
            l_i = logit_scale * im @ txt.t()
            source_logits.append(l_i)
        source_logits = torch.stack(source_logits)

        target_logits = []
        for txt, im in zip(target_text_features, target_image_features):
            l_i = logit_scale * im @ txt.t()
            target_logits.append(l_i)
        target_logits = torch.stack(target_logits)

        target_probs = torch.nn.functional.softmax(target_logits, dim=1)

        return source_logits, target_probs, source_domaintokens, source_style_mappingtokens, \
            source_text_features, target_text_features


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, target_prob):
        full_enp = torch.zeros(target_prob.shape[0])
        target_prob = nn.functional.normalize(target_prob, dim=0)
        for i in range(len(target_prob)):
            total_en = 0
            for j in range(target_prob.shape[1]):
                total_en = total_en - target_prob[i][j] * torch.log(target_prob[i][j] + 1e-8)
            full_enp[i] = total_en
        avg_full_enp = torch.mean(full_enp)
        return avg_full_enp
