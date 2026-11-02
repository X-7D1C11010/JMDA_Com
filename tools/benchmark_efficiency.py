#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified efficiency benchmark for all DA methods in JMDA-Net comparison.

Layer 1: static + micro-benchmark (params, FLOPs, latency, memory, train step).
Uses thop for FLOPs (1 MAC = 2 FLOPs). All methods use the same device,
precision, warmup/iters, and input sizes.

Usage:
    python tools/benchmark_efficiency.py --methods dann,deepcoral,cdan
    python tools/benchmark_efficiency.py --methods all --benchmark_repeats 3
    python tools/benchmark_efficiency.py --methods jmda_baseline,jmda_full

Method adapters map each method name to its model constructor and input factory.
"""
import os, sys, json, time, argparse, io, csv, platform
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is importable
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, 'tools'))

from efficiency_common import (
    count_parameters, get_state_dict_size_mb, measure_flops,
    measure_inference_latency, measure_inference_memory, measure_train_step,
    save_environment_json, save_metrics_json, append_efficiency_csv,
    generate_summary_files, get_run_id, _cuda_sync, _reset_memory,
    run_profiler_summary,
)

# Classifier width used by the UDA adapters. UDA training now uses a
# target-driven label space (tools/uda_data.py), so +/- one linear layer vs the
# dataset-wide 14 classes; set from --num_classes to match the target domain.
NUM_CLASSES = 14
IMAGE_BACKBONE = 'method_default'


# ═══════════════════════════════════════════════════════════════════════════
#  ADAPTER REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
# Each adapter is a dict with keys:
#   name, group, modalities, build(inputs), make_input(bs,device), forward_loss
# group ∈ {'trimodal','visual_vlm','jmda'}
# ─────────────────────────────────────────────────────────────────────────────

def _make_vis_input(bs, device):
    return torch.randn(bs, 3, 224, 224, device=device)

def _make_ir_input(bs, device):
    return torch.randn(bs, 3, 224, 224, device=device)

def _make_ais_input(bs, device, dim=23040):
    return torch.randn(bs, dim, device=device)

def _make_mcft_patch(bs, device, channels=3):
    """MCFT's architecture contract: tokens are a patch_size x patch_size grid
    (15x15), which is what the released model reshapes its position embedding to."""
    return torch.randn(bs, channels, 15, 15, device=device)

def _load_module_from_file(module_name, path):
    """Load a .py file as a module; needed for d-SNE/ whose directory name is not
    a valid Python identifier."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# ── DANN ────────────────────────────────────────────────────────────────────
def adapter_dann():
    if IMAGE_BACKBONE == 'resnet50':
        from DANN.train_dann import ResNetDANN
        constructor = lambda: ResNetDANN(num_classes=NUM_CLASSES)
        input_size = 224
    else:
        model_dir = os.path.join(PROJ, 'DANN', 'models')
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from model import CNNModel
        constructor = lambda: CNNModel(num_classes=NUM_CLASSES)
        input_size = 28
    def build():
        return constructor()
    def make_input(bs, device):
        return torch.randn(bs, 3, input_size, input_size, device=device)
    def forward_loss(model, inp):
        x = inp
        cls_out, dom_out = model(x, alpha=0.5)
        labels = torch.randint(0, NUM_CLASSES, (x.size(0),), device=x.device)
        dom_labels = torch.randint(0, 2, (x.size(0),), device=x.device)
        return nn.CrossEntropyLoss()(cls_out, labels) + nn.CrossEntropyLoss()(dom_out, dom_labels)
    def forward_infer(model, inp):
        return model(inp, alpha=0.0)[0]
    return {'name':'DANN','group':'visual_vlm','modalities':'VIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':f'(1,3,{input_size},{input_size})'}

# ── DeepCORAL ───────────────────────────────────────────────────────────────
def adapter_deepcoral():
    if IMAGE_BACKBONE == 'resnet50':
        from DeepCORAL.train_deepcoral import ResNetDeepCORAL
        constructor = lambda: ResNetDeepCORAL(num_classes=NUM_CLASSES)
        input_size = 224
    else:
        from DeepCORAL.models import DeepCORAL
        constructor = lambda: DeepCORAL(num_classes=NUM_CLASSES)
        input_size = 227
    from DeepCORAL.models import CORAL
    def build():
        return constructor()
    def make_input(bs, device):
        return torch.randn(bs, 3, input_size, input_size, device=device)
    def forward_loss(model, inp):
        src = inp
        tgt = torch.randn_like(src)
        s_out, t_out = model(src, tgt)
        labels = torch.randint(0, NUM_CLASSES, (src.size(0),), device=src.device)
        loss = nn.CrossEntropyLoss()(s_out, labels) + CORAL(s_out, t_out)
        return loss
    def forward_infer(model, inp):
        return model.fc(model.sharedNet(inp))
    return {'name':'DeepCORAL','group':'visual_vlm','modalities':'VIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':f'(1,3,{input_size},{input_size})'}

# ── CDAN ────────────────────────────────────────────────────────────────────
def adapter_cdan():
    # train_cdan.py imports sibling helpers (optimizer_utils) by bare name, so
    # the CDAN directory must be importable when loaded as a package.
    sys.path.insert(0, os.path.join(PROJ, 'CDAN'))
    from CDAN.train_cdan import ResNetFc, AdversarialNetwork, RandomLayer
    class CDANWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            backbone_name = 'ResNet50' if IMAGE_BACKBONE == 'resnet50' else 'ResNet18'
            self.backbone = ResNetFc(backbone_name, True, 256, True, NUM_CLASSES)
            self.random_layer = RandomLayer([256, NUM_CLASSES], 1024)
            self.ad_net = AdversarialNetwork(1024, 1024)
        def forward(self, x):
            feat, out = self.backbone(x)
            return out  # for FLOPs: only return logits
    def build():
        return CDANWrapper()
    def make_input(bs, device):
        return _make_vis_input(bs, device)
    def forward_loss(model, inp):
        feat, out = model.backbone(inp)
        labels = torch.randint(0, NUM_CLASSES, (inp.size(0),), device=inp.device)
        loss_cls = nn.CrossEntropyLoss()(out, labels)
        # AdversarialNetwork outputs 1 value (sigmoid); use BCEWithLogitsLoss
        dom_labels = torch.randint(0, 2, (inp.size(0),), device=inp.device).float().unsqueeze(1)
        if torch.cuda.is_available():
            model.random_layer.cuda()
        feat_combined = model.random_layer.forward([feat, out])
        dom_out = model.ad_net(feat_combined)
        loss_dom = nn.BCEWithLogitsLoss()(dom_out, dom_labels)
        return loss_cls + loss_dom
    return {'name':'CDAN','group':'visual_vlm','modalities':'VIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'vis_shape':'(1,3,224,224)'}

# ── MDD (tllib) ──────────────────────────────────────────────────────────────
def adapter_mdd():
    sys.path.insert(0, os.path.join(PROJ, 'Transfer-Learning-Library'))
    from tllib.alignment.mdd import ImageClassifier
    from tllib.vision.models import resnet18, resnet50
    def build():
        backbone = resnet50() if IMAGE_BACKBONE == 'resnet50' else resnet18()
        return ImageClassifier(backbone, NUM_CLASSES, 256, 256)
    def make_input(bs, device):
        return _make_vis_input(bs, device)
    def forward_loss(model, inp):
        out, out_adv = model(inp)
        labels = torch.randint(0, NUM_CLASSES, (inp.size(0),), device=inp.device)
        return nn.CrossEntropyLoss()(out, labels)
    return {'name':'MDD','group':'visual_vlm','modalities':'VIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'vis_shape':'(1,3,224,224)'}

# ── MCC (tllib) ──────────────────────────────────────────────────────────────
def adapter_mcc():
    sys.path.insert(0, os.path.join(PROJ, 'Transfer-Learning-Library'))
    from tllib.self_training.mcc import ImageClassifier, MinimumClassConfusionLoss
    def build():
        from tllib.vision.models import resnet18, resnet50
        backbone = resnet50() if IMAGE_BACKBONE == 'resnet50' else resnet18()
        return ImageClassifier(backbone, NUM_CLASSES, 256)
    def make_input(bs, device):
        return _make_vis_input(bs, device)
    def forward_loss(model, inp):
        # In train mode, ImageClassifier returns (predictions, features)
        preds, feats = model(inp)
        labels = torch.randint(0, NUM_CLASSES, (inp.size(0),), device=inp.device)
        loss = nn.CrossEntropyLoss()(preds, labels)
        loss = loss + MinimumClassConfusionLoss(temperature=2.5)(preds)
        return loss
    return {'name':'MCC','group':'visual_vlm','modalities':'VIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'vis_shape':'(1,3,224,224)'}

# ── CCSA (SDA) ───────────────────────────────────────────────────────────────
def adapter_ccsa():
    from CCSA.train_ccsa import CCSA, euclidean_distance, contrastive_loss
    def build():
        return CCSA(num_classes=NUM_CLASSES)
    def make_input(bs, device):
        return {
            'rgb': _make_vis_input(bs, device),
            'nir': torch.randn(bs, 1, 224, 224, device=device),
        }
    def forward_loss(model, inp):
        count, device = inp['rgb'].size(0), inp['rgb'].device
        tgt = make_input(count, device)
        logits_s, logits_t, f_s, f_t = model(inp, tgt)
        labels_s = torch.randint(0, NUM_CLASSES, (count,), device=device)
        labels_t = torch.randint(0, NUM_CLASSES, (count,), device=device)
        yc = labels_s.eq(labels_t).float()
        distance = euclidean_distance(f_s, f_t)
        # One released source-side update; the training entry point performs
        # the corresponding target-side update immediately afterwards.
        ce = nn.CrossEntropyLoss()(logits_s, labels_s)
        return 0.75 * ce + 0.25 * contrastive_loss(yc, distance)
    def forward_infer(model, inp):
        return model(inp)
    return {'name':'CCSA','group':'multimodal','modalities':'VIS+IR',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':'(1,3,224,224)','ir_shape':'(1,1,224,224)'}

# ── d-SNE (SDA) ──────────────────────────────────────────────────────────────
def adapter_dsne():
    mod = _load_module_from_file('dsne_train_module',
                                 os.path.join(PROJ, 'd-SNE', 'train_dsne.py'))
    def build():
        return mod.DSNE(num_classes=NUM_CLASSES)
    def make_input(bs, device):
        return {
            'rgb': _make_vis_input(bs, device),
            'nir': torch.randn(bs, 1, 224, 224, device=device),
        }
    def forward_loss(model, inp):
        count, device = inp['rgb'].size(0), inp['rgb'].device
        tgt = make_input(count, device)
        labels_s = torch.randint(0, NUM_CLASSES, (count,), device=device)
        labels_t = torch.randint(0, NUM_CLASSES, (count,), device=device)
        logits_s, f_s = model(inp, return_embedding=True)
        logits_t, f_t = model(tgt, return_embedding=True)
        # Released default: target CE plus target->source d-SNE.  The optional
        # source update is enabled only by train_dsne.py --train-src.
        xent = nn.CrossEntropyLoss()(logits_t, labels_t)
        aux = mod.dsne_loss(f_t, labels_t, f_s, labels_s, fn=True)
        return 0.9 * xent + 0.1 * aux
    def forward_infer(model, inp):
        return model(inp)
    return {'name':'d-SNE','group':'multimodal','modalities':'VIS+IR',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':'(1,3,224,224)','ir_shape':'(1,1,224,224)'}

# ── MCFT (VIS+IR supervised) ─────────────────────────────────────────────────
def adapter_mcft():
    from MCFT.train_mcft import build_model
    def build():
        return build_model(NUM_CLASSES)
    def make_input(bs, device):
        return {
            'vis': _make_mcft_patch(bs, device, channels=3),
            'ir': _make_mcft_patch(bs, device, channels=1),
        }
    def forward_loss(model, inp):
        labels = torch.randint(0, NUM_CLASSES, (inp['vis'].size(0),), device=inp['vis'].device)
        logits, loss_ita, loss_itm = model(inp['vis'], inp['ir'], labels)
        return nn.CrossEntropyLoss()(logits, labels) + 0.1 * loss_ita + loss_itm
    def forward_infer(model, inp):
        # forward() needs a label for the ITM hard-negative mining; inference
        # uses zeros, which the ITM path treats as "no negatives available".
        labels = torch.zeros(inp['vis'].size(0), dtype=torch.long, device=inp['vis'].device)
        logits, _, _ = model(inp['vis'], inp['ir'], labels)
        return logits
    return {'name':'MCFT','group':'multimodal','modalities':'VIS+IR',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':'(1,3,15,15)','ir_shape':'(1,1,15,15)'}


# ── VDS2Raw paper configurations ────────────────────────────────────────────
def adapter_vds_ccsa():
    from vds2raw.models import CCSAOfficeModel, ccsa_contrastive_loss, euclidean_distance
    def build():
        return CCSAOfficeModel(
            num_classes=3, pretrained=False, multimodal=True, use_ais=True
        )
    def make_input(bs, device):
        return {
            'rgb': torch.randn(bs, 3, 224, 224, device=device),
            'nir': torch.randn(bs, 1, 224, 224, device=device),
            'ais': torch.randn(bs, 13, device=device),
            'ais_mask': torch.ones(bs, 13, device=device),
        }
    def forward_loss(model, inp):
        other = make_input(inp['rgb'].size(0), inp['rgb'].device)
        logits, _, first, second = model(inp, other)
        labels = torch.randint(0, 3, (inp['rgb'].size(0),), device=inp['rgb'].device)
        pair = torch.randint(0, 2, (inp['rgb'].size(0),), device=inp['rgb'].device).float()
        return 0.75 * nn.CrossEntropyLoss()(logits, labels) + 0.25 * ccsa_contrastive_loss(
            pair, euclidean_distance(first, second)
        )
    def forward_infer(model, inp):
        return model(inp)
    return {'name':'VDS2Raw-CCSA','group':'multimodal','modalities':'RGB+NIR+AIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':'(1,3,224,224)','ir_shape':'(1,1,224,224)',
            'ais_shape':'(1,13)'}


def adapter_vds_dsne():
    from vds2raw.models import DSNEOfficeModel, dsne_loss
    def build():
        return DSNEOfficeModel(
            num_classes=3, pretrained=False, multimodal=True, use_ais=True
        )
    def make_input(bs, device):
        return {
            'rgb': torch.randn(bs, 3, 224, 224, device=device),
            'nir': torch.randn(bs, 1, 224, 224, device=device),
            'ais': torch.randn(bs, 13, device=device),
            'ais_mask': torch.ones(bs, 13, device=device),
        }
    def forward_loss(model, inp):
        other = make_input(inp['rgb'].size(0), inp['rgb'].device)
        labels = torch.randint(0, 3, (inp['rgb'].size(0),), device=inp['rgb'].device)
        other_labels = torch.randint(0, 3, (inp['rgb'].size(0),), device=inp['rgb'].device)
        logits, features = model(inp, return_embedding=True)
        _, other_features = model(other, return_embedding=True)
        return 0.9 * nn.CrossEntropyLoss()(logits, labels) + 0.1 * dsne_loss(
            features, labels, other_features, other_labels,
            feature_normalization=True,
        )
    def forward_infer(model, inp):
        return model(inp)
    return {'name':'VDS2Raw-d-SNE','group':'multimodal','modalities':'RGB+NIR+AIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':'(1,3,224,224)','ir_shape':'(1,1,224,224)',
            'ais_shape':'(1,13)'}


def adapter_vds_mcft3():
    from vds2raw.models import MCFTWithAIS
    def build():
        return MCFTWithAIS(num_classes=3)
    def make_input(bs, device):
        return {
            'rgb': torch.randn(bs, 3, 15, 15, device=device),
            'nir': torch.randn(bs, 1, 15, 15, device=device),
            'ais': torch.randn(bs, 13, device=device),
            'ais_mask': torch.ones(bs, 13, device=device),
        }
    def forward_loss(model, inp):
        labels = torch.randint(0, 3, (inp['rgb'].size(0),), device=inp['rgb'].device)
        logits, ita, itm = model(
            inp['rgb'], inp['nir'], inp['ais'], inp['ais_mask'], labels
        )
        return nn.CrossEntropyLoss()(logits, labels) + 0.1 * ita + itm
    def forward_infer(model, inp):
        return model(
            inp['rgb'], inp['nir'], inp['ais'], inp['ais_mask'],
            labels=None, compute_auxiliary=False,
        )[0]
    return {'name':'VDS2Raw-MCFT3','group':'multimodal','modalities':'RGB+NIR+AIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':'(1,3,15,15)','ir_shape':'(1,1,15,15)'}

# ── PDA (UDA, CLIP) ──────────────────────────────────────────────────────────
def adapter_pda():
    import copy
    from PDA.train_pda import CustomCLIP, load_clip_to_cpu
    classnames = [f'class_{i + 1}' for i in range(NUM_CLASSES)]  # same placeholder prompts as training
    clip_model = load_clip_to_cpu('ViT-B/16')
    def build():
        # CustomCLIP holds the CLIP submodules by reference, so the caller's
        # .to(device) would mutate the shared template. Each repeat gets its own
        # copy, keeping construction (CPU token ids) and the dumped model
        # consistent.
        clip_local = copy.deepcopy(clip_model)
        model = CustomCLIP(classnames, clip_local, n_ctx=2, num_tokens=2,
                           ctx_init='a_photo_of_a',
                           num_layer=len(clip_local.transformer.resblocks),
                           hidden_size=clip_local.visual.conv1.out_channels,
                           location='middle', share_layer=[0, 5], deep_layer=None)
        for name, param in model.named_parameters():
            param.requires_grad_(False)
            if 'prompt_learner' in name:
                param.requires_grad_(True)
            if 'bank' in name:
                param.requires_grad_(False)
        return model
    def make_input(bs, device):
        return _make_vis_input(bs, device)
    def forward_loss(model, inp):
        labels = torch.randint(0, NUM_CLASSES, (inp.size(0),), device=inp.device)
        _, loss = model(inp, label=labels, epoch=0, train=True)
        return loss
    return {'name':'PDA','group':'visual_vlm','modalities':'VIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'vis_shape':'(1,3,224,224)(fp16)'}

# ── Source-only ──────────────────────────────────────────────────────────────
def adapter_source_only():
    from torchvision import models as tvm
    def build():
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 14)
        return m
    def make_input(bs, device):
        return _make_vis_input(bs, device)
    def forward_loss(model, inp):
        out = model(inp)
        labels = torch.randint(0, 14, (inp.size(0),), device=inp.device)
        return nn.CrossEntropyLoss()(out, labels)
    return {'name':'Source-only','group':'visual_vlm','modalities':'VIS',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'vis_shape':'(1,3,224,224)'}

# ── JMDA-Net (various ablation modes) ──────────────────────────────────────────
JMDA_NET_DIR = os.environ.get('JMDA_NET_DIR', os.path.join(PROJ, 'JMDA-Net'))
def adapter_jmda(mode='full'):
    """mode: 'baseline' (no tensor, no OT), 'tensor_only' (no OT), 'ot_only' (no tensor), 'full'"""
    sys.path.insert(0, JMDA_NET_DIR)
    from Models import VisualFeatureExtractor, IRFeatureExtractor, Classifier
    from Tensor import TensorBasedAlignmentStable
    from Generator import NeuralOptimalTransportGenerator
    import torch.nn.functional as F_func
    VIS_DIM = 512; IR_DIM = 512; PROJ_DIM = 256; FUSED_DIM = 512

    class JMDANetBundle(nn.Module):
        def __init__(self):
            super().__init__()
            self.net_vis = VisualFeatureExtractor(output_dim=VIS_DIM)
            self.net_ir = IRFeatureExtractor(output_dim=IR_DIM)
            self.use_tensor = mode in ('tensor_only', 'full')
            self.use_ot = mode in ('ot_only', 'full')
            # Feature dim after tensor (if used) is PROJ_DIM*2=512, else 1024
            self.feat_dim = FUSED_DIM if self.use_tensor else VIS_DIM + IR_DIM
            if self.use_tensor:
                self.tal = TensorBasedAlignmentStable(
                    input_dims=[VIS_DIM, IR_DIM], output_dims=[PROJ_DIM, PROJ_DIM],
                    num_modalities=2, max_svd_sweeps=3, svd_tolerance=1e-4)
            if self.use_ot:
                self.generator = NeuralOptimalTransportGenerator(
                    feature_dim=self.feat_dim, transport_mode='sinkhorn',
                    epsilon=0.1, sinkhorn_iterations=10, correction_scale=1.0)
            self.classifier = Classifier(input_dim=self.feat_dim, num_classes=14)

    def build():
        return JMDANetBundle()

    def make_input(bs, device):
        return {
            'vis': _make_vis_input(bs, device),
            'ir': _make_ir_input(bs, device),
        }

    def forward_loss(model, inp):
        f_vis = model.net_vis(inp['vis'])
        f_ir = model.net_ir(inp['ir'])
        if model.use_tensor:
            (p_vis, p_ir), _, _ = model.tal([f_vis, f_ir], [f_vis, f_ir])
            if isinstance(p_vis, list): p_vis = p_vis[0]
            if isinstance(p_ir, list): p_ir = p_ir[0]
            feat = torch.cat([p_vis, p_ir], dim=1)
        else:
            feat = torch.cat([f_vis, f_ir], dim=1)
        if model.use_ot:
            feat_src = feat
            feat_tgt = feat.clone()
            feat_mid, _ = model.generator(feat_src, feat_tgt, return_details=True)
            logits = model.classifier(feat_mid)
        else:
            logits = model.classifier(feat)
        labels = torch.randint(0, 14, (inp['vis'].size(0),), device=inp['vis'].device)
        return nn.CrossEntropyLoss()(logits, labels)

    def forward_infer(model, inp):
        f_vis = model.net_vis(inp['vis'])
        f_ir = model.net_ir(inp['ir'])
        if model.use_tensor:
            # Formal JMDA inference uses the fixed target-side projections and
            # classifier only. Sinkhorn/TransNet is a training-time module.
            p_vis = f_vis.matmul(model.tal.V_matrices[0])
            p_ir = f_ir.matmul(model.tal.V_matrices[1])
            feat = torch.cat([p_vis, p_ir], dim=1)
        else:
            feat = torch.cat([f_vis, f_ir], dim=1)
        return model.classifier(feat)

    name_map = {'baseline':'JMDA-Net-Baseline','tensor_only':'JMDA-Net-TensorOnly',
                'ot_only':'JMDA-Net-OTOnly','full':'JMDA-Net-Full'}
    return {'name':name_map.get(mode, f'JMDA-Net-{mode}'),
            'group':'jmda','modalities':'VIS+IR',
            'build':build,'make_input':make_input,'forward_loss':forward_loss,
            'forward_infer':forward_infer,
            'vis_shape':'(1,3,224,224)','ir_shape':'(1,3,224,224)',
            'jmda_mode':mode}


# ── ADAPTER MAP ────────────────────────────────────────────────────────────────
# Backbone / pretraining metadata recorded in every efficiency row.
BACKBONE_INFO = {
    'source_only':    ('ResNet-18', 'ImageNet'),
    'dann':           ('Official CNN (28x28)', 'None'),
    'deepcoral':      ('AlexNet', 'ImageNet'),
    'cdan':           ('ResNet-18', 'ImageNet'),
    'mdd':            ('ResNet-18', 'ImageNet'),
    'mcc':            ('ResNet-18', 'ImageNet'),
    'ccsa':           ('VGG-16 conv + FC1024/128', 'ImageNet -> source_train'),
    'dsne':           ('ResNetV2-101 + released 4096/512 head', 'ImageNet -> source_train'),
    'mcft':           ('MCFT MViT (patch 15x15, dim=64)', 'None (from scratch)'),
    'vds_ccsa':       ('VGG-16 conv + FC1024/128', 'ImageNet -> source_train'),
    'vds_dsne':       ('ResNetV2-101 + released 4096/512 head', 'ImageNet -> source_train'),
    'vds_mcft3':      ('MCFT MViT + AIS/mask late fusion', 'None (from scratch)'),
    'pda':            ('CLIP ViT-B/16 + PDA prompts', 'CLIP zero-shot'),
    'jmda_baseline':  ('ResNet-18 x2 (VIS/IR)', 'ImageNet'),
    'jmda_tensor_only': ('ResNet-18 x2 (VIS/IR)', 'ImageNet'),
    'jmda_ot_only':   ('ResNet-18 x2 (VIS/IR)', 'ImageNet'),
    'jmda_full':      ('ResNet-18 x2 (VIS/IR)', 'ImageNet'),
}

ADAPTERS = {
    'source_only': adapter_source_only,
    'dann': adapter_dann,
    'deepcoral': adapter_deepcoral,
    'cdan': adapter_cdan,
    'mdd': adapter_mdd,
    'mcc': adapter_mcc,
    'ccsa': adapter_ccsa,
    'dsne': adapter_dsne,
    'mcft': adapter_mcft,
    'vds_ccsa': adapter_vds_ccsa,
    'vds_dsne': adapter_vds_dsne,
    'vds_mcft3': adapter_vds_mcft3,
    'pda': adapter_pda,
    'jmda_baseline': lambda: adapter_jmda('baseline'),
    'jmda_tensor_only': lambda: adapter_jmda('tensor_only'),
    'jmda_ot_only': lambda: adapter_jmda('ot_only'),
    'jmda_full': lambda: adapter_jmda('full'),
}

# Source-only remains available for an explicit historical benchmark, but it
# is temporarily excluded from the default formal run together with the
# Joint/Target-only training baselines.
ALL_METHODS = [key for key in ADAPTERS if key != 'source_only']


# ═══════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def run_benchmark(adapter, args, device, output_dir):
    """Run one full benchmark for one adapter. Returns metrics dict."""
    name = adapter['name']
    group = adapter['group']
    modalities = adapter['modalities']
    forward_infer = adapter.get('forward_infer', None)
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {name}  ({modalities})")
    print(f"{'='*60}")

    # Build model for params + FLOPs (separate instance to avoid hook contamination)
    model_flops = adapter['build']().to(device)
    model_flops.eval()

    # ── Static: parameters ──
    total_params, trainable_params, buffer_bytes = count_parameters(model_flops)
    buffer_mb = buffer_bytes / (1024 * 1024)
    model_size_mb = get_state_dict_size_mb(model_flops)
    print(f"  total_params={total_params:,}  trainable={trainable_params:,}  "
          f"buffer={buffer_mb:.2f}MB  state_dict={model_size_mb:.2f}MB")

    # ── FLOPs (batch=1) on dedicated model ──
    inp_flops = adapter['make_input'](1, device)
    flops_info = measure_flops(model_flops, inp_flops, device, forward_fn=forward_infer)
    macs = flops_info['MACs']
    gflops = flops_info['inference_GFLOPs']
    unsupported = flops_info['unsupported_ops']
    print(f"  MACs={macs:,}  GFLOPs={gflops}  unsupported_ops={unsupported}")
    del model_flops
    _reset_memory()

    # ── Fresh model for inference / train benchmarks ──
    model = adapter['build']().to(device)
    model.eval()

    # ── Inference latency (batch=1) ──
    inp_lat = adapter['make_input'](1, device)
    lat_info = measure_inference_latency(model, inp_lat, device,
                                          warmup=args.benchmark_warmup,
                                          iters=args.benchmark_iters,
                                          forward_fn=forward_infer)
    print(f"  latency: mean={lat_info['latency_mean_ms']}ms  "
          f"p50={lat_info['latency_p50_ms']}ms  p95={lat_info['latency_p95_ms']}ms  "
          f"throughput={lat_info['throughput_samples_s']} samples/s")

    # ── Inference memory (batch=1) ──
    mem_info = measure_inference_memory(model, inp_lat, device, batch_size=1,
                                        forward_fn=forward_infer)
    print(f"  infer mem: alloc={mem_info['peak_infer_allocated_gb']}GB  "
          f"reserved={mem_info['peak_infer_reserved_gb']}GB")

    # ── Train step timing ──
    train_bs = args.benchmark_train_batch_size
    inp_train = adapter['make_input'](train_bs, device)
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=1e-4)
    try:
        step_info = measure_train_step(model, inp_train, optimizer,
                                        adapter['forward_loss'], device,
                                        warmup=args.benchmark_train_warmup,
                                        iters=args.benchmark_train_iters)
        print(f"  train step: mean={step_info['train_step_mean_ms']}ms  "
              f"fwd={step_info['train_fwd_ms']}ms  bwd={step_info['train_bwd_ms']}ms  "
              f"opt={step_info['train_opt_ms']}ms")
        print(f"  train mem: alloc={step_info['peak_train_allocated_gb']}GB  "
              f"reserved={step_info['peak_train_reserved_gb']}GB")
    except Exception as e:
        print(f"  [WARN] train step failed: {e}")
        step_info = {k: 0 for k in
                     ['train_step_mean_ms','train_fwd_ms','train_bwd_ms',
                      'train_opt_ms','peak_train_allocated_gb','peak_train_reserved_gb']}

    # ── Build output row ──
    run_id = get_run_id()
    env = {
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'pytorch': torch.__version__,
        'cuda': torch.version.cuda or 'N/A',
    }
    # Shapes
    vis_shape = adapter.get('vis_shape', '')
    ir_shape = adapter.get('ir_shape', '')
    ais_shape = adapter.get('ais_shape', '')

    row = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'run_id': run_id,
        'method': name,
        'protocol': args.protocol,
        'modalities': modalities,
        'seed': args.seed,
        'device': str(device),
        'gpu_name': env['gpu_name'],
        'pytorch': env['pytorch'],
        'cuda': env['cuda'],
        'precision': args.benchmark_precision,
        'batch_size': train_bs,
        'vis_shape': vis_shape,
        'ir_shape': ir_shape,
        'ais_shape': ais_shape,
        'target_label_count': '',
        'target_labels_per_class': (args.target_labels_per_class
                                    if args.target_labels_per_class is not None else ''),
        'backbone': BACKBONE_INFO.get(adapter.get('_key', ''), ('', ''))[0],
        'pretraining': BACKBONE_INFO.get(adapter.get('_key', ''), ('', ''))[1],
        'total_params': total_params,
        'trainable_params': trainable_params,
        'buffer_mb': round(buffer_mb, 4),
        'model_size_mb': round(model_size_mb, 4),
        'MACs': macs,
        'inference_GFLOPs': gflops,
        'unsupported_ops': json.dumps(unsupported),
        'latency_mean_ms': lat_info['latency_mean_ms'],
        'latency_std_ms': lat_info['latency_std_ms'],
        'latency_p50_ms': lat_info['latency_p50_ms'],
        'latency_p95_ms': lat_info['latency_p95_ms'],
        'throughput_samples_s': lat_info['throughput_samples_s'],
        'train_step_mean_ms': step_info.get('train_step_mean_ms', 0),
        'train_fwd_ms': step_info.get('train_fwd_ms', 0),
        'train_bwd_ms': step_info.get('train_bwd_ms', 0),
        'train_opt_ms': step_info.get('train_opt_ms', 0),
        'peak_train_allocated_gb': step_info.get('peak_train_allocated_gb', 0),
        'peak_train_reserved_gb': step_info.get('peak_train_reserved_gb', 0),
        'peak_infer_allocated_gb': mem_info['peak_infer_allocated_gb'],
        'peak_infer_reserved_gb': mem_info['peak_infer_reserved_gb'],
        'svd_update_ms': 0,
        'sinkhorn_ms': 0,
        'encoder_forward_ms': 0,
        'tensor_forward_ms': 0,
        'backward_ms': step_info.get('train_bwd_ms', 0),
        'optimizer_step_ms': step_info.get('train_opt_ms', 0),
    }
    # JMDA-specific: measure module-level timings
    if group == 'jmda':
        mod_times = _measure_jmda_modules(adapter, model, device, args)
        for k, v in mod_times.items():
            row[k] = v

    # Save
    # Sanitize method name for directory (no * etc. on Windows)
    safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)
    method_dir = os.path.join(output_dir, safe_name)
    run_dir = save_metrics_json(method_dir, run_id, row)
    append_efficiency_csv(output_dir, row)

    # Save benchmark.log
    log_path = os.path.join(method_dir, run_id, 'benchmark.log')
    with open(log_path, 'w') as f:
        f.write(f"Benchmark: {name}\n")
        f.write(f"Time: {row['timestamp']}\n")
        f.write(f"Device: {device}  GPU: {env['gpu_name']}\n")
        f.write(f"Precision: {args.benchmark_precision}\n\n")
        f.write(f"num_classes: {NUM_CLASSES}\n")
        f.write(f"total_params: {total_params}\n")
        f.write(f"trainable_params: {trainable_params}\n")
        f.write(f"buffer_mb: {buffer_mb:.4f}\n")
        f.write(f"model_size_mb: {model_size_mb:.4f}\n")
        f.write(f"MACs: {macs}\n")
        f.write(f"inference_GFLOPs: {gflops} (1 MAC = 2 FLOPs)\n")
        f.write(f"unsupported_ops: {unsupported}\n")
        f.write(f"latency_mean_ms: {lat_info['latency_mean_ms']}\n")
        f.write(f"latency_p50_ms: {lat_info['latency_p50_ms']}\n")
        f.write(f"latency_p95_ms: {lat_info['latency_p95_ms']}\n")
        f.write(f"throughput_samples_s: {lat_info['throughput_samples_s']}\n")
        f.write(f"train_step_mean_ms: {step_info.get('train_step_mean_ms',0)}\n")
        f.write(f"peak_train_allocated_gb: {step_info.get('peak_train_allocated_gb',0)}\n")
        f.write(f"peak_infer_allocated_gb: {mem_info['peak_infer_allocated_gb']}\n")
        if group == 'jmda':
            for k, v in mod_times.items():
                f.write(f"{k}: {v}\n")

    # Save profiler summary
    try:
        model.eval()
        run_profiler_summary(model, inp_lat, device, run_dir,
                             forward_fn=forward_infer, iters=10)
    except Exception as e:
        with open(os.path.join(run_dir, 'profiler_summary.txt'), 'w') as f:
            f.write(f"Profiler failed: {e}\n")

    # Cleanup
    del model
    _reset_memory()
    return row


def _measure_jmda_modules(adapter, model, device, args):
    """Measure JMDA-Net module-level timings."""
    import time as _time
    timings = {}
    model.eval()
    inp = adapter['make_input'](1, device)

    # encoder forward
    times = []
    with torch.inference_mode():
        for _ in range(10):  # warmup
            f_vis = model.net_vis(inp['vis'])
            f_ir = model.net_ir(inp['ir'])
        for _ in range(50):
            _cuda_sync()
            t0 = _time.perf_counter()
            f_vis = model.net_vis(inp['vis'])
            f_ir = model.net_ir(inp['ir'])
            _cuda_sync()
            t1 = _time.perf_counter()
            times.append((t1-t0)*1000)
    timings['encoder_forward_ms'] = round(float(np.mean(times)), 4)

    # tensor forward (if applicable)
    if hasattr(model, 'tal'):
        times = []
        with torch.inference_mode():
            f_vis = model.net_vis(inp['vis'])
            f_ir = model.net_ir(inp['ir'])
            for _ in range(10):
                model.tal([f_vis, f_ir], [f_vis, f_ir])
            for _ in range(50):
                _cuda_sync()
                t0 = _time.perf_counter()
                model.tal([f_vis, f_ir], [f_vis, f_ir])
                _cuda_sync()
                t1 = _time.perf_counter()
                times.append((t1-t0)*1000)
        timings['tensor_forward_ms'] = round(float(np.mean(times)), 4)

        # SVD update (per epoch). Tensor.forward intentionally never changes
        # projection matrices, so time the real update_projections() API.
        svd_times = []
        sample_count = max(model.tal.output_dims) + 1
        source_bank = [
            torch.randn(sample_count, dim, device=device)
            for dim in model.tal.input_dims
        ]
        target_bank = [
            torch.randn(sample_count, dim, device=device)
            for dim in model.tal.input_dims
        ]
        for _ in range(3):
            _cuda_sync()
            t0 = _time.perf_counter()
            model.tal.update_projections(source_bank, target_bank)
            _cuda_sync()
            t1 = _time.perf_counter()
            svd_times.append((t1-t0)*1000)
        timings['svd_update_ms'] = round(float(np.mean(svd_times)), 4)

    # sinkhorn (if applicable, training-only)
    if hasattr(model, 'generator'):
        gen_times = []
        model.train()
        # Use larger batch (train_batch_size) since generator's BatchNorm
        # requires batch_size > 1 in training mode; Sinkhorn operates on
        # source/target batches in actual training (batch > 1)
        gen_bs = getattr(args, 'benchmark_train_batch_size', 8)
        gen_inp = adapter['make_input'](gen_bs, device)
        f_vis = model.net_vis(gen_inp['vis'])
        f_ir = model.net_ir(gen_inp['ir'])
        # If tensor module is used, apply TAL first to match generator's feat_dim
        if hasattr(model, 'tal') and model.use_tensor:
            (p_vis, p_ir), _, _ = model.tal([f_vis, f_ir], [f_vis, f_ir])
            if isinstance(p_vis, list): p_vis = p_vis[0]
            if isinstance(p_ir, list): p_ir = p_ir[0]
            feat = torch.cat([p_vis, p_ir], dim=1)
        else:
            feat = torch.cat([f_vis, f_ir], dim=1)
        for _ in range(5):  # warmup
            model.generator(feat, feat.clone())
        for _ in range(20):
            _cuda_sync()
            t0 = _time.perf_counter()
            model.generator(feat, feat.clone())
            _cuda_sync()
            t1 = _time.perf_counter()
            gen_times.append((t1-t0)*1000)
        timings['sinkhorn_ms'] = round(float(np.mean(gen_times)), 4)
    else:
        timings['sinkhorn_ms'] = 0

    return timings


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Unified efficiency benchmark')
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated method names or "all"')
    parser.add_argument('--benchmark_efficiency', action='store_true', default=True)
    parser.add_argument('--benchmark_warmup', type=int, default=100)
    parser.add_argument('--benchmark_iters', type=int, default=500)
    parser.add_argument('--benchmark_repeats', type=int, default=3)
    parser.add_argument('--benchmark_batch_sizes', type=str, default='1,8')
    parser.add_argument('--benchmark_precision', type=str, default='fp32')
    parser.add_argument('--benchmark_train_batch_size', type=int, default=8)
    parser.add_argument('--benchmark_train_warmup', type=int, default=5)
    parser.add_argument('--benchmark_train_iters', type=int, default=20)
    parser.add_argument('--efficiency_output_dir', type=str,
                        default=os.path.join(PROJ, 'efficiency_results'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--protocol', type=str, default='benchmark',
                        choices=['UDA', 'SDA', 'Supervised', 'VLM', 'benchmark'],
                        help='Supervision protocol recorded in the output rows; '
                             'SDA/UDA results are kept in separate tables.')
    parser.add_argument('--target-labels-per-class', dest='target_labels_per_class',
                        type=int, default=None,
                        help='SDA label budget per class (recorded when --protocol SDA).')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick smoke test (reduced warmup/iters)')
    parser.add_argument('--num-classes', type=int, default=14,
                        help='Classifier width for the UDA adapters. UDA training '
                             'uses a target-driven label space (tools/uda_data.py), '
                             'so pass the class count of the target domain.')
    parser.add_argument('--image-backbone', choices=['method_default', 'resnet50'],
                        default='method_default',
                        help='backbone override for DANN/DeepCORAL/CDAN/MDD/MCC')
    args = parser.parse_args()

    global NUM_CLASSES, IMAGE_BACKBONE
    NUM_CLASSES = args.num_classes
    IMAGE_BACKBONE = args.image_backbone
    if IMAGE_BACKBONE == 'resnet50':
        for key in ('dann', 'deepcoral', 'cdan', 'mdd', 'mcc'):
            BACKBONE_INFO[key] = ('ResNet-50', 'ImageNet')

    if args.smoke:
        args.benchmark_warmup = 5
        args.benchmark_iters = 20
        args.benchmark_repeats = 1
        args.benchmark_train_warmup = min(args.benchmark_train_warmup, 1)
        args.benchmark_train_iters = min(args.benchmark_train_iters, 3)

    os.makedirs(args.efficiency_output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Precision: {args.benchmark_precision}  Warmup: {args.benchmark_warmup}  "
          f"Iters: {args.benchmark_iters}  Repeats: {args.benchmark_repeats}")

    # Save environment
    env_info = save_environment_json(args.efficiency_output_dir)
    print(f"Environment saved: {env_info}")

    # Select methods
    if args.methods == 'all':
        methods = ALL_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(',')]

    # Run benchmarks
    all_rows = []
    for method_key in methods:
        if method_key not in ADAPTERS:
            print(f"[WARN] Unknown method: {method_key}, skipping")
            continue
        for rep in range(args.benchmark_repeats):
            print(f"\n>>> {method_key} repeat {rep+1}/{args.benchmark_repeats}")
            torch.manual_seed(args.seed + rep)
            try:
                adapter = ADAPTERS[method_key]()
                adapter['_key'] = method_key
                row = run_benchmark(adapter, args, device, args.efficiency_output_dir)
                all_rows.append(row)
            except Exception as e:
                import traceback
                print(f"[ERROR] {method_key} failed: {e}")
                traceback.print_exc()

    # Generate summary
    generate_summary_files(args.efficiency_output_dir)
    print(f"\n{'='*60}")
    print(f"  All benchmarks complete. Results in: {args.efficiency_output_dir}")
    print(f"  Methods tested: {len(set(r['method'] for r in all_rows))}")
    print(f"  Total runs: {len(all_rows)}")
    print(f"{'='*60}")

    # Print method mapping
    print(f"\nMethod → Script → Model Class → Data Loader mapping:")
    print(f"{'─'*80}")
    mappings = [
        ('Source-only', 'benchmark adapter', 'resnet18+fc', 'ImageList (VIS)'),
        ('DANN', 'DANN/train_dann.py', 'DANN(nn.Module)', 'ImageListDataset (VIS)'),
        ('DeepCORAL', 'DeepCORAL/train_deepcoral.py', 'DeepCORAL(nn.Module)', 'ImageListDataset (VIS)'),
        ('CDAN', 'CDAN/train_cdan.py', 'ResNetFc+AdversarialNet', 'ImageListDataset (VIS)'),
        ('MDD', 'MDD/train_mdd.py', 'tllib ImageClassifier', 'ImageListDataset (VIS)'),
        ('MCC', 'MCC/train_mcc.py', 'tllib ImageClassifier', 'ImageListDataset (VIS)'),
        ('CCSA', 'CCSA/train_ccsa.py', 'VGG16 + IR adapter + FC1024/128', 'Paired source/target table (VIS+IR, SDA)'),
        ('d-SNE', 'd-SNE/train_dsne.py', 'ResNetV2-101 + IR adapter + 512 embed', 'Paired source/target table (VIS+IR, SDA)'),
        ('MCFT', 'MCFT/train_mcft.py', 'MViT (VIS+IR)', 'PairedVisIrDataset (VIS+IR, 15x15)'),
        ('VDS2Raw-CCSA', 'vds2raw/train_ccsa.py', 'VGG16 + NIR/AIS adapters + FC1024/128', 'VDS2Raw RGB+NIR+AIS'),
        ('VDS2Raw-d-SNE', 'vds2raw/train_dsne.py', 'ResNetV2-101 + NIR/AIS adapters + 4096/512', 'VDS2Raw RGB+NIR+AIS'),
        ('VDS2Raw-MCFT3', 'vds2raw/train_mcft.py', 'MViT + AIS/mask late fusion', 'VDS2Raw RGB+NIR+AIS'),
        ('PDA', 'PDA/train_pda.py', 'CustomCLIP(ViT-B/16)', 'ImageListDataset (VIS)'),
        ('JMDA-Net', os.path.join(JMDA_NET_DIR, 'Ablation', 'module_ablation.py'),
         'VisFeat+IRFeat+TAL+OTGen', 'MultiModalDomainDataset (VIS+IR)'),
    ]
    for m, s, c, d in mappings:
        print(f"  {m:18s} | {s:42s} | {c:28s} | {d}")


if __name__ == '__main__':
    main()
