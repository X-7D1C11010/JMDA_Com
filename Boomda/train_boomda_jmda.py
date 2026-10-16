"""
Boomda adapted for JMDA-Net dataset (IR / VIS / AIS three modalities).

Based on the author's implementation: https://github.com/sunjunaimer/Boomda
(train_boomda.py + models/Boomda_model.py). The original acoustic (Wavlm) /
visual (APViT) / lexical (Bert) encoders are replaced by IR (ResNet-18) /
VIS (ResNet-18) / AIS (1-D CNN over concatenated I/Q) encoders, while the
core Boomda pipeline is preserved unchanged:
  1. Information-bottleneck regularisation      -> entropy_re on features
     (sum of log-variance, models/myfunc/en_re.py);
  2. Pseudo-labeling                            -> fused one-hot voting
     (1.5 * fused + each modality), index_rely threshold filtering and
     loss_CE_tpl on selected target samples;
  3. Correlation alignment                      -> CORAL loss per modality and
     on the joint feature (models/coralutils/coral.py);
  4. Multi-objective optimisation with modality-balance weight solving ->
     per-modality CORAL gradients captured at the joint feature via a hook,
     gradient_normalizers('loss') normalisation and the author's closed-form
     `use_full` solver (models/min_norm_solvers.py), with EMA (beta=0.2) of the
     modality scales;
  5. Domain discriminator on the joint feature with gradient reversal
     (models/networks/discriminator.py), alpha = p**2 schedule.

Usage:
    python train_boomda_jmda.py --target backlight --epochs 3
"""
import os
import sys
import argparse
import copy
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms as T
from torchvision import models as tvm
from PIL import Image
import h5py

BOOMDA_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOOMDA_ROOT)

# Reuse the author's modules as-is (avoid models.myfunc package __init__ which
# pulls in heavy LoRA deps).
from models.networks.classifier import SimpleFC2
from models.networks.discriminator import DomainDiscriminator
from models.min_norm_solvers import gradient_normalizers
from models.coralutils.coral import coral

_spec = importlib.util.spec_from_file_location(
    'boomda_en_re', os.path.join(BOOMDA_ROOT, 'models', 'myfunc', 'en_re.py'))
_en_re_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_en_re_mod)
entropy_re = _en_re_mod.entropy_re

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from data_paths import remap_data_path, data_path, exact_ir_partner
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Data: paired (VIS, IR) images from the data lists + AIS row assigned per class
# --------------------------------------------------------------------------
class PairedMultimodalDataset(Dataset):
    """Read the shared VIS cohort and recover its exact IR observation.

    The raw split may place the two modalities of one observation in different
    train/val folders. Search their union by filename stem, matching JMDA-Net's
    paired cohort rule. AIS assignment remains deterministic by class.
    """

    IMG_EXT = ('.jpg', '.jpeg', '.png')

    @staticmethod
    def _find_ir_partner(vis_path):
        return exact_ir_partner(vis_path)

    def __init__(self, list_path, ais_data=None, ais_labels=None, transform=None,
                 use_ais=False, ais_dim=23040):
        self.transform = transform
        self.use_ais = bool(use_ais)
        self.ais_data = ais_data
        self.ais_dim = int(ais_dim)
        self.samples = []  # (vis_path, ir_path, label, ais_index)
        if self.use_ais:
            if ais_data is None or ais_labels is None:
                raise ValueError('use_ais=True requires AIS data and labels')
            self.ais_dim = int(ais_data.shape[1])
            per_class = {
                c: np.where(ais_labels == c)[0] for c in range(14)
            }
        else:
            per_class = {c: np.empty(0, dtype=np.int64) for c in range(14)}
        counters = {c: 0 for c in range(14)}
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                vis_path, label = remap_data_path(parts[0]), int(parts[1])
                ir_path = self._find_ir_partner(vis_path)
                if ir_path is None:
                    continue
                if not 0 <= label < 14:
                    raise ValueError(
                        f'Label {label} is outside the declared 14-class space.'
                    )
                c = label
                pool = per_class[c]
                if not self.use_ais:
                    ais_idx = -1
                elif len(pool) == 0:
                    ais_idx = 0
                else:
                    ais_idx = int(pool[counters[c] % len(pool)])
                    counters[c] += 1
                self.samples.append((vis_path, ir_path, label, ais_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vis_path, ir_path, label, ais_idx = self.samples[idx]
        vis = Image.open(vis_path).convert('RGB')
        ir = Image.open(ir_path).convert('RGB')
        if self.transform:
            # Replay one RNG state so paired modalities receive the same
            # random spatial augmentation.
            transform_rng_state = torch.get_rng_state()
            vis = self.transform(vis)
            torch.set_rng_state(transform_rng_state)
            ir = self.transform(ir)
        if self.use_ais:
            ais = torch.from_numpy(self.ais_data[ais_idx].copy())
        else:
            # Weather data currently has no trustworthy sample-level AIS
            # pairing. Keep the AIS branch shape-compatible without injecting
            # label-selected signals; --use-ais re-enables the real input.
            ais = torch.zeros(self.ais_dim, dtype=torch.float32)
        return {'vision': vis, 'ir': ir, 'ais': ais, 'label': label}


def get_transforms():
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def load_ais(mat_path):
    """Load the AIS .mat (v7.3, HDF5): balanced_rcv_I/Q are (11520, 1572) each,
    sample axis is the second dimension -> transpose to (1572, 11520), concat
    I|Q -> (1572, 23040), per-row z-score normalisation."""
    with h5py.File(mat_path, 'r') as f:
        I = np.array(f['balanced_rcv_I']).T.astype(np.float32)   # (1572, 11520)
        Q = np.array(f['balanced_rcv_Q']).T.astype(np.float32)
        labels = np.array(f['new_balanced_label']).reshape(-1).astype(np.int64)
    ais = np.concatenate([I, Q], axis=1)  # (1572, 23040)
    mean = ais.mean(axis=1, keepdims=True)
    std = ais.std(axis=1, keepdims=True) + 1e-8
    ais = (ais - mean) / std
    return ais, labels


# --------------------------------------------------------------------------
# Encoders (replacing the original Wavlm / APViT / Bert encoders)
# --------------------------------------------------------------------------
class IRVisEncoder(nn.Module):
    """ResNet-18 backbone -> 512-d feature (same role as the original encoders)."""

    def __init__(self, out_dim=512):
        super().__init__()
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.backbone = net
        self.out_features = out_dim

    def forward(self, x):
        return self.backbone(x)


class AISEncoder(nn.Module):
    """1-D CNN over the concatenated I/Q vector (23040-d) -> embd-dim feature."""

    def __init__(self, in_dim=23040, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=15, stride=8, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_features = out_dim

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def p2oh(prob_matrix):
    max_indices = torch.argmax(prob_matrix, dim=1)
    one_hot = torch.zeros_like(prob_matrix)
    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
    return one_hot


def index_rely(matrix, threshold=3):
    large = matrix >= threshold
    rows = torch.any(large, dim=1)
    return torch.nonzero(rows).squeeze(1)


# --------------------------------------------------------------------------
# Boomda model (faithful reproduction of BoomdaModel, encoders swapped)
# --------------------------------------------------------------------------
class BoomdaJMDA(nn.Module):
    # letters kept compatible with the author's solver: A=AIS, V=VIS, L=IR
    MODALITIES = 'AVL'

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.use_ais = bool(opt.use_ais)
        self.modality = self.MODALITIES if self.use_ais else 'VL'
        all_len = len(self.modality) + 1
        scalet = [1.0 if self.use_ais else 0.0, 1.0, 1.0, 1.0]
        self.scale = np.array(scalet) / all_len
        self.d_f = 32
        self.pl_num = 0

        embd_a = opt.embd_size_a  # AIS
        embd_v = opt.embd_size_v  # VIS
        embd_l = opt.embd_size_l  # IR
        cls_input_size = embd_a + embd_v + embd_l

        # joint classifier + joint domain discriminator (author's structure)
        self.netC = SimpleFC2(input_dim=cls_input_size, hidden=[128],
                              output_dim=opt.output_dim, d_f=self.d_f)
        self.netdiscCom = DomainDiscriminator(cls_input_size, 1)

        # per-modality classifiers (author's structure: netCA/netCV hidden 64, netCL hidden 128)
        self.netCA = SimpleFC2(input_dim=embd_a, hidden=[64], output_dim=opt.output_dim, d_f=self.d_f)
        self.netCV = SimpleFC2(input_dim=embd_v, hidden=[64], output_dim=opt.output_dim, d_f=self.d_f)
        self.netCL = SimpleFC2(input_dim=embd_l, hidden=[128], output_dim=opt.output_dim, d_f=self.d_f)

        # encoders (replacing Wavlm / APViT / Bert)
        self.netA_encoder = AISEncoder(out_dim=embd_a)
        self.netV_encoder = IRVisEncoder(out_dim=embd_v)
        self.netL_encoder = IRVisEncoder(out_dim=embd_l)
        if not self.use_ais:
            self.netA_encoder.requires_grad_(False)
            self.netCA.requires_grad_(False)

        if opt.isTrain:
            self.criterion_ce = nn.CrossEntropyLoss()
            self.criterion_domdisc = nn.CrossEntropyLoss()
            parameters = [
                {'params': self.netC.parameters(), 'lr': opt.lr},
                {'params': self.netCV.parameters(), 'lr': opt.lr},
                {'params': self.netCL.parameters(), 'lr': opt.lr},
                {'params': self.netdiscCom.parameters(), 'lr': opt.lr},
                {'params': self.netV_encoder.parameters(), 'lr': opt.enc_lr},
                {'params': self.netL_encoder.parameters(), 'lr': opt.enc_lr},
            ]
            if self.use_ais:
                parameters.extend([
                    {'params': self.netCA.parameters(), 'lr': opt.lr},
                    {'params': self.netA_encoder.parameters(), 'lr': opt.enc_lr},
                ])
            self.optimizer = torch.optim.Adam(parameters, betas=(opt.beta1, 0.999))
        self.scaler = GradScaler()
        self.is_amp = 1
        self.save_dir = opt.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def set_input(self, data, data_t, alpha):
        self.ais, self.ais_t = data['ais'].to(device), data_t['ais'].to(device)
        self.visual, self.visual_t = data['vision'].to(device), data_t['vision'].to(device)
        self.ir, self.ir_t = data['ir'].to(device), data_t['ir'].to(device)
        self.label = data['label'].to(device)
        self.alpha = alpha
        domain_source = torch.zeros(self.label.shape[0]).type(torch.LongTensor)
        domain_target = torch.ones(self.visual_t.shape[0]).type(torch.LongTensor)
        self.domain_combined_label = torch.cat((domain_source, domain_target), 0).to(device)
        self.bs_s, self.bs_t = self.label.shape[0], self.visual_t.shape[0]

    def forward(self):
        if self.use_ais:
            self.feat_A = self.netA_encoder(self.ais)
            self.feat_A_t = self.netA_encoder(self.ais_t)
        else:
            self.feat_A = self.visual.new_zeros((self.bs_s, self.opt.embd_size_a))
            self.feat_A_t = self.visual_t.new_zeros((self.bs_t, self.opt.embd_size_a))
        self.feat_V = self.netV_encoder(self.visual)
        self.feat_V_t = self.netV_encoder(self.visual_t)
        self.feat_L = self.netL_encoder(self.ir)
        self.feat_L_t = self.netL_encoder(self.ir_t)

        bs_s, bs_t = self.bs_s, self.bs_t
        bs_a = bs_s + bs_t
        self.feat = torch.cat([self.feat_A, self.feat_V, self.feat_L], dim=-1)
        self.feat_t = torch.cat([self.feat_A_t, self.feat_V_t, self.feat_L_t], dim=-1)
        self.feat_all = torch.cat([self.feat, self.feat_t], dim=0)
        self.feat_all.register_hook(self.save_gradients)
        self.feat_r = self.feat_all[0:bs_s, :]
        self.feat_t_r = self.feat_all[bs_s:bs_a, :]

        self.logits, self.fall = self.netC(self.feat_r)
        self.pred = F.softmax(self.logits, dim=-1)
        self.logits_t, self.fall_t = self.netC(self.feat_t_r)
        self.pred_t = F.softmax(self.logits_t, dim=-1)
        # ---- pseudo labels: fused voting (author's flow) ----
        self.pred_toh = p2oh(self.pred_t.detach())
        self.pred_tf = 1.5 * self.pred_toh

        # Author's flow: per-modality classifiers read SLICES of feat_all so
        # that their losses are connected to feat_all and the weight-solving
        # hook below can capture their per-modality gradients.
        dim_a = self.opt.embd_size_a
        dim_av = dim_a + self.opt.embd_size_v
        dim_avl = dim_av + self.opt.embd_size_l

        if self.use_ais:
            self.feat_A_r = self.feat_all[0:bs_s, 0:dim_a]
            self.feat_A_t_r = self.feat_all[bs_s:bs_a, 0:dim_a]
            self.logits_A, self.fa = self.netCA(self.feat_A_r)
            self.pred_A = F.softmax(self.logits_A, dim=-1)
            self.logits_A_t, self.fa_t = self.netCA(self.feat_A_t_r)
            self.pred_A_t = F.softmax(self.logits_A_t, dim=-1)
            self.pred_tf += p2oh(self.pred_A_t.detach())

        self.feat_V_r = self.feat_all[0:bs_s, dim_a:dim_av]
        self.feat_V_t_r = self.feat_all[bs_s:bs_a, dim_a:dim_av]
        self.logits_V, self.fv = self.netCV(self.feat_V_r)
        self.pred_V = F.softmax(self.logits_V, dim=-1)
        self.logits_V_t, self.fv_t = self.netCV(self.feat_V_t_r)
        self.pred_V_t = F.softmax(self.logits_V_t, dim=-1)
        self.pred_tf += p2oh(self.pred_V_t.detach())

        self.feat_L_r = self.feat_all[0:bs_s, dim_av:dim_avl]
        self.feat_L_t_r = self.feat_all[bs_s:bs_a, dim_av:dim_avl]
        self.logits_L, self.fl = self.netCL(self.feat_L_r)
        self.pred_L = F.softmax(self.logits_L, dim=-1)
        self.logits_L_t, self.fl_t = self.netCL(self.feat_L_t_r)
        self.pred_L_t = F.softmax(self.logits_L_t, dim=-1)
        self.pred_tf += p2oh(self.pred_L_t.detach())

        self.domainCom = self.netdiscCom(self.feat_all, self.alpha)
        # Preserve the original 3/4.5 consensus ratio when AIS is inactive:
        # active max vote becomes 3.5, so the nearest attainable threshold is 2.5.
        vote_threshold = 3.0 if self.use_ais else 2.5
        self.index_pl = index_rely(self.pred_tf, threshold=vote_threshold)
        self.pred_tf = self.pred_tf.argmax(dim=1)
        self.pl = self.pred_tf[self.index_pl]
        if self.index_pl.shape[0] >= 1:
            self.pl_num += self.index_pl.shape[0]

    def save_gradients(self, grad):
        self.feat_all_gradients = grad

    def weight_cal(self):
        """Author's multi-objective modality-balance weight solving (use_full=1)."""
        loss_data, loss_data_m = {}, {}
        grads, grads_m = {}, {}
        self.loss_coral_all = 0
        mul = 1
        with autocast():
            if self.use_ais:
                self.loss_coral_a = mul * coral(self.fa, self.fa_t)
                loss_data['A'] = self.loss_coral_a.item()
                self.loss_coral_all = self.loss_coral_all + self.loss_coral_a

            self.loss_coral_v = mul * coral(self.fv, self.fv_t)
            loss_data['V'] = self.loss_coral_v.item()
            self.loss_coral_all = self.loss_coral_all + self.loss_coral_v

            self.loss_coral_l = mul * coral(self.fl, self.fl_t)
            loss_data['L'] = self.loss_coral_l.item()
            self.loss_coral_all = self.loss_coral_all + self.loss_coral_l

            self.scaler.scale(self.loss_coral_all).backward(retain_graph=True)
            grads_all = [copy.deepcopy(self.feat_all_gradients)]
            self.feat_all_gradients.zero_()

            self.loss_coral_com = mul * coral(self.fall, self.fall_t)
            loss_data_m['A'] = loss_data_m['V'] = loss_data_m['L'] = self.loss_coral_com.item()
            self.scaler.scale(self.loss_coral_com).backward(retain_graph=True)
            grads_com = [copy.deepcopy(self.feat_all_gradients)]
            self.feat_all_gradients.zero_()

        grads_all = torch.sum(grads_all[0], dim=0)
        grads_com = torch.sum(grads_com[0], dim=0)
        dim_a = self.opt.embd_size_a
        dim_av = dim_a + self.opt.embd_size_v
        ranges = {
            'A': (0, dim_a),
            'V': (dim_a, dim_av),
            'L': (dim_av, dim_av + self.opt.embd_size_l),
        }
        for t in self.modality:
            start, end = ranges[t]
            grads[t] = grads_all[start:end]
            grads_m[t] = grads_com[start:end]

        gn = gradient_normalizers(grads, loss_data, 'loss')
        for t in self.modality:
            grads[t] = grads[t] / gn[t]
        gn_m = gradient_normalizers(grads_m, loss_data_m, 'loss')
        for t in self.modality:
            grads_m[t] = grads_m[t] / gn[t]

        modality = self.modality
        nc = len(self.modality)
        M = nc
        # author's `use_full` closed-form solver
        pd_1 = torch.tensor([torch.dot(grads[t], grads[t]) for t in modality])
        pd_2 = torch.tensor([torch.dot(grads_m[t], grads_m[t]) for t in modality])
        pd_x = torch.tensor([torch.dot(grads[t], grads_m[t]) for t in modality])
        diag = [i for i in pd_1]
        diag.append(sum(pd_2))
        Q_t = torch.tensor(diag)
        Q = torch.diag(Q_t)
        Q[M, 0:M] = pd_x
        Q[0:M, M] = pd_x
        Q_t_inv = 1 / Q_t
        sol = Q_t_inv / (sum(Q_t_inv))

        scale = {'A': 0.0, 'V': 0.0, 'L': 0.0, 'J': 0.0}
        for i, t in enumerate(self.modality + 'J'):
            scale[t] = float(sol[i])
        return scale

    def loss_cal(self, scale):
        scale_current = [0.0] * 4
        for i, t in enumerate('AVLJ'):
            scale_current[i] = scale[t]
        scale_current = np.array(scale_current)
        beta = 0.2
        if not np.isnan(scale_current).any():
            self.scale = (1 - beta) * self.scale + beta * scale_current
        scalea = {}
        scalea['A'], scalea['V'], scalea['L'], scalea['J'] = \
            self.scale[0], self.scale[1], self.scale[2], self.scale[3]

        self.loss_CE = self.criterion_ce(self.logits, self.label)
        if self.index_pl.shape[0] >= 1:
            self.loss_CE_tpl = self.criterion_ce(self.logits_t[self.index_pl], self.pl)
        else:
            self.loss_CE_tpl = 0.0

        mul = 1e3
        mul2 = 1e-3
        self.loss_ee = mul2 * entropy_re(self.fall)
        self.loss_ee_A = mul2 * entropy_re(self.fa) if self.use_ais else self.fall.new_zeros(())
        self.loss_ee_V = mul2 * entropy_re(self.fv)
        self.loss_ee_L = mul2 * entropy_re(self.fl)

        self.loss_domdisc_com = self.criterion_domdisc(self.domainCom, self.domain_combined_label)
        self.loss_coral = mul * coral(self.fall, self.fall_t)
        self.loss_coral_a = mul * coral(self.fa, self.fa_t) if self.use_ais else self.fall.new_zeros(())
        self.loss_coral_v = mul * coral(self.fv, self.fv_t)
        self.loss_coral_l = mul * coral(self.fl, self.fl_t)

        self.loss_coralall = scalea['J'] * self.loss_coral + scalea['A'] * self.loss_coral_a + \
                             scalea['V'] * self.loss_coral_v + scalea['L'] * self.loss_coral_l

        self.loss = self.loss_CE + 0.5 * self.alpha * 0.1 * 10 * self.loss_CE_tpl + \
                    0.1 * 20 * 0.05 * (self.loss_CE_A_placeholder()) + \
                    0.1 * 10 * 0.1 * self.loss_coralall + \
                    0.001 * 0.01 * (self.loss_ee + self.loss_ee_A + self.loss_ee_V + self.loss_ee_L)

    def loss_CE_A_placeholder(self):
        # author: 0.1*20*0.05*(loss_CE_A + loss_CE_V + loss_CE_L)
        loss = self.criterion_ce(self.logits_V, self.label) + \
               self.criterion_ce(self.logits_L, self.label)
        if self.use_ais:
            loss = loss + self.criterion_ce(self.logits_A, self.label)
        return loss

    def optimize_parameters(self, epoch):
        self.optimizer.zero_grad()
        with autocast():
            self.forward()
            scale = self.weight_cal()
            self.loss_cal(scale)
        self.optimizer.zero_grad()
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def test(self):
        self.forward_eval()

    def forward_eval(self):
        if self.use_ais:
            self.feat_A = self.netA_encoder(self.ais_eval)
        else:
            self.feat_A = self.visual_eval.new_zeros(
                (self.visual_eval.shape[0], self.opt.embd_size_a))
        self.feat_V = self.netV_encoder(self.visual_eval)
        self.feat_L = self.netL_encoder(self.ir_eval)
        self.feat = torch.cat([self.feat_A, self.feat_V, self.feat_L], dim=-1)
        self.logits, _ = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)


def evaluate(model, loader):
    model.eval()
    total_pred, total_label = [], []
    with torch.no_grad():
        for data in loader:
            model.ais_eval = data['ais'].to(device)
            model.visual_eval = data['vision'].to(device)
            model.ir_eval = data['ir'].to(device)
            model.test()
            pred = model.pred.argmax(dim=1).detach().cpu().numpy()
            total_pred.append(pred)
            total_label.append(data['label'].numpy())
    model.train()
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    return classification_metrics(total_label.tolist(), total_pred.tolist())


def main():
    parser = argparse.ArgumentParser(description='Boomda (author code) adapted for JMDA-Net')
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4, help='head lr (Adam)')
    parser.add_argument('--enc-lr', type=float, default=1e-5, help='encoder lr (Adam)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--embd-size-a', type=int, default=256, help='AIS embedding dim')
    parser.add_argument('--embd-size-v', type=int, default=512, help='VIS embedding dim')
    parser.add_argument('--embd-size-l', type=int, default=512, help='IR embedding dim')
    parser.add_argument('--output-dim', type=int, default=14)
    parser.add_argument('--ais-mat', type=str,
                        default=data_path('AIS', 'balanced_AIS-dataset_16classes_100persample.mat'))
    parser.add_argument('--use-ais', action='store_true',
                        help='Enable AIS .mat input. Default: disabled for the weather dataset.')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(BOOMDA_ROOT, 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.seed = seed_everything(args.seed)
    print(f'Controlled random seed: {args.seed}')
    if args.use_ais:
        parser.error(
            '--use-ais is not valid in strict UDA: the current AIS assignment '
            'selects samples by target ground-truth class. Use only genuinely '
            'sample-aligned, label-free AIS data in a separate implementation.'
        )
    args.isTrain = True
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    if args.use_ais:
        ais_data, ais_labels = load_ais(args.ais_mat)
        print(f'AIS enabled: data={ais_data.shape}, labels={ais_labels.shape}')
    else:
        ais_data = ais_labels = None
        print('AIS disabled for weather data; using a zero placeholder. '
              'Pass --use-ais to enable the preserved AIS input.')

    train_tf, val_tf = get_transforms()
    src_ds = PairedMultimodalDataset(os.path.join(args.data_dir, 'sunny_train.txt'), ais_data, ais_labels, train_tf, args.use_ais)
    tgt_ds = PairedMultimodalDataset(os.path.join(args.data_dir, f'{args.target}_train.txt'), ais_data, ais_labels, train_tf, args.use_ais)
    tgt_val_ds = PairedMultimodalDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'), ais_data, ais_labels, val_tf, args.use_ais)
    print(f'Source paired: {len(src_ds)}, target paired: {len(tgt_ds)}, target val paired: {len(tgt_val_ds)}')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BoomdaJMDA(args).to(device)

    best_acc = -float('inf')
    final_acc = None
    num_epoch = args.epochs
    batch_step = max(len(src_loader), len(tgt_loader))
    total_steps = num_epoch * batch_step
    it = 0
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        modalities = 'VIS+IR+AIS' if args.use_ais else 'VIS+IR (AIS disabled)'
        eff = EfficiencyLogger(args.efficiency_output_dir, 'Boomda',
                                seed=args.seed, device=str(device), modalities=modalities)
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(1, num_epoch + 1):
        if eff:
            eff.epoch_start(epoch)
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        for i in range(batch_step):
            if eff:
                eff.step_start()
            try:
                data = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                data = next(src_iter)
            try:
                data_t = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                data_t = next(tgt_iter)
            # The dataset carries labels for validation compatibility, but
            # strict UDA training must not expose target truth to the model.
            data_t = {key: value for key, value in data_t.items() if key != 'label'}
            p = float(it) / max(total_steps, 1)
            alpha = p ** 2
            it += 1
            model.set_input(data, data_t, alpha)
            if eff:
                eff.dataloader_done()
            model.optimize_parameters(epoch)
            if eff:
                eff.step_end()
            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch}/{num_epoch}] batch [{i + 1}/{batch_step}] '
                      f'loss_CE={float(model.loss_CE):.4f} pl_num={model.pl_num} '
                      f'scale={np.round(model.scale, 3).tolist()}')
        val_metrics = evaluate(model, tgt_val_loader)
        acc = val_metrics['accuracy']
        final_acc = acc
        best_acc = max(best_acc, acc)
        print(f'Epoch [{epoch}/{num_epoch}] target val accuracy: {acc:.4f}')
        print(format_classification_metrics(val_metrics))
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'boomda_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': final_acc,
                                    'diagnostic_peak_acc': best_acc,
                                    'target': args.target})
    print(f'Final target val accuracy ({args.target}): {final_acc:.4f}')
    print(f'Diagnostic peak target val accuracy (not used for model selection): {best_acc:.4f}')


if __name__ == '__main__':
    main()
