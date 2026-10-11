"""
MODfinity (CVPR 2025) for JMDA-Net dataset -- OUR REIMPLEMENTATION.

No public code exists for "MODfinity: Unsupervised Domain Adaptation with
Multimodal Information Flow Intertwining" (Liu et al., CVPR 2025). This file is
a faithful reimplementation from the paper (main paper + Fig. 2-5, Eq. 1-8).
Optimizer details are given only in the (image-only) supplementary, so the
optimiser / temperature below are documented choices.

Method summary (paper notation, M modalities):
  * Labels are modeled as an independent modality: a class encoder E_c maps the
    one-hot label to a feature L, and the affinity is
        phi(F, L) = exp(F . L / tau)                                (Eq. 1)
  * S1 Modal-Affinity Metric Learning (MOML): InfoNCE-style loss between
    (detached) modality features F^m_si and label features L_sj, positives =
    same label in the batch; it updates only E_c (features are detached copies,
    unidirectional w.r.t. the modality encoders)                   (Eq. 2)
  * S2 Fine-Grained Sample Filtering: pseudo label (m_i, k_i) =
    argmax_{m,k} phi(F^m_ti, L_k), confidence R_ti = max_{m,k} phi; the top
    alpha% most confident target samples form the Learning Material LM  (Eq. 3-5)
  * S3 Modal-Affinity Distillation (MOD): for each student modality m, use
    LM^(m) = {<ti, m_i> in LM | m_i != m}; distill from each sample's teacher
    feature (detached copy, unidirectional flow): positives = LM^(m) samples
    sharing the same pseudo label                                  (Eq. 6-7)
  * S4 Continuous Optimization: L_all = sum_m (L^m_c + lambda*L^m_MOML +
    beta*L^m_MOD); E_c keeps being updated with MOML during adaptation (Eq. 8)

Paper hyper-parameters (Sec. 4.3.4): alpha = 0.15 (top 15% -> LM), beta = 1.2
(MOD weight), lambda = 4.5 (MOML weight, all tasks).

JMDA adaptations (documented in REPRODUCTION_RECORD.md):
  * 3 modalities: IR (ResNet-18, 512-d), VIS (ResNet-18, 512-d), AIS (1-D CNN
    over concatenated I/Q, projected to 512-d so a SINGLE shared class encoder
    E_c can score affinities across modalities, as in the paper where all
    backbones share one feature dim);
  * dataset = our ImageList-based paired loader (same protocol as the other
    reproduced methods);
  * evaluation reports per-modality accuracy and the fusion accuracy (mean of
    per-modality softmax probabilities); the fusion accuracy is the headline
    "Best target val accuracy";
  * features are L2-normalised inside phi for numerical stability;
  * Windows: DataLoader num_workers=0.

Usage:
    python train_modfinity.py --target backlight --epochs 3
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import models as tvm
from PIL import Image
import h5py

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from data_paths import remap_data_path, data_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Data: paired (VIS, IR) images from the data lists + AIS row per class
# --------------------------------------------------------------------------
class PairedMultimodalDataset(Dataset):
    """Reads VIS paths (+labels) from a list file and pairs each VIS image with
    an IR image following the canonical JMDA-Net pairing rule (DataLoad.py):
    within the same class folder, sorted(VIS files) and sorted(IR files) are
    paired by index (truncated to the shorter list). Assigns an AIS sample from
    the class pool (deterministic cycling; class-index mapping assumption
    i -> class i). Samples without an IR partner are skipped.
    """

    IMG_EXT = ('.jpg', '.jpeg', '.png')

    @staticmethod
    def _find_ir_partner(vis_path):
        vis_dir = os.path.dirname(vis_path)          # ...\class\可见光
        ir_dir = vis_dir.replace('可见光', '红外')
        if not os.path.isdir(ir_dir):
            return None
        ext = PairedMultimodalDataset.IMG_EXT
        vis_files = sorted(f for f in os.listdir(vis_dir)
                           if f.lower().endswith(ext))
        ir_files = sorted(f for f in os.listdir(ir_dir)
                          if f.lower().endswith(ext))
        name = os.path.basename(vis_path)
        if name not in vis_files:
            return None
        i = vis_files.index(name)
        if i >= len(ir_files):
            return None
        return os.path.join(ir_dir, ir_files[i])

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
            per_class = {c: np.where(ais_labels == c)[0] for c in range(14)}
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
                c = min(label, 13)
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
            vis = self.transform(vis)
            ir = self.transform(ir)
        if self.use_ais:
            ais = torch.from_numpy(self.ais_data[ais_idx].copy())
        else:
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
    """AIS .mat (v7.3): balanced_rcv_I/Q (11520, 1572) -> transpose -> concat
    I|Q -> (1572, 23040), per-row z-score normalisation."""
    with h5py.File(mat_path, 'r') as f:
        I = np.array(f['balanced_rcv_I']).T.astype(np.float32)
        Q = np.array(f['balanced_rcv_Q']).T.astype(np.float32)
        labels = np.array(f['new_balanced_label']).reshape(-1).astype(np.int64)
    ais = np.concatenate([I, Q], axis=1)
    mean = ais.mean(axis=1, keepdims=True)
    std = ais.std(axis=1, keepdims=True) + 1e-8
    ais = (ais - mean) / std
    return ais, labels


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class ImageModalityEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.backbone = net
        self.out_dim = out_dim

    def forward(self, x):
        return self.backbone(x)


class AISEncoder(nn.Module):
    """1-D CNN over the concatenated I/Q vector -> out_dim feature (512 to
    share the affinity space with the image modalities)."""

    def __init__(self, in_dim=23040, out_dim=512):
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
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class ClassEncoder(nn.Module):
    """E_c: one-hot label -> feature L in the shared affinity space.
    A bias-free linear map means L_k is a learned class prototype."""

    def __init__(self, num_classes, feat_dim=512):
        super().__init__()
        self.fc = nn.Linear(num_classes, feat_dim, bias=False)

    def forward(self, onehot):
        return self.fc(onehot)


class ModalityClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# --------------------------------------------------------------------------
# Losses (paper Eq. 1, 2, 7)
# --------------------------------------------------------------------------
def affinity(F_feat, L_feat, tau):
    """phi(F, L) = exp(F . L / tau) with L2-normalised vectors (Eq. 1)."""
    F_n = F.normalize(F_feat, dim=1)
    L_n = F.normalize(L_feat, dim=1)
    return torch.exp(F_n @ L_n.t() / tau)


def moml_loss(F_detached, L_feats, labels, tau):
    """MOML (Eq. 2): global InfoNCE between detached modality features and
    class-encoder features; positives share the same (pseudo-)label.
    Gradient flows into E_c only (F is a detached copy)."""
    phi = affinity(F_detached, L_feats, tau)          # (N, N)
    pos_mask = (labels[:, None] == labels[None, :]).float()
    num = (phi * pos_mask).sum()
    den = phi.sum() + 1e-8
    return -torch.log(num / den)


def mod_loss(student_F, LM_idx, LM_plabel, teacher_F, tau):
    """MOD (Eq. 7) for one student modality m over LM^(m):
    -log( sum_{i, j in t+_i} phi(F^m_ti, F^m̄j_tj.detach()) /
          sum_{i, j in LM^(m)} phi(F^m_ti, F^m̄j_tj.detach()) ).
    teacher_F (K, d) holds, for each LM sample i, its own teacher-modality
    feature (detached copy, unidirectional flow); student_F is the full
    (N, d) student-modality feature matrix."""
    stu = student_F[LM_idx]                            # (K, d), gradient flows
    tea = teacher_F.detach()                           # (K, d), unidirectional
    phi = affinity(stu, tea, tau)                      # (K, K)
    pl = LM_plabel
    pos_mask = (pl[:, None] == pl[None, :]).float()
    num = (phi * pos_mask).sum()
    den = phi.sum() + 1e-8
    return -torch.log(num / den)


class MODfinity(nn.Module):
    ALL_MODALITIES = ('ir', 'vision', 'ais')

    def __init__(self, num_classes=14, feat_dim=512, tau=0.07, use_ais=False):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.use_ais = bool(use_ais)
        self.MODALITIES = self.ALL_MODALITIES if self.use_ais else ('ir', 'vision')
        all_encoders = {
            'ir': ImageModalityEncoder(feat_dim),
            'vision': ImageModalityEncoder(feat_dim),
            'ais': AISEncoder(out_dim=feat_dim),
        }
        self.encoders = nn.ModuleDict({m: all_encoders[m] for m in self.MODALITIES})
        self.classifiers = nn.ModuleDict({
            m: ModalityClassifier(feat_dim, num_classes) for m in self.MODALITIES
        })
        self.class_encoder = ClassEncoder(num_classes, feat_dim)

    def features(self, data):
        return {m: self.encoders[m](data[m]) for m in self.MODALITIES}

    def label_features(self, labels):
        """L = E_c(one-hot(label)) for a tensor of label indices."""
        onehot = F.one_hot(labels, self.num_classes).float()
        return self.class_encoder(onehot)

    def class_prototypes(self):
        """L_k for every class (used by the S2 filtering)."""
        classes = torch.arange(self.num_classes, device=device)
        return self.label_features(classes)            # (C, d)

    def forward(self, data):
        feats = self.features(data)
        logits = {m: self.classifiers[m](feats[m]) for m in self.MODALITIES}
        return logits


# --------------------------------------------------------------------------
# Evaluation: per-modality accuracy + fusion (mean softmax)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = {m: 0 for m in model.MODALITIES}
    correct_fusion = 0
    total = 0
    for data in loader:
        data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
        labels = data['label']
        probs = []
        feats = model.features(data)
        for m in model.MODALITIES:
            logits = model.classifiers[m](feats[m])
            probs.append(F.softmax(logits, dim=1))
            correct[m] += (logits.argmax(1) == labels).sum().item()
        fusion = torch.stack(probs, dim=0).mean(dim=0)
        correct_fusion += (fusion.argmax(1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    per_mod = {m: correct[m] / total for m in model.MODALITIES}
    return correct_fusion / total, per_mod


def main():
    parser = argparse.ArgumentParser(
        description='MODfinity (our reimplementation) for JMDA-Net dataset')
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Adam lr (our choice; paper gives no optimiser '
                             'details outside the image-only supplementary)')
    parser.add_argument('--tau', type=float, default=0.07,
                        help='affinity temperature (our choice)')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='top-alpha fraction forming LM (paper: 0.15)')
    parser.add_argument('--lambda-moml', type=float, default=4.5,
                        help='MOML weight lambda (paper: 4.5)')
    parser.add_argument('--beta-mod', type=float, default=1.2,
                        help='MOD weight beta (paper: 1.2)')
    parser.add_argument('--output-dim', type=int, default=14)
    parser.add_argument('--feat-dim', type=int, default=512)
    parser.add_argument('--ais-mat', type=str,
                        default=data_path('AIS', 'balanced_AIS-dataset_16classes_100persample.mat'))
    parser.add_argument('--use-ais', action='store_true',
                        help='Enable AIS .mat input. Default: disabled for the weather dataset.')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('[our reimplementation]', args)

    if args.use_ais:
        ais_data, ais_labels = load_ais(args.ais_mat)
        print(f'AIS enabled: data={ais_data.shape}, labels={ais_labels.shape}')
    else:
        ais_data = ais_labels = None
        print('AIS disabled for weather data; using a zero placeholder. '
              'Pass --use-ais to enable the preserved AIS input.')

    train_tf, val_tf = get_transforms()
    src_ds = PairedMultimodalDataset(os.path.join(args.data_dir, 'sunny_train.txt'),
                                     ais_data, ais_labels, train_tf, args.use_ais)
    tgt_ds = PairedMultimodalDataset(os.path.join(args.data_dir, f'{args.target}_train.txt'),
                                     ais_data, ais_labels, train_tf, args.use_ais)
    tgt_val_ds = PairedMultimodalDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'),
                                         ais_data, ais_labels, val_tf, args.use_ais)
    print(f'Source paired: {len(src_ds)}, target paired: {len(tgt_ds)}, '
          f'target val paired: {len(tgt_val_ds)}')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
    args.iters_per_epoch = max(len(src_loader), len(tgt_loader))

    model = MODfinity(args.output_dim, args.feat_dim, args.tau,
                      use_ais=args.use_ais).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        modalities = 'VIS+IR+AIS' if args.use_ais else 'VIS+IR (AIS disabled)'
        eff = EfficiencyLogger(args.efficiency_output_dir, 'MODfinity',
                                seed=args.seed, device=str(device), modalities=modalities)
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(1, args.epochs + 1):
        if eff:
            eff.epoch_start(epoch)
        tgt_iter = iter(tgt_loader)
        for i, data in enumerate(src_loader):
            if eff:
                eff.step_start()
            try:
                data_t = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                data_t = next(tgt_iter)
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            data_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_t.items()}
            if eff:
                eff.dataloader_done()
            labels_s = data['label']

            # ---- features & source classification losses ----
            feats_s = model.features(data)
            feats_t = model.features(data_t)
            loss_ce = 0.0
            for m in model.MODALITIES:
                logits_s = model.classifiers[m](feats_s[m])
                loss_ce = loss_ce + F.cross_entropy(logits_s, labels_s)

            # ---- S1 MOML on source (true labels); updates E_c only ----
            L_s = model.label_features(labels_s)
            loss_moml_src = 0.0
            for m in model.MODALITIES:
                loss_moml_src = loss_moml_src + \
                    moml_loss(feats_s[m].detach(), L_s, labels_s, args.tau)

            # ---- S2 fine-grained sample filtering on the target batch ----
            with torch.no_grad():
                protos = model.class_prototypes()          # (C, d)
                phi_stack = torch.stack([
                    affinity(feats_t[m].detach(), protos, args.tau)
                    for m in model.MODALITIES], dim=0)     # (M, N, C)
                best_k_val, k_idx = phi_stack.max(dim=2)   # (M, N) max over k
                conf, m_bar = best_k_val.max(dim=0)        # (N,) max over m
                n_t = conf.shape[0]
                k_bar = k_idx[m_bar, torch.arange(n_t, device=conf.device)]
                k_sel = max(1, int(args.alpha * n_t))
                LM_idx = torch.topk(conf, k_sel).indices   # top alpha%

            # ---- MOML on target with pseudo labels (continuous E_c update) ----
            L_t = model.label_features(k_bar)
            loss_moml_tgt = 0.0
            for m in model.MODALITIES:
                loss_moml_tgt = loss_moml_tgt + \
                    moml_loss(feats_t[m].detach(), L_t, k_bar, args.tau)

            # ---- S3 MOD: distillation per student modality over LM^(m) ----
            # teacher feature of LM sample i = its own teacher-modality feature
            tea_idx = m_bar[LM_idx]
            teacher_F = torch.stack([
                feats_t[model.MODALITIES[tea_idx[j].item()]][LM_idx[j]]
                for j in range(LM_idx.shape[0])], dim=0)   # (K, d)
            loss_mod = 0.0
            for si, m in enumerate(model.MODALITIES):
                keep = tea_idx != si                        # LM^(m): teacher != m
                if keep.sum() >= 1:
                    loss_mod = loss_mod + mod_loss(
                        feats_t[m], LM_idx[keep], k_bar[LM_idx[keep]],
                        teacher_F, args.tau)

            # ---- Eq. 8: L_all = sum_m (L^m_c + lambda*MOML + beta*MOD) ----
            loss = loss_ce + args.lambda_moml * (loss_moml_src + loss_moml_tgt) \
                + args.beta_mod * loss_mod

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch}/{args.epochs}] batch [{i + 1}/{args.iters_per_epoch}] '
                      f'loss={loss.item():.4f} ce={loss_ce.item():.4f} '
                      f'moml={float(loss_moml_src) + float(loss_moml_tgt):.4f} '
                      f'mod={float(loss_mod):.4f} |LM|={k_sel}')

        acc, per_mod = evaluate(model, tgt_val_loader)
        best_acc = max(best_acc, acc)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, f'modfinity_{args.target}_best.pth'))
        print(f'Epoch [{epoch}/{args.epochs}] target val accuracy: {acc:.4f} '
              f'(fusion) | ' + ' '.join(f'{m}={per_mod[m]:.4f}' for m in per_mod))
        if eff:
            eff.epoch_end(epoch)

    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'target': args.target})
    print(f'\nMODfinity (our reimplementation) smoke test finished. '
          f'Best target val accuracy ({args.target}): {best_acc:.4f}')


if __name__ == '__main__':
    main()
