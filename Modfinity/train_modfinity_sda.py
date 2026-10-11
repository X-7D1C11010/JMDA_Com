"""
MODfinity-SDA: supervised-domain-adaptation variant of MODfinity (CVPR 2025,
our reimplementation) for the paper's SDA protocol. This is a NEW file; the
original UDA script train_modfinity.py is left unchanged.

SDA protocol:
  * shared labeled target subset: data_lists/<target>_sda_<N>shot_seed<S>.txt;
  * the original MODfinity mechanism is kept EXACTLY as in the UDA script:
      - S1 MOML on source (true source labels) AND on target (PSEUDO labels
        k_bar from S2 filtering -- never replaced by ground truth);
      - S2 fine-grained sample filtering (top alpha% by class-prototype
        affinity) -> LM;
      - S3 modal-affinity distillation over LM^(m) with pseudo labels;
      - Eq. 8 weighting lambda / beta.
  * the ONLY addition is the target classification loss on the labeled subset:
        L_cls_target = w_cls_target * sum_m CE(classifier_m(feats_t[m]), t_label)

For model / loss details see the docstring of train_modfinity.py.

Usage:
    python train_modfinity_sda.py --target backlight --epochs 3
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
    """Same pairing rule as the UDA script (canonical JMDA-Net pairing)."""

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

    def __init__(self, list_path, ais_data, ais_labels, transform=None):
        self.transform = transform
        self.samples = []  # (vis_path, ir_path, label, ais_index)
        per_class = {c: np.where(ais_labels == c)[0] for c in range(14)}
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
                if len(pool) == 0:
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
        ais = torch.from_numpy(ais_data[ais_idx].copy())
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
# Model (identical to the UDA script)
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


def affinity(F_feat, L_feat, tau):
    F_n = F.normalize(F_feat, dim=1)
    L_n = F.normalize(L_feat, dim=1)
    return torch.exp(F_n @ L_n.t() / tau)


def moml_loss(F_detached, L_feats, labels, tau):
    phi = affinity(F_detached, L_feats, tau)
    pos_mask = (labels[:, None] == labels[None, :]).float()
    num = (phi * pos_mask).sum()
    den = phi.sum() + 1e-8
    return -torch.log(num / den)


def mod_loss(student_F, LM_idx, LM_plabel, teacher_F, tau):
    stu = student_F[LM_idx]
    tea = teacher_F.detach()
    phi = affinity(stu, tea, tau)
    pl = LM_plabel
    pos_mask = (pl[:, None] == pl[None, :]).float()
    num = (phi * pos_mask).sum()
    den = phi.sum() + 1e-8
    return -torch.log(num / den)


class MODfinity(nn.Module):
    MODALITIES = ('ir', 'vision', 'ais')  # M = 3

    def __init__(self, num_classes=14, feat_dim=512, tau=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.encoders = nn.ModuleDict({
            'ir': ImageModalityEncoder(feat_dim),
            'vision': ImageModalityEncoder(feat_dim),
            'ais': AISEncoder(out_dim=feat_dim),
        })
        self.classifiers = nn.ModuleDict({
            m: ModalityClassifier(feat_dim, num_classes) for m in self.MODALITIES
        })
        self.class_encoder = ClassEncoder(num_classes, feat_dim)

    def features(self, data):
        return {m: self.encoders[m](data[m]) for m in self.MODALITIES}

    def label_features(self, labels):
        onehot = F.one_hot(labels, self.num_classes).float()
        return self.class_encoder(onehot)

    def class_prototypes(self):
        classes = torch.arange(self.num_classes, device=device)
        return self.label_features(classes)


# --------------------------------------------------------------------------
# Evaluation: fusion (mean softmax) accuracy
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
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
        fusion = torch.stack(probs, dim=0).mean(dim=0)
        correct_fusion += (fusion.argmax(1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct_fusion / total


def main():
    parser = argparse.ArgumentParser(
        description='MODfinity-SDA (our reimplementation) for JMDA-Net dataset')
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='top-alpha fraction forming LM (paper: 0.15)')
    parser.add_argument('--lambda-moml', type=float, default=4.5,
                        help='MOML weight lambda (paper: 4.5)')
    parser.add_argument('--beta-mod', type=float, default=1.2,
                        help='MOD weight beta (paper: 1.2)')
    parser.add_argument('--output-dim', type=int, default=14)
    parser.add_argument('--feat-dim', type=int, default=512)
    parser.add_argument('--target-labels-per-class', type=int, default=10,
                        help='SDA label budget per class (selects the shared subset list)')
    parser.add_argument('--sda-seed', type=int, default=42,
                        help='Seed used by tools/make_sda_lists.py')
    parser.add_argument('--w-cls-target', type=float, default=1.0,
                        help='Weight of L_cls_target (the ONLY added loss)')
    parser.add_argument('--ais-mat', type=str,
                        default=data_path('AIS', 'balanced_AIS-dataset_16classes_100persample.mat'))
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_sda'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('[our reimplementation, SDA protocol]', args)

    global ais_data
    ais_data, ais_labels = load_ais(args.ais_mat)
    print(f'AIS data: {ais_data.shape}, labels: {ais_labels.shape}')

    train_tf, val_tf = get_transforms()
    src_ds = PairedMultimodalDataset(os.path.join(args.data_dir, 'sunny_train.txt'),
                                     ais_data, ais_labels, train_tf)
    tgt_list = os.path.join(
        args.data_dir,
        f'{args.target}_sda_{args.target_labels_per_class}shot_seed{args.sda_seed}.txt')
    assert os.path.isfile(tgt_list), (
        f'SDA subset list missing: {tgt_list}. '
        f'Run: python tools/make_sda_lists.py --n-per-class {args.target_labels_per_class} '
        f'--seed {args.sda_seed}')
    tgt_ds = PairedMultimodalDataset(tgt_list, ais_data, ais_labels, train_tf)
    tgt_val_ds = PairedMultimodalDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'),
                                         ais_data, ais_labels, val_tf)
    print(f'Source paired: {len(src_ds)}, target LABELED subset (SDA) paired: {len(tgt_ds)}, '
          f'target val paired: {len(tgt_val_ds)}')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
    args.iters_per_epoch = max(len(src_loader), len(tgt_loader))

    model = MODfinity(args.output_dim, args.feat_dim, args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'MODfinity-SDA',
                                seed=args.seed, device=str(device), modalities='VIS+IR+AIS',
                                protocol='SDA',
                                target_label_count=len(tgt_ds),
                                target_labels_per_class=args.target_labels_per_class,
                                backbone='ResNet-18 x2 + 1D-CNN (VIS/IR/AIS)',
                                pretraining='ImageNet')
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
            labels_t = data_t['label']  # SDA: ground truth of the LABELED subset

            # ---- features & source classification losses (unchanged) ----
            feats_s = model.features(data)
            feats_t = model.features(data_t)
            loss_ce = 0.0
            for m in model.MODALITIES:
                logits_s = model.classifiers[m](feats_s[m])
                loss_ce = loss_ce + F.cross_entropy(logits_s, labels_s)

            # ---- S1 MOML on source (true labels); updates E_c only (unchanged) ----
            L_s = model.label_features(labels_s)
            loss_moml_src = 0.0
            for m in model.MODALITIES:
                loss_moml_src = loss_moml_src + \
                    moml_loss(feats_s[m].detach(), L_s, labels_s, args.tau)

            # ---- S2 fine-grained sample filtering (PSEUDO labels, unchanged) ----
            with torch.no_grad():
                protos = model.class_prototypes()
                phi_stack = torch.stack([
                    affinity(feats_t[m].detach(), protos, args.tau)
                    for m in model.MODALITIES], dim=0)
                best_k_val, k_idx = phi_stack.max(dim=2)
                conf, m_bar = best_k_val.max(dim=0)
                n_t = conf.shape[0]
                k_bar = k_idx[m_bar, torch.arange(n_t, device=conf.device)]
                k_sel = max(1, int(args.alpha * n_t))
                LM_idx = torch.topk(conf, k_sel).indices

            # ---- MOML on target with PSEUDO labels (unchanged) ----
            L_t = model.label_features(k_bar)
            loss_moml_tgt = 0.0
            for m in model.MODALITIES:
                loss_moml_tgt = loss_moml_tgt + \
                    moml_loss(feats_t[m].detach(), L_t, k_bar, args.tau)

            # ---- S3 MOD distillation with PSEUDO labels (unchanged) ----
            tea_idx = m_bar[LM_idx]
            teacher_F = torch.stack([
                feats_t[model.MODALITIES[tea_idx[j].item()]][LM_idx[j]]
                for j in range(LM_idx.shape[0])], dim=0)
            loss_mod = 0.0
            for si, m in enumerate(model.MODALITIES):
                keep = tea_idx != si
                if keep.sum() >= 1:
                    loss_mod = loss_mod + mod_loss(
                        feats_t[m], LM_idx[keep], k_bar[LM_idx[keep]],
                        teacher_F, args.tau)

            # ---- SDA addition: L_cls_target (per-modality CE on the subset) ----
            loss_cls_target = 0.0
            for m in model.MODALITIES:
                logits_t_m = model.classifiers[m](feats_t[m])
                loss_cls_target = loss_cls_target + F.cross_entropy(logits_t_m, labels_t)

            # ---- Eq. 8 (unchanged) + L_cls_target ----
            loss = loss_ce + args.lambda_moml * (loss_moml_src + loss_moml_tgt) \
                + args.beta_mod * loss_mod + args.w_cls_target * loss_cls_target

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch}/{args.epochs}] batch [{i + 1}/{args.iters_per_epoch}] '
                      f'loss={loss.item():.4f} ce={loss_ce.item():.4f} '
                      f'moml={float(loss_moml_src) + float(loss_moml_tgt):.4f} '
                      f'mod={float(loss_mod):.4f} '
                      f'L_cls_target={float(loss_cls_target):.4f} |LM|={k_sel}')

        acc = evaluate(model, tgt_val_loader)
        best_acc = max(best_acc, acc)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, f'modfinity_sda_{args.target}_best.pth'))
        print(f'Epoch [{epoch}/{args.epochs}] target val accuracy: {acc:.4f} (fusion)')
        if eff:
            eff.epoch_end(epoch)

    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'target': args.target,
                                    'protocol': 'SDA'})
    print(f'\nMODfinity-SDA (our reimplementation) finished. '
          f'Best target val accuracy ({args.target}): {best_acc:.4f}')


if __name__ == '__main__':
    main()
