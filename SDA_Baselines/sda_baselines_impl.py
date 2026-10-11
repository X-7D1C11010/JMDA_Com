"""
SDA protocol baselines for the JMDA paper (NEW folder; no original file is
modified). This module holds the shared implementation; the three entry
scripts are:

  * train_source_only_sda.py : Source-only — train on the labeled source
    (sunny) only, never touch target data or labels. Identical in the UDA and
    SDA tracks; reported in the SDA main table for reference.
  * train_joint_training.py  : Joint Training — CE on source + CE on the
    shared labeled target subset, NO domain-alignment module of any kind.
  * train_target_only.py     : Target-only Oracle — train on the shared
    labeled target subset only (same label budget as every SDA method; use
    --full-target for a full-target-label reference run with a DIFFERENT
    label budget, reported separately).

All three use the same ResNet-18 backbone / optimizer / transforms as the
reproduced baselines.

Usage:
    python train_source_only_sda.py --target backlight --epochs 3
    python train_joint_training.py  --target backlight --epochs 3
    python train_target_only.py     --target backlight --epochs 3
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
_TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from data_paths import remap_data_path


class ImageListDataset(Dataset):
    def __init__(self, list_path, transform=None):
        self.samples = []
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                self.samples.append((remap_data_path(parts[0]), int(parts[1])))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class ResNetClassifier(nn.Module):
    """Plain ResNet-18 classifier (same backbone as the reproduced baselines)."""

    def __init__(self, num_classes=14):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.net = resnet

    def forward(self, x):
        return self.net(x)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def shared_args(mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--target-labels-per-class', type=int, default=10,
                        help='SDA label budget per class (selects the shared subset list)')
    parser.add_argument('--sda-seed', type=int, default=42,
                        help='Seed used by tools/make_sda_lists.py')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    if mode == 'target_only':
        parser.add_argument('--full-target', action='store_true',
                            help='Train on the FULL target train set (oracle reference '
                                 'with a different label budget; reported separately).')
    return parser.parse_args()


def sda_list_path(args, target):
    path = os.path.join(
        args.data_dir,
        f'{target}_sda_{args.target_labels_per_class}shot_seed{args.sda_seed}.txt')
    assert os.path.isfile(path), (
        f'SDA subset list missing: {path}. '
        f'Run: python tools/make_sda_lists.py --n-per-class {args.target_labels_per_class} '
        f'--seed {args.sda_seed}')
    return path


def run(mode, args, train_lists, eff_name):
    """mode: 'source_only' | 'joint' | 'target_only'."""
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Protocol: SDA | Mode: {mode}')

    train_tf, val_tf = get_transforms()
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')
    tgt_val_ds = ImageListDataset(tgt_val_list, transform=val_tf)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    loaders = []
    label_counts = []
    for name, path in train_lists:
        ds = ImageListDataset(path, transform=train_tf)
        print(f'{name}: {len(ds)} images ({path})')
        label_counts.append(len(ds))
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, drop_last=len(ds) >= args.batch_size))

    model = ResNetClassifier(num_classes=14).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, eff_name,
                                seed=args.seed, device=str(device), modalities='VIS',
                                protocol='SDA',
                                target_label_count=label_counts[-1],
                                target_labels_per_class=(
                                    args.target_labels_per_class
                                    if mode != 'source_only' else 0),
                                backbone='ResNet-18', pretraining='ImageNet')
        print(f'Efficiency logging -> {eff.run_dir}')

    # Primary loader drives the epoch; extra loaders are cycled (joint mode).
    primary = loaders[0]
    extra_iters = [iter(l) for l in loaders[1:]]

    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        for i, (imgs, labels) in enumerate(primary):
            if eff:
                eff.step_start()
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)

            if mode == 'joint':
                for j, it in enumerate(extra_iters):
                    try:
                        t_imgs, t_labels = next(it)
                    except StopIteration:
                        extra_iters[j] = iter(loaders[j + 1])
                        t_imgs, t_labels = next(extra_iters[j])
                    t_imgs, t_labels = t_imgs.to(device), t_labels.to(device)
                    loss = loss + criterion(model(t_imgs), t_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 20 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{len(primary)}] '
                      f'loss={loss.item():.4f}')

        acc = evaluate(model, tgt_val_loader, device)
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'{mode}_{args.target}_best.pth'))
        if eff:
            eff.epoch_end(epoch)

    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'target': args.target,
                                    'protocol': 'SDA'})
    print(f'\n{eff_name} finished. Best target val accuracy ({args.target}): {best_acc:.4f}')
    return best_acc
