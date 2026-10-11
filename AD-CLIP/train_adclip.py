"""
AD-CLIP (Adapting Domains in Prompt Space Using CLIP) adapted for JMDA-Net.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天
Backbone: CLIP RN50 (cached locally)

Usage:
    python train_adclip.py --target backlight --epochs 3
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add AD-CLIP path to import its clip module and components
ADCLIP_DIR = os.path.dirname(__file__)
sys.path.insert(0, ADCLIP_DIR)

# Import AD-CLIP's clip package under an alias to avoid conflict with installed clip
import importlib.util
import importlib.machinery

# Load AD-CLIP's clip package (with __init__.py) as 'adclip_clip_pkg'
_spec = importlib.util.spec_from_file_location(
    'adclip_clip_pkg',
    os.path.join(ADCLIP_DIR, 'clip', '__init__.py'),
    submodule_search_locations=[os.path.join(ADCLIP_DIR, 'clip')]
)
adclip_clip_pkg = importlib.util.module_from_spec(_spec)
sys.modules['adclip_clip_pkg'] = adclip_clip_pkg
_spec.loader.exec_module(adclip_clip_pkg)

# Now load clip.py and model.py as submodules of adclip_clip_pkg
def _load_submodule(pkg, mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        f'adclip_clip_pkg.{mod_name}',
        os.path.join(ADCLIP_DIR, 'clip', file_name))
    mod = importlib.util.module_from_spec(spec)
    setattr(pkg, mod_name, mod)
    sys.modules[f'adclip_clip_pkg.{mod_name}'] = mod
    spec.loader.exec_module(mod)
    return mod

adclip_clip_model = _load_submodule(adclip_clip_pkg, 'model', 'model.py')
adclip_clip_clip = _load_submodule(adclip_clip_pkg, 'clip', 'clip.py')
# Fix the relative import in clip.py: `from .model import build_model`
adclip_clip_clip.model = adclip_clip_model
adclip_clip_clip.build_model = adclip_clip_model.build_model

clip = adclip_clip_clip
from adclip_model import (
    PromptLearner, TextEncoder, CustomCLIP, entropy_loss
)

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
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


def load_clip_rn50(device):
    """Load CLIP RN50 from local cache using AD-CLIP's modified model."""
    model, _ = clip.load('RN50', device='cpu')
    model.float()
    return model


class SimpleCfg:
    """Minimal config object for AD-CLIP components."""
    class INPUT:
        SIZE = [224, 224]

    class MODEL:
        class BACKBONE:
            NAME = 'RN50'
            PATH = os.path.expanduser('~/.cache/clip')


def evaluate(model, loader, device, classnames):
    """Evaluate on target domain (zero-shot style with learned prompts)."""
    model.eval()
    correct = 0
    total = 0
    # Build text features using classnames
    with torch.no_grad():
        # Use source prompts for evaluation (style mapping from source)
        # For simplicity, evaluate using the model's forward with target images
        # We need a "source" image to generate prompts; use the first batch
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Use the same images as both source and target for eval prompt generation
            _, target_probs, _, _, _, _ = model(imgs, imgs)
            pred = target_probs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Class names (14 classes)
    classnames = [f'class_{i}' for i in range(1, 15)]

    # CLIP transforms
    _, preprocess = clip.load('RN50', device='cpu')
    transform = preprocess

    # Load data
    src_list = os.path.join(args.data_dir, 'sunny_train.txt')
    tgt_train_list = os.path.join(args.data_dir, f'{args.target}_train.txt')
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')

    src_ds = ImageListDataset(src_list, transform=transform)
    tgt_ds = ImageListDataset(tgt_train_list, transform=transform)
    tgt_val_ds = ImageListDataset(tgt_val_list, transform=transform)

    print(f'Source train: {len(src_ds)} images')
    print(f'Target train: {len(tgt_ds)} images')
    print(f'Target val:   {len(tgt_val_ds)} images')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

    # Load CLIP model
    print('Loading CLIP RN50...')
    clip_model = load_clip_rn50(device)

    cfg = SimpleCfg()
    model = CustomCLIP(cfg, classnames, clip_model).to(device)

    # Freeze everything except prompt_learner
    for name, param in model.named_parameters():
        if 'prompt_learner' not in name:
            param.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable parameters: {sum(p.numel() for p in trainable_params)}')

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    entropy_criterion = entropy_loss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        n_batches = min(len(src_loader), len(tgt_loader))

        for i in range(n_batches):
            try:
                src_imgs, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_imgs, src_labels = next(src_iter)
            try:
                tgt_imgs, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_imgs, _ = next(tgt_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)

            source_logits, target_probs, source_domaintokens, source_style_mappingtokens, \
                source_text_features, target_text_features = model(src_imgs, tgt_imgs)

            # CE loss on source
            loss_ce = F.cross_entropy(source_logits, src_labels)

            # KL divergence between source and target text features
            source_textfeat = F.log_softmax(source_text_features, dim=1)
            target_textfeat = F.softmax(target_text_features, dim=1)
            loss_kl = kl_loss(source_textfeat, target_textfeat)

            # Style mapping loss (MSE between domain tokens and style mapping tokens)
            loss_smn = F.mse_loss(source_domaintokens, source_style_mappingtokens)

            # Entropy loss on target predictions
            loss_entropy = entropy_criterion(target_probs)

            loss = loss_ce + 0.1 * loss_smn + 0.01 * loss_entropy + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'loss_ce={loss_ce.item():.4f} loss_kl={loss_kl.item():.4f} '
                      f'loss_smn={loss_smn.item():.4f} loss_ent={loss_entropy.item():.4f}')

        acc = evaluate(model, tgt_val_loader, device, classnames)
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.prompt_learner.state_dict(),
                       os.path.join(args.save_dir, f'adclip_{args.target}_best.pth'))

    print(f'\nAD-CLIP smoke test finished. Best target val accuracy ({args.target}): {best_acc:.4f}')


if __name__ == '__main__':
    main()
