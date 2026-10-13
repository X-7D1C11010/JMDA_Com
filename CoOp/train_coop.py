"""
CoOp (Context Optimization) adapted for JMDA-Net dataset.
VLM domain adaptation via prompt learning.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天
Backbone: CLIP RN50

Usage:
    python train_coop.py --target backlight --epochs 3
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

COOP_DIR = os.path.dirname(__file__)
sys.path.insert(0, COOP_DIR)

# Use the standard clip package (CoOp uses unmodified CLIP)
import clip as clip_pkg
from coop_model import CustomCLIP

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from class_names import load_class_names
from classification_metrics import classification_metrics, format_classification_metrics
from data_paths import remap_data_path


class SimpleCfg:
    class INPUT:
        SIZE = [224, 224]

    class MODEL:
        class BACKBONE:
            NAME = 'RN50'
        INIT_WEIGHTS = ''

    class TRAINER:
        class COOP:
            N_CTX = 16
            CTX_INIT = ''
            CSC = False
            CLASS_TOKEN_POSITION = 'end'
            PREC = 'fp32'


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


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            pred = logits.argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(pred.cpu().tolist())
    return classification_metrics(all_labels, all_predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument(
        '--class-names-file', type=str,
        default=os.environ.get('JMDA_CLASS_NAMES_FILE'),
        help='14 real English class names in zero-based label order',
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load CLIP RN50
    print('Loading CLIP RN50...')
    clip_model, preprocess = clip_pkg.load('RN50', device='cpu')
    clip_model.float()

    classnames = load_class_names(args.class_names_file, expected_count=14)
    print(f'Class names: {classnames}')

    # Build CoOp model
    cfg = SimpleCfg()
    model = CustomCLIP(cfg, classnames, clip_model).to(device)

    # Freeze everything except prompt_learner
    for name, param in model.named_parameters():
        if 'prompt_learner' not in name:
            param.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable parameters: {sum(p.numel() for p in trainable_params)}')

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Load data
    src_list = os.path.join(args.data_dir, 'sunny_train.txt')
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')

    src_ds = ImageListDataset(src_list, transform=preprocess)
    tgt_val_ds = ImageListDataset(tgt_val_list, transform=preprocess)

    print(f'Source train: {len(src_ds)} images')
    print(f'Target val:   {len(tgt_val_ds)} images')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

    best_acc = -float('inf')
    final_acc = None
    for epoch in range(args.epochs):
        model.train()
        for i, (imgs, labels) in enumerate(src_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                acc = (logits.argmax(1) == labels).float().mean().item()
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{len(src_loader)}] '
                      f'loss={loss.item():.4f} acc={acc:.4f}')

        val_metrics = evaluate(model, tgt_val_loader, device)
        acc = val_metrics['accuracy']
        final_acc = acc
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        print(format_classification_metrics(val_metrics))
        best_acc = max(best_acc, acc)

    torch.save(
        {
            'prompt_learner': model.prompt_learner.state_dict(),
            'class_names': classnames,
            'target': args.target,
            'seed': args.seed,
            'epoch': args.epochs,
            'final_acc': final_acc,
            'final_metrics': val_metrics,
        },
        os.path.join(args.save_dir, f'coop_{args.target}_last.pth'),
    )
    print(f'Final target val accuracy ({args.target}): {final_acc:.4f}')
    print(f'Diagnostic peak target val accuracy (not used for model selection): {best_acc:.4f}')


if __name__ == '__main__':
    main()
