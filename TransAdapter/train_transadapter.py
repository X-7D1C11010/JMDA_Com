"""
TransAdapter (Vision Transformer for Unsupervised Domain Adaptation) adapted for JMDA-Net.
Uses Swin Transformer backbone with adversarial domain alignment.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

Usage:
    python train_transadapter.py --target backlight --epochs 3
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Function
from PIL import Image
import timm

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from data_paths import remap_data_path


SWIN_MODEL_NAME = 'swin_tiny_patch4_window7_224'


def build_swin_backbone(pretrained=True, weights_path=None):
    """Build Swin-T without making Hugging Face the only weight source.

    Newer timm releases prefer Hugging Face even though this model also has an
    official Swin release URL.  The training server cannot reach Hugging Face,
    so use a caller-provided checkpoint first, then timm's normal cache/source,
    and finally retry against the official Swin GitHub release.  Formal runs
    must not silently fall back to random initialization.
    """
    common = dict(num_classes=0)
    if not pretrained:
        return timm.create_model(SWIN_MODEL_NAME, pretrained=False, **common)

    weights_path = (
        weights_path
        or os.environ.get('TRANSADAPTER_BACKBONE_WEIGHTS', '')
    ).strip()
    if weights_path:
        weights_path = os.path.abspath(os.path.expanduser(weights_path))
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f'TransAdapter backbone checkpoint not found: {weights_path}')
        print(f'Loading TransAdapter Swin-T weights: {weights_path}')
        model = timm.create_model(
            SWIN_MODEL_NAME, pretrained=False, **common)
        # Official Swin checkpoints include the 1000-class classification
        # head, while this feature extractor is built with num_classes=0.
        # Non-strict loading correctly ignores that unused head.
        timm.models.load_checkpoint(model, weights_path, strict=False)
        return model

    errors = []
    try:
        return timm.create_model(SWIN_MODEL_NAME, pretrained=True, **common)
    except Exception as exc:
        errors.append(f'timm/HuggingFace: {type(exc).__name__}: {exc}')
        print('[WARN] timm/Hugging Face pretrained loading failed; '
              'retrying the official Swin GitHub release.')

    try:
        # Removing hf_hub_id makes timm use the official URL already stored in
        # this model's pretrained configuration.
        return timm.create_model(
            SWIN_MODEL_NAME,
            pretrained=True,
            pretrained_cfg_overlay={'hf_hub_id': None},
            **common,
        )
    except Exception as exc:
        errors.append(f'official Swin URL: {type(exc).__name__}: {exc}')

    details = '\n  - '.join(errors)
    raise RuntimeError(
        'Unable to load ImageNet-pretrained Swin-T weights for TransAdapter. '
        'Download swin_tiny_patch4_window7_224.pth on a connected machine, '
        'upload it to the server, and set TRANSADAPTER_BACKBONE_WEIGHTS to '
        'that file. Random initialization is intentionally not used because '
        f'it would make the formal comparison unfair.\n  - {details}'
    )


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


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


class TransAdapter(nn.Module):
    """TransAdapter: Swin Transformer + adversarial domain adaptation."""

    def __init__(self, num_classes=14, pretrained=True, weights_path=None):
        super(TransAdapter, self).__init__()
        self.backbone = build_swin_backbone(
            pretrained=pretrained, weights_path=weights_path)
        feat_dim = self.backbone.num_features  # 768

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Domain discriminator (adversarial)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        class_out = self.classifier(features)
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_out = self.domain_discriminator(reverse_features)
        return class_out, domain_out, features


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
            class_out, _, _ = model(imgs, alpha=0.0)
            pred = class_out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--backbone-weights', type=str,
        default=os.environ.get('TRANSADAPTER_BACKBONE_WEIGHTS'),
        help='Optional local Swin-T ImageNet checkpoint. Also configurable '
             'through TRANSADAPTER_BACKBONE_WEIGHTS.',
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_tf, val_tf = get_transforms()

    src_list = os.path.join(args.data_dir, 'sunny_train.txt')
    tgt_train_list = os.path.join(args.data_dir, f'{args.target}_train.txt')
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')

    src_ds = ImageListDataset(src_list, transform=train_tf)
    tgt_ds = ImageListDataset(tgt_train_list, transform=train_tf)
    tgt_val_ds = ImageListDataset(tgt_val_list, transform=val_tf)

    print(f'Source train: {len(src_ds)} images')
    print(f'Target train: {len(tgt_ds)} images')
    print(f'Target val:   {len(tgt_val_ds)} images')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    model = TransAdapter(
        num_classes=14, pretrained=True,
        weights_path=args.backbone_weights,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'TransAdapter',
                                seed=args.seed, device=str(device), modalities='VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        p = (epoch + 1) / args.epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        n_batches = min(len(src_loader), len(tgt_loader))

        for i in range(n_batches):
            if eff:
                eff.step_start()
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

            if eff:
                eff.dataloader_done()

            # Source classification
            src_class_out, src_domain_out, _ = model(src_imgs, alpha)
            loss_cls = criterion(src_class_out, src_labels)

            # Domain adversarial
            src_domain_label = torch.zeros(src_imgs.size(0), dtype=torch.long).to(device)
            tgt_domain_label = torch.ones(tgt_imgs.size(0), dtype=torch.long).to(device)
            _, tgt_domain_out, _ = model(tgt_imgs, alpha)

            loss_domain = criterion(src_domain_out, src_domain_label) + \
                          criterion(tgt_domain_out, tgt_domain_label)

            loss = loss_cls + loss_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 5 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'loss_cls={loss_cls.item():.4f} loss_domain={loss_domain.item():.4f}')

        acc = evaluate(model, tgt_val_loader, device)
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'transadapter_{args.target}_best.pth'))
        if eff:
            eff.epoch_end(epoch)

    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'target': args.target})
    print(f'\nTransAdapter smoke test finished. Best target val accuracy ({args.target}): {best_acc:.4f}')


if __name__ == '__main__':
    main()
