"""
Deep CORAL (Correlation Alignment) adapted for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

Usage:
    python train_deepcoral.py --target backlight --epochs 3
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

# Import CORAL loss from original repo
sys.path.insert(0, os.path.dirname(__file__))
from models import CORAL

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


class DeepCORAL(nn.Module):
    """Deep CORAL with ResNet-18 backbone."""

    def __init__(self, num_classes=14):
        super(DeepCORAL, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.sharedNet = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features
        self.fc = nn.Linear(feat_dim, num_classes)
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target=None):
        source_feat = self.sharedNet(source).view(source.size(0), -1)
        source_out = self.fc(source_feat)
        if target is not None:
            target_feat = self.sharedNet(target).view(target.size(0), -1)
            target_out = self.fc(target_feat)
            return source_out, target_out, source_feat, target_feat
        return source_out


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
            out = model(imgs)
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lamb', type=float, default=1.0, help='CORAL loss weight')
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

    model = DeepCORAL(num_classes=14).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'DeepCORAL',
                                seed=args.seed, device=str(device), modalities='VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
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

            src_out, tgt_out, src_feat, tgt_feat = model(src_imgs, tgt_imgs)

            loss_cls = criterion(src_out, src_labels)
            loss_coral = CORAL(src_feat, tgt_feat)
            loss = loss_cls + args.lamb * loss_coral

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'loss_cls={loss_cls.item():.4f} loss_coral={loss_coral.item():.4f}')

        acc = evaluate(model, tgt_val_loader, device)
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'deepcoral_{args.target}_best.pth'))
        if eff:
            eff.epoch_end(epoch)

    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'target': args.target})
    print(f'\nDeep CORAL smoke test finished. Best target val accuracy ({args.target}): {best_acc:.4f}')


if __name__ == '__main__':
    main()
