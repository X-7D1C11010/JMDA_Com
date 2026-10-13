"""
CDAN (Conditional Domain Adversarial Network) adapted for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

Usage:
    python train_cdan.py --target backlight --epochs 3
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Import CDAN components from original repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
from network import ResNetFc, AdversarialNetwork, RandomLayer, calc_coeff
from loss import CDAN, Entropy

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from data_paths import remap_data_path
from classification_metrics import classification_metrics, format_classification_metrics


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
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, out = model(imgs)
            pred = out.argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(pred.cpu().tolist())
    return classification_metrics(all_labels, all_predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--method', type=str, default='CDAN',
                        choices=['CDAN', 'CDAN+E'])
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
    print(f'Device: {device}, Method: {args.method}')

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

    num_classes = 14
    # ResNet-18 backbone with bottleneck
    model = ResNetFc(resnet_name='ResNet18', use_bottleneck=True,
                     bottleneck_dim=256, new_cls=True, class_num=num_classes).to(device)
    # When using RandomLayer, ad_net input dim = random_layer output_dim
    random_layer = RandomLayer([256, num_classes], output_dim=1024)
    if torch.cuda.is_available():
        random_layer.cuda()
    ad_net = AdversarialNetwork(in_feature=1024, hidden_size=1024).to(device)

    optimizer = optim.SGD(model.get_parameters() + ad_net.get_parameters(),
                          lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = -float('inf')
    final_acc = None
    iter_num = 0
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'CDAN',
                                seed=args.seed, device=str(device), modalities='VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        ad_net.train()
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        n_batches = max(len(src_loader), len(tgt_loader))

        for i in range(n_batches):
            iter_num += 1
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

            # Concatenate source and target
            inputs = torch.cat((src_imgs, tgt_imgs), dim=0)
            features, outputs = model(inputs)
            softmax_out = nn.Softmax(dim=1)(outputs)

            # Classification loss on source
            loss_cls = criterion(outputs.narrow(0, 0, src_imgs.size(0)), src_labels)

            # CDAN loss
            if args.method == 'CDAN+E':
                entropy = Entropy(softmax_out)
                loss_transfer = CDAN([features, softmax_out], ad_net, entropy,
                                     calc_coeff(iter_num), random_layer=random_layer)
            else:
                loss_transfer = CDAN([features, softmax_out], ad_net,
                                     random_layer=random_layer)

            loss = loss_cls + loss_transfer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'loss_cls={loss_cls.item():.4f} loss_transfer={loss_transfer.item():.4f}')

        val_metrics = evaluate(model, tgt_val_loader, device)
        acc = val_metrics['accuracy']
        final_acc = acc
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        print(format_classification_metrics(val_metrics))
        best_acc = max(best_acc, acc)
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'cdan_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': final_acc,
                                    'diagnostic_peak_acc': best_acc,
                                    'target': args.target})
    print(f'Final target val accuracy ({args.target}): {final_acc:.4f}')
    print(f'Diagnostic peak target val accuracy (not used for model selection): {best_acc:.4f}')


if __name__ == '__main__':
    main()
