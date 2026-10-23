"""
DANN (Domain-Adversarial Neural Network) adapted for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

Usage:
    python train_dann.py --target backlight --epochs 3

2026-10-20 变更（见 REPRODUCTION_RECORD.md 第 14 节）：
  * 标签空间以目标域为准（tools/uda_data.py）：只保留目标域存在的类别，分类头
    输出维度 = 该目标域类别数；
  * 优化器改为原参考实现 DANN/train/main.py 的 Adam(lr=1e-3)。

2026-10-21 修复（逐项核对原参考实现 DANN/train/main.py 后）：
  * 对抗系数 α 恢复 main.py:100-101 的**逐迭代**调度：
    p = float(i + epoch·len_dataloader) / n_epoch / len_dataloader，
    α = 2/(1+exp(−10p)) − 1（原实现每个 epoch 只算一次且首轮即 p=1/epochs）；
  * 每轮步数恢复 main.py:93 的 min(len(source), len(target))，不再对较短
    一侧循环取样。
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Use the original DANN gradient reversal layer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from functions import ReverseLayerF

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
# UDA 共用数据加载：以目标域类别集为准裁剪源域/目标域并重映射标签
from uda_data import UdaImageListDataset, uda_label_space, format_class_space, format_dropped


class DANN(nn.Module):
    """DANN model with ResNet-18 backbone."""

    def __init__(self, num_classes=14):
        super(DANN, self).__init__()
        # Feature extractor: ResNet-18 (remove final FC)
        resnet = models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features  # 512

        # Label classifier
        self.class_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x, alpha=1.0):
        feature = self.feature(x).view(x.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output


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


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            class_out, _ = model(imgs, alpha=0.0)
            pred = class_out.argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(pred.cpu().tolist())
    return classification_metrics(all_labels, all_predictions, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists_vis'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.seed = seed_everything(args.seed)
    print(f'Controlled random seed: {args.seed}')

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_tf, val_tf = get_transforms()

    # Load source (sunny) and target domain data.
    # 标签空间以目标域为准：只保留目标域出现的类别，并把标签压到 0..|T|-1。
    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    src_list = os.path.join(args.data_dir, 'sunny_train.txt')
    tgt_train_list = os.path.join(args.data_dir, f'{args.target}_train.txt')
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')

    src_ds = UdaImageListDataset(src_list, transform=train_tf, label_map=label_map)
    tgt_ds = UdaImageListDataset(tgt_train_list, transform=train_tf, label_map=label_map)
    tgt_val_ds = UdaImageListDataset(tgt_val_list, transform=val_tf, label_map=label_map)

    print(format_class_space(class_ids, num_classes))
    print(f'Source train: {len(src_ds)} images')
    print(f'Target train: {len(tgt_ds)} images')
    print(f'Target val:   {len(tgt_val_ds)} images')
    print(format_dropped('source train', src_ds))

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    model = DANN(num_classes=num_classes).to(device)
    # 原参考实现 DANN/train/main.py:76 使用 Adam(lr=1e-3)（无 momentum/weight_decay）
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -float('inf')
    final_acc = None
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'DANN',
                                seed=args.seed, device=str(device), modalities='VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        # 原参考实现 main.py:93：len_dataloader = min(len(source), len(target))，
        # 每轮新建迭代器、不循环取样。
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        n_batches = min(len(src_loader), len(tgt_loader))

        for i in range(n_batches):
            # 原参考实现 main.py:100-101：逐迭代调度 α（0 基 i 与 epoch）
            p = float(i + epoch * n_batches) / args.epochs / n_batches
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            if eff:
                eff.step_start()
            src_imgs, src_labels = next(src_iter)
            tgt_imgs, _ = next(tgt_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)

            if eff:
                eff.dataloader_done()

            # Source classification loss
            src_class_out, src_domain_out = model(src_imgs, alpha)
            loss_cls = criterion(src_class_out, src_labels)

            # Domain adversarial loss
            src_domain_label = torch.zeros(src_imgs.size(0), dtype=torch.long).to(device)
            tgt_domain_label = torch.ones(tgt_imgs.size(0), dtype=torch.long).to(device)
            _, tgt_domain_out = model(tgt_imgs, alpha)

            loss_domain = criterion(src_domain_out, src_domain_label) + \
                          criterion(tgt_domain_out, tgt_domain_label)

            loss = loss_cls + loss_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'loss_cls={loss_cls.item():.4f} loss_domain={loss_domain.item():.4f} '
                      f'alpha={alpha:.4f}')

        # Evaluate on target val
        val_metrics = evaluate(model, tgt_val_loader, device, num_classes)
        acc = val_metrics['accuracy']
        final_acc = acc
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        print(format_classification_metrics(val_metrics))
        best_acc = max(best_acc, acc)
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'dann_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': final_acc,
                                    'diagnostic_peak_acc': best_acc,
                                    'target': args.target})
    print(f'Final target val accuracy ({args.target}): {final_acc:.4f}')
    print(f'Diagnostic peak target val accuracy (not used for model selection): {best_acc:.4f}')


if __name__ == '__main__':
    main()
