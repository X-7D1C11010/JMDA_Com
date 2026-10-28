"""
DANN (Domain-Adversarial Neural Network) adapted for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

Usage:
    python train_dann.py --target backlight

2026-10-20 变更（见 REPRODUCTION_RECORD.md 第 14 节）：
  * 标签空间以目标域为准（tools/uda_data.py）：只保留目标域存在的类别，分类头
    输出维度 = 该目标域类别数；
  * 网络直接使用原参考实现 DANN/models/model.py 的 MNIST CNN；仅把末层
    的 10 类改为当前目标域的类别数，这是数据集接口所必需的变化。
  * 优化器、batch=128、100 epochs、NLLLoss 与逐迭代 alpha 调度均保持
    DANN/train/main.py 的官方设置。

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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Use the original DANN gradient reversal layer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from model import CNNModel

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
# UDA 共用数据加载：以目标域类别集为准裁剪源域/目标域并重映射标签
from uda_data import UdaImageListDataset, uda_label_space, format_class_space, format_dropped
from data_loader_options import (
    add_data_loader_args,
    data_loader_kwargs,
    evaluation_loader_kwargs,
)


def get_transforms():
    # The official network has a fixed 50*4*4 flattening layer and therefore
    # requires 28x28 inputs. Both weather domains are RGB, so the MNIST-M
    # normalization is applied consistently to source and target.
    train_tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数（随附 DANN 参考实现默认 100）')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='官方 DANN/train/main.py 设置为 128')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists_vis'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    add_data_loader_args(parser, default_workers=8)
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

    loader_kwargs = data_loader_kwargs(args)
    eval_loader_kwargs = evaluation_loader_kwargs(args)
    print(f'DataLoader settings: train={loader_kwargs}, eval={eval_loader_kwargs}')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=False, **loader_kwargs)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=False, **loader_kwargs)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, **eval_loader_kwargs)

    model = CNNModel(num_classes=num_classes).to(device)
    print('DANN architecture: official DANN/models/model.py CNNModel (28x28 input)')
    # 原参考实现 DANN/train/main.py:76 使用 Adam(lr=1e-3)（无 momentum/weight_decay）
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss()

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
