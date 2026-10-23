"""
MDD (Margin Disparity Discrepancy, ICML 2019) adapted for JMDA-Net dataset,
using the modern PyTorch implementation from thuml/Transfer-Learning-Library
(tllib). Training loop faithfully follows the official example:
    examples/domain_adaptation/image_classification/mdd.py
Source domain: 晴天 (Sunny); Target domains: 逆光/雨天/雾天/黑天.

Adaptations (documented in REPRODUCTION_RECORD.md):
  * dataset replaced by the shared target-driven UDA loader
    (tools/uda_data.py: only target-domain classes are kept, labels remapped
    to 0..|T|-1, so the classifier head width equals that target's class count);
  * ResizeImage implemented locally (tllib.vision.transforms not downloaded);
  * Windows: DataLoader num_workers=0.

2026-10-21 修复（逐项核对 tllib 官方 mdd.py 后）：
  * bottleneck_dim 恢复官方默认 1024（mdd.py:227），原为 256；
  * iters_per_epoch 恢复官方默认 1000（mdd.py:249），不再用数据加载器长度
    （原写法每轮仅 ~12–16 次更新，比官方 20×1000 少约两个数量级）。

优化器/调度/超参保持与 tllib 官方示例 mdd.py 完全一致（未做任何改动）。

Usage:
    python train_mdd.py --target backlight
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Transfer-Learning-Library'))
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy \
    as MarginDisparityDiscrepancy, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.vision.models import resnet18

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
# UDA 共用数据加载：以目标域类别集为准裁剪源域/目标域并重映射标签
from uda_data import UdaImageListDataset, uda_label_space, format_class_space, format_dropped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResizeImage:
    """Same semantics as tllib.vision.transforms.ResizeImage: resize to size x size."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size, self.size), Image.BILINEAR)


def get_train_transform(resize_size=224):
    return T.Compose([
        ResizeImage(256),
        T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_val_transform():
    return T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def validate(val_loader, classifier, num_classes):
    classifier.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            output = classifier(images)
            pred = output.argmax(dim=1)
            all_labels.extend(target.cpu().tolist())
            all_predictions.extend(pred.cpu().tolist())
    classifier.train()
    return classification_metrics(all_labels, all_predictions, num_classes=num_classes)


def train_epoch(train_source_iter, train_target_iter, classifier, mdd,
                optimizer, lr_scheduler, epoch, args, eff=None):
    classifier.train()
    mdd.train()
    for i in range(args.iters_per_epoch):
        if eff:
            eff.step_start()
        optimizer.zero_grad()
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]
        x_s, x_t = x_s.to(device), x_t.to(device)
        labels_s = labels_s.to(device)

        if eff:
            eff.dataloader_done()

        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = -mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss * args.trade_off
        classifier.step()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if eff:
            eff.step_end()

        if (i + 1) % args.print_freq == 0:
            acc = accuracy(y_s, labels_s)[0]
            print(f'  Epoch [{epoch}/{args.epochs}] iter [{i + 1}/{args.iters_per_epoch}] '
                  f'loss={loss.item():.4f} cls={cls_loss.item():.4f} '
                  f'mdd={transfer_loss.item():.4f} src_acc={acc.item():.2f}')


def main():
    parser = argparse.ArgumentParser(description='MDD (tllib) for JMDA-Net dataset')
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数（tllib 官方默认 20）')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--margin', type=float, default=4.)
    parser.add_argument('--trade-off', default=1., type=float)
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('-p', '--print-freq', default=100, type=int)
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch (tllib official default: 1000)')
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
    print(args)

    train_tf, val_tf = get_train_transform(), get_val_transform()
    # 标签空间以目标域为准：只保留目标域出现的类别，并把标签压到 0..|T|-1。
    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    src_ds = UdaImageListDataset(os.path.join(args.data_dir, 'sunny_train.txt'),
                                 train_tf, label_map)
    tgt_ds = UdaImageListDataset(os.path.join(args.data_dir, f'{args.target}_train.txt'),
                                 train_tf, label_map)
    tgt_val_ds = UdaImageListDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'),
                                     val_tf, label_map)
    print(format_class_space(class_ids, num_classes))
    print(f'Source train: {len(src_ds)}, target train: {len(tgt_ds)}, target val: {len(tgt_val_ds)}')
    print(format_dropped('source train', src_ds))

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
    train_source_iter = ForeverDataIterator(src_loader)
    train_target_iter = ForeverDataIterator(tgt_loader)

    backbone = resnet18(pretrained=True)
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 width=args.bottleneck_dim).to(device)
    mdd = MarginDisparityDiscrepancy(args.margin).to(device)
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    best_acc = -float('inf')
    final_acc = None
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'MDD',
                                seed=args.seed, device=str(device), modalities='VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(1, args.epochs + 1):
        if eff:
            eff.epoch_start(epoch)
        train_epoch(train_source_iter, train_target_iter, classifier, mdd,
                    optimizer, lr_scheduler, epoch, args, eff=eff)
        val_metrics = validate(tgt_val_loader, classifier, num_classes)
        acc = val_metrics['accuracy']
        final_acc = acc
        best_acc = max(best_acc, acc)
        print(f'Epoch [{epoch}/{args.epochs}] target val accuracy: {acc:.4f}')
        print(format_classification_metrics(val_metrics))
        if eff:
            eff.epoch_end(epoch)

    torch.save(classifier.state_dict(), os.path.join(args.save_dir, f'mdd_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': final_acc,
                                    'diagnostic_peak_acc': best_acc,
                                    'target': args.target})
    print(f'Final target val accuracy ({args.target}): {final_acc:.4f}')
    print(f'Diagnostic peak target val accuracy (not used for model selection): {best_acc:.4f}')


if __name__ == '__main__':
    main()
