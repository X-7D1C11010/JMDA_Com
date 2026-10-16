"""
MDD (Margin Disparity Discrepancy, ICML 2019) adapted for JMDA-Net dataset,
using the modern PyTorch implementation from thuml/Transfer-Learning-Library
(tllib). Training loop faithfully follows the official example:
    examples/domain_adaptation/image_classification/mdd.py
Source domain: 晴天 (Sunny); Target domains: 逆光/雨天/雾天/黑天.

Adaptations (documented in REPRODUCTION_RECORD.md):
  * dataset replaced by our ImageList-based paired loader (same protocol as the
    other reproduced methods);
  * iters_per_epoch = max(len(src_loader), len(tgt_loader)) instead of 1000;
  * ResizeImage implemented locally (tllib.vision.transforms not downloaded);
  * Windows: DataLoader num_workers=0.

Usage:
    python train_mdd.py --target backlight --epochs 3
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
from data_paths import remap_data_path
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageListDataset(torch.utils.data.Dataset):
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


def validate(val_loader, classifier):
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
    return classification_metrics(all_labels, all_predictions)


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
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--margin', type=float, default=4.)
    parser.add_argument('--trade-off', default=1., type=float)
    parser.add_argument('--bottleneck-dim', default=256, type=int)
    parser.add_argument('-p', '--print-freq', default=20, type=int)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
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
    src_ds = ImageListDataset(os.path.join(args.data_dir, 'sunny_train.txt'), train_tf)
    tgt_ds = ImageListDataset(os.path.join(args.data_dir, f'{args.target}_train.txt'), train_tf)
    tgt_val_ds = ImageListDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'), val_tf)
    print(f'Source train: {len(src_ds)}, target train: {len(tgt_ds)}, target val: {len(tgt_val_ds)}')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
    train_source_iter = ForeverDataIterator(src_loader)
    train_target_iter = ForeverDataIterator(tgt_loader)
    args.iters_per_epoch = max(len(src_loader), len(tgt_loader))

    backbone = resnet18(pretrained=True)
    classifier = ImageClassifier(backbone, 14, bottleneck_dim=args.bottleneck_dim,
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
        val_metrics = validate(tgt_val_loader, classifier)
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
