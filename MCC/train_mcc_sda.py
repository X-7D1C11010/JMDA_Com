"""
MCC-SDA: supervised-domain-adaptation variant of MCC (ECCV 2020, tllib
implementation) for the paper's SDA protocol. This is a NEW file; the
original UDA script train_mcc.py is left unchanged.

SDA protocol:
  * shared labeled target subset: data_lists/<target>_sda_<N>shot_seed<S>.txt;
  * the original Minimum Class Confusion mechanism (temperature, trade-off,
    lr schedule) is kept exactly as in the UDA script;
  * the ONLY addition is L_cls_target = w_cls_target * CE(y_t, labels_t) on
    the labeled target subset.

Usage:
    python train_mcc_sda.py --target backlight --epochs 3
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Transfer-Learning-Library'))
from tllib.self_training.mcc import MinimumClassConfusionLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.vision.models import resnet18

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from data_paths import remap_data_path
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


def get_train_transform():
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
    correct, total = 0, 0
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            output = classifier(images)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    classifier.train()
    return correct / total if total else 0.0


def train_epoch(train_source_iter, train_target_iter, classifier, mcc,
                optimizer, lr_scheduler, epoch, args, eff=None):
    classifier.train()
    for i in range(args.iters_per_epoch):
        if eff:
            eff.step_start()
        optimizer.zero_grad()
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, labels_t = next(train_target_iter)[:2]  # SDA: keep target labels
        x_s, x_t = x_s.to(device), x_t.to(device)
        labels_s, labels_t = labels_s.to(device), labels_t.to(device)

        if eff:
            eff.dataloader_done()

        y_s, f_s = classifier(x_s)
        y_t, f_t = classifier(x_t)
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = mcc(y_t)
        # SDA addition: L_cls_target on the labeled target subset
        cls_loss_target = F.cross_entropy(y_t, labels_t)
        loss = cls_loss + transfer_loss * args.trade_off + args.w_cls_target * cls_loss_target
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if eff:
            eff.step_end()

        if (i + 1) % args.print_freq == 0:
            acc = accuracy(y_s, labels_s)[0]
            print(f'  Epoch [{epoch}/{args.epochs}] iter [{i + 1}/{args.iters_per_epoch}] '
                  f'loss={loss.item():.4f} cls={cls_loss.item():.4f} '
                  f'mcc={transfer_loss.item():.4f} L_cls_target={cls_loss_target.item():.4f} '
                  f'src_acc={acc.item():.2f}')


def main():
    parser = argparse.ArgumentParser(description='MCC-SDA (tllib) for JMDA-Net dataset')
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('-b', '--batch-size', default=36, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--lr-gamma', default=0.001, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--temperature', default=2.5, type=float)
    parser.add_argument('--trade-off', default=1., type=float)
    parser.add_argument('--bottleneck-dim', default=256, type=int)
    parser.add_argument('-p', '--print-freq', default=20, type=int)
    parser.add_argument('--target-labels-per-class', type=int, default=10,
                        help='SDA label budget per class (selects the shared subset list)')
    parser.add_argument('--sda-seed', type=int, default=42,
                        help='Seed used by tools/make_sda_lists.py')
    parser.add_argument('--w-cls-target', type=float, default=1.0,
                        help='Weight of L_cls_target (the ONLY added loss)')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_sda'))
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
    tgt_list = os.path.join(
        args.data_dir,
        f'{args.target}_sda_{args.target_labels_per_class}shot_seed{args.sda_seed}.txt')
    assert os.path.isfile(tgt_list), (
        f'SDA subset list missing: {tgt_list}. '
        f'Run: python tools/make_sda_lists.py --n-per-class {args.target_labels_per_class} '
        f'--seed {args.sda_seed}')
    tgt_ds = ImageListDataset(tgt_list, train_tf)
    tgt_val_ds = ImageListDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'), val_tf)
    print(f'Source train: {len(src_ds)}, target LABELED subset (SDA): {len(tgt_ds)} '
          f'(budget {args.target_labels_per_class}/class), target val: {len(tgt_val_ds)}')

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
    classifier = ImageClassifier(backbone, 14, bottleneck_dim=args.bottleneck_dim).to(device)
    mcc = MinimumClassConfusionLoss(temperature=args.temperature)
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    best_acc = 0.
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'MCC-SDA',
                                seed=args.seed, device=str(device), modalities='VIS',
                                protocol='SDA',
                                target_label_count=len(tgt_ds),
                                target_labels_per_class=args.target_labels_per_class,
                                backbone='ResNet-18', pretraining='ImageNet')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(1, args.epochs + 1):
        if eff:
            eff.epoch_start(epoch)
        train_epoch(train_source_iter, train_target_iter, classifier, mcc,
                    optimizer, lr_scheduler, epoch, args, eff=eff)
        acc = validate(tgt_val_loader, classifier)
        best_acc = max(best_acc, acc)
        torch.save(classifier.state_dict(), os.path.join(args.save_dir, f'mcc_sda_{args.target}_best.pth'))
        print(f'Epoch [{epoch}/{args.epochs}] target val accuracy: {acc:.4f}')
        if eff:
            eff.epoch_end(epoch)

    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'target': args.target,
                                    'protocol': 'SDA'})
    print(f'\nMCC-SDA finished. Best target val accuracy ({args.target}): {best_acc:.4f}')


if __name__ == '__main__':
    main()
