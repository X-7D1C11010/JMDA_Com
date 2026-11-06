"""
MCC (Minimum Class Confusion, ECCV 2020) adapted for JMDA-Net dataset,
using the modern PyTorch implementation from thuml/Transfer-Learning-Library
(tllib). Training loop faithfully follows the official example:
    examples/domain_adaptation/image_classification/mcc.py
Source domain: 晴天 (Sunny); Target domains: 逆光/雨天/雾天/黑天.

Adaptations (documented in REPRODUCTION_RECORD.md):
  * dataset replaced by the shared target-driven UDA loader
    (tools/uda_data.py: only target-domain classes are kept, labels remapped
    to 0..|T|-1, so the classifier head width equals that target's class count);
  * ResizeImage implemented locally (tllib.vision.transforms not downloaded);
  * DataLoader workers/prefetch/pinned memory 可配置；
  * 多模态使用 JMDA 非张量部分的 VIS ResNet-18 + IR U-Net，512+512 直接拼接；
  * 先纯源域监督预热，再逐步启用 MCC，避免随机分类头产生的低质量目标
    概率立即主导类别混淆矩阵。

2026-10-21 修复（逐项核对 tllib 官方 mcc.py 后）：
  * iters_per_epoch 恢复官方默认 1000（mcc.py:248），不再用数据加载器长度
    （原写法每轮仅 ~11–16 次更新，比官方 20×1000 少约两个数量级）；
  * 前向恢复官方 mcc.py:166-168 的 ``torch.cat((x_s, x_t))`` 一次前向 +
    ``y.chunk(2)``（分别前向在含 BatchNorm 的 ResNet 下与官方不等价）。

MCC 损失与联合前向保持 tllib 官方实现；源域预热是数据集迁移所需的显式阶段。

Usage:
    python train_mcc.py --target backlight
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Transfer-Learning-Library'))
from tllib.self_training.mcc import ImageClassifier, MinimumClassConfusionLoss
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.vision.models import resnet18, resnet50

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
from peak_timing import diagnose_peak_timing, format_peak_timing

# UDA 共用数据加载：以目标域类别集为准裁剪源域/目标域并重映射标签
from uda_data import (
    UdaImageListDataset,
    assert_training_label_space,
    format_class_space,
    format_dropped,
    uda_label_space,
)
from data_loader_options import (
    add_data_loader_args,
    data_loader_kwargs,
    evaluation_loader_kwargs,
)
from uda_multimodal import (
    JmdaTllibBackbone,
    configure_jmda_visual_finetuning,
    describe_pairing,
    jmda_train_transform,
    jmda_val_transform,
    label_free_target_paired_dataset,
    paired_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResizeImage:
    """Same semantics as tllib.vision.transforms.ResizeImage: resize to size x size."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size, self.size), Image.BILINEAR)


def get_train_transform(enhanced_augmentation=False):
    operations = [
        ResizeImage(256),
        T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        T.RandomHorizontalFlip(),
    ]
    if enhanced_augmentation:
        operations.append(T.ColorJitter(brightness=0.25, contrast=0.25,
                                        saturation=0.2, hue=0.05))
    operations += [T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return T.Compose(operations)


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


def train_epoch(train_source_iter, train_target_iter, classifier, mcc,
                optimizer, lr_scheduler, epoch, args, eff=None):
    classifier.train()
    cls_loss_sum = 0.0
    transfer_loss_sum = 0.0
    source_correct = 0
    source_samples = 0
    target_confidence_sum = 0.0
    target_entropy_sum = 0.0
    target_steps = 0
    target_class_counts = None
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

        source_warmup = epoch <= args.source_warmup_epochs
        if source_warmup:
            # Establish a source-trained classifier before MCC is allowed to
            # trust and sharpen its target predictions.
            y_s, _ = classifier(x_s)
            y_t = None
        else:
            # tllib official MCC joint forward once adaptation starts.
            x = torch.cat((x_s, x_t), dim=0)
            y, _ = classifier(x)
            y_s, y_t = y.chunk(2, dim=0)
        cls_loss = F.cross_entropy(
            y_s, labels_s, label_smoothing=args.label_smoothing
        )
        if source_warmup:
            transfer_loss = cls_loss.new_zeros(())
            transfer_weight = 0.0
        elif args.trade_off_warmup_epochs > 0:
            transfer_loss = mcc(y_t)
            completed_updates = ((epoch - args.source_warmup_epochs - 1)
                                 * args.iters_per_epoch + i + 1)
            warmup_updates = args.trade_off_warmup_epochs * args.iters_per_epoch
            transfer_weight = args.trade_off * min(1.0, completed_updates / warmup_updates)
        else:
            transfer_loss = mcc(y_t)
            transfer_weight = args.trade_off
        loss = cls_loss + transfer_loss * transfer_weight
        with torch.no_grad():
            cls_loss_sum += cls_loss.item()
            transfer_loss_sum += transfer_loss.item()
            source_correct += int(y_s.argmax(dim=1).eq(labels_s).sum().item())
            source_samples += int(labels_s.numel())
            if target_class_counts is None:
                target_class_counts = torch.zeros(
                    y_s.size(1), dtype=torch.long)
            if y_t is not None:
                target_probabilities = F.softmax(y_t, dim=1)
                target_confidence_sum += target_probabilities.max(
                    dim=1).values.mean().item()
                target_entropy_sum += (-(target_probabilities *
                                         target_probabilities.clamp_min(1e-8).log())
                                       .sum(dim=1).mean().item())
                target_steps += 1
                target_class_counts += torch.bincount(
                    target_probabilities.argmax(dim=1),
                    minlength=target_probabilities.size(1)).cpu()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if eff:
            eff.step_end()

        if (i + 1) % args.print_freq == 0 or i + 1 == args.iters_per_epoch:
            acc = accuracy(y_s, labels_s)[0]
            target_confidence = (float('nan') if y_t is None else
                                 F.softmax(y_t.detach(), dim=1).max(dim=1).values.mean().item())
            learning_rates = lr_scheduler.get_last_lr()
            print(f'  Epoch [{epoch}/{args.epochs}] iter [{i + 1}/{args.iters_per_epoch}] '
                  f'loss={loss.item():.4f} cls={cls_loss.item():.4f} '
                  f'mcc={transfer_loss.item():.4f} transfer_weight={transfer_weight:.4f} '
                  f'src_acc={acc.item():.2f} target_conf={target_confidence:.4f} '
                  f'lr=[{min(learning_rates):.3e},{max(learning_rates):.3e}]',
                  flush=True)
    return {
        'avg_cls': cls_loss_sum / args.iters_per_epoch,
        'avg_mcc': transfer_loss_sum / args.iters_per_epoch,
        'source_acc': source_correct / max(source_samples, 1),
        'target_conf': (target_confidence_sum / target_steps
                        if target_steps else float('nan')),
        'target_entropy': (target_entropy_sum / target_steps
                           if target_steps else float('nan')),
        'target_class_coverage': int((target_class_counts > 0).sum().item()),
        'target_class_hist': target_class_counts.tolist(),
        'transfer_weight': transfer_weight,
        'adaptation_steps': target_steps,
    }


def main():
    parser = argparse.ArgumentParser(description='MCC (tllib) for JMDA-Net dataset')
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数（tllib 官方默认 20）')
    parser.add_argument('-b', '--batch-size', default=36, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--lr-gamma', default=0.001, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--temperature', default=2.5, type=float)
    parser.add_argument('--trade-off', default=1., type=float)
    parser.add_argument('--trade-off-warmup-epochs', default=0, type=int,
                        help='linearly ramp MCC weight for N epochs after source warmup')
    parser.add_argument('--source-warmup-epochs', default=5, type=int,
                        help='pure source-supervised epochs before MCC target adaptation')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        help='source cross-entropy label smoothing; 0 preserves baseline')
    parser.add_argument('--bottleneck-dim', default=256, type=int)
    parser.add_argument('-p', '--print-freq', default=100, type=int)
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch (tllib official default: 1000)')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet50', 'jmda'],
                        default='resnet18')
    parser.add_argument('--enhanced-augmentation', action='store_true')
    parser.add_argument('--multimodal', action='store_true')
    parser.add_argument('--save-model', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--early-stop-patience', type=int, default=0,
                        help='target-val epochs without improvement before stopping; 0 disables')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists_vis'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    add_data_loader_args(parser, default_workers=2)
    args = parser.parse_args()
    if (args.iters_per_epoch < 1 or args.trade_off < 0 or
            args.trade_off_warmup_epochs < 0 or
            args.source_warmup_epochs < 0 or
            not 0 <= args.label_smoothing < 1):
        raise ValueError('invalid MCC iterations/trade-off/warmup setting')
    if args.early_stop_patience < 0:
        raise ValueError('--early-stop-patience must be >= 0')
    if args.early_stop_patience:
        raise ValueError('target-validation early stopping is disabled by this protocol')
    args.seed = seed_everything(args.seed)
    print(f'Controlled random seed: {args.seed}')
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    train_tf, val_tf = get_train_transform(args.enhanced_augmentation), get_val_transform()
    if args.multimodal:
        train_tf, val_tf = jmda_train_transform(), jmda_val_transform()
        print('Multimodal preprocessing: JMDA paired Resize(256) + '
              'RandomCrop(224) + HFlip; validation Resize(224)')
    # 标签空间以目标域为准：只保留目标域出现的类别，并把标签压到 0..|T|-1。
    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    src_list = os.path.join(args.data_dir, 'sunny_train.txt')
    tgt_train_list = os.path.join(args.data_dir, f'{args.target}_train.txt')
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')
    if args.multimodal:
        src_ds = paired_dataset(src_list, train_tf, label_map)
        tgt_ds = label_free_target_paired_dataset(
            tgt_train_list, train_tf, label_map)
        tgt_val_ds = paired_dataset(tgt_val_list, val_tf, label_map)
    else:
        src_ds = UdaImageListDataset(src_list, train_tf, label_map)
        tgt_ds = UdaImageListDataset(tgt_train_list, train_tf, label_map)
        tgt_val_ds = UdaImageListDataset(tgt_val_list, val_tf, label_map)
    assert_training_label_space(src_ds, tgt_ds, class_ids)
    print(format_class_space(class_ids, num_classes))
    print(f'Source train: {len(src_ds)}, target train: {len(tgt_ds)}, target val: {len(tgt_val_ds)}')
    print(format_dropped('source train', src_ds))
    if args.multimodal:
        print(describe_pairing('Source train', src_ds))
        print(describe_pairing('Target train', tgt_ds))
        print(describe_pairing('Target val', tgt_val_ds))

    loader_kwargs = data_loader_kwargs(args)
    eval_loader_kwargs = evaluation_loader_kwargs(args)
    print(f'DataLoader settings: train={loader_kwargs}, eval={eval_loader_kwargs}')
    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, **loader_kwargs)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, **loader_kwargs)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, **eval_loader_kwargs)
    train_source_iter = ForeverDataIterator(src_loader)
    train_target_iter = ForeverDataIterator(tgt_loader)

    if args.multimodal:
        if args.backbone != 'jmda':
            raise ValueError('--multimodal requires --backbone jmda')
        backbone = JmdaTllibBackbone()
        tuning = configure_jmda_visual_finetuning(
            backbone, mode='legacy_supervised')
        print('MCC encoder: JMDA non-tensor VIS ResNet-18 + IR U-Net, '
              '512+512 direct concatenation')
        print('JMDA VIS fine-tuning: projection only; frozen ResNet '
              'parameters with legacy train-mode BN; '
              f'VIS trainable={tuning["vis_trainable"]:,}/'
              f'{tuning["vis_total"]:,}; IR fully trainable')
    else:
        if args.backbone == 'jmda':
            raise ValueError('--backbone jmda requires --multimodal')
        backbone_fn = resnet50 if args.backbone == 'resnet50' else resnet18
        backbone = backbone_fn(pretrained=True)
        print(f'MCC backbone: {args.backbone} (ImageNet pretrained), modalities=VIS')
    print(f'MCC schedule: source-only warmup={args.source_warmup_epochs} epochs; '
          f'MCC ramp={args.trade_off_warmup_epochs} epochs')
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)
    mcc = MinimumClassConfusionLoss(temperature=args.temperature)
    if args.multimodal:
        parameter_groups = [
            {'params': backbone.vis_encoder.parameters(), 'lr': 0.1},
            {'params': backbone.ir_encoder.parameters(), 'lr': 1.0},
            {'params': classifier.bottleneck.parameters(), 'lr': 1.0},
            {'params': classifier.head.parameters(), 'lr': 1.0},
        ]
    else:
        parameter_groups = classifier.get_parameters()
    optimizer = SGD(parameter_groups, args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    best_acc = -float('inf')
    best_epoch = 0
    best_metrics = None
    last_metrics = None
    epochs_without_improvement = 0
    last_epoch = 0
    accuracy_history = []
    best_path = os.path.join(args.save_dir, f'mcc_{args.target}_best.pth')
    last_path = os.path.join(args.save_dir, f'mcc_{args.target}_last.pth')
    print('WARNING: labelled target validation is evaluated after every epoch and '
          'the highest-accuracy epoch is selected. Training remains target-label-free, '
          'but reporting is target-validation-selected UDA.')
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir,
                                'MCC-MM' if args.multimodal else 'MCC',
                                seed=args.seed, device=str(device),
                                modalities='VIS+IR' if args.multimodal else 'VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        if eff:
            eff.epoch_start(epoch)
        train_diagnostics = train_epoch(
            train_source_iter, train_target_iter, classifier, mcc,
            optimizer, lr_scheduler, epoch, args, eff=eff)
        print(
            f'Epoch [{epoch}/{args.epochs}] train summary: '
            f'avg_cls={train_diagnostics["avg_cls"]:.4f} '
            f'avg_mcc={train_diagnostics["avg_mcc"]:.4f} '
            f'source_acc={train_diagnostics["source_acc"]:.4f} '
            f'target_conf={train_diagnostics["target_conf"]:.4f} '
            f'target_entropy={train_diagnostics["target_entropy"]:.4f} '
            f'target_class_coverage={train_diagnostics["target_class_coverage"]}/{num_classes} '
            f'target_class_hist={train_diagnostics["target_class_hist"]} '
            f'transfer_weight={train_diagnostics["transfer_weight"]:.4f} '
            f'adaptation_steps={train_diagnostics["adaptation_steps"]}',
            flush=True,
        )
        val_metrics = validate(tgt_val_loader, classifier, num_classes)
        last_metrics = dict(val_metrics)
        acc = val_metrics['accuracy']
        accuracy_history.append((epoch, acc))
        print(f'Epoch [{epoch}/{args.epochs}] target val accuracy: {acc:.4f}',
              flush=True)
        print(format_classification_metrics(val_metrics), flush=True)
        improved = acc > best_acc + args.early_stop_min_delta
        tied_but_better_f1 = (
            best_metrics is not None and
            abs(acc - best_acc) <= 1e-12 and
            val_metrics['f1_macro'] > best_metrics['f1_macro']
        )
        if improved or tied_but_better_f1:
            best_acc = acc
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            if args.save_model:
                torch.save({'epoch': epoch, 'state_dict': classifier.state_dict(),
                            'metrics': best_metrics, 'target': args.target}, best_path)
        else:
            epochs_without_improvement += 1
        if eff:
            eff.epoch_end(epoch)
        if (args.early_stop_patience and
                epochs_without_improvement >= args.early_stop_patience):
            print(f'Early stopping at epoch {epoch}; best epoch={best_epoch}, '
                  f'best target val accuracy={best_acc:.4f}', flush=True)
            break

    if best_metrics is None or last_metrics is None:
        raise RuntimeError('MCC completed without target-validation metrics')
    if args.save_model:
        torch.save({'epoch': last_epoch, 'state_dict': classifier.state_dict(),
                    'metrics': last_metrics, 'target': args.target}, last_path)
    selected_metrics = best_metrics
    if eff:
        eff.finalize(extra_metrics={'final_acc': selected_metrics['accuracy'],
                                    'best_acc': best_acc,
                                    'selected_epoch': best_epoch,
                                    'last_epoch_acc': last_metrics['accuracy'],
                                    'target': args.target})
    print(f'Best target val accuracy ({args.target}, selected): {best_acc:.4f} '
          f'at epoch {best_epoch}', flush=True)
    print('Selected target val metrics:', flush=True)
    print(format_classification_metrics(selected_metrics), flush=True)
    print(f'Last-epoch target val accuracy ({args.target}, epoch {last_epoch}): '
          f'{last_metrics["accuracy"]:.4f}', flush=True)
    print(f'Final target val accuracy ({args.target}, selected epoch {best_epoch}): '
          f'{selected_metrics["accuracy"]:.4f}', flush=True)
    print(format_peak_timing(
        diagnose_peak_timing(accuracy_history, args.epochs), unit='epoch'
    ), flush=True)


if __name__ == '__main__':
    main()
