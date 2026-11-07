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
  * 多模态协议使用 JMDA 的非张量异构编码器；本数据集适配完整遍历源域并
    循环较短目标域，避免丢弃有标签源样本。
"""
import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Use the original DANN gradient reversal layer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from model import CNNModel
from functions import ReverseLayerF

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
# UDA 共用数据加载：以目标域类别集为准裁剪源域/目标域并重映射标签
from uda_data import (
    UdaImageListDataset,
    assert_training_label_space,
    uda_label_space,
    format_class_space,
    format_dropped,
)
from data_loader_options import (
    add_data_loader_args,
    data_loader_kwargs,
    evaluation_loader_kwargs,
)
from uda_multimodal import (
    JMDA_GRAD_CLIP_NORM,
    JMDA_OTHER_LR,
    JMDA_SOURCE_WARMUP_EPOCHS,
    JMDA_VIS_LR,
    JMDA_WEIGHT_DECAY,
    JmdaMultimodalEncoder,
    configure_jmda_visual_finetuning,
    gradient_l2_norm,
    jmda_cosine_lr_multiplier,
    jmda_train_transform,
    jmda_val_transform,
    label_free_target_paired_dataset,
    paired_dataset,
    describe_pairing,
    unpaired_domain_loader,
)


def label_smoothed_nll_loss(log_probabilities, targets, smoothing=0.1):
    """JMDA's source label smoothing for DANN's log-softmax classifier."""
    nll = F.nll_loss(log_probabilities, targets)
    smooth = -log_probabilities.mean(dim=1).mean()
    return (1.0 - smoothing) * nll + smoothing * smooth


class ResNetDANN(nn.Module):
    """DANN over JMDA's non-tensor VIS/IR feature concatenation."""

    def __init__(self, num_classes, multimodal=False):
        super().__init__()
        self.multimodal = bool(multimodal)
        if self.multimodal:
            self.encoder = JmdaMultimodalEncoder()
            self.backbone = self.encoder.vis_encoder
            self.ir_backbone = self.encoder.ir_encoder
            feature_dim = self.encoder.out_features
        else:
            self.encoder = None
            backbone = models.resnet50(pretrained=True)
            branch_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.ir_backbone = None
            feature_dim = branch_dim
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes), nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024, 2), nn.LogSoftmax(dim=1),
        )

    def forward(self, input_data, alpha):
        if self.multimodal:
            feature = self.encoder(input_data)
        else:
            feature = self.backbone(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        return (self.class_classifier(feature),
                self.domain_classifier(reverse_feature))


def get_transforms(backbone='cnn28', enhanced_augmentation=False):
    # The official network has a fixed 50*4*4 flattening layer and therefore
    # requires 28x28 inputs. Both weather domains are RGB, so the MNIST-M
    # normalization is applied consistently to source and target.
    if backbone in {'resnet50', 'jmda'}:
        train_ops = [transforms.Resize((256, 256))]
        if enhanced_augmentation:
            train_ops += [
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                       saturation=0.2, hue=0.05),
            ]
        else:
            train_ops += [transforms.CenterCrop(224)]
        train_ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        train_tf = transforms.Compose(train_ops)
        val_tf = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        train_ops = [transforms.Resize((32, 32))]
        if enhanced_augmentation:
            train_ops += [
                transforms.RandomCrop(28), transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.15, hue=0.03),
            ]
        else:
            train_ops = [transforms.Resize((28, 28))]
        train_ops += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        train_tf = transforms.Compose(train_ops)
        val_tf = transforms.Compose([
            transforms.Resize((28, 28)), transforms.ToTensor(),
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
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='exact optimizer-update budget; 0 uses --epochs')
    parser.add_argument(
        '--grl-max-iterations', type=int, default=0,
        help=('gradient-reversal ramp length; 0 uses the optimizer-update '
              'budget. This allows a shorter training budget without '
              'artificially accelerating the official logistic GRL ramp.'),
    )
    parser.add_argument('--batch-size', type=int, default=128,
                        help='官方 DANN/train/main.py 设置为 128')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-scheduler', choices=['constant', 'cosine'],
                        default='constant')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='final/base LR ratio for cosine scheduling')
    parser.add_argument('--domain-loss-weight', type=float, default=1.0,
                        help='maximum weight of the source+target domain loss')
    parser.add_argument('--source-warmup-epochs', type=int,
                        default=JMDA_SOURCE_WARMUP_EPOCHS,
                        help='multimodal source-only warm-up (JMDA: 10 epochs)')
    parser.add_argument('--target-eval-interval', type=int, default=0,
                        help='diagnostic target-val interval in epochs; 0 evaluates only the final model')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--backbone', choices=['cnn28', 'resnet50', 'jmda'], default='cnn28',
                        help=('cnn28 reproduces the released digit DANN; resnet50 is the '
                              'single-modal image variant; jmda selects the non-tensor '
                              'VIS ResNet-18 + IR U-Net encoder'))
    parser.add_argument('--enhanced-augmentation', action='store_true',
                        help='enable random crop, horizontal flip, and color jitter')
    parser.add_argument('--multimodal', action='store_true',
                        help='exact-paired VIS+IR dual ResNet branches with feature concatenation')
    parser.add_argument('--save-model', action=argparse.BooleanOptionalAction,
                        default=False, help='save checkpoints (disabled by default)')
    parser.add_argument('--early-stop-patience', type=int, default=0,
                        help='target-val checks without improvement before stopping; 0 disables')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
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

    if args.early_stop_patience < 0:
        raise ValueError('--early-stop-patience must be >= 0')
    if args.early_stop_patience:
        raise ValueError('target-validation early stopping is disabled by this protocol')
    if args.epochs < 1 or args.max_iterations < 0:
        raise ValueError('--epochs must be >=1 and --max-iterations must be >=0')
    if (args.domain_loss_weight < 0 or args.target_eval_interval < 0 or
            args.source_warmup_epochs < 0 or
            args.grl_max_iterations < 0 or
            args.print_freq < 1 or not 0 <= args.min_lr_ratio <= 1):
        raise ValueError('domain loss weight/evaluation interval/print frequency is invalid')
    train_tf, val_tf = get_transforms(args.backbone, args.enhanced_augmentation)
    if args.multimodal:
        train_tf, val_tf = jmda_train_transform(), jmda_val_transform()
        print('Multimodal preprocessing: JMDA paired Resize(256) + '
              'RandomCrop(224) + HFlip; validation Resize(224)')

    # Load source (sunny) and target domain data.
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
        src_ds = UdaImageListDataset(
            src_list, transform=train_tf, label_map=label_map)
        tgt_ds = UdaImageListDataset(
            tgt_train_list, transform=train_tf, label_map=label_map)
        tgt_val_ds = UdaImageListDataset(
            tgt_val_list, transform=val_tf, label_map=label_map)
    assert_training_label_space(src_ds, tgt_ds, class_ids)

    print(format_class_space(class_ids, num_classes))
    print(f'Source train: {len(src_ds)} images')
    print(f'Target train: {len(tgt_ds)} images')
    print(f'Target val:   {len(tgt_val_ds)} images')
    print(format_dropped('source train', src_ds))
    if args.multimodal:
        print(describe_pairing('Source train', src_ds))
        print(describe_pairing('Target train', tgt_ds))
        print(describe_pairing('Target val', tgt_val_ds))

    loader_kwargs = data_loader_kwargs(args)
    eval_loader_kwargs = evaluation_loader_kwargs(args)
    print(f'DataLoader settings: train={loader_kwargs}, eval={eval_loader_kwargs}')
    train_loader = unpaired_domain_loader(
        src_ds, tgt_ds, args.batch_size, args.seed, **loader_kwargs)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, **eval_loader_kwargs)

    if args.backbone in {'resnet50', 'jmda'}:
        if args.backbone == 'jmda' and not args.multimodal:
            raise ValueError('--backbone jmda requires --multimodal')
        if args.multimodal and args.backbone != 'jmda':
            raise ValueError('--multimodal requires --backbone jmda')
        model = ResNetDANN(num_classes=num_classes,
                           multimodal=args.multimodal).to(device)
        if args.multimodal:
            tuning = configure_jmda_visual_finetuning(
                model.encoder, mode='layer2_4')
            print('JMDA VIS fine-tuning: ResNet layer2-layer4 + projection; '
                  f'VIS trainable={tuning["vis_trainable"]:,}/'
                  f'{tuning["vis_total"]:,}; IR fully trainable')
        print('DANN architecture: '
              + ('JMDA non-tensor encoder (VIS ResNet-18 + IR U-Net, '
                 '512+512 direct concatenation)'
                 if args.multimodal else 'ImageNet ResNet-50 VIS branch'))
    else:
        if args.multimodal:
            raise ValueError('--multimodal requires --backbone jmda')
        model = CNNModel(num_classes=num_classes).to(device)
        print('DANN architecture: official DANN/models/model.py CNNModel (28x28 input)')
    # Multimodal comparisons share JMDA's optimizer envelope.  Single-modal
    # runs retain the released DANN optimizer.
    if args.backbone in {'resnet50', 'jmda'}:
        if args.multimodal:
            parameter_groups = [
                {'params': [p for p in model.encoder.vis_encoder.parameters()
                            if p.requires_grad], 'lr': JMDA_VIS_LR},
                {'params': list(model.encoder.ir_encoder.parameters()) +
                           list(model.class_classifier.parameters()) +
                           list(model.domain_classifier.parameters()),
                 'lr': JMDA_OTHER_LR},
            ]
            optimizer = optim.AdamW(
                parameter_groups, weight_decay=JMDA_WEIGHT_DECAY)
        else:
            optimizer = optim.Adam([
                {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
                {'params': model.class_classifier.parameters(), 'lr': args.lr},
                {'params': model.domain_classifier.parameters(), 'lr': args.lr},
            ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss()
    domain_criterion = torch.nn.NLLLoss()

    best_acc = -float('inf')
    best_epoch = 0
    best_metrics = None
    last_metrics = None
    epochs_without_improvement = 0
    final_acc = None
    best_path = os.path.join(args.save_dir, f'dann_{args.target}_{args.backbone}_best.pth')
    if args.early_stop_patience:
        print('WARNING: early stopping monitors labelled target validation accuracy; '
              'this is target-validation-selected UDA, not strict label-free UDA.')
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'DANN-MM' if args.multimodal else 'DANN',
                                seed=args.seed, device=str(device),
                                modalities='VIS+IR' if args.multimodal else 'VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    # Match JMDA strict UDA exactly: cover the larger domain and independently
    # reshuffle/cycle the smaller domain, with a full equal batch on both sides.
    n_batches = len(train_loader)
    if n_batches < 1:
        raise ValueError('DANN requires at least one source and target batch')
    planned_updates = args.max_iterations or args.epochs * n_batches
    grl_max_iterations = args.grl_max_iterations or planned_updates
    outer_epochs = math.ceil(planned_updates / n_batches)
    source_warmup_epochs = (JMDA_SOURCE_WARMUP_EPOCHS
                            if args.multimodal else args.source_warmup_epochs)
    if args.multimodal and outer_epochs <= source_warmup_epochs:
        raise ValueError('JMDA staged training requires more than 10 epochs')
    scheduler = None
    if args.multimodal:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch_index: jmda_cosine_lr_multiplier(
                epoch_index, outer_epochs),
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: jmda_cosine_lr_multiplier(
                step, planned_updates, args.min_lr_ratio),
        )
    print(f'DANN optimizer-update budget: {planned_updates} '
          f'({"explicit iterations" if args.max_iterations else f"{args.epochs} epochs"})')
    print(f'DANN domain sampling: JMDA strict-UDA larger-domain epoch; '
          f'{n_batches} full paired batches/epoch')
    print(f'DANN GRL ramp length: {grl_max_iterations} updates')
    print('DANN multimodal staged training: '
          f'source-only warmup={source_warmup_epochs} epochs; '
          f'VIS lr={JMDA_VIS_LR:g}; IR+heads lr={JMDA_OTHER_LR:g}; '
          f'AdamW wd={JMDA_WEIGHT_DECAY:g}; epoch cosine; '
          f'grad clip={JMDA_GRAD_CLIP_NORM:g}')
    global_step = 0
    for epoch in range(outer_epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        train_iter = iter(train_loader)
        batches_this_epoch = min(n_batches, planned_updates - global_step)
        epoch_cls_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_source_correct = 0
        epoch_source_samples = 0
        epoch_domain_correct = 0
        epoch_domain_samples = 0
        epoch_vis_grad = 0.0
        epoch_ir_grad = 0.0
        epoch_head_grad = 0.0
        source_warmup = args.multimodal and epoch < source_warmup_epochs

        for i in range(batches_this_epoch):
            # 原参考实现 main.py:100-101：逐迭代调度 α（0 基 i 与 epoch）
            adaptation_step = max(
                global_step - source_warmup_epochs * n_batches, 0)
            adaptation_budget = max(
                planned_updates - source_warmup_epochs * n_batches, 1)
            p = min(float(adaptation_step) /
                    max(min(grl_max_iterations, adaptation_budget), 1), 1.0)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            if eff:
                eff.step_start()
            (src_imgs, src_labels), (tgt_imgs, _) = next(train_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            if not source_warmup:
                tgt_imgs = tgt_imgs.to(device)

            if eff:
                eff.dataloader_done()

            # Source classification loss
            src_class_out, src_domain_out = model(
                src_imgs, 0.0 if source_warmup else alpha)
            loss_cls = (label_smoothed_nll_loss(src_class_out, src_labels)
                        if args.multimodal else criterion(src_class_out, src_labels))

            if source_warmup:
                loss_domain = loss_cls.new_zeros(())
                loss = loss_cls
                src_domain_label = tgt_domain_label = tgt_domain_out = None
            else:
                src_domain_label = torch.zeros(
                    src_imgs.size(0), dtype=torch.long, device=device)
                tgt_domain_label = torch.ones(
                    tgt_imgs.size(0), dtype=torch.long, device=device)
                _, tgt_domain_out = model(tgt_imgs, alpha)
                loss_domain = (
                    domain_criterion(src_domain_out, src_domain_label) +
                    domain_criterion(tgt_domain_out, tgt_domain_label)
                )
                loss = loss_cls + args.domain_loss_weight * loss_domain
            with torch.no_grad():
                epoch_cls_loss += loss_cls.item()
                epoch_domain_loss += loss_domain.item()
                epoch_source_correct += int(
                    src_class_out.argmax(dim=1).eq(src_labels).sum().item())
                epoch_source_samples += int(src_labels.numel())
                if not source_warmup:
                    epoch_domain_correct += int(
                        src_domain_out.argmax(dim=1).eq(src_domain_label).sum().item())
                    epoch_domain_correct += int(
                        tgt_domain_out.argmax(dim=1).eq(tgt_domain_label).sum().item())
                    epoch_domain_samples += int(src_domain_label.numel() +
                                                tgt_domain_label.numel())
            optimizer.zero_grad()
            loss.backward()
            if args.multimodal:
                epoch_vis_grad += gradient_l2_norm(
                    model.encoder.vis_encoder.parameters())
                epoch_ir_grad += gradient_l2_norm(
                    model.encoder.ir_encoder.parameters())
                epoch_head_grad += gradient_l2_norm(
                    list(model.class_classifier.parameters()) +
                    list(model.domain_classifier.parameters()))
                torch.nn.utils.clip_grad_norm_(
                    model.encoder.vis_encoder.parameters(), JMDA_GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(
                    model.encoder.ir_encoder.parameters(), JMDA_GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(
                    model.class_classifier.parameters(), JMDA_GRAD_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(
                    model.domain_classifier.parameters(), JMDA_GRAD_CLIP_NORM)
            optimizer.step()
            if scheduler is not None and not args.multimodal:
                scheduler.step()
            global_step += 1
            if eff:
                eff.step_end()

            if (i + 1) % args.print_freq == 0 or i + 1 == batches_this_epoch:
                print(f'  Epoch [{epoch+1}/{outer_epochs}] batch [{i+1}/{batches_this_epoch}] '
                      f'update [{global_step}/{planned_updates}] '
                      f'loss_cls={loss_cls.item():.4f} loss_domain={loss_domain.item():.4f} '
                      f'domain_weight={0.0 if source_warmup else args.domain_loss_weight:.3f} '
                      f'warmup={source_warmup} alpha={alpha:.4f} '
                      f'lr={[round(group["lr"], 8) for group in optimizer.param_groups]}')

        print(
            f'Epoch [{epoch+1}/{outer_epochs}] train summary: '
            f'avg_cls={epoch_cls_loss / batches_this_epoch:.4f} '
            f'avg_domain={epoch_domain_loss / batches_this_epoch:.4f} '
            f'source_acc={epoch_source_correct / max(epoch_source_samples, 1):.4f} '
            f'domain_acc={epoch_domain_correct / max(epoch_domain_samples, 1):.4f} '
            + (f'grad_VIS={epoch_vis_grad / batches_this_epoch:.4e} '
               f'grad_IR={epoch_ir_grad / batches_this_epoch:.4e} '
               f'grad_heads={epoch_head_grad / batches_this_epoch:.4e} '
               if args.multimodal else '') +
            f'alpha={alpha:.4f}',
            flush=True,
        )
        if scheduler is not None and args.multimodal:
            scheduler.step()

        is_final = global_step >= planned_updates
        should_evaluate = is_final or (
            args.target_eval_interval > 0 and
            (epoch + 1) % args.target_eval_interval == 0
        )
        if should_evaluate:
            val_metrics = evaluate(model, tgt_val_loader, device, num_classes)
            last_metrics = dict(val_metrics)
            acc = val_metrics['accuracy']
            if is_final:
                final_acc = acc
            print(f'Epoch [{epoch + 1}/{outer_epochs}] update '
                  f'[{global_step}/{planned_updates}] target val accuracy '
                  f'(diagnostic): {acc:.4f}')
            print(format_classification_metrics(val_metrics))
            improved = acc > best_acc + args.early_stop_min_delta
            tied_but_better_f1 = (
                best_metrics is not None and
                abs(acc - best_acc) <= 1e-12 and
                val_metrics['f1_macro'] > best_metrics['f1_macro']
            )
            if improved or tied_but_better_f1:
                best_acc = acc
                best_epoch = epoch + 1
                best_metrics = dict(val_metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        if eff:
            eff.epoch_end(epoch)
        if not is_final:
            continue
        if (args.early_stop_patience and
                epochs_without_improvement >= args.early_stop_patience):
            print(f'Early stopping at epoch {epoch + 1}; best epoch={best_epoch}, '
                  f'best target val accuracy={best_acc:.4f}')
            break

    if best_metrics is None or last_metrics is None:
        raise RuntimeError('DANN completed without target-validation metrics')
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'dann_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': best_metrics['accuracy'],
                                    'best_acc': best_acc,
                                    'selected_epoch': best_epoch,
                                    'last_epoch_acc': last_metrics['accuracy'],
                                    'target': args.target})
    print(f'Best target val accuracy ({args.target}, selected): {best_acc:.4f} '
          f'at epoch {best_epoch}')
    print('Selected target val metrics:')
    print(format_classification_metrics(best_metrics))
    print(f'Last-epoch target val accuracy ({args.target}): '
          f'{last_metrics["accuracy"]:.4f}')
    print(f'Final target val accuracy ({args.target}, selected epoch {best_epoch}): '
          f'{best_metrics["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
