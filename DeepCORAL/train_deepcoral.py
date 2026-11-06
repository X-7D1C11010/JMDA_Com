"""
Deep CORAL (Correlation Alignment) adapted for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

Usage:
    python train_deepcoral.py --target backlight

2026-10-20 变更（见 REPRODUCTION_RECORD.md 第 14 节）：
  * 标签空间以目标域为准（tools/uda_data.py）：只保留目标域存在的类别，分类头
    输出维度 = 该目标域类别数；
  * 优化器使用 2 参数组（主干 = lr，分类头 fc = 10×lr）；
  * CORAL 权重恢复原参考实现 main.py:148 的线性递增 λ=(epoch+1)/EPOCHS，
    取代原先固定的 --lamb。

2026-10-21 变更（逐项核对官方实现后的修复）：
  * CORAL 损失作用点改为分类器输出（main.py:46 ``models.CORAL(out1, out2)``，
    即 fc 之后、softmax 之前的 logits），不再作用于 512 维池化特征；
  * 多模态协议使用 JMDA 的非张量异构编码器；本数据集适配完整遍历源域并
    循环较短目标域。末批由完整覆盖采样器保证至少2个样本，
    因而可在不丢样本的前提下计算协方差。

2026-10-27 官方网络/数据流程恢复：
  * 直接使用随附官方端口 models.py 的 AlexNet + 4096 维分类器；
  * 从官方 AlexNet ImageNet 权重加载 features 与前两层全连接层；
  * source/target batch 恢复为 200/56，输入恢复为 Resize(227) +
    ToTensor + ImageNet Normalize，不额外做随机增强。

论文公式修正（保留）：
  * models.py 的 CORAL 公式按论文 Eq. (1)--(3) 计算无偏协方差及 Frobenius
    范数；不沿用第三方 PyTorch 端口遗漏 1/(n-1) 且重复除 d² 的实现；
  * 默认训练预算为官方端口的 20 epochs。优化器严格复现其实际执行路径：
    虽然源码定义 WEIGHT_DECAY=5e-4，但构造 SGD 时没有传入，故默认值为 0。
"""
import os
import sys
import argparse
import hashlib
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import model_zoo
from torchvision import transforms, models as tv_models

# Import CORAL loss from original repo
sys.path.insert(0, os.path.dirname(__file__))
from models import CORAL, DeepCORAL

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
    JmdaMultimodalEncoder,
    configure_jmda_visual_finetuning,
    gradient_l2_norm,
    jmda_train_transform,
    jmda_val_transform,
    label_free_target_paired_dataset,
    paired_dataset,
    describe_pairing,
    unpaired_domain_loader,
)


ALEXNET_URL = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
ALEXNET_SHA256 = '4df8aa717fd11ab2d34e9483c5984ef7cdbefd732366a836076020db08909c9a'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrained_alexnet(backbone, weights_path=''):
    """Official main.py:107-117, with an optional offline checkpoint path."""
    if weights_path:
        actual_sha256 = _sha256(weights_path)
        if actual_sha256 != ALEXNET_SHA256:
            raise RuntimeError(
                'Refusing legacy AlexNet checkpoint deserialization because its '
                f'SHA256 is not the official value: {actual_sha256}'
            )
        try:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        except (TypeError, RuntimeError) as exc:
            # torchvision's official AlexNet file is a legacy tar checkpoint,
            # which PyTorch cannot open with weights_only=True.  Unsafe pickle
            # loading is allowed only after the exact official SHA256 check.
            if isinstance(exc, RuntimeError) and 'legacy .tar format' not in str(exc):
                raise
            if isinstance(exc, TypeError):
                # PyTorch < 2.0 does not expose the weights_only keyword.
                state_dict = torch.load(weights_path, map_location='cpu')
            else:
                state_dict = torch.load(weights_path, map_location='cpu',
                                        weights_only=False)
        source = weights_path
    else:
        state_dict = model_zoo.load_url(ALEXNET_URL)
        source = ALEXNET_URL
    own = backbone.state_dict()
    compatible = {
        key: value for key, value in state_dict.items()
        if key in own and own[key].shape == value.shape
    }
    missing = sorted(set(own) - set(compatible))
    backbone.load_state_dict(compatible, strict=False)
    print(f'Loaded official AlexNet ImageNet initialization from {source}; '
          f'compatible tensors={len(compatible)}, randomly initialized={missing}')


class ResNetDeepCORAL(nn.Module):
    """Deep CORAL over JMDA's non-tensor VIS/IR feature concatenation."""

    def __init__(self, num_classes, multimodal=False):
        super().__init__()
        self.multimodal = bool(multimodal)
        if self.multimodal:
            self.encoder = JmdaMultimodalEncoder()
            self.sharedNet = self.encoder.vis_encoder
            self.irNet = self.encoder.ir_encoder
            feature_dim = self.encoder.out_features
        else:
            self.encoder = None
            backbone = tv_models.resnet50(pretrained=True)
            branch_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.sharedNet = backbone
            self.irNet = None
            feature_dim = branch_dim
        self.fc = nn.Linear(feature_dim, num_classes)
        self.fc.weight.data.normal_(0, 0.005)

    def features(self, images):
        if not self.multimodal:
            return self.sharedNet(images)
        return self.encoder(images)

    def predict_components(self, images):
        return {'fused': self.fc(self.features(images))}

    def forward(self, source, target):
        return self.predict_components(source), self.predict_components(target)


def get_transforms(backbone='alexnet', enhanced_augmentation=False):
    # DeepCORAL/data_loader.py:get_office31_dataloader: Scale(227), ToTensor,
    # ImageNet Normalize. Scale is replaced by its current torchvision name.
    crop_size = 224 if backbone in {'resnet50', 'jmda'} else 227
    train_ops = [transforms.Resize((256, 256))]
    if enhanced_augmentation:
        train_ops += [
            transforms.RandomResizedCrop(crop_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.2, hue=0.05),
        ]
    else:
        train_ops = [transforms.Resize((crop_size, crop_size))]
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    train_tf = transforms.Compose(train_ops)
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_labels = []
    predictions = {'fused': [], 'vis': [], 'ir': []}
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if hasattr(model, 'predict_components'):
                outputs = model.predict_components(imgs)
            else:
                outputs = {'fused': model.fc(model.sharedNet(imgs))}
            all_labels.extend(labels.cpu().tolist())
            for name, output in outputs.items():
                predictions[name].extend(output.argmax(1).cpu().tolist())
    metrics = {
        name: classification_metrics(all_labels, values, num_classes=num_classes)
        for name, values in predictions.items() if values
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数（Deep CORAL 参考实验为 20）')
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='exact optimizer-update budget; 0 uses --epochs')
    parser.add_argument('--source-batch-size', type=int, default=200,
                        help='官方 main.py BATCH_SIZE[0]')
    parser.add_argument('--target-batch-size', type=int, default=56,
                        help='官方 main.py BATCH_SIZE[1]')
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--head-lr-mult', type=float, default=10.0,
                        help='classifier-head learning-rate multiplier')
    parser.add_argument('--vis-lr-mult', type=float, default=1.0,
                        help='VIS encoder learning-rate multiplier')
    parser.add_argument('--ir-lr-mult', type=float, default=1.0,
                        help='IR encoder learning-rate multiplier')
    parser.add_argument('--lr-scheduler', choices=['constant', 'cosine'],
                        default='constant')
    parser.add_argument('--min-lr-ratio', type=float, default=0.05,
                        help='final/base LR ratio for cosine scheduling')
    parser.add_argument('--coral-max-weight', type=float, default=1.0,
                        help='maximum CORAL loss weight after linear ramp-up')
    parser.add_argument('--coral-warmup-epochs', type=int, default=0,
                        help='source-only epochs before the CORAL ramp starts')
    parser.add_argument('--target-eval-interval', type=int, default=0,
                        help='diagnostic target-val interval in epochs; 0 means final only')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='官方 main.py 定义 5e-4 但实际未传给 SGD，故默认 0')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='source cross-entropy label smoothing; 0 preserves baseline')
    parser.add_argument('--alexnet-weights', type=str, default='',
                        help='离线 torchvision AlexNet ImageNet state_dict；为空则使用官方 URL/cache')
    parser.add_argument('--backbone', choices=['alexnet', 'resnet50', 'jmda'], default='alexnet',
                        help=('alexnet reproduces released Deep CORAL; resnet50 is the '
                              'single-modal variant; jmda selects the non-tensor '
                              'VIS ResNet-18 + IR U-Net encoder'))
    parser.add_argument('--enhanced-augmentation', action='store_true',
                        help='enable random crop, horizontal flip, and color jitter')
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
    if (args.coral_max_weight < 0 or args.coral_warmup_epochs < 0 or
            args.coral_warmup_epochs >= args.epochs or args.target_eval_interval < 0 or
            args.head_lr_mult <= 0 or args.vis_lr_mult <= 0 or
            args.ir_lr_mult <= 0 or not 0 <= args.min_lr_ratio <= 1 or
            not 0 <= args.label_smoothing < 1):
        raise ValueError('invalid CORAL weight/eval interval/head LR/scheduler setting')
    train_tf, val_tf = get_transforms(args.backbone, args.enhanced_augmentation)
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
        src_eval_ds = paired_dataset(src_list, val_tf, label_map)
        tgt_ds = label_free_target_paired_dataset(
            tgt_train_list, train_tf, label_map)
        tgt_val_ds = paired_dataset(tgt_val_list, val_tf, label_map)
    else:
        src_ds = UdaImageListDataset(src_list, transform=train_tf,
                                     label_map=label_map)
        src_eval_ds = UdaImageListDataset(src_list, transform=val_tf,
                                          label_map=label_map)
        tgt_ds = UdaImageListDataset(tgt_train_list, transform=train_tf,
                                     label_map=label_map)
        tgt_val_ds = UdaImageListDataset(tgt_val_list, transform=val_tf,
                                         label_map=label_map)
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
    if args.source_batch_size != args.target_batch_size:
        raise ValueError(
            'JMDA strict-UDA sampling requires equal source/target batch sizes')
    train_loader = unpaired_domain_loader(
        src_ds, tgt_ds, args.source_batch_size, args.seed, **loader_kwargs)
    src_eval_loader = DataLoader(src_eval_ds, batch_size=args.test_batch_size,
                                 shuffle=False, **eval_loader_kwargs)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.test_batch_size,
                                shuffle=False, **eval_loader_kwargs)

    if args.backbone in {'resnet50', 'jmda'}:
        if args.backbone == 'jmda' and not args.multimodal:
            raise ValueError('--backbone jmda requires --multimodal')
        if args.multimodal and args.backbone != 'jmda':
            raise ValueError('--multimodal requires --backbone jmda')
        model = ResNetDeepCORAL(num_classes=num_classes,
                                multimodal=args.multimodal).to(device)
        if args.multimodal:
            tuning = configure_jmda_visual_finetuning(
                model.encoder, mode='layer2_4')
            print('JMDA VIS fine-tuning: ResNet layer2-layer4 + projection; '
                  f'VIS trainable={tuning["vis_trainable"]:,}/'
                  f'{tuning["vis_total"]:,}; IR fully trainable')
        print('DeepCORAL architecture: '
              + ('JMDA non-tensor encoder (VIS ResNet-18 + IR U-Net, '
                 '512+512 direct concatenation)'
                 if args.multimodal else 'ImageNet ResNet-50 VIS branch'))
    else:
        if args.multimodal:
            raise ValueError('--multimodal requires --backbone jmda')
        model = DeepCORAL(num_classes=num_classes).to(device)
        load_pretrained_alexnet(model.sharedNet, args.alexnet_weights)
        print('DeepCORAL architecture: official DeepCORAL/models.py AlexNet')
    # The released recipe uses a 10x head. Keep that as the default while
    # allowing small target datasets to slow the head independently.
    optimizer_groups = [{
        'params': model.sharedNet.parameters(),
        'lr': args.vis_lr_mult * args.lr,
    }]
    if getattr(model, 'irNet', None) is not None:
        optimizer_groups.append({
            'params': model.irNet.parameters(),
            'lr': args.ir_lr_mult * args.lr,
        })
    optimizer_groups.append({'params': model.fc.parameters(),
                             'lr': args.head_lr_mult * args.lr})
    optimizer = optim.SGD(optimizer_groups, lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    print(f'DeepCORAL optimizer group learning rates: '
          f'{[group["lr"] for group in optimizer.param_groups]}')
    print(f'DeepCORAL LR multipliers: VIS={args.vis_lr_mult}, '
          f'IR={args.ir_lr_mult}, head={args.head_lr_mult}')
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f'DeepCORAL source label smoothing: {args.label_smoothing}')

    best_acc = -float('inf')
    best_epoch = 0
    best_metrics = None
    last_metrics = None
    epochs_without_improvement = 0
    final_acc = None
    best_path = os.path.join(args.save_dir,
                             f'deepcoral_{args.target}_{args.backbone}_best.pth')
    if args.early_stop_patience:
        print('WARNING: early stopping monitors labelled target validation accuracy; '
              'this is target-validation-selected UDA, not strict label-free UDA.')
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir,
                                'DeepCORAL-MM' if args.multimodal else 'DeepCORAL',
                                seed=args.seed, device=str(device),
                                modalities='VIS+IR' if args.multimodal else 'VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    n_batches = len(train_loader)
    if n_batches < 1:
        raise ValueError('DeepCORAL requires at least one full source and target batch')
    planned_updates = args.max_iterations or args.epochs * n_batches
    outer_epochs = math.ceil(planned_updates / n_batches)
    scheduler = None
    if args.lr_scheduler == 'cosine':
        def lr_factor(step):
            progress = min(float(step) / max(planned_updates, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    print(f'DeepCORAL optimizer-update budget: {planned_updates} '
          f'({"explicit iterations" if args.max_iterations else f"{args.epochs} epochs"})')
    print(f'DeepCORAL domain sampling: JMDA strict-UDA larger-domain epoch; '
          f'{n_batches} full paired batches/epoch')
    print(f'DeepCORAL LR schedule: {args.lr_scheduler}; '
          f'head multiplier={args.head_lr_mult}; min ratio={args.min_lr_ratio}')
    global_step = 0
    for epoch in range(outer_epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        # Small target datasets can benefit from a source-only warmup before
        # CORAL alignment. Defaults preserve the released no-warmup behavior.
        warmup_updates = min(args.coral_warmup_epochs * n_batches,
                             max(planned_updates - 1, 0))
        alignment_updates = max(planned_updates - warmup_updates, 1)
        alignment_progress = max(global_step + 1 - warmup_updates, 0) / alignment_updates
        _lambda = args.coral_max_weight * min(alignment_progress, 1.0)
        train_iter = iter(train_loader)
        batches_this_epoch = min(n_batches, planned_updates - global_step)
        epoch_cls_loss = 0.0
        epoch_coral_loss = 0.0
        epoch_weighted_coral = 0.0
        epoch_source_confidence = 0.0
        epoch_target_confidence = 0.0
        epoch_vis_grad = 0.0
        epoch_ir_grad = 0.0
        epoch_head_grad = 0.0

        for i in range(batches_this_epoch):
            alignment_progress = max(global_step + 1 - warmup_updates, 0) / alignment_updates
            _lambda = args.coral_max_weight * min(alignment_progress, 1.0)
            if eff:
                eff.step_start()
            (src_imgs, src_labels), (tgt_imgs, _) = next(train_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)

            if eff:
                eff.dataloader_done()

            src_outputs, tgt_outputs = model(src_imgs, tgt_imgs)
            # The bundled official AlexNet returns raw tensors, while the
            # ResNet wrapper returns a named fused output.
            if not isinstance(src_outputs, dict):
                src_outputs = {'fused': src_outputs}
                tgt_outputs = {'fused': tgt_outputs}

            loss_cls = criterion(src_outputs['fused'], src_labels)
            # 原参考实现 main.py:46：CORAL 作用于 fc 输出（softmax 前 logits）。
            loss_coral = CORAL(src_outputs['fused'], tgt_outputs['fused'])
            loss = loss_cls + _lambda * loss_coral
            with torch.no_grad():
                epoch_cls_loss += loss_cls.item()
                epoch_coral_loss += loss_coral.item()
                epoch_weighted_coral += _lambda * loss_coral.item()
                epoch_source_confidence += torch.softmax(
                    src_outputs['fused'], dim=1).max(dim=1).values.mean().item()
                epoch_target_confidence += torch.softmax(
                    tgt_outputs['fused'], dim=1).max(dim=1).values.mean().item()

            optimizer.zero_grad()
            loss.backward()
            if args.multimodal:
                epoch_vis_grad += gradient_l2_norm(
                    model.encoder.vis_encoder.parameters())
                epoch_ir_grad += gradient_l2_norm(
                    model.encoder.ir_encoder.parameters())
                epoch_head_grad += gradient_l2_norm(model.fc.parameters())
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0 or i + 1 == batches_this_epoch:
                print(f'  Epoch [{epoch+1}/{outer_epochs}] batch [{i+1}/{batches_this_epoch}] '
                      f'update [{global_step}/{planned_updates}] '
                      f'lambda={_lambda:.4f} '
                      f'loss_cls={loss_cls.item():.4f} '
                      f'loss_coral={loss_coral.item():.4f} '
                      f'lr={[round(group["lr"], 8) for group in optimizer.param_groups]}')

        print(
            f'Epoch [{epoch+1}/{outer_epochs}] train summary: '
            f'avg_cls={epoch_cls_loss / batches_this_epoch:.4f} '
            f'avg_coral={epoch_coral_loss / batches_this_epoch:.4f} '
            f'avg_weighted_coral={epoch_weighted_coral / batches_this_epoch:.4f} '
            f'source_conf={epoch_source_confidence / batches_this_epoch:.4f} '
            f'target_conf={epoch_target_confidence / batches_this_epoch:.4f} '
            + (f'grad_VIS={epoch_vis_grad / batches_this_epoch:.4e} '
               f'grad_IR={epoch_ir_grad / batches_this_epoch:.4e} '
               f'grad_head={epoch_head_grad / batches_this_epoch:.4e} '
               if args.multimodal else '') +
            f'lambda={_lambda:.4f}',
            flush=True,
        )

        source_metrics = evaluate(model, src_eval_loader, device, num_classes)
        source_fused_acc = source_metrics['fused']['accuracy']
        print(f'  Source deterministic accuracy after epoch {epoch + 1}: '
              f'{source_fused_acc:.4f}')

        is_final = global_step >= planned_updates
        should_evaluate = is_final or (
            args.target_eval_interval > 0 and
            (epoch + 1) % args.target_eval_interval == 0
        )
        if should_evaluate:
            val_metrics = evaluate(model, tgt_val_loader, device, num_classes)
            fused_metrics = val_metrics['fused']
            last_metrics = dict(fused_metrics)
            acc = fused_metrics['accuracy']
            if is_final:
                final_acc = acc
            print(f'Epoch [{epoch + 1}/{outer_epochs}] update '
                  f'[{global_step}/{planned_updates}] target val accuracy '
                  f'(diagnostic): {acc:.4f}')
            print(format_classification_metrics(fused_metrics))
            improved = acc > best_acc + args.early_stop_min_delta
            tied_but_better_f1 = (
                best_metrics is not None and
                abs(acc - best_acc) <= 1e-12 and
                fused_metrics['f1_macro'] > best_metrics['f1_macro']
            )
            if improved or tied_but_better_f1:
                best_acc = acc
                best_epoch = epoch + 1
                best_metrics = dict(fused_metrics)
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
        raise RuntimeError('DeepCORAL completed without target-validation metrics')
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'deepcoral_{args.target}_last.pth'))
    if eff:
        extra_metrics = {'final_acc': best_metrics['accuracy'],
                         'best_acc': best_acc,
                         'selected_epoch': best_epoch,
                         'last_epoch_acc': last_metrics['accuracy'],
                         'target': args.target,
                         'source_fused_acc': source_metrics['fused']['accuracy']}
        eff.finalize(extra_metrics=extra_metrics)
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
