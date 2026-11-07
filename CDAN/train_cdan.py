"""
CDAN (Conditional Domain Adversarial Network) adapted for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

2026-10-20 变更（见 REPRODUCTION_RECORD.md 第 14 节）：
  * 标签空间以目标域为准（tools/uda_data.py）：只保留目标域存在的类别，分类头
    与 RandomLayer 的类别维度 = 该目标域类别数；
  * 优化器恢复原参考实现 CDAN/pytorch/train_image.py:244-246 的
    SGD(nesterov=True, weight_decay=5e-4)，并按 pytorch/lr_schedule.py 的
    ``inv`` 策略**逐迭代**更新 lr = lr·(1+0.001·iter)^(−0.75)，保留
    lr_mult/decay_mult 分组语义。

2026-10-21 修复（逐项核对 CDAN/pytorch/train_image.py 后）：
  * RandomLayer 改为可选（--random，默认 False，对应 train_image.py:208 的
    ``--random type=bool default=False`` 与 :113-118 的分支）；默认走官方
    random=False 分支：ad_net 输入维度 = bottleneck×类别数（外积）；
  * 前向恢复 train_image.py:175-178：源/目标**分别**前向后拼接 features/
    outputs（原先把输入拼接一次前向，在含 BatchNorm 的 ResNet 下不等价）；
  * 训练预算恢复官方迭代制：num_iterations=100004、test_interval=500
    （train_image.py:204,217），取代 epochs×loader 长度；按项目规定不使用
    目标域准确率选 best checkpoint，仅报告末次并记录诊断峰值；
  * 预处理恢复 pre_process.py 的 image_train(256,224)；评估恢复官方默认的
    test_10crop=True 十裁剪协议（train_image.py:228），softmax 十裁剪求和；
  * 训练 batch 恢复官方 36（train_image.py:249-250）。

Usage:
    python train_cdan.py --target backlight --num-iterations 2000   # 冒烟
    python train_cdan.py --target backlight                         # 官方预算
"""
import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

# Import CDAN components from original repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch'))
from network import ResNetFc, AdversarialNetwork, RandomLayer, calc_coeff
from loss import CDAN, Entropy
from lr_schedule import inv_lr_scheduler
import pre_process as prep

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
from data_loader_options import add_data_loader_args, data_loader_kwargs
from uda_multimodal import (
    JMDA_GRAD_CLIP_NORM,
    JMDA_OTHER_LR,
    JMDA_SOURCE_WARMUP_EPOCHS,
    JMDA_VIS_LR,
    JMDA_WEIGHT_DECAY,
    JmdaMultimodalEncoder,
    configure_jmda_visual_finetuning,
    describe_pairing,
    gradient_l2_norm,
    jmda_cosine_lr_multiplier,
    jmda_train_transform,
    jmda_val_transform,
    label_free_target_paired_dataset,
    paired_dataset,
    unpaired_domain_loader,
)


class DualResNetFc(nn.Module):
    """CDAN head over JMDA's non-tensor VIS/IR direct concatenation."""

    def __init__(self, class_num, bottleneck_dim=256,
                 head_lr_mult=10.0):
        super().__init__()
        self.head_lr_mult = float(head_lr_mult)
        self.encoder = JmdaMultimodalEncoder()
        # This is CDAN's official bottleneck after the already-concatenated
        # multimodal feature; it is not an additional fusion mechanism.
        self.bottleneck = nn.Linear(self.encoder.out_features, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        nn.init.xavier_normal_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        features = self.bottleneck(self.encoder(images))
        return features, self.fc(features)

    def get_parameters(self):
        return [
            {'params': self.encoder.vis_encoder.parameters(), 'lr_mult': 0.1, 'decay_mult': 2},
            {'params': self.encoder.ir_encoder.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.bottleneck.parameters(), 'lr_mult': self.head_lr_mult, 'decay_mult': 2},
            {'params': self.fc.parameters(), 'lr_mult': self.head_lr_mult, 'decay_mult': 2},
        ]


def evaluate_10crop(model, list_path, label_map, device, num_classes,
                    batch_size=4, num_workers=4, prefetch_factor=2,
                    pin_memory=True, worker_timeout_seconds=300,
                    multimodal=False):
    """官方 train_image.py:22-45 的十裁剪测试协议（test_10crop=True 默认）：
    10 个裁剪视角分别前向，softmax 概率求和后取 argmax。

    The released implementation creates ten loaders at once.  On Windows each
    loader's workers are independent spawned PyTorch processes, so 10 crops x
    4 workers can exhaust committed memory.  Processing the views sequentially
    is mathematically identical and bounds the active worker pool to one view.
    """
    model.eval()
    # Current multimodal comparison follows JMDA's single deterministic
    # Resize(224) validation view. Single-modal CDAN retains its official
    # 10-crop evaluation protocol.
    transform_list = ([jmda_val_transform()] if multimodal else
                      prep.image_test_10crop(resize_size=256, crop_size=224))
    eval_loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        eval_loader_kwargs['prefetch_factor'] = prefetch_factor
        eval_loader_kwargs['timeout'] = worker_timeout_seconds

    summed_probabilities = None
    reference_labels = None
    with torch.no_grad():
        for transform in transform_list:
            dataset = (paired_dataset(list_path, transform, label_map)
                       if multimodal else UdaImageListDataset(
                           list_path, transform=transform, label_map=label_map))
            loader = DataLoader(
                dataset,
                batch_size=batch_size, shuffle=False, **eval_loader_kwargs,
            )
            view_probabilities = []
            view_labels = []
            for imgs, labels in loader:
                _, out = model(imgs.to(device))
                view_probabilities.append(nn.Softmax(dim=1)(out).cpu())
                view_labels.append(labels.cpu())

            probabilities = torch.cat(view_probabilities, dim=0)
            labels = torch.cat(view_labels, dim=0)
            if reference_labels is None:
                reference_labels = labels
                summed_probabilities = probabilities
            else:
                if not torch.equal(reference_labels, labels):
                    raise RuntimeError('10-crop loaders returned inconsistent sample order')
                summed_probabilities += probabilities

    all_labels = reference_labels.tolist()
    all_predictions = summed_probabilities.argmax(1).tolist()
    return classification_metrics(all_labels, all_predictions,
                                  num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--num-iterations', type=int, default=2000,
                        help='JMDA multimodal protocol budget (default: 2000 updates)')
    parser.add_argument('--test-interval', type=int, default=500,
                        help='评估间隔迭代数（原参考实现 --test_interval 默认 500）')
    parser.add_argument('--batch-size', type=int, default=36,
                        help='每个域的 batch（原参考实现 train_image.py:249-250 为 36）')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-gamma', type=float, default=0.001,
                        help='inv 调度参数 γ（原参考实现 0.001）')
    parser.add_argument('--lr-decay', type=float, default=0.75,
                        help='inv 调度参数 power（原参考实现 0.75）')
    parser.add_argument('--head-lr-mult', type=float, default=10.0,
                        help='multiplier for multimodal bottleneck/fusion/classifier groups')
    parser.add_argument('--ad-lr-mult', type=float, default=10.0,
                        help='domain-discriminator learning-rate multiplier')
    parser.add_argument('--trade-off', type=float, default=1.0,
                        help='weight applied to the conditional adversarial loss')
    parser.add_argument('--source-warmup-iterations', type=int, default=0,
                        help='source-only updates before conditional adversarial adaptation')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='source cross-entropy label smoothing')
    parser.add_argument('--coeff-max-iter', type=int, default=0,
                        help='GRL ramp length; 0 uses --num-iterations')
    parser.add_argument('--coeff-alpha', type=float, default=10.0,
                        help='steepness of the GRL logistic ramp')
    parser.add_argument('--method', type=str, default='CDAN+E',
                        choices=['CDAN', 'CDAN+E'],
                        help='原参考实现 train_image.py:198 默认 CDAN+E（熵条件化，论文最优变体）')
    parser.add_argument('--random', action='store_true',
                        help='使用随机多线性映射（原参考实现 --random 默认 False；'
                             '论文仅在外积维度 >4096 时启用）')
    parser.add_argument('--backbone', choices=['ResNet18', 'ResNet50', 'JMDA'],
                        default='ResNet18')
    parser.add_argument('--enhanced-augmentation', action='store_true',
                        help='add color jitter to the official resize/crop/flip pipeline')
    parser.add_argument('--multimodal', action='store_true')
    parser.add_argument('--save-model', action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument('--early-stop-patience', type=int, default=0,
                        help='target-val evaluations without improvement before stopping; 0 disables')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists_vis'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None,
                        help='Directory for efficiency logging output')
    parser.add_argument('--seed', type=int, default=42)
    add_data_loader_args(parser, default_workers=4, default_eval_workers=0)
    args = parser.parse_args()
    if args.early_stop_patience < 0:
        raise ValueError('--early-stop-patience must be >= 0')
    if args.early_stop_patience:
        raise ValueError('target-validation early stopping is disabled by this protocol')
    if (args.num_iterations < 1 or args.test_interval < 1 or
            args.head_lr_mult <= 0 or args.ad_lr_mult <= 0 or
            args.trade_off < 0 or args.coeff_max_iter < 0 or
            args.source_warmup_iterations < 0 or
            args.source_warmup_iterations >= args.num_iterations or
            not 0 <= args.label_smoothing < 1 or args.coeff_alpha <= 0):
        raise ValueError('invalid CDAN iteration/evaluation/lr-mult/ramp setting')
    args.seed = seed_everything(args.seed)
    print(f'Controlled random seed: {args.seed}')

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, Method: {args.method}')

    # 预处理恢复原参考实现 pre_process.py:image_train（Resize 256 + 随机裁剪 224）；
    # 评估由 evaluate_10crop 按官方十裁剪协议自建数据加载。
    if args.enhanced_augmentation:
        train_tf = T.Compose([
            prep.ResizeImage(256),
            T.RandomResizedCrop(224, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.25, contrast=0.25,
                          saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = prep.image_train(resize_size=256, crop_size=224)
    if args.multimodal:
        train_tf = jmda_train_transform()
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
    else:
        src_ds = UdaImageListDataset(src_list, transform=train_tf,
                                     label_map=label_map)
        tgt_ds = UdaImageListDataset(tgt_train_list, transform=train_tf,
                                     label_map=label_map)
    assert_training_label_space(src_ds, tgt_ds, class_ids)

    print(format_class_space(class_ids, num_classes))
    print(f'Source train: {len(src_ds)} images')
    print(f'Target train: {len(tgt_ds)} images')
    print(format_dropped('source train', src_ds))
    if args.multimodal:
        print(describe_pairing('Source train', src_ds))
        print(describe_pairing('Target train', tgt_ds))

    loader_kwargs = data_loader_kwargs(args)
    eval_protocol = ('JMDA deterministic single-view' if args.multimodal
                     else 'official CDAN 10-crop')
    print(f'DataLoader settings: train={loader_kwargs}, '
          f'eval={eval_protocol}, workers/view={args.eval_num_workers}')
    train_loader = unpaired_domain_loader(
        src_ds, tgt_ds, args.batch_size, args.seed, **loader_kwargs)

    if args.multimodal:
        if args.backbone != 'JMDA':
            raise ValueError('--multimodal requires --backbone JMDA')
        model = DualResNetFc(num_classes,
                             head_lr_mult=args.head_lr_mult).to(device)
        tuning = configure_jmda_visual_finetuning(
            model.encoder, mode='layer2_4')
        print('JMDA VIS fine-tuning: ResNet layer2-layer4 + projection; '
              f'VIS trainable={tuning["vis_trainable"]:,}/'
              f'{tuning["vis_total"]:,}; IR fully trainable')
        print('CDAN-MM encoder: JMDA non-tensor VIS ResNet-18 + IR U-Net, '
              '512+512 direct concatenation; CDAN bottleneck=256')
    else:
        if args.backbone == 'JMDA':
            raise ValueError('--backbone JMDA requires --multimodal')
        model = ResNetFc(resnet_name=args.backbone, use_bottleneck=True,
                         bottleneck_dim=256, new_cls=True,
                         class_num=num_classes).to(device)
        print(f'CDAN backbone: official CDAN/pytorch/network.py ResNetFc({args.backbone})')
    # 原参考实现 train_image.py:113-118：--random 默认 False 时不建 RandomLayer，
    # 判别器输入 = bottleneck×类别数 的外积；仅 --random 时用 1024 维随机映射。
    if args.random:
        random_layer = RandomLayer([256, num_classes], output_dim=1024)
        if torch.cuda.is_available():
            random_layer.cuda()
        ad_net = AdversarialNetwork(in_feature=1024, hidden_size=1024).to(device)
    else:
        random_layer = None
        ad_net = AdversarialNetwork(in_feature=256 * num_classes,
                                    hidden_size=1024).to(device)
    ad_net.alpha = float(args.coeff_alpha)
    print(f'Random projection (--random): {args.random}; '
          f'ad_net input dim={256 * num_classes if not args.random else 1024}')

    if args.multimodal:
        optimizer = optim.AdamW([
            {'params': [p for p in model.encoder.vis_encoder.parameters()
                        if p.requires_grad], 'lr': JMDA_VIS_LR},
            {'params': list(model.encoder.ir_encoder.parameters()) +
                       list(model.bottleneck.parameters()) +
                       list(model.fc.parameters()) +
                       list(ad_net.parameters()), 'lr': JMDA_OTHER_LR},
        ], weight_decay=JMDA_WEIGHT_DECAY)
        args.label_smoothing = 0.1
    else:
        ad_parameters = ad_net.get_parameters()
        for group in ad_parameters:
            group['lr_mult'] = args.ad_lr_mult
        optimizer = optim.SGD(
            model.get_parameters() + ad_parameters, lr=args.lr,
            momentum=0.9, weight_decay=5e-4, nesterov=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = -float('inf')
    best_iteration = 0
    best_metrics = None
    last_metrics = None
    evaluations_without_improvement = 0
    final_acc = None
    best_path = os.path.join(args.save_dir,
                             f'cdan_{args.target}_{args.backbone}_best.pth')
    if args.early_stop_patience:
        print('WARNING: early stopping monitors labelled target validation accuracy; '
              'this is target-validation-selected UDA, not strict label-free UDA.')
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir,
                                'CDAN-MM' if args.multimodal else 'CDAN',
                                seed=args.seed, device=str(device),
                                modalities='VIS+IR' if args.multimodal else 'VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    # 原参考实现 train_image.py:145-192 的迭代制训练：
    #   for i in range(100004)，迭代器耗尽即重建（:168-171），
    #   每个 iteration 开头 inv 调 lr（:166，0 基迭代号），
    #   源/目标分别前向后拼接 features/outputs（:175-178）。
    updates_per_domain_epoch = len(train_loader)
    total_domain_epochs = math.ceil(args.num_iterations /
                                    updates_per_domain_epoch)
    if args.multimodal:
        args.source_warmup_iterations = (
            JMDA_SOURCE_WARMUP_EPOCHS * updates_per_domain_epoch)
        if args.source_warmup_iterations >= args.num_iterations:
            raise ValueError(
                'JMDA staged training requires more than 10 domain epochs')
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch_index: jmda_cosine_lr_multiplier(
                epoch_index, total_domain_epochs),
        )
    else:
        scheduler = None
    coeff_max_iter = (args.coeff_max_iter or
                      max(args.num_iterations -
                          args.source_warmup_iterations, 1))
    ad_net.max_iter = float(coeff_max_iter)
    print('CDAN multimodal staged training: '
          f'source-only warmup={JMDA_SOURCE_WARMUP_EPOCHS} epochs/'
          f'{args.source_warmup_iterations} updates; '
          f'VIS lr={JMDA_VIS_LR:g}; IR+heads+ad lr={JMDA_OTHER_LR:g}; '
          f'AdamW wd={JMDA_WEIGHT_DECAY:g}; epoch cosine; '
          f'grad clip={JMDA_GRAD_CLIP_NORM:g}; '
          f'label smoothing={args.label_smoothing}')
    train_iterator = iter(train_loader)
    print('CDAN domain sampling: JMDA strict-UDA larger-domain epoch; '
          f'{updates_per_domain_epoch} full paired batches/domain epoch')
    block = 0
    for iter_num in range(args.num_iterations):
        if iter_num == 0 and eff:
            eff.epoch_start(block)
        model.train()
        ad_net.train()
        if not args.multimodal:
            optimizer = inv_lr_scheduler(
                optimizer, iter_num, gamma=args.lr_gamma,
                power=args.lr_decay, lr=args.lr, weight_decay=5e-4)
        if iter_num % updates_per_domain_epoch == 0:
            train_iterator = iter(train_loader)
        if eff:
            eff.step_start()
        (src_imgs, src_labels), (tgt_imgs, _) = next(train_iterator)
        src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
        source_warmup = iter_num < args.source_warmup_iterations
        if not source_warmup:
            tgt_imgs = tgt_imgs.to(device)
        if eff:
            eff.dataloader_done()

        # Establish a discriminative source classifier before CDAN conditions
        # its discriminator on target predictions. Target labels stay unused.
        features_source, outputs_source = model(src_imgs)
        loss_cls = criterion(outputs_source, src_labels)
        if source_warmup:
            loss_transfer = loss_cls.new_zeros(())
            transfer_weight = 0.0
            coeff = 0.0
            domain_details = {
                'domain_accuracy': float('nan'),
                'domain_source_accuracy': float('nan'),
                'domain_target_accuracy': float('nan'),
            }
            target_confidence = float('nan')
            target_entropy = float('nan')
            target_class_coverage = 0
            target_class_hist = []
        else:
            features_target, outputs_target = model(tgt_imgs)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            target_probabilities = nn.Softmax(dim=1)(outputs_target.detach())
            target_confidence = target_probabilities.max(
                dim=1).values.mean().item()
            target_entropy = (-(target_probabilities *
                                target_probabilities.clamp_min(1e-8).log())
                              .sum(dim=1).mean().item())
            target_class_counts = torch.bincount(
                target_probabilities.argmax(dim=1),
                minlength=num_classes).cpu()
            target_class_coverage = int((target_class_counts > 0).sum().item())
            target_class_hist = target_class_counts.tolist()
            adaptation_iteration = iter_num - args.source_warmup_iterations
            coeff = calc_coeff(adaptation_iteration, alpha=args.coeff_alpha,
                               max_iter=coeff_max_iter)
            if args.method == 'CDAN+E':
                entropy = Entropy(softmax_out)
                loss_transfer, domain_details = CDAN(
                    [features, softmax_out], ad_net, entropy, coeff,
                    random_layer, return_details=True)
            else:
                loss_transfer, domain_details = CDAN(
                    [features, softmax_out], ad_net, None, None,
                    random_layer, return_details=True)
            transfer_weight = args.trade_off

        loss = loss_cls + transfer_weight * loss_transfer
        optimizer.zero_grad()
        loss.backward()
        if args.multimodal:
            vis_grad = gradient_l2_norm(model.encoder.vis_encoder.parameters())
            ir_grad = gradient_l2_norm(model.encoder.ir_encoder.parameters())
            head_grad = gradient_l2_norm(
                list(model.bottleneck.parameters()) + list(model.fc.parameters()))
            adversary_grad = gradient_l2_norm(ad_net.parameters())
            torch.nn.utils.clip_grad_norm_(
                model.encoder.vis_encoder.parameters(), JMDA_GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(
                model.encoder.ir_encoder.parameters(), JMDA_GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(
                model.bottleneck.parameters(), JMDA_GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(
                model.fc.parameters(), JMDA_GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(
                ad_net.parameters(), JMDA_GRAD_CLIP_NORM)
        else:
            vis_grad = ir_grad = head_grad = adversary_grad = float('nan')
        optimizer.step()
        if (scheduler is not None and
                ((iter_num + 1) % updates_per_domain_epoch == 0 or
                 iter_num == args.num_iterations - 1)):
            scheduler.step()
        if eff:
            eff.step_end()

        if ((iter_num + 1) % 100 == 0 or
                iter_num == args.num_iterations - 1):
            print(f'  iter [{iter_num + 1}/{args.num_iterations}] '
                  f'loss_cls={loss_cls.item():.4f} loss_transfer={loss_transfer.item():.4f} '
                  f'trade_off={transfer_weight:.3f} warmup={source_warmup} '
                  f'coeff={coeff:.4f} '
                  f'domain_acc={domain_details["domain_accuracy"]:.4f} '
                  f'domain_src_acc={domain_details["domain_source_accuracy"]:.4f} '
                  f'domain_tgt_acc={domain_details["domain_target_accuracy"]:.4f} '
                  f'target_conf={target_confidence:.4f} '
                  f'target_entropy={target_entropy:.4f} '
                  f'target_class_coverage={target_class_coverage}/{num_classes} '
                  f'target_class_hist={target_class_hist} '
                  f'grad_VIS={vis_grad:.4e} grad_IR={ir_grad:.4e} '
                  f'grad_head={head_grad:.4e} grad_ad={adversary_grad:.4e} '
                  f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        # Diagnostic target evaluation is periodic, but never controls
        # gradients, stopping, checkpoint choice, or the final in-memory model.
        should_evaluate = ((iter_num + 1) % args.test_interval == 0 or
                           iter_num == args.num_iterations - 1)
        if should_evaluate:
            val_metrics = evaluate_10crop(model, tgt_val_list, label_map,
                                          device, num_classes,
                                          num_workers=args.eval_num_workers,
                                          prefetch_factor=args.prefetch_factor,
                                          pin_memory=args.pin_memory,
                                          worker_timeout_seconds=args.worker_timeout_seconds,
                                          multimodal=args.multimodal)
            acc = val_metrics['accuracy']
            last_metrics = dict(val_metrics)
            if iter_num == args.num_iterations - 1:
                final_acc = acc
            print(f'iter [{iter_num + 1}/{args.num_iterations}] '
                  f'target val accuracy ({eval_protocol}): {acc:.4f}')
            print(format_classification_metrics(val_metrics))
            improved = acc > best_acc + args.early_stop_min_delta
            tied_but_better_f1 = (
                best_metrics is not None and
                abs(acc - best_acc) <= 1e-12 and
                val_metrics['f1_macro'] > best_metrics['f1_macro']
            )
            if improved or tied_but_better_f1:
                best_acc = acc
                best_iteration = iter_num + 1
                best_metrics = dict(val_metrics)
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            if eff and iter_num == args.num_iterations - 1:
                eff.epoch_end(block)
            if (args.early_stop_patience and
                    evaluations_without_improvement >= args.early_stop_patience):
                print(f'Early stopping at iteration {iter_num + 1}; '
                      f'best iteration={best_iteration}, '
                      f'best target val accuracy={best_acc:.4f}')
                break

    if best_metrics is None or last_metrics is None:
        raise RuntimeError('CDAN completed without target-validation metrics')
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'cdan_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': best_metrics['accuracy'],
                                    'best_acc': best_acc,
                                    'selected_iteration': best_iteration,
                                    'last_iteration_acc': last_metrics['accuracy'],
                                    'target': args.target})
    print(f'Best target val accuracy ({args.target}, selected): {best_acc:.4f} '
          f'at iteration {best_iteration}')
    print('Selected target val metrics:')
    print(format_classification_metrics(best_metrics))
    print(f'Last-iteration target val accuracy ({args.target}): '
          f'{last_metrics["accuracy"]:.4f}')
    print(f'Final target val accuracy ({args.target}, selected iteration '
          f'{best_iteration}): {best_metrics["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
