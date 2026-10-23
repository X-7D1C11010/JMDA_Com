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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
from uda_data import UdaImageListDataset, uda_label_space, format_class_space, format_dropped


def evaluate_10crop(model, list_path, label_map, device, num_classes, batch_size=4):
    """官方 train_image.py:22-45 的十裁剪测试协议（test_10crop=True 默认）：
    10 个裁剪视角分别前向，softmax 概率求和后取 argmax。"""
    model.eval()
    transform_list = prep.image_test_10crop(resize_size=256, crop_size=224)
    loaders = [
        DataLoader(UdaImageListDataset(list_path, transform=tf, label_map=label_map),
                   batch_size=batch_size, shuffle=False, num_workers=0)
        for tf in transform_list
    ]
    all_labels, all_predictions = [], []
    with torch.no_grad():
        iterators = [iter(loader) for loader in loaders]
        for _ in range(len(loaders[0])):
            summed, labels = None, None
            for it in iterators:
                imgs, lbl = next(it)
                _, out = model(imgs.to(device))
                prob = nn.Softmax(dim=1)(out)
                summed = prob if summed is None else summed + prob
                labels = lbl
            all_labels.extend(labels.tolist())
            all_predictions.extend(summed.argmax(1).cpu().tolist())
    return classification_metrics(all_labels, all_predictions, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--num-iterations', type=int, default=100004,
                        help='总训练迭代数（原参考实现 train_image.py:217 固定 100004）')
    parser.add_argument('--test-interval', type=int, default=500,
                        help='评估间隔迭代数（原参考实现 --test_interval 默认 500）')
    parser.add_argument('--batch-size', type=int, default=36,
                        help='每个域的 batch（原参考实现 train_image.py:249-250 为 36）')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-gamma', type=float, default=0.001,
                        help='inv 调度参数 γ（原参考实现 0.001）')
    parser.add_argument('--lr-decay', type=float, default=0.75,
                        help='inv 调度参数 power（原参考实现 0.75）')
    parser.add_argument('--method', type=str, default='CDAN+E',
                        choices=['CDAN', 'CDAN+E'],
                        help='原参考实现 train_image.py:198 默认 CDAN+E（熵条件化，论文最优变体）')
    parser.add_argument('--random', action='store_true',
                        help='使用随机多线性映射（原参考实现 --random 默认 False；'
                             '论文仅在外积维度 >4096 时启用）')
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
    print(f'Device: {device}, Method: {args.method}')

    # 预处理恢复原参考实现 pre_process.py:image_train（Resize 256 + 随机裁剪 224）；
    # 评估由 evaluate_10crop 按官方十裁剪协议自建数据加载。
    train_tf = prep.image_train(resize_size=256, crop_size=224)

    # 标签空间以目标域为准：只保留目标域出现的类别，并把标签压到 0..|T|-1。
    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    src_list = os.path.join(args.data_dir, 'sunny_train.txt')
    tgt_train_list = os.path.join(args.data_dir, f'{args.target}_train.txt')
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')

    src_ds = UdaImageListDataset(src_list, transform=train_tf, label_map=label_map)
    tgt_ds = UdaImageListDataset(tgt_train_list, transform=train_tf, label_map=label_map)

    print(format_class_space(class_ids, num_classes))
    print(f'Source train: {len(src_ds)} images')
    print(f'Target train: {len(tgt_ds)} images')
    print(format_dropped('source train', src_ds))

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)

    # ResNet-18 backbone with bottleneck（官方 ResNetFc 配置：bottleneck_dim=256）
    model = ResNetFc(resnet_name='ResNet18', use_bottleneck=True,
                     bottleneck_dim=256, new_cls=True, class_num=num_classes).to(device)
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
    print(f'Random projection (--random): {args.random}; '
          f'ad_net input dim={256 * num_classes if not args.random else 1024}')

    # 原参考实现把带 lr_mult/decay_mult 的参数组直接交给 SGD，再由 inv 调度逐步
    # 按 lr_mult 缩放学习率、按 decay_mult 缩放权重衰减；PyTorch 允许参数组携带
    # 这两个自定义字段，故这里保持原样而不预先 materialize。
    optimizer = optim.SGD(model.get_parameters() + ad_net.get_parameters(),
                          lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    print(
        'CDAN optimizer group lr_mult: '
        f'{[group.get("lr_mult") for group in optimizer.param_groups]}'
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -float('inf')
    final_acc = None
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'CDAN',
                                seed=args.seed, device=str(device), modalities='VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    # 原参考实现 train_image.py:145-192 的迭代制训练：
    #   for i in range(100004)，迭代器耗尽即重建（:168-171），
    #   每个 iteration 开头 inv 调 lr（:166，0 基迭代号），
    #   源/目标分别前向后拼接 features/outputs（:175-178）。
    len_train_source = len(src_loader)
    len_train_target = len(tgt_loader)
    iter_source = iter(src_loader)
    iter_target = iter(tgt_loader)
    block = -1
    for iter_num in range(args.num_iterations):
        if iter_num % args.test_interval == 0:
            block += 1
            if eff:
                eff.epoch_start(block)
        model.train()
        ad_net.train()
        # 原参考实现 train_image.py:166：逐迭代 inv 调度（0 基迭代号）
        optimizer = inv_lr_scheduler(optimizer, iter_num,
                                     gamma=args.lr_gamma, power=args.lr_decay,
                                     lr=args.lr, weight_decay=5e-4)
        if iter_num % len_train_source == 0:
            iter_source = iter(src_loader)
        if iter_num % len_train_target == 0:
            iter_target = iter(tgt_loader)
        if eff:
            eff.step_start()
        src_imgs, src_labels = next(iter_source)
        tgt_imgs, _ = next(iter_target)
        src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
        tgt_imgs = tgt_imgs.to(device)
        if eff:
            eff.dataloader_done()

        # 原参考实现 train_image.py:175-178：源/目标分别前向，再拼接
        features_source, outputs_source = model(src_imgs)
        features_target, outputs_target = model(tgt_imgs)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        # Classification loss on source
        loss_cls = criterion(outputs_source, src_labels)

        # CDAN loss（trade_off=1.0，train_image.py:229）
        if args.method == 'CDAN+E':
            entropy = Entropy(softmax_out)
            loss_transfer = CDAN([features, softmax_out], ad_net, entropy,
                                 calc_coeff(iter_num), random_layer)
        else:
            loss_transfer = CDAN([features, softmax_out], ad_net,
                                 None, None, random_layer)

        loss = loss_cls + loss_transfer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if eff:
            eff.step_end()

        if (iter_num + 1) % 100 == 0:
            print(f'  iter [{iter_num + 1}/{args.num_iterations}] '
                  f'loss_cls={loss_cls.item():.4f} loss_transfer={loss_transfer.item():.4f} '
                  f'coeff={calc_coeff(iter_num):.4f} '
                  f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        if (iter_num % args.test_interval == args.test_interval - 1 or
                iter_num == args.num_iterations - 1):
            val_metrics = evaluate_10crop(model, tgt_val_list, label_map,
                                          device, num_classes)
            acc = val_metrics['accuracy']
            final_acc = acc
            print(f'iter [{iter_num + 1}/{args.num_iterations}] '
                  f'target val accuracy (10-crop): {acc:.4f}')
            print(format_classification_metrics(val_metrics))
            best_acc = max(best_acc, acc)
            if eff:
                eff.epoch_end(block)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'cdan_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': final_acc,
                                    'diagnostic_peak_acc': best_acc,
                                    'target': args.target})
    print(f'Final target val accuracy ({args.target}): {final_acc:.4f}')
    print(f'Diagnostic peak target val accuracy (not used for model selection): {best_acc:.4f}')


if __name__ == '__main__':
    main()
