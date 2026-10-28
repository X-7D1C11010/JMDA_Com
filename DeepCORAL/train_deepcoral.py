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
  * 每轮步数改为 min(len(source), len(target))（main.py:29），不再对较短
    一侧循环取样。

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import model_zoo
from torchvision import transforms

# Import CORAL loss from original repo
sys.path.insert(0, os.path.dirname(__file__))
from models import CORAL, DeepCORAL

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


def get_transforms():
    # DeepCORAL/data_loader.py:get_office31_dataloader: Scale(227), ToTensor,
    # ImageNet Normalize. Scale is replaced by its current torchvision name.
    train_tf = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((227, 227)),
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
            out = model.fc(model.sharedNet(imgs))
            pred = out.argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(pred.cpu().tolist())
    return classification_metrics(all_labels, all_predictions, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数（Deep CORAL 参考实验为 20）')
    parser.add_argument('--source-batch-size', type=int, default=200,
                        help='官方 main.py BATCH_SIZE[0]')
    parser.add_argument('--target-batch-size', type=int, default=56,
                        help='官方 main.py BATCH_SIZE[1]')
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='官方 main.py 定义 5e-4 但实际未传给 SGD，故默认 0')
    parser.add_argument('--alexnet-weights', type=str, default='',
                        help='离线 torchvision AlexNet ImageNet state_dict；为空则使用官方 URL/cache')
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

    train_tf, val_tf = get_transforms()

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
    src_loader = DataLoader(src_ds, batch_size=args.source_batch_size, shuffle=True,
                            drop_last=True, **loader_kwargs)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.target_batch_size, shuffle=True,
                            drop_last=True, **loader_kwargs)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.test_batch_size,
                                shuffle=False, **eval_loader_kwargs)

    model = DeepCORAL(num_classes=num_classes).to(device)
    load_pretrained_alexnet(model.sharedNet, args.alexnet_weights)
    print('DeepCORAL architecture: official DeepCORAL/models.py AlexNet')
    # 官方 main.py：新分类层为 10x lr，主干为 base lr，momentum=0.9。
    optimizer = optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.fc.parameters(), 'lr': 10 * args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    print(f'DeepCORAL optimizer group learning rates: '
          f'{[group["lr"] for group in optimizer.param_groups]}')
    criterion = nn.CrossEntropyLoss()

    best_acc = -float('inf')
    final_acc = None
    # Efficiency logging (optional, does not affect training)
    eff = None
    if args.efficiency_output_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'DeepCORAL',
                                seed=args.seed, device=str(device), modalities='VIS')
        print(f'Efficiency logging -> {eff.run_dir}')

    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        # 原参考实现 main.py:148：CORAL 权重 λ=(e+1)/EPOCHS 随 epoch 线性递增
        _lambda = (epoch + 1) / args.epochs
        # 原参考实现 main.py:29：train_steps = min(len(source), len(target))，
        # 每轮新建迭代器、不循环取样。
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        n_batches = min(len(src_loader), len(tgt_loader))

        for i in range(n_batches):
            if eff:
                eff.step_start()
            src_imgs, src_labels = next(src_iter)
            tgt_imgs, _ = next(tgt_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)

            if eff:
                eff.dataloader_done()

            src_out, tgt_out = model(src_imgs, tgt_imgs)

            loss_cls = criterion(src_out, src_labels)
            # 原参考实现 main.py:46：CORAL 作用于 fc 输出（softmax 前 logits）。
            loss_coral = CORAL(src_out, tgt_out)
            loss = loss_cls + _lambda * loss_coral

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'lambda={_lambda:.4f} '
                      f'loss_cls={loss_cls.item():.4f} loss_coral={loss_coral.item():.4f}')

        val_metrics = evaluate(model, tgt_val_loader, device, num_classes)
        acc = val_metrics['accuracy']
        final_acc = acc
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {acc:.4f}')
        print(format_classification_metrics(val_metrics))
        best_acc = max(best_acc, acc)
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'deepcoral_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'final_acc': final_acc,
                                    'diagnostic_peak_acc': best_acc,
                                    'target': args.target})
    print(f'Final target val accuracy ({args.target}): {final_acc:.4f}')
    print(f'Diagnostic peak target val accuracy (not used for model selection): {best_acc:.4f}')


if __name__ == '__main__':
    main()
