"""
Deep CORAL (Correlation Alignment) adapted for JMDA-Net dataset.
Source domain: 晴天 (Sunny)
Target domains: 逆光, 雨天, 雾天, 黑天

Usage:
    python train_deepcoral.py --target backlight --epochs 3

2026-10-20 变更（见 REPRODUCTION_RECORD.md 第 14 节）：
  * 标签空间以目标域为准（tools/uda_data.py）：只保留目标域存在的类别，分类头
    输出维度 = 该目标域类别数；
  * 优化器恢复原参考实现 DeepCORAL/main.py:130-133 的 2 参数组（主干 = lr，
    分类头 fc = 10×lr），并按其原样**不传入** weight_decay（该仓库虽定义了
    WEIGHT_DECAY=5e-4 但未传给优化器）；
  * CORAL 权重恢复原参考实现 main.py:148 的线性递增 λ=(epoch+1)/EPOCHS，
    取代原先固定的 --lamb。

2026-10-21 变更（逐项核对官方实现后的修复）：
  * CORAL 损失作用点改为分类器输出（main.py:46 ``models.CORAL(out1, out2)``，
    即 fc 之后、softmax 之前的 logits），不再作用于 512 维池化特征；
    models.py 的 CORAL 公式与官方仓库逐字节一致，未改动；
  * 每轮步数改为 min(len(source), len(target))（main.py:29），不再对较短
    一侧循环取样。
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Import CORAL loss from original repo
sys.path.insert(0, os.path.dirname(__file__))
from models import CORAL

# 跨平台数据路径适配（服务器：export JMDA_DATA_ROOT=/mnt/data/lixiang/data/Data）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything
# UDA 共用数据加载：以目标域类别集为准裁剪源域/目标域并重映射标签
from uda_data import UdaImageListDataset, uda_label_space, format_class_space, format_dropped


class DeepCORAL(nn.Module):
    """Deep CORAL with ResNet-18 backbone."""

    def __init__(self, num_classes=14):
        super(DeepCORAL, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.sharedNet = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features
        self.fc = nn.Linear(feat_dim, num_classes)
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target=None):
        source_feat = self.sharedNet(source).view(source.size(0), -1)
        source_out = self.fc(source_feat)
        if target is not None:
            target_feat = self.sharedNet(target).view(target.size(0), -1)
            target_out = self.fc(target_feat)
            return source_out, target_out, source_feat, target_feat
        return source_out


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
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
            out = model(imgs)
            pred = out.argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(pred.cpu().tolist())
    return classification_metrics(all_labels, all_predictions, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
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

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    model = DeepCORAL(num_classes=num_classes).to(device)
    # 原参考实现 DeepCORAL/main.py:130-133：2 个参数组，分类头 fc 用 10× 学习率；
    # 该实现未把 WEIGHT_DECAY 传给优化器，这里保持一致。
    optimizer = optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.fc.parameters(), 'lr': 10 * args.lr},
    ], lr=args.lr, momentum=0.9)
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

            src_out, tgt_out, _, _ = model(src_imgs, tgt_imgs)

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
