"""d-SNE on the JMDA weather dataset using the released SDA recipe.

Reference: https://github.com/aws-samples/d-SNE.  This port follows
``train_val/training_sda.py``, ``train_val/custom_layers.py`` and the released
ResNet embedding head.  The default update is target-supervised and
single-directional, exactly as the release; ``--train-src`` enables its optional
second source update.  The source set is restricted to the target weather's
known class support and all labels are compactly remapped.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from classification_metrics import classification_metrics, format_classification_metrics
from domain_pairing import CrossDomainPairDataset
from metrics_io import save_metric_history
from reproducibility import seed_everything
from uda_data import UdaImageListDataset, format_dropped, uda_label_space


def get_transforms(args):
    train_steps = [transforms.Resize(args.resize)]
    if args.random_crop:
        train_steps.append(transforms.RandomResizedCrop(args.size, scale=(0.8, 1.0)))
    else:
        train_steps.append(transforms.CenterCrop(args.size))
    if args.flip:
        train_steps.append(transforms.RandomHorizontalFlip())
    if args.random_color:
        train_steps.append(
            transforms.ColorJitter(
                brightness=args.color_jitter, contrast=args.color_jitter,
                saturation=args.color_jitter, hue=0.1,
            )
        )
    normalise = transforms.Normalize([args.mean] * 3, [args.std] * 3)
    train_steps.extend([transforms.ToTensor(), normalise])
    eval_steps = [
        transforms.Resize(args.resize), transforms.CenterCrop(args.size),
        transforms.ToTensor(), normalise,
    ]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def dsne_loss(fts, ys, ftt, yt, margin=1.0, fn=False):
    """Line-for-line PyTorch equivalent of the released ``dSNELoss``."""
    if fn:
        fts = F.normalize(fts, p=2, dim=1)
        ftt = F.normalize(ftt, p=2, dim=1)

    distances = ((ftt.unsqueeze(1) - fts.unsqueeze(0)) ** 2).sum(dim=2)
    same = yt.view(-1, 1).eq(ys.view(1, -1))
    intra = distances * same.float()
    inter = distances * (~same).float()
    row_max = distances.max(dim=1, keepdim=True).values.expand_as(distances)
    revised_inter = torch.where(same, row_max, inter)
    max_intra = intra.max(dim=1).values
    min_inter = revised_inter.min(dim=1).values
    return F.relu(max_intra - min_inter + margin).mean()


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels else None
        )

    def forward(self, inputs):
        activated = F.relu(self.bn1(inputs), inplace=True)
        residual = self.projection(activated) if self.projection is not None else inputs
        output = self.conv1(activated)
        output = self.conv2(F.relu(self.bn2(output), inplace=True))
        return output + residual


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        hidden_channels = out_channels // self.expansion
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels else None
        )

    def forward(self, inputs):
        activated = F.relu(self.bn1(inputs), inplace=True)
        residual = self.projection(activated) if self.projection is not None else inputs
        output = self.conv1(activated)
        output = self.conv2(F.relu(self.bn2(output), inplace=True))
        output = self.conv3(F.relu(self.bn3(output), inplace=True))
        return output + residual


class ResNetV2Features(nn.Module):
    """PyTorch translation of the release's MXNet ``ResNetV2.features``."""

    SPECS = {
        18: (BasicBlockV2, [2, 2, 2, 2], [64, 64, 128, 256, 512]),
        50: (BottleneckV2, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
        101: (BottleneckV2, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
        152: (BottleneckV2, [3, 8, 36, 3], [64, 256, 512, 1024, 2048]),
    }

    def __init__(self, nlayers):
        super().__init__()
        if nlayers not in self.SPECS:
            raise ValueError(f'unsupported ResNet depth {nlayers}; choose {sorted(self.SPECS)}')
        block, layer_counts, channels = self.SPECS[nlayers]
        # MXNet release starts with BatchNorm(scale=False, center=False).
        self.input_norm = nn.BatchNorm2d(3, affine=False)
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stages = []
        in_channels = channels[0]
        for stage_index, (count, out_channels) in enumerate(
                zip(layer_counts, channels[1:])):
            stride = 1 if stage_index == 0 else 2
            blocks = [block(in_channels, out_channels, stride=stride)]
            blocks.extend(block(out_channels, out_channels) for _ in range(count - 1))
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.final_bn = nn.BatchNorm2d(in_channels)
        self.output_channels = in_channels
        self._initialise()

    def _initialise(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d) and module.affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images):
        features = self.input_norm(images)
        features = self.pool(F.relu(self.bn1(self.conv1(features)), inplace=True))
        features = self.stages(features)
        features = F.relu(self.final_bn(features), inplace=True)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)


class DSNE(nn.Module):
    """Released ResNetV2 plus its 4096 -> embedding -> classifier head."""

    def __init__(self, num_classes, embed_size=512, nlayers=101,
                 pretrained=False, use_dropout=False, use_l2n=False):
        super().__init__()
        if pretrained:
            raise ValueError(
                'The release preloads a source-domain checkpoint, not generic '
                'ImageNet weights. Use --backbone-checkpoint.'
            )
        self.features = ResNetV2Features(nlayers)
        feature_dim = self.features.output_channels
        embedding_layers = [nn.Linear(feature_dim, 4096), nn.ReLU(inplace=True)]
        if use_dropout:
            embedding_layers.append(nn.Dropout(0.5))
        embedding_layers.extend([nn.Linear(4096, embed_size), nn.ReLU(inplace=True)])
        if use_dropout:
            embedding_layers.append(nn.Dropout(0.5))
        self.embeds = nn.Sequential(*embedding_layers)
        self.use_l2n = use_l2n
        self.output = nn.Linear(embed_size, num_classes)
        for module in self.embeds.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, images, return_embedding=False):
        features = self.embeds(self.features(images))
        if self.use_l2n:
            features = F.normalize(features, p=2, dim=1)
        logits = self.output(features)
        return (logits, features) if return_embedding else logits

    def load_backbone_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        own_state = self.state_dict()
        compatible = {
            key: value for key, value in checkpoint.items()
            if key.startswith('features.') and key in own_state
            and own_state[key].shape == value.shape
        }
        if not compatible:
            raise ValueError(f'no compatible ResNetV2 backbone tensors in {path}')
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        print(
            f'Loaded {len(compatible)} backbone tensors from {path}; '
            f'head remains freshly initialised (missing={len(missing)}, '
            f'unexpected={len(unexpected)})'
        )


def evaluate(model, loader, device, num_classes):
    model.eval()
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions)
    return classification_metrics(labels_all, predictions_all, num_classes=num_classes)


def train_direction(model, optimizer, criterion, first_images, first_labels,
                    second_images, second_labels, args):
    """One released ``train_batch`` call and its immediately following step."""
    first_logits, first_features = model(first_images, return_embedding=True)
    _, second_features = model(second_images, return_embedding=True)
    xent = criterion(first_logits, first_labels)
    auxiliary = dsne_loss(
        first_features, first_labels, second_features, second_labels,
        margin=args.margin, fn=args.fn,
    )
    total = (1.0 - args.alpha) * xent + args.alpha * auxiliary
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return xent.detach(), auxiliary.detach(), total.detach()


def main():
    parser = argparse.ArgumentParser(description='Official-recipe d-SNE SDA port')
    parser.add_argument('--target', default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--ratio', type=int, default=1,
                        help='Released negative/positive pair ratio.')
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument('--nlayers', type=int, default=101,
                        choices=[18, 50, 101, 152])
    parser.add_argument('--pretrained', action='store_true',
                        help='Deprecated compatibility flag; use --backbone-checkpoint.')
    parser.add_argument('--backbone-checkpoint', default='',
                        help='Optional source-domain PyTorch ResNetV2 checkpoint.')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--l2n', action='store_true')
    parser.add_argument('--fn', action='store_true')
    parser.add_argument('--train-src', action='store_true',
                        help='Enable the optional released source-direction update.')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--mean', type=float, default=0.5)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--color-jitter', type=float, default=0.2)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--random-crop', action='store_true')
    parser.add_argument('--random-color', action='store_true')
    parser.add_argument('--target-labels-per-class', type=int, default=10)
    parser.add_argument('--sda-seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-train-batches', type=int, default=0,
                        help='Smoke-test only: cap batches per epoch; 0 uses all pairs.')
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data_lists'))
    parser.add_argument('--save-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--efficiency_output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.seed = seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Controlled random seed: {args.seed}')
    print(f'Device: {device}')
    if args.max_train_batches:
        print(f'[SMOKE MODE] limiting each epoch to {args.max_train_batches} pair batches')

    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    print(
        f'Target-driven label space: {num_classes} classes, original ids '
        f'{class_ids}, remapped to 0..{num_classes - 1}'
    )

    train_tf, eval_tf = get_transforms(args)
    source_list = os.path.join(args.data_dir, 'sunny_train.txt')
    target_list = os.path.join(
        args.data_dir,
        f'{args.target}_sda_{args.target_labels_per_class}shot_seed{args.sda_seed}.txt',
    )
    target_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')
    if not os.path.isfile(target_list):
        raise FileNotFoundError(
            f'SDA subset list missing: {target_list}. Run tools/make_sda_lists.py first.'
        )

    source = UdaImageListDataset(source_list, train_tf, label_map)
    target = UdaImageListDataset(target_list, train_tf, label_map)
    target_val = UdaImageListDataset(target_val_list, eval_tf, label_map)
    print(format_dropped('source train', source))
    print(format_dropped('target labelled train', target))
    print(format_dropped('target validation', target_val))

    pair_dataset = CrossDomainPairDataset(
        source, target, negative_ratio=args.ratio, seed=args.seed
    )
    print(
        f'Official d-SNE pair table: {len(pair_dataset)} pairs '
        f'({pair_dataset.positive_pairs} positive, '
        f'{pair_dataset.negative_pairs} negative)'
    )
    pair_loader = DataLoader(
        pair_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    target_val_loader = DataLoader(
        target_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = DSNE(
        num_classes=num_classes, embed_size=args.embed_size,
        nlayers=args.nlayers, pretrained=args.pretrained,
        use_dropout=args.dropout, use_l2n=args.l2n,
    ).to(device)
    if args.backbone_checkpoint:
        model.load_backbone_checkpoint(args.backbone_checkpoint)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    efficiency = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        efficiency = EfficiencyLogger(
            args.efficiency_output_dir, 'd-SNE', seed=args.seed,
            device=str(device), modalities='VIS', protocol='SDA',
            target_label_count=len(target),
            target_labels_per_class=args.target_labels_per_class,
            backbone=f'ResNet-{args.nlayers} + released 4096/{args.embed_size} head',
            pretraining=(args.backbone_checkpoint or 'None (release default)'),
        )
        print(f'Efficiency logging -> {efficiency.run_dir}')

    history = []
    best_metrics = None
    best_epoch = None
    run_stem = f'dsne_{args.target}_seed{args.seed}'
    best_checkpoint = os.path.join(args.save_dir, f'{run_stem}_best.pth')
    for epoch in range(1, args.epochs + 1):
        if efficiency:
            efficiency.epoch_start(epoch - 1)
        model.train()
        for batch_index, (source_images, source_labels, target_images,
                          target_labels, _) in enumerate(pair_loader, 1):
            if efficiency:
                efficiency.step_start()
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)
            target_labels = target_labels.to(device)
            if efficiency:
                efficiency.dataloader_done()

            if args.train_src:
                train_direction(
                    model, optimizer, criterion,
                    source_images, source_labels, target_images, target_labels, args,
                )
            target_xent, target_aux, _ = train_direction(
                model, optimizer, criterion,
                target_images, target_labels, source_images, source_labels, args,
            )
            if efficiency:
                efficiency.step_end()
            if batch_index % 10 == 0:
                print(
                    f'  Epoch [{epoch}/{args.epochs}] '
                    f'batch [{batch_index}/{len(pair_loader)}] '
                    f'target_xent={target_xent.item():.4f} '
                    f'dsne={target_aux.item():.4f} train_src={args.train_src}'
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        metrics = evaluate(model, target_val_loader, device, num_classes)
        history.append((epoch, metrics))
        print(f'Epoch [{epoch}/{args.epochs}] target val accuracy: {metrics["accuracy"]:.4f}')
        print(f'  {format_classification_metrics(metrics)}')
        if best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics = dict(metrics)
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint)
        if efficiency:
            efficiency.epoch_end(epoch - 1)

    last_checkpoint = os.path.join(args.save_dir, f'{run_stem}_last.pth')
    torch.save(model.state_dict(), last_checkpoint)
    metrics_csv, metrics_json = save_metric_history(
        args.save_dir,
        run_stem,
        history,
        metadata={
            'method': 'd-SNE', 'protocol': 'SDA', 'target': args.target,
            'source': 'sunny', 'class_ids_original_zero_based': class_ids,
            'label_map': label_map, 'epochs': args.epochs, 'seed': args.seed,
            'target_labels_per_class': args.target_labels_per_class,
            'backbone': f'ResNetV2-{args.nlayers}',
            'backbone_checkpoint': args.backbone_checkpoint or None,
            'train_src': args.train_src, 'negative_positive_ratio': args.ratio,
            'max_train_batches': args.max_train_batches,
        },
    )
    final_metrics = history[-1][1]
    if efficiency:
        efficiency.finalize(extra_metrics={
            'best_acc': best_metrics['accuracy'],
            'final_acc': final_metrics['accuracy'],
            'precision_macro': final_metrics['precision_macro'],
            'recall_macro': final_metrics['recall_macro'],
            'f1_macro': final_metrics['f1_macro'],
            'target': args.target, 'protocol': 'SDA',
        })

    print(f'Final target val accuracy: {final_metrics["accuracy"]:.4f}')
    print(f'Best target val accuracy: {best_metrics["accuracy"]:.4f} (epoch {best_epoch})')
    print(f'Metrics CSV: {metrics_csv}')
    print(f'Metrics JSON: {metrics_json}')


if __name__ == '__main__':
    main()
