"""d-SNE on the JMDA weather dataset using the released SDA recipe.

Reference: https://github.com/aws-samples/d-SNE.  This port follows
``train_val/training_sda.py``, ``train_val/custom_layers.py`` and the released
ResNet embedding head.  The default update is target-supervised and
single-directional, exactly as the release; ``--train-src`` enables its optional
second source update.  The source set is restricted to the target weather's
known class support and all labels are compactly remapped.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import timm
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
    if args.vertical_flip:
        train_steps.append(transforms.RandomVerticalFlip())
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
    """Official ResNetV2 head with a strictly loadable PyTorch backbone.

    The repository's portable ImageNet checkpoint was exported from timm's
    ``resnetv2_101.a1h_in1k``.  Using that exact module is essential: the old
    hand-written 101-layer feature extractor shared no state-dict keys with the
    checkpoint and previously allowed a silent partial load.  The lightweight
    hand-written variants remain available only for smoke tests.
    """

    def __init__(self, num_classes, embed_size=512, nlayers=101,
                 pretrained=False, use_dropout=False, use_l2n=False):
        super().__init__()
        self.nlayers = int(nlayers)
        if self.nlayers == 101:
            self.features = timm.create_model(
                'resnetv2_101.a1h_in1k', pretrained=pretrained,
                num_classes=0, global_pool='avg',
            )
            feature_dim = 2048
        else:
            if pretrained:
                raise ValueError('ImageNet initialization is supported only for nlayers=101')
            self.features = ResNetV2Features(self.nlayers)
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
        source = Path(path).expanduser().resolve()
        payload = torch.load(source, map_location='cpu', weights_only=False)
        state = payload.get('state_dict', payload.get('model_state', payload)) \
            if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError(f'checkpoint does not contain a state dict: {source}')
        prefixed = {
            key[len('features.'):]: value
            for key, value in state.items() if key.startswith('features.')
        }
        if prefixed:
            state = prefixed
        expected = self.features.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(state)
            if tuple(expected[key].shape) != tuple(state[key].shape)
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                'ResNetV2 backbone checkpoint is not structurally compatible: '
                f'expected={len(expected)} supplied={len(state)} '
                f'missing={missing[:8]} unexpected={unexpected[:8]} '
                f'shape_mismatch={mismatched[:8]}'
            )
        self.features.load_state_dict(state, strict=True)
        print(f'Loaded all {len(state)} backbone tensors strictly from {source}')

    def load_source_checkpoint(self, path):
        source = Path(path).expanduser().resolve()
        payload = torch.load(source, map_location='cpu', weights_only=False)
        state = payload.get('model_state', payload.get('state_dict', payload)) \
            if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError(f'source checkpoint does not contain a state dict: {source}')
        self.load_state_dict(state, strict=True)
        print(f'Loaded source-tuned d-SNE model strictly from {source}')


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


def train_source(model, loader, eval_loader, device, criterion, args, num_classes):
    """Supervised source fine-tuning used by the released d-SNE pipeline."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    history = []
    for epoch in range(1, args.source_epochs + 1):
        model.train()
        running_loss = 0.0
        items = 0
        for batch_index, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]
            items += images.shape[0]
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        metrics = evaluate(model, eval_loader, device, num_classes)
        history.append((epoch, metrics))
        print(
            f'Source fine-tune [{epoch}/{args.source_epochs}] '
            f'CE={running_loss / max(items, 1):.4f} '
            f'ACC={metrics["accuracy"]:.4f}'
        )
    return history


def checkpoint_payload(model, epoch, metrics, metadata):
    return {
        'model_state': model.state_dict(),
        'epoch': int(epoch),
        'metrics': dict(metrics),
        'metadata': dict(metadata),
    }


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


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
    parser.add_argument('--source-epochs', type=int, default=50,
                        help='Released source-only fine-tuning budget.')
    parser.add_argument('--source-batch-size', type=int, default=32)
    parser.add_argument('--source-checkpoint', default='',
                        help='Strict full-model source-tuned checkpoint; skips source fine-tuning.')
    parser.add_argument('--pretrained-backbone', '--backbone-checkpoint',
                        dest='pretrained_backbone', default='',
                        help='Strict ImageNet ResNetV2-101 backbone checkpoint.')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Smoke-test only: initialize the backbone randomly.')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--l2n', action='store_true')
    parser.add_argument('--fn', dest='fn', action='store_true')
    parser.add_argument('--no-fn', dest='fn', action='store_false')
    parser.add_argument('--train-src', action='store_true',
                        help='Enable the optional released source-direction update.')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--mean', type=float, default=0.5)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--color-jitter', type=float, default=0.2)
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.add_argument('--no-flip', dest='flip', action='store_false')
    parser.add_argument('--vertical-flip', dest='vertical_flip', action='store_true')
    parser.add_argument('--no-vertical-flip', dest='vertical_flip', action='store_false')
    parser.add_argument('--random-crop', dest='random_crop', action='store_true')
    parser.add_argument('--no-random-crop', dest='random_crop', action='store_false')
    parser.add_argument('--random-color', dest='random_color', action='store_true')
    parser.add_argument('--no-random-color', dest='random_color', action='store_false')
    parser.add_argument('--target-labels-per-class', type=int, default=10)
    parser.add_argument('--sda-seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-train-batches', type=int, default=0,
                        help='Smoke-test only: cap batches per epoch; 0 uses all pairs.')
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data_lists'))
    parser.add_argument('--save-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--efficiency_output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.set_defaults(
        fn=True, flip=True, vertical_flip=True,
        random_crop=True, random_color=True,
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.source_epochs <= 0:
        raise ValueError('epochs and source-epochs must be positive')
    if args.batch_size <= 0 or args.source_batch_size <= 0:
        raise ValueError('batch sizes must be positive')
    if args.embed_size != 512 and not args.max_train_batches:
        raise ValueError('formal d-SNE reproduction requires embed-size=512')
    if args.nlayers != 101 and not args.max_train_batches:
        raise ValueError('formal d-SNE reproduction requires ResNetV2-101')
    if args.source_checkpoint and args.pretrained_backbone:
        raise ValueError('source-checkpoint and pretrained-backbone are mutually exclusive')

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
    source_eval = UdaImageListDataset(source_list, eval_tf, label_map)
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

    source_generator = torch.Generator().manual_seed(args.seed)
    source_loader = DataLoader(
        source, batch_size=args.source_batch_size, shuffle=True,
        num_workers=args.num_workers, generator=source_generator,
    )
    source_eval_loader = DataLoader(
        source_eval, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    default_backbone = Path(ROOT) / 'pretrained_weights' / \
        'dsne_resnetv2_101_a1h_in1k_backbone.pth'
    if (not args.source_checkpoint and not args.pretrained_backbone
            and not args.no_pretrained and args.nlayers == 101
            and default_backbone.is_file()):
        args.pretrained_backbone = str(default_backbone)
    timm_download = (
        not args.source_checkpoint and not args.pretrained_backbone
        and not args.no_pretrained and args.nlayers == 101
    )
    if args.no_pretrained and not args.max_train_batches:
        raise ValueError('--no-pretrained is restricted to smoke tests')

    model = DSNE(
        num_classes=num_classes, embed_size=args.embed_size,
        nlayers=args.nlayers, pretrained=timm_download,
        use_dropout=args.dropout, use_l2n=args.l2n,
    ).to(device)
    if args.pretrained_backbone:
        model.load_backbone_checkpoint(args.pretrained_backbone)
    criterion = nn.CrossEntropyLoss()

    run_stem = f'dsne_{args.target}_seed{args.seed}'
    source_checkpoint = os.path.join(args.save_dir, f'{run_stem}_source_tuned.pth')
    source_elapsed = 0.0
    source_history = []
    if args.source_checkpoint:
        model.load_source_checkpoint(args.source_checkpoint)
        source_provenance = str(Path(args.source_checkpoint).expanduser().resolve())
    else:
        source_started = time.perf_counter()
        source_history = train_source(
            model, source_loader, source_eval_loader, device, criterion,
            args, num_classes,
        )
        source_elapsed = time.perf_counter() - source_started
        source_metadata = {
            'stage': 'supervised source_train fine-tuning',
            'source': 'sunny', 'target': args.target, 'seed': args.seed,
            'epochs': args.source_epochs,
            'pretrained_backbone': args.pretrained_backbone or (
                'timm ImageNet download' if timm_download else None
            ),
        }
        torch.save(
            checkpoint_payload(
                model, args.source_epochs, source_history[-1][1], source_metadata
            ),
            source_checkpoint,
        )
        save_metric_history(
            args.save_dir, f'{run_stem}_source', source_history,
            metadata=source_metadata,
        )
        model.load_source_checkpoint(source_checkpoint)
        source_provenance = str(Path(source_checkpoint).resolve())
        print(
            f'Saved and strictly reloaded source-tuned checkpoint: '
            f'{source_checkpoint}'
        )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    efficiency = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        efficiency = EfficiencyLogger(
            args.efficiency_output_dir, 'd-SNE', seed=args.seed,
            device=str(device), modalities='VIS', protocol='SDA',
            target_label_count=len(target),
            target_labels_per_class=args.target_labels_per_class,
            backbone=f'ResNet-{args.nlayers} + released 4096/{args.embed_size} head',
            pretraining=source_provenance,
        )
        print(f'Efficiency logging -> {efficiency.run_dir}')

    history = []
    budgets = sorted(set(
        [budget for budget in (10, 20) if budget <= args.epochs]
        + [args.epochs]
    ))
    best_by_budget = {budget: None for budget in budgets}
    best_epoch_by_budget = {budget: None for budget in budgets}
    checkpoint_by_budget = {
        budget: os.path.join(args.save_dir, f'{run_stem}_e{budget}_best.pth')
        for budget in budgets
    }
    metadata = {
        'method': 'd-SNE', 'protocol': 'SDA', 'target': args.target,
        'source': 'sunny', 'class_ids_original_zero_based': class_ids,
        'label_map': label_map, 'epochs': args.epochs, 'seed': args.seed,
        'source_epochs': args.source_epochs,
        'source_checkpoint': source_provenance,
        'source_stage_elapsed_seconds': source_elapsed,
        'target_labels_per_class': args.target_labels_per_class,
        'backbone': f'ResNetV2-{args.nlayers}',
        'pretrained_backbone': args.pretrained_backbone or (
            'timm ImageNet download' if timm_download else None
        ),
        'strict_checkpoint_loading': True,
        'train_src_during_adaptation': args.train_src,
        'feature_normalization_in_dsne_loss': args.fn,
        'negative_positive_ratio': args.ratio,
        'augmentation': {
            'resize': args.resize, 'crop': args.size,
            'random_resized_crop': args.random_crop,
            'horizontal_flip': args.flip, 'vertical_flip': args.vertical_flip,
            'color_jitter': args.random_color,
        },
        'normalization': {'mean': args.mean, 'std': args.std},
        'max_train_batches': args.max_train_batches,
    }
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
        for budget in budgets:
            if epoch > budget:
                continue
            current_best = best_by_budget[budget]
            if current_best is None or metrics['accuracy'] > current_best['accuracy']:
                best_by_budget[budget] = dict(metrics)
                best_epoch_by_budget[budget] = epoch
                torch.save(
                    checkpoint_payload(model, epoch, metrics, metadata),
                    checkpoint_by_budget[budget],
                )
        if efficiency:
            efficiency.epoch_end(epoch - 1)

    last_checkpoint = os.path.join(args.save_dir, f'{run_stem}_last.pth')
    torch.save(
        checkpoint_payload(model, args.epochs, history[-1][1], metadata),
        last_checkpoint,
    )
    budget_results = {}
    for budget in budgets:
        checkpoint = checkpoint_by_budget[budget]
        model.load_source_checkpoint(checkpoint)
        budget_history = history[:budget]
        budget_metadata = dict(metadata)
        budget_metadata.update({
            'method': f'd-SNE@{budget}', 'budget_epochs': budget,
            'selected_epoch': best_epoch_by_budget[budget],
            'checkpoint': checkpoint,
        })
        metrics_csv, metrics_json = save_metric_history(
            args.save_dir, f'{run_stem}_e{budget}', budget_history,
            metadata=budget_metadata,
        )
        selected = best_by_budget[budget]
        budget_results[str(budget)] = {
            'method': f'd-SNE@{budget}',
            'ACC': selected['accuracy'],
            'Precision_macro': selected['precision_macro'],
            'Recall_macro': selected['recall_macro'],
            'F1_macro': selected['f1_macro'],
            'best_epoch': best_epoch_by_budget[budget],
            'checkpoint': checkpoint,
            'metrics_csv': metrics_csv,
            'metrics_json': metrics_json,
        }
        print(
            f'd-SNE@{budget}: ACC={selected["accuracy"]:.4f} '
            f'Precision_macro={selected["precision_macro"]:.4f} '
            f'Recall_macro={selected["recall_macro"]:.4f} '
            f'F1_macro={selected["f1_macro"]:.4f} '
            f'best_epoch={best_epoch_by_budget[budget]}'
        )

    result_manifest = os.path.join(args.save_dir, f'{run_stem}_budget_results.json')
    atomic_json(result_manifest, {
        'metadata': metadata,
        'budgets': budget_results,
        'last_checkpoint': last_checkpoint,
    })
    final_metrics = history[-1][1]
    final_budget = budgets[-1]
    if efficiency:
        efficiency.finalize(extra_metrics={
            'best_acc': best_by_budget[final_budget]['accuracy'],
            'final_acc': final_metrics['accuracy'],
            'precision_macro': final_metrics['precision_macro'],
            'recall_macro': final_metrics['recall_macro'],
            'f1_macro': final_metrics['f1_macro'],
            'target': args.target, 'protocol': 'SDA',
        })

    print(f'Final target val accuracy: {final_metrics["accuracy"]:.4f}')
    print(f'Budget result manifest: {result_manifest}')
    print(f'Last checkpoint: {last_checkpoint}')


if __name__ == '__main__':
    main()
