"""CCSA on the JMDA weather dataset using the released SDA training recipe.

Reference: https://github.com/samotiian/CCSA (``Initialization.py``).

The released method is preserved where it defines the algorithm: exhaustive
cross-domain pairs (all positives plus 3x negatives), the released CNN/MLP
topology, CSA weight 0.25, Adadelta, 256 pair batches, 80 epochs, and two
successive optimiser updates for every pair batch (source branch, then target
branch).  The unavoidable dataset adaptation is RGB 224x224 input instead of
16x16 grayscale digits.  Source examples are restricted to the target weather's
known class support and labels are remapped to a compact target-specific space.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from classification_metrics import classification_metrics, format_classification_metrics
from domain_pairing import CrossDomainPairDataset
from metrics_io import save_metric_history
from multimodal_batches import batch_size, batch_slice, batch_to_device
from paired_transforms import PairedNaturalImageTransform
from reproducibility import seed_everything
from uda_data import PairedUdaImageListDataset, format_dropped, uda_label_space


IMAGE_SIZE = 224


def auxiliary_ir_encoder(output_dim):
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm2d(32), nn.GELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.GELU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, output_dim),
    )


def get_transforms():
    """Static, synchronized VIS/IR preprocessing for the fixed pair table."""
    kwargs = {
        'size': IMAGE_SIZE,
        'resize': IMAGE_SIZE,
        'rgb_mean': (0.485, 0.456, 0.406),
        'rgb_std': (0.229, 0.224, 0.225),
    }
    return PairedNaturalImageTransform(**kwargs), PairedNaturalImageTransform(**kwargs)


def euclidean_distance(x, y, eps=1e-8):
    """Released ``euclidean_distance`` translated directly to PyTorch."""
    return torch.sqrt(torch.clamp(((x - y) ** 2).sum(dim=1, keepdim=True), min=eps))


def contrastive_loss(y_true, y_pred, margin=1.0):
    """Released CCSA contrastive semantic alignment loss."""
    y_true = y_true.float().view(-1, 1)
    return (
        y_true * y_pred.square()
        + (1.0 - y_true) * torch.clamp(margin - y_pred, min=0.0).square()
    ).mean()


def _strict_load_backbone(module, checkpoint):
    source = Path(checkpoint).expanduser().resolve()
    payload = torch.load(source, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload.get('model_state', payload)) \
        if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f'CCSA backbone checkpoint has no state dict: {source}')
    prefixed = {
        key[len('features.'):]: value
        for key, value in state.items() if key.startswith('features.')
    }
    if prefixed:
        state = prefixed
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            'VGG-16 backbone checkpoint is not structurally compatible: '
            f'expected={len(expected)} supplied={len(state)} '
            f'missing={missing[:8]} unexpected={unexpected[:8]} '
            f'shape_mismatch={mismatched[:8]}'
        )
    module.load_state_dict(state, strict=True)
    print(f'Loaded all {len(state)} VGG-16 backbone tensors strictly from {source}')


class CCSA(nn.Module):
    """Natural-image CCSA architecture reported for Office-31.

    The digit release's 16x16 CNN is not the configuration reported for
    natural images.  The paper uses ImageNet VGG-16 convolutions followed by
    1024- and 128-dimensional fully connected layers.
    """

    def __init__(self, num_classes, pretrained=False, pretrained_backbone=''):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained and not pretrained_backbone else None
        backbone = vgg16(weights=weights)
        self.features = backbone.features
        if pretrained_backbone:
            _strict_load_backbone(self.features, pretrained_backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.ir_encoder = auxiliary_ir_encoder(128)
        self.ir_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, num_classes)
        for module in list(self.encoder.modules()) + [self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def embed(self, images):
        if torch.is_tensor(images):
            rgb, ir = images, None
        else:
            rgb, ir = images['rgb'], images.get('nir', images.get('ir'))
        embedding = self.encoder(self.avgpool(self.features(rgb)))
        if ir is not None:
            if ir.shape[1] == 3:
                ir = ir.mean(dim=1, keepdim=True)
            embedding = embedding + self.ir_residual_scale * self.ir_encoder(ir)
        return embedding

    def forward(self, first, second=None):
        first_features = self.embed(first)
        first_logits = self.classifier(first_features)
        if second is None:
            return first_logits
        second_features = self.embed(second)
        second_logits = self.classifier(second_features)
        return first_logits, second_logits, first_features, second_features

    def set_source_initialization_stage(self, enabled):
        for parameter in self.features.parameters():
            parameter.requires_grad = not enabled

    def source_initialization_parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.classifier.parameters())
            + list(self.ir_encoder.parameters())
            + [self.ir_residual_scale]
        )

    def load_source_checkpoint(self, checkpoint):
        source = Path(checkpoint).expanduser().resolve()
        payload = torch.load(source, map_location='cpu', weights_only=False)
        state = payload.get('model_state', payload.get('state_dict', payload)) \
            if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError(f'CCSA source checkpoint has no state dict: {source}')
        self.load_state_dict(state, strict=True)
        print(f'Loaded CCSA source-initialized model strictly from {source}')


def evaluate(model, loader, device, num_classes):
    model.eval()
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(batch_to_device(images, device)).argmax(1).cpu().tolist()
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions)
    return classification_metrics(labels_all, predictions_all, num_classes=num_classes)


def train_source_initialization(model, loader, device, criterion, args):
    """Initialize the new FC layers on all source data, as in the paper."""
    model.set_source_initialization_stage(True)
    optimizer = torch.optim.Adadelta(
        model.source_initialization_parameters(),
        lr=args.lr, rho=args.rho, eps=args.eps,
    )
    for epoch in range(1, args.source_epochs + 1):
        model.train()
        loss_sum, item_count = 0.0, 0
        for batch_index, (images, labels) in enumerate(loader, 1):
            images, labels = batch_to_device(images, device), labels.to(device)
            item_count = batch_size(images)
            optimizer.zero_grad()
            for start in range(0, item_count, args.micro_batch_size):
                stop = min(start + args.micro_batch_size, item_count)
                loss = criterion(
                    model(batch_slice(images, start, stop)), labels[start:stop]
                )
                ((stop - start) / item_count * loss).backward()
                loss_sum += loss.item() * (stop - start)
                item_count += stop - start
            optimizer.step()
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        print(
            f'CCSA source FC initialization [{epoch}/{args.source_epochs}] '
            f'CE={loss_sum / max(item_count, 1):.4f}'
        )
    model.set_source_initialization_stage(False)


def train_pair_direction(model, optimizer, criterion, first_images, first_labels,
                         second_images, pair_labels, args):
    """One released CCSA direction with an effective pair batch."""
    optimizer.zero_grad()
    count = batch_size(first_images)
    ce_sum, csa_sum = 0.0, 0.0
    for start in range(0, count, args.micro_batch_size):
        stop = min(start + args.micro_batch_size, count)
        logits, _, first_features, second_features = model(
            batch_slice(first_images, start, stop),
            batch_slice(second_images, start, stop),
        )
        ce = criterion(logits, first_labels[start:stop])
        csa = contrastive_loss(
            pair_labels[start:stop],
            euclidean_distance(first_features, second_features),
        )
        loss = (1.0 - args.alpha) * ce + args.alpha * csa
        ((stop - start) / count * loss).backward()
        ce_sum += ce.item() * (stop - start)
        csa_sum += csa.item() * (stop - start)
    optimizer.step()
    return ce_sum / count, csa_sum / count


DEFAULT_TARGET_LABELS_PER_CLASS = 0


def target_training_list_path(data_dir, target, labels_per_class, sda_seed):
    """Resolve the labelled target cohort; zero means every target-train label."""
    if labels_per_class < 0:
        raise ValueError('target labels per class cannot be negative')
    if labels_per_class == 0:
        return os.path.join(data_dir, f'{target}_train.txt')
    return os.path.join(
        data_dir,
        f'{target}_sda_{labels_per_class}shot_seed{sda_seed}.txt',
    )


def main():
    parser = argparse.ArgumentParser(description='Official-recipe CCSA SDA port')
    parser.add_argument('--target', default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=80,
                        help='Released CCSA budget: 80 epochs.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Released pair batch size: 256.')
    parser.add_argument('--source-epochs', type=int, default=50)
    parser.add_argument('--source-batch-size', type=int, default=256)
    parser.add_argument('--micro-batch-size', type=int, default=16)
    parser.add_argument('--source-checkpoint', default='')
    parser.add_argument('--pretrained-backbone', default='')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0.95,
                        help='Keras Adadelta default used by the release.')
    parser.add_argument('--eps', type=float, default=1e-7,
                        help='Keras Adadelta epsilon used by the release.')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument(
        '--target-labels-per-class', type=int,
        default=DEFAULT_TARGET_LABELS_PER_CLASS,
        help='Labelled target samples per class; 0 uses all target-train labels.',
    )
    parser.add_argument('--sda-seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-train-batches', type=int, default=0,
                        help='Smoke-test only: cap batches per epoch; 0 uses all pairs.')
    parser.add_argument('--early-stopping-patience', type=int, default=8)
    parser.add_argument('--early-stopping-min-epochs', type=int, default=5)
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data_lists'))
    parser.add_argument('--save-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--efficiency_output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if min(
        args.epochs, args.source_epochs, args.batch_size,
        args.source_batch_size, args.micro_batch_size,
    ) <= 0:
        raise ValueError('epoch and batch counts must be positive')
    if args.target_labels_per_class < 0:
        raise ValueError('target labels per class cannot be negative')
    if args.source_checkpoint and args.pretrained_backbone:
        raise ValueError('source-checkpoint and pretrained-backbone are mutually exclusive')
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError('early stopping settings cannot be negative')

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

    train_tf, eval_tf = get_transforms()
    source_list = os.path.join(args.data_dir, 'sunny_train.txt')
    target_list = target_training_list_path(
        args.data_dir,
        args.target,
        args.target_labels_per_class,
        args.sda_seed,
    )
    target_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')
    if not os.path.isfile(target_list):
        raise FileNotFoundError(
            f'Labelled target-training list missing: {target_list}. '
            'Prepare the protocol lists first.'
        )

    source = PairedUdaImageListDataset(source_list, train_tf, label_map)
    target = PairedUdaImageListDataset(target_list, train_tf, label_map)
    target_val = PairedUdaImageListDataset(target_val_list, eval_tf, label_map)
    print(format_dropped('source train', source))
    print(format_dropped('target labelled train', target))
    print(format_dropped('target validation', target_val))
    if args.target_labels_per_class == 0:
        print('CCSA target label mode: all available target-training labels')
    else:
        print(
            f'CCSA target label mode: {args.target_labels_per_class}-shot '
            f'per class (subset seed {args.sda_seed})'
        )

    pair_dataset = CrossDomainPairDataset(
        source, target, negative_ratio=3, seed=args.seed
    )
    print(
        f'Official CCSA pair table: {len(pair_dataset)} pairs '
        f'({pair_dataset.positive_pairs} positive, '
        f'{pair_dataset.negative_pairs} negative)'
    )
    pair_loader = DataLoader(
        pair_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True,
    )
    if len(pair_loader) == 0:
        raise ValueError('pair table is smaller than one full pair batch')
    target_val_loader = DataLoader(
        target_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    source_generator = torch.Generator().manual_seed(args.seed)
    source_loader = DataLoader(
        source, batch_size=args.source_batch_size, shuffle=True,
        num_workers=args.num_workers, generator=source_generator,
    )
    default_backbone = Path(ROOT) / 'pretrained_weights' / \
        'ccsa_vgg16_imagenet1k_v1_backbone.pth'
    if (not args.source_checkpoint and not args.pretrained_backbone
            and not args.no_pretrained and default_backbone.is_file()):
        args.pretrained_backbone = str(default_backbone)
    download_pretrained = (
        not args.source_checkpoint and not args.pretrained_backbone
        and not args.no_pretrained
    )
    if args.no_pretrained and not args.max_train_batches:
        raise ValueError('--no-pretrained is restricted to smoke tests')
    model = CCSA(
        num_classes=num_classes,
        pretrained=download_pretrained,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    source_init_checkpoint = os.path.join(
        args.save_dir, f'ccsa_{args.target}_seed{args.seed}_source_init.pth'
    )
    criterion = nn.CrossEntropyLoss()
    if args.source_checkpoint:
        model.load_source_checkpoint(args.source_checkpoint)
        source_provenance = str(Path(args.source_checkpoint).expanduser().resolve())
    else:
        train_source_initialization(model, source_loader, device, criterion, args)
        torch.save(
            {
                'model_state': model.state_dict(),
                'metadata': {
                    'stage': 'source_train FC initialization',
                    'source_epochs': args.source_epochs,
                    'pretrained_backbone': args.pretrained_backbone or 'torchvision download',
                },
            },
            source_init_checkpoint,
        )
        model.load_source_checkpoint(source_init_checkpoint)
        source_provenance = str(Path(source_init_checkpoint).resolve())
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, rho=args.rho, eps=args.eps
    )

    efficiency = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        efficiency = EfficiencyLogger(
            args.efficiency_output_dir, 'CCSA', seed=args.seed,
            device=str(device), modalities='VIS+IR', protocol='SDA',
            target_label_count=len(target),
            target_labels_per_class=args.target_labels_per_class,
            backbone='VGG-16 conv + 1024/128 head (paper Office configuration)',
            pretraining=source_provenance,
        )
        print(f'Efficiency logging -> {efficiency.run_dir}')

    history = []
    best_metrics = None
    best_epoch = None
    evaluations_without_improvement = 0
    stopped_early = False
    completed_epoch = 0
    run_stem = f'ccsa_{args.target}_seed{args.seed}'
    best_checkpoint = os.path.join(args.save_dir, f'{run_stem}_best.pth')
    for epoch in range(1, args.epochs + 1):
        if efficiency:
            efficiency.epoch_start(epoch - 1)
        model.train()
        for batch_index, (source_images, source_labels, target_images,
                          target_labels, pair_labels) in enumerate(pair_loader, 1):
            if efficiency:
                efficiency.step_start()
            source_images = batch_to_device(source_images, device)
            source_labels = source_labels.to(device)
            target_images = batch_to_device(target_images, device)
            target_labels = target_labels.to(device)
            pair_labels = pair_labels.to(device)
            if efficiency:
                efficiency.dataloader_done()

            source_ce, source_csa = train_pair_direction(
                model, optimizer, criterion,
                source_images, source_labels, target_images, pair_labels, args,
            )
            target_ce, target_csa = train_pair_direction(
                model, optimizer, criterion,
                target_images, target_labels, source_images, pair_labels, args,
            )
            if efficiency:
                efficiency.step_end()

            if batch_index % 10 == 0:
                print(
                    f'  Epoch [{epoch}/{args.epochs}] '
                    f'batch [{batch_index}/{len(pair_loader)}] '
                    f'source_ce={source_ce:.4f} '
                    f'target_ce={target_ce:.4f} '
                    f'csa={0.5 * (source_csa + target_csa):.4f}'
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        metrics = evaluate(model, target_val_loader, device, num_classes)
        history.append((epoch, metrics))
        completed_epoch = epoch
        print(f'Epoch [{epoch}/{args.epochs}] target val accuracy: {metrics["accuracy"]:.4f}')
        print(f'  {format_classification_metrics(metrics)}')
        if best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics = dict(metrics)
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint)
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        if efficiency:
            efficiency.epoch_end(epoch - 1)
        if (args.early_stopping_patience
                and epoch >= args.early_stopping_min_epochs
                and evaluations_without_improvement >= args.early_stopping_patience):
            stopped_early = True
            print(
                f'Early stopping after {evaluations_without_improvement} '
                'target-val epochs without ACC improvement.'
            )
            break

    last_checkpoint = os.path.join(args.save_dir, f'{run_stem}_last.pth')
    torch.save(model.state_dict(), last_checkpoint)
    best_state = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(best_state, strict=True)
    print(f'Strictly reloaded best checkpoint: {best_checkpoint}')
    metrics_csv, metrics_json = save_metric_history(
        args.save_dir,
        run_stem,
        history,
        metadata={
            'method': 'CCSA', 'protocol': 'SDA', 'target': args.target,
            'source': 'sunny', 'class_ids_original_zero_based': class_ids,
            'label_map': label_map, 'epochs': args.epochs, 'seed': args.seed,
            'target_labels_per_class': args.target_labels_per_class,
            'architecture': 'paper Office-31 VGG-16 conv + FC1024/128',
            'input_modalities': 'VIS_RGB+IR_grayscale',
            'multimodal_adapter': (
                'official RGB VGG-16 plus a lightweight IR residual encoder; '
                'CCSA objective and official 1024/128 head unchanged'
            ),
            'source_initialization_epochs': args.source_epochs,
            'source_initialization_checkpoint': source_provenance,
            'pretrained_backbone': args.pretrained_backbone or (
                'torchvision ImageNet download' if download_pretrained else None
            ),
            'strict_checkpoint_loading': True,
            'pair_protocol': (
                'all positives + up to 3x unique negatives; '
                'two updates per batch'
            ),
            'positive_pairs': pair_dataset.positive_pairs,
            'negative_pairs': pair_dataset.negative_pairs,
            'max_train_batches': args.max_train_batches,
            'completed_epochs': completed_epoch,
            'stopped_early': stopped_early,
            'early_stopping': {
                'monitor': 'target_val ACC',
                'patience': args.early_stopping_patience,
                'min_epochs': args.early_stopping_min_epochs,
            },
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
