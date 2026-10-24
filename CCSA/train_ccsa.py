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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from classification_metrics import classification_metrics, format_classification_metrics
from domain_pairing import CrossDomainPairDataset
from metrics_io import save_metric_history
from reproducibility import seed_everything
from uda_data import UdaImageListDataset, format_dropped, uda_label_space


IMAGE_SIZE = 224


def get_transforms():
    """Static preprocessing, matching the release's non-augmented pair table."""
    common = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(common), transforms.Compose(common)


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


class CCSA(nn.Module):
    """Released ``Create_Model`` encoder and Dropout-Dense classifier."""

    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * 110 * 110, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(84, num_classes))

    def embed(self, images):
        return self.encoder(images)

    def forward(self, first, second=None):
        first_features = self.embed(first)
        first_logits = self.classifier(first_features)
        if second is None:
            return first_logits
        second_features = self.embed(second)
        second_logits = self.classifier(second_features)
        return first_logits, second_logits, first_features, second_features


def evaluate(model, loader, device, num_classes):
    model.eval()
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            labels_all.extend(labels.tolist())
            predictions_all.extend(predictions)
    return classification_metrics(labels_all, predictions_all, num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser(description='Official-recipe CCSA SDA port')
    parser.add_argument('--target', default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=80,
                        help='Released CCSA budget: 80 epochs.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Released pair batch size: 256.')
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0.95,
                        help='Keras Adadelta default used by the release.')
    parser.add_argument('--eps', type=float, default=1e-7,
                        help='Keras Adadelta epsilon used by the release.')
    parser.add_argument('--alpha', type=float, default=0.25)
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

    train_tf, eval_tf = get_transforms()
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

    model = CCSA(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, rho=args.rho, eps=args.eps
    )
    criterion = nn.CrossEntropyLoss()

    efficiency = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        efficiency = EfficiencyLogger(
            args.efficiency_output_dir, 'CCSA', seed=args.seed,
            device=str(device), modalities='VIS', protocol='SDA',
            target_label_count=len(target),
            target_labels_per_class=args.target_labels_per_class,
            backbone='CCSA released CNN/MLP topology (RGB input adaptation)',
            pretraining='None (trained from scratch)',
        )
        print(f'Efficiency logging -> {efficiency.run_dir}')

    history = []
    best_metrics = None
    best_epoch = None
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
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)
            target_labels = target_labels.to(device)
            pair_labels = pair_labels.to(device)
            if efficiency:
                efficiency.dataloader_done()

            source_logits, _, source_features, target_features = model(
                source_images, target_images
            )
            source_ce = criterion(source_logits, source_labels)
            source_csa = contrastive_loss(
                pair_labels, euclidean_distance(source_features, target_features)
            )
            source_total = (1.0 - args.alpha) * source_ce + args.alpha * source_csa
            optimizer.zero_grad()
            source_total.backward()
            optimizer.step()

            target_logits, _, target_features, source_features = model(
                target_images, source_images
            )
            target_ce = criterion(target_logits, target_labels)
            target_csa = contrastive_loss(
                pair_labels, euclidean_distance(target_features, source_features)
            )
            target_total = (1.0 - args.alpha) * target_ce + args.alpha * target_csa
            optimizer.zero_grad()
            target_total.backward()
            optimizer.step()
            if efficiency:
                efficiency.step_end()

            if batch_index % 10 == 0:
                print(
                    f'  Epoch [{epoch}/{args.epochs}] '
                    f'batch [{batch_index}/{len(pair_loader)}] '
                    f'source_ce={source_ce.item():.4f} '
                    f'target_ce={target_ce.item():.4f} '
                    f'csa={0.5 * (source_csa.item() + target_csa.item()):.4f}'
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
            'method': 'CCSA', 'protocol': 'SDA', 'target': args.target,
            'source': 'sunny', 'class_ids_original_zero_based': class_ids,
            'label_map': label_map, 'epochs': args.epochs, 'seed': args.seed,
            'target_labels_per_class': args.target_labels_per_class,
            'pair_protocol': 'all positives + 3x negatives; two updates per batch',
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
