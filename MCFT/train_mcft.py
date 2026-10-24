"""MCFT multimodal Transformer training on paired JMDA VIS+IR images.

Reference: https://github.com/SYFYN0317/MCFT.  The model and optimiser recipe
follow ``MViT_pytorch_upload.py`` and ``MViT_demo_h2013.py``: 15x15 inputs,
dim=64, depth=6, four heads, CE + 0.1*ITA + ITM, Adam 5e-4, StepLR(20, 0.9),
batch 64, 500 epochs and evaluation every 10 epochs.  HSI+LiDAR cannot be
recreated from this dataset, so the explicit modality adaptation is HSI->RGB
VIS (3 channels) and LiDAR->grayscale IR (1 channel).  Training remains
source-supervised, with sunny examples restricted to the target weather's
known class support.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classification_metrics import classification_metrics, format_classification_metrics
from data_paths import exact_ir_partner, remap_data_path
from metrics_io import save_metric_history
from reproducibility import seed_everything
from uda_data import uda_label_space

from mcft_model import MViT


PATCH_SIZE = 15


class PairedVisIrDataset(Dataset):
    """Exact-stem VIS/IR pairs filtered into a compact target label space."""

    def __init__(self, list_path, label_map, patch_size=PATCH_SIZE,
                 horizontal_flip=False):
        self.samples = []
        self.dropped_unpaired = 0
        self.skipped = 0
        with open(list_path, 'r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                path, raw_label = line.split()
                raw_label = int(raw_label)
                if raw_label not in label_map:
                    self.skipped += 1
                    continue
                vis_path = remap_data_path(path)
                ir_path = exact_ir_partner(vis_path)
                if ir_path is None:
                    self.dropped_unpaired += 1
                    continue
                self.samples.append((vis_path, ir_path, label_map[raw_label]))
        if not self.samples:
            raise ValueError(f'no paired sample remains after filtering {list_path}')
        self.patch_size = patch_size
        self.horizontal_flip = horizontal_flip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        vis_path, ir_path, label = self.samples[index]
        vis_image = Image.open(vis_path).convert('RGB')
        ir_image = Image.open(ir_path).convert('L')
        output_size = [self.patch_size, self.patch_size]
        vis_image = TF.resize(vis_image, output_size)
        ir_image = TF.resize(ir_image, output_size)
        if self.horizontal_flip and bool(torch.rand(()) < 0.5):
            vis_image = TF.hflip(vis_image)
            ir_image = TF.hflip(ir_image)
        # The release min-max normalises HSI/LiDAR bands to [0, 1].  ToTensor
        # provides the same numeric range for 8-bit VIS/IR without imposing
        # unrelated ImageNet statistics.
        return TF.to_tensor(vis_image), TF.to_tensor(ir_image), label


def evaluate(model, loader, device, num_classes):
    model.eval()
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for vis, ir, labels in loader:
            vis = vis.to(device)
            ir = ir.to(device)
            logits, _, _ = model(
                vis, ir, label=None, compute_auxiliary=False
            )
            labels_all.extend(labels.tolist())
            predictions_all.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(labels_all, predictions_all, num_classes=num_classes)


def build_model(num_classes):
    """Released H2013 configuration with dataset-specific input channels."""
    return MViT(
        patch_size=PATCH_SIZE,
        num_patches=[3, 1],
        num_classes=num_classes,
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=32,
        dropout=0.1,
        emb_dropout=0.1,
        mode='MViT',
    )


def main():
    parser = argparse.ArgumentParser(description='Official-recipe MCFT VIS+IR port')
    parser.add_argument('--target', default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=500,
                        help='Released MCFT budget: 500 epochs.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-frequency', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--loss-ita-weight', type=float, default=0.1)
    parser.add_argument('--horizontal-flip', action='store_true',
                        help='Optional VIS/IR augmentation; off in the release-equivalent default.')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-train-batches', type=int, default=0,
                        help='Smoke-test only: cap batches per epoch; 0 uses all data.')
    parser.add_argument('--data-dir', default=os.path.join(ROOT, 'data_lists'))
    parser.add_argument('--save-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--efficiency_output_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.eval_frequency <= 0:
        raise ValueError('--eval-frequency must be positive')
    args.seed = seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Controlled random seed: {args.seed}')
    print(f'Device: {device}')
    if args.max_train_batches:
        print(f'[SMOKE MODE] limiting each epoch to {args.max_train_batches} batches')

    class_ids, label_map = uda_label_space(args.data_dir, args.target)
    num_classes = len(class_ids)
    print(
        f'Target-driven label space: {num_classes} classes, original ids '
        f'{class_ids}, remapped to 0..{num_classes - 1}'
    )

    source = PairedVisIrDataset(
        os.path.join(args.data_dir, 'sunny_train.txt'), label_map,
        horizontal_flip=args.horizontal_flip,
    )
    target_val = PairedVisIrDataset(
        os.path.join(args.data_dir, f'{args.target}_val.txt'), label_map,
        horizontal_flip=False,
    )
    print(
        f'Source train (sunny VIS+IR): kept={len(source)}, '
        f'dropped_label={source.skipped}, dropped_unpaired={source.dropped_unpaired}'
    )
    print(
        f'Target val ({args.target} VIS+IR): kept={len(target_val)}, '
        f'dropped_label={target_val.skipped}, '
        f'dropped_unpaired={target_val.dropped_unpaired}'
    )

    source_loader = DataLoader(
        source, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    target_val_loader = DataLoader(
        target_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    efficiency = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        efficiency = EfficiencyLogger(
            args.efficiency_output_dir, 'MCFT', seed=args.seed,
            device=str(device), modalities='VIS+IR', protocol='Supervised',
            target_label_count=0, target_labels_per_class=0,
            backbone='MCFT MViT released architecture (VIS3 + IR1 adaptation)',
            pretraining='None (trained from scratch)',
        )
        print(f'Efficiency logging -> {efficiency.run_dir}')

    history = []
    best_metrics = None
    best_epoch = None
    run_stem = f'mcft_{args.target}_seed{args.seed}'
    best_checkpoint = os.path.join(args.save_dir, f'{run_stem}_best.pth')
    for epoch in range(1, args.epochs + 1):
        if efficiency:
            efficiency.epoch_start(epoch - 1)
        model.train()
        for batch_index, (vis, ir, labels) in enumerate(source_loader, 1):
            if efficiency:
                efficiency.step_start()
            vis = vis.to(device)
            ir = ir.to(device)
            labels = labels.to(device)
            if efficiency:
                efficiency.dataloader_done()

            logits, loss_ita, loss_itm = model(vis, ir, labels)
            classification_loss = criterion(logits, labels)
            loss = classification_loss + args.loss_ita_weight * loss_ita + loss_itm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if efficiency:
                efficiency.step_end()

            if batch_index % 5 == 0:
                print(
                    f'  Epoch [{epoch}/{args.epochs}] '
                    f'batch [{batch_index}/{len(source_loader)}] '
                    f'ce={classification_loss.item():.4f} '
                    f'ita={loss_ita.item():.4f} itm={loss_itm.item():.4f}'
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        scheduler.step()

        should_evaluate = ((epoch - 1) % args.eval_frequency == 0) or epoch == args.epochs
        if should_evaluate:
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
            'method': 'MCFT', 'protocol': 'source-supervised',
            'source': 'sunny', 'target': args.target,
            'class_ids_original_zero_based': class_ids, 'label_map': label_map,
            'epochs': args.epochs, 'eval_frequency': args.eval_frequency,
            'seed': args.seed, 'input_modalities': 'VIS_RGB+IR_grayscale',
            'official_modality_adaptation': 'HSI->VIS; LiDAR->IR',
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
            'target': args.target, 'protocol': 'Supervised',
        })

    print(f'Final target val accuracy: {final_metrics["accuracy"]:.4f}')
    print(f'Best target val accuracy: {best_metrics["accuracy"]:.4f} (epoch {best_epoch})')
    print(f'Metrics CSV: {metrics_csv}')
    print(f'Metrics JSON: {metrics_json}')


if __name__ == '__main__':
    main()
