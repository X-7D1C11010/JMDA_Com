"""
MCFT training entry point for the JMDA-Net weather dataset (multimodal
VIS + IR classification).

What MCFT is (and is not):  the paper ("MCFT: Multimodal Contrastive Fusion
Transformer", IEEE TGRS 2024) is a *supervised* HSI + LiDAR land-cover
classifier with a contrastive fusion transformer -- it is not a domain
adaptation method.  The weather analogue used here keeps the architecture and
the training recipe and swaps the modalities: HSI -> VIS, LiDAR -> IR.

Reference: https://github.com/SYFYN0317/MCFT
    ``MViT_pytorch_upload.py`` (model, ported to ``mcft_model.py``)
    ``MViT_demo_h2013.py``      (training loop)

Faithfulness notes:

  * Training recipe is the release's: Adam(lr=5e-4, weight_decay=0),
    StepLR(step_size=20, gamma=0.9), batch 64, loss
    ``CE(logits, y) + 0.1 * loss_ita + loss_itm``.
  * Hyper-parameters are the release's: patch_size=15, dim=64, depth=6, heads=4,
    mlp_dim=32, dropout=0.1, emb_dropout=0.1.  ``dim`` must stay equal to
    ``patch_size*patch_size*nout*4``/patch tokens (64) because the release's
    ``vision_proj_m`` linear layer is sized for exactly that flattened width.
  * The release feeds 15x15 HSI/LiDAR patches; here both modalities are resized
    to the model's required 15x15 patch grid.  This is the architecture's input
    contract, not a choice, and it is reported with every result.
  * Modalities are paired by VIS/IR filename stem, the same convention the
    project uses for JMDA-Net's ``PairedMultiModalDomainDataset``; samples whose
    stem has no partner are dropped (count printed, never silently mislabelled).
  * Supervision: the model is trained on the **source (sunny) training split
    only** and evaluated on each target weather split.  Test-time target labels
    are used for reporting only, never for model selection, and the saved
    checkpoint is always the last epoch.

Usage:
    python MCFT/train_mcft.py --target backlight --epochs 20 --seed 42 \
        --efficiency_output_dir efficiency_results/uda_sunny
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # MCFT/ has a hyphen-free name
from data_paths import remap_data_path, exact_ir_partner
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything

from mcft_model import MViT

NUM_CLASSES = 14
PATCH_SIZE = 15
VIS_MEAN, VIS_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IR_MEAN, IR_STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


# ── paired VIS/IR dataset ───────────────────────────────────────────────────
class PairedVisIrDataset(Dataset):
    """VIS (from the list file) paired with its exact-stem IR counterpart."""

    def __init__(self, list_path, patch_size=PATCH_SIZE, train=True):
        self.samples = []
        self.dropped_unpaired = 0
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                vis_path = remap_data_path(parts[0])
                ir_path = exact_ir_partner(vis_path)
                if ir_path is None:
                    self.dropped_unpaired += 1
                    continue
                self.samples.append((vis_path, ir_path, int(parts[1])))
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vis_path, ir_path, label = self.samples[idx]
        vis_img = Image.open(vis_path).convert('RGB')
        ir_img = Image.open(ir_path).convert('L').convert('RGB')
        size = [self.patch_size, self.patch_size]
        vis_img = TF.resize(vis_img, size)
        ir_img = TF.resize(ir_img, size)
        if self.train and bool(torch.rand(()) < 0.5):
            vis_img = TF.hflip(vis_img)
            ir_img = TF.hflip(ir_img)
        vis = transforms.Normalize(VIS_MEAN, VIS_STD)(TF.to_tensor(vis_img))
        ir = transforms.Normalize(IR_MEAN, IR_STD)(TF.to_tensor(ir_img))
        return vis, ir, label


def evaluate(model, loader, device, amp_enabled=False):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for vis, ir, labels in loader:
            vis, ir = vis.to(device), ir.to(device)
            logits, _, _ = model(vis, ir, labels.to(device))
            all_labels.extend(labels.tolist())
            all_predictions.extend(logits.argmax(1).cpu().tolist())
    if not all_labels:
        return None
    return classification_metrics(all_labels, all_predictions)


def build_model(num_classes=NUM_CLASSES, pretrained=False):
    """MViT with the release's H2013 demo hyper-parameters."""
    return MViT(
        patch_size=PATCH_SIZE,
        num_patches=[3, 3],
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--loss-ita-weight', type=float, default=0.1)
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             'data_lists'))
    parser.add_argument('--save-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
    parser.add_argument('--efficiency_output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.seed = seed_everything(args.seed)
    print(f'Controlled random seed: {args.seed}')

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    src_ds = PairedVisIrDataset(os.path.join(args.data_dir, 'sunny_train.txt'), train=True)
    tgt_val_ds = PairedVisIrDataset(os.path.join(args.data_dir, f'{args.target}_val.txt'), train=False)
    print(f'Source train (sunny, VIS+IR pairs): {len(src_ds)} samples '
          f'(dropped unpaired VIS: {src_ds.dropped_unpaired})')
    print(f'Target val ({args.target}, VIS+IR pairs): {len(tgt_val_ds)} samples '
          f'(dropped unpaired VIS: {tgt_val_ds.dropped_unpaired})')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                                gamma=args.gamma)

    eff = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'MCFT',
                               seed=args.seed, device=str(device), modalities='VIS+IR',
                               protocol='Supervised',
                               target_label_count=0, target_labels_per_class=0,
                               backbone='MCFT MViT (separable CNN + transformer, dim=64)',
                               pretraining='None (trained from scratch)')
        print(f'Efficiency logging -> {eff.run_dir}')

    final_metrics = None
    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        for i, (vis, ir, labels) in enumerate(src_loader):
            if eff:
                eff.step_start()
            vis, ir, labels = vis.to(device), ir.to(device), labels.to(device)
            if eff:
                eff.dataloader_done()

            logits, loss_ita, loss_itm = model(vis, ir, labels)
            loss = criterion(logits, labels) + args.loss_ita_weight * loss_ita + loss_itm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 5 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{len(src_loader)}] '
                      f'loss={loss.item():.4f} ita={loss_ita.item():.4f} itm={loss_itm.item():.4f}')
        scheduler.step()

        final_metrics = evaluate(model, tgt_val_loader, device)
        if final_metrics:
            print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: '
                  f'{final_metrics["accuracy"]:.4f}')
            print(f'  {format_classification_metrics(final_metrics)}')
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'mcft_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={
            'best_acc': final_metrics['accuracy'] if final_metrics else None,
            'final_acc': final_metrics['accuracy'] if final_metrics else None,
            'target': args.target, 'protocol': 'Supervised',
        })
    if final_metrics:
        print(f'\nMCFT finished. Target val accuracy ({args.target}, source-only supervised): '
              f'{final_metrics["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
