"""
CCSA (Contrastive Semantic Alignment, ICCV 2017) ported to the JMDA-Net weather
dataset under the paper's SDA (few-shot supervised domain adaptation) protocol.

Official implementation used as the reference:
    https://github.com/samotiian/CCSA   (Keras, Python 2: Initialization.py / main.py)

Faithfulness notes -- every deviation from the released code is listed so the
port can be audited:

  * Encoder g.  ``Create_Model()`` is ported with its layer topology unchanged:
        conv(32, 3x3, valid) -> relu -> conv(32, 3x3, valid) -> relu
        -> maxpool(2) -> dropout(0.25) -> flatten
        -> dense(120) -> relu -> dense(84) -> relu
    The original consumes 16x16 grayscale digits; here the same topology is
    applied to 224x224 RGB weather images.  Efficiency benchmarking in this
    project requires an identical input size across methods, so no resizing is
    allowed -- the unavoidable consequence is that the first dense layer holds
    ~46.5M weights (387200 -> 120).  This is reported, not hidden.
  * Classifier h.  ``Dropout(0.5) -> Dense(nb_classes) -> softmax`` is kept.
  * Loss.  ``(1 - alpha) * classification + alpha * CSA`` with ``alpha = 0.25``
    is kept.  ``euclidean_distance`` and ``contrastive_loss`` (margin = 1) are
    ported verbatim from ``Initialization.py``.
  * Pair construction.  ``Create_Pairs()`` semantics are kept: every same-class
    source/target pair is a positive, negatives are sampled at a 3:1 ratio, and
    every pair carries ``yc = 1`` when the two samples share a label.  The
    original materialises the whole pair table before training because its
    target pool is tiny (<= 7 images/class); here pairs are rebuilt on the fly
    from the current source/target mini-batches, which is the same construction
    sampled per step instead of exhaustively.
  * Update order.  The original runs TWO ``train_on_batch`` calls per pair batch
    (source branch then target branch) over the *same* pair set and the *same*
    CSA term.  We keep the identical pair set and CSA term, but combine the two
    branch classification terms into a single optimizer step with the unchanged
    (1 - alpha) / alpha weighting.
  * Optimizer.  Adadelta (Keras' 'adadelta' default) with batch size 256, as in
    ``training_the_model()``.

Protocol: source = sunny (labelled), target = <weather> SDA labelled subset
(``data_lists/<target>_sda_<N>shot_seed<S>.txt``).  The target classifier branch
is trained only on that labelled subset; the unlabelled target split is used
exclusively for evaluation, exactly like the other SDA scripts.

Usage:
    python CCSA/train_ccsa.py --target backlight --epochs 20 --seed 42 \
        --efficiency_output_dir efficiency_results/uda_sunny
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Cross-platform data path remapping (server: export JMDA_DATA_ROOT=...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from data_paths import remap_data_path
from classification_metrics import classification_metrics, format_classification_metrics
from reproducibility import seed_everything

NUM_CLASSES = 14


# ── data ────────────────────────────────────────────────────────────────────
class ImageListDataset(Dataset):
    """Reads (path, label) pairs from a text list file."""

    def __init__(self, list_path, transform=None):
        self.samples = []
        with open(list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                self.samples.append((remap_data_path(parts[0]), int(parts[1])))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# ── losses ported verbatim from Initialization.py ───────────────────────────
def euclidean_distance(x, y, eps=1e-08):
    """``K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))``"""
    return torch.sqrt(torch.clamp(((x - y) ** 2).sum(dim=1, keepdim=True), min=eps))


def contrastive_loss(y_true, y_pred, margin=1.0):
    """``y_true * y_pred**2 + (1 - y_true) * max(margin - y_pred, 0)**2``"""
    y_true = y_true.float().view(-1, 1)
    return (y_true * y_pred ** 2 +
            (1.0 - y_true) * torch.clamp(margin - y_pred, min=0.0) ** 2).mean()


# ── model ───────────────────────────────────────────────────────────────────
class CCSA(nn.Module):
    """Encoder g (Create_Model) + classifier h (Dropout-Dense-softmax)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # g: identical topology to Initialization.Create_Model()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),   # Convolution2D(32,(3,3),'valid')
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
        # h: Dropout(0.5) -> Dense(num_classes) -> softmax (softmax folded into CE)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(84, num_classes),
        )

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x1, x2=None):
        """Single-stream returns logits; two-stream returns both branches + embeddings."""
        f1 = self.embed(x1)
        logits1 = self.classifier(f1)
        if x2 is None:
            return logits1
        f2 = self.embed(x2)
        logits2 = self.classifier(f2)
        return logits1, logits2, f1, f2


# ── pair construction (Create_Pairs semantics) ──────────────────────────────
def build_pairs(src_labels, tgt_labels, negative_ratio=3):
    """Return (src_idx, tgt_idx, yc) with all positives + 3:1 negatives."""
    eq = src_labels.view(-1, 1).eq(tgt_labels.view(1, -1))
    pos = eq.nonzero(as_tuple=False)
    neg = (~eq).nonzero(as_tuple=False)
    neg = neg[torch.randperm(neg.size(0), device=neg.device)]
    keep = min(neg.size(0), negative_ratio * max(pos.size(0), 1))
    neg = neg[:keep]
    pairs = torch.cat([pos, neg], dim=0)
    yc = torch.cat([
        torch.ones(pos.size(0), device=eq.device),
        torch.zeros(neg.size(0), device=eq.device),
    ])
    order = torch.randperm(pairs.size(0), device=pairs.device)
    return pairs[order, 0], pairs[order, 1], yc[order]


# ── evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            all_labels.extend(labels.tolist())
            all_predictions.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(all_labels, all_predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='backlight',
                        choices=['backlight', 'rainy', 'foggy', 'dark'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Per-domain mini-batch used to sample pairs; the '
                             'original uses 256 pairs per update.')
    parser.add_argument('--lr', type=float, default=1.0,
                        help="Adadelta learning rate (Keras' 'adadelta' default).")
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='Loss = (1-alpha)*classification + alpha*CSA')
    parser.add_argument('--target-labels-per-class', type=int, default=10)
    parser.add_argument('--sda-seed', type=int, default=42)
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

    train_tf, eval_tf = get_transforms()
    src_list = os.path.join(args.data_dir, 'sunny_train.txt')
    tgt_list = os.path.join(
        args.data_dir,
        f'{args.target}_sda_{args.target_labels_per_class}shot_seed{args.sda_seed}.txt')
    tgt_val_list = os.path.join(args.data_dir, f'{args.target}_val.txt')
    assert os.path.isfile(tgt_list), (
        f'SDA subset list missing: {tgt_list}. Run tools/make_sda_lists.py first.')

    src_ds = ImageListDataset(src_list, transform=train_tf)
    tgt_ds = ImageListDataset(tgt_list, transform=train_tf)
    tgt_val_ds = ImageListDataset(tgt_val_list, transform=eval_tf)
    print(f'Source train: {len(src_ds)} images')
    print(f'Target train ({args.target} SDA {args.target_labels_per_class}-shot): {len(tgt_ds)} images')
    print(f'Target val:   {len(tgt_val_ds)} images')

    src_loader = DataLoader(src_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)
    # The SDA subset can be smaller than one full batch (e.g. rainy 10-shot has
    # only ~60 images), so drop_last must stay False for the target loader.
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=False)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    model = CCSA().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    eff = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'CCSA',
                               seed=args.seed, device=str(device), modalities='VIS',
                               protocol='SDA',
                               target_label_count=len(tgt_ds),
                               target_labels_per_class=args.target_labels_per_class,
                               backbone='CCSA-CNN (conv32x2 + fc120/fc84)',
                               pretraining='None (trained from scratch)')
        print(f'Efficiency logging -> {eff.run_dir}')

    best_acc, final_acc = -float('inf'), None
    for epoch in range(args.epochs):
        if eff:
            eff.epoch_start(epoch)
        model.train()
        src_iter, tgt_iter = iter(src_loader), iter(tgt_loader)
        n_batches = max(len(src_loader), len(tgt_loader))

        for i in range(n_batches):
            if eff:
                eff.step_start()
            try:
                src_imgs, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_imgs, src_labels = next(src_iter)
            try:
                tgt_imgs, tgt_labels = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_imgs, tgt_labels = next(tgt_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs, tgt_labels = tgt_imgs.to(device), tgt_labels.to(device)
            if eff:
                eff.dataloader_done()

            # One forward pass per domain; embeddings are shared with the CSA term.
            logits_s, logits_t, f_s, f_t = model(src_imgs, tgt_imgs)
            src_idx, tgt_idx, yc = build_pairs(src_labels, tgt_labels)
            distance = euclidean_distance(f_s[src_idx], f_t[tgt_idx])

            cls_loss = 0.5 * (criterion(logits_s, src_labels) +
                              criterion(logits_t, tgt_labels))
            csa_loss = contrastive_loss(yc, distance)
            loss = (1.0 - args.alpha) * cls_loss + args.alpha * csa_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'cls={cls_loss.item():.4f} csa={csa_loss.item():.4f} '
                      f'pairs={src_idx.numel()}')

        metrics = evaluate(model, tgt_val_loader, device)
        final_acc = metrics['accuracy']
        best_acc = max(best_acc, final_acc)
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {final_acc:.4f}')
        print(f'  {format_classification_metrics(metrics)}')
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'ccsa_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'final_acc': final_acc,
                                    'target': args.target, 'protocol': 'SDA'})
    print(f'\nCCSA finished. Target val accuracy ({args.target}): final={final_acc:.4f} '
          f'best={best_acc:.4f}')


if __name__ == '__main__':
    main()
