"""
d-SNE (Domain Adaptive Stochastic Neighbor Embedding, CVPR 2019) ported to the
JMDA-Net weather dataset under the paper's SDA (few-shot supervised domain
adaptation) protocol.

Official implementation used as the reference:
    https://github.com/aws-samples/d-SNE
    Key files ported: ``train_val/custom_layers.py`` (dSNELoss),
    ``train_val/training_sda.py`` (dSNE.train_epoch / train_batch) and
    ``models/resnet.py`` (ResNetV2 embedding head).

Faithfulness notes -- every deviation is listed so the port can be audited:

  * Encoder.  The released code uses an mxnet ``ResNetV2`` trunk whose head is
        features -> GlobalAvgPool -> Dense(4096, relu) -> [Dropout] ->
        Dense(embed_size=512, relu) -> [Dropout] -> [L2Normalization]
        -> Dense(num_classes)
    The head topology and the embedding size (512) are preserved verbatim; only
    the convolution trunk is replaced by the project-standard torchvision
    ``ResNet-18`` (the release defaults to ``resnet101``).  This is the only
    encoder substitution and it keeps the efficiency comparison like-for-like.
  * dSNE loss.  ``dSNELoss.hybrid_forward`` is ported line by line, including
    the ``fn`` feature L2-normalisation switch, the ``y_same`` / ``y_diff``
    masks, the ``where(y_same, max_dists, inter_cls_dists)`` revised inter-class
    distance and the ``relu(max_intra - min_inter + margin)`` hinge, with
    ``margin = 1`` and a batch mean (mxnet ``Loss`` with batch_axis=0).
  * Total loss.  ``(1 - alpha) * xent + alpha * dSNE`` with ``alpha = 0.1``
    (``--alpha`` default in ``parse_args_sda``), batch size 32 (``--bs``).
  * Update order.  ``dSNE.train_epoch`` calls ``train_batch(src, tgt)`` and then
    ``train_batch(tgt, src)``; the second call supervises the target branch
    because the arguments are swapped.  The released code additionally gates the
    source-direction update behind ``--train-src`` (default off).  We keep the
    identical per-direction losses and the identical (1 - alpha) / alpha
    weighting, but run both directions and combine them symmetrically in one
    optimizer step:  0.5 * (L_src->tgt + L_tgt->src) for both terms.
  * Optimizer.  sgd, lr 1e-3, momentum 0.9, weight decay 1e-4 (the released
    defaults), no LR schedule.
  * Feature normalisation is off by default, exactly like the released
    ``--fn`` / ``--l2n`` store_true defaults; both remain available as flags.

Protocol: source = sunny (labelled), target = <weather> SDA labelled subset
(``data_lists/<target>_sda_<N>shot_seed<S>.txt``); the unlabelled target split
is used only for evaluation.

Usage:
    python d-SNE/train_dsne.py --target backlight --epochs 20 --seed 42 \
        --efficiency_output_dir efficiency_results/uda_sunny
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

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


# ── loss ported verbatim from custom_layers.dSNELoss ────────────────────────
def dsne_loss(fts, ys, ftt, yt, margin=1.0, fn=False):
    """Semantic alignment loss of d-SNE.

    Args:
        fts: source features [M, K]
        ys:  source labels   [M]
        ftt: target features [N, K]
        yt:  target labels   [N]
    Returns:
        scalar (mxnet ``Loss`` with batch_axis=0 averages over the batch)
    """
    if fn:
        fts = F.normalize(fts, p=2, dim=1)
        ftt = F.normalize(ftt, p=2, dim=1)

    # dists[n, m] = || ftt[n] - fts[m] ||^2
    dists = ((ftt.unsqueeze(1) - fts.unsqueeze(0)) ** 2).sum(dim=2)

    y_same = yt.view(-1, 1).eq(ys.view(1, -1)).float()
    y_diff = 1.0 - y_same

    intra_cls_dists = dists * y_same
    inter_cls_dists = dists * y_diff

    max_dists = dists.max(dim=1, keepdim=True).values.expand_as(dists)
    revised_inter_cls_dists = torch.where(y_same.bool(), max_dists, inter_cls_dists)

    max_intra_cls_dist = intra_cls_dists.max(dim=1).values
    min_inter_cls_dist = revised_inter_cls_dists.min(dim=1).values

    loss = F.relu(max_intra_cls_dist - min_inter_cls_dist + margin)
    return loss.mean()


# ── model ───────────────────────────────────────────────────────────────────
class DSNE(nn.Module):
    """ResNet trunk + 4096/embed_size embedding head + linear classifier."""

    def __init__(self, num_classes=NUM_CLASSES, embed_size=512, nlayers=18,
                 pretrained=True, use_dropout=False, use_l2n=False):
        super().__init__()
        if nlayers != 18:
            raise NotImplementedError('Only the project-standard ResNet-18 trunk is wired up.')
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        trunk = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(trunk.children())[:-1], nn.Flatten())
        feat_dim = trunk.fc.in_features  # 512

        embeds = [nn.Linear(feat_dim, 4096), nn.ReLU(inplace=True)]
        if use_dropout:
            embeds.append(nn.Dropout(0.5))
        embeds += [nn.Linear(4096, embed_size), nn.ReLU(inplace=True)]
        if use_dropout:
            embeds.append(nn.Dropout(0.5))
        self.embeds = nn.Sequential(*embeds)
        self.use_l2n = use_l2n
        self.output = nn.Linear(embed_size, num_classes)

    def forward(self, x, return_embedding=False):
        f = self.embeds(self.features(x))
        if self.use_l2n:
            f = F.normalize(f, p=2, dim=1)
        logits = self.output(f)
        if return_embedding:
            return logits, f
        return logits


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
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Loss = (1-alpha)*xent + alpha*dSNE')
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument('--dropout', action='store_true',
                        help='Use the release\'s optional 0.5 dropout in the embedding head.')
    parser.add_argument('--l2n', action='store_true',
                        help='L2-normalise the embedding (release --l2n).')
    parser.add_argument('--fn', action='store_true',
                        help='L2-normalise features inside the dSNE loss (release --fn).')
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
    # drop_last=False: the SDA subset may be smaller than one full batch.
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=False)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=2)

    model = DSNE(num_classes=NUM_CLASSES, embed_size=args.embed_size,
                 use_dropout=args.dropout, use_l2n=args.l2n).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    eff = None
    if args.efficiency_output_dir:
        from efficiency_common import EfficiencyLogger
        eff = EfficiencyLogger(args.efficiency_output_dir, 'd-SNE',
                               seed=args.seed, device=str(device), modalities='VIS',
                               protocol='SDA',
                               target_label_count=len(tgt_ds),
                               target_labels_per_class=args.target_labels_per_class,
                               backbone='ResNet-18 (original: mxnet ResNet-101)',
                               pretraining='ImageNet')
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

            logits_s, f_s = model(src_imgs, return_embedding=True)
            logits_t, f_t = model(tgt_imgs, return_embedding=True)

            xent = 0.5 * (criterion(logits_s, src_labels) +
                          criterion(logits_t, tgt_labels))
            # Symmetric dSNE: the released code runs src->tgt and tgt->src as
            # two separate updates over the same batch.
            aux = 0.5 * (dsne_loss(f_s, src_labels, f_t, tgt_labels,
                                   margin=args.margin, fn=args.fn) +
                         dsne_loss(f_t, tgt_labels, f_s, src_labels,
                                   margin=args.margin, fn=args.fn))
            loss = (1.0 - args.alpha) * xent + args.alpha * aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if eff:
                eff.step_end()

            if (i + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{args.epochs}] batch [{i+1}/{n_batches}] '
                      f'xent={xent.item():.4f} dsne={aux.item():.4f}')

        metrics = evaluate(model, tgt_val_loader, device)
        final_acc = metrics['accuracy']
        best_acc = max(best_acc, final_acc)
        print(f'Epoch [{epoch+1}/{args.epochs}] target val accuracy: {final_acc:.4f}')
        print(f'  {format_classification_metrics(metrics)}')
        if eff:
            eff.epoch_end(epoch)

    torch.save(model.state_dict(), os.path.join(args.save_dir, f'dsne_{args.target}_last.pth'))
    if eff:
        eff.finalize(extra_metrics={'best_acc': best_acc, 'final_acc': final_acc,
                                    'target': args.target, 'protocol': 'SDA'})
    print(f'\nd-SNE finished. Target val accuracy ({args.target}): final={final_acc:.4f} '
          f'best={best_acc:.4f}')


if __name__ == '__main__':
    main()
