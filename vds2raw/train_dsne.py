"""d-SNE on the reviewed VDS2Raw cross-day supervised-DA protocol."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.domain_pairing import CrossDomainPairDataset
from tools.reproducibility import seed_everything
from vds2raw.common import (
    better_metrics,
    evaluate_rgb,
    load_selected_checkpoint,
    metric_line,
    save_checkpoint,
    save_experiment_results,
)
from vds2raw.dataset import (
    RGBView,
    VDS2RawDataset,
    audit_dataset,
    load_or_compute_statistics,
)
from vds2raw.models import DSNE96, dsne_loss


def train_direction(
    model,
    optimizer,
    criterion,
    first_images,
    first_labels,
    second_images,
    second_labels,
    args,
):
    first_logits, first_features = model(first_images, return_embedding=True)
    _, second_features = model(second_images, return_embedding=True)
    cross_entropy = criterion(first_logits, first_labels)
    # Preserve the release's argument direction exactly: the branch receiving
    # classification supervision is passed as fts/ys, and the paired branch
    # is passed as ftt/yt.
    adaptation = dsne_loss(
        first_features,
        first_labels,
        second_features,
        second_labels,
        margin=args.margin,
        feature_normalization=args.fn,
    )
    total = (1.0 - args.alpha) * cross_entropy + args.alpha * adaptation
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return cross_entropy.detach(), adaptation.detach()


def main():
    parser = argparse.ArgumentParser(description="d-SNE on VDS2Raw cross-day SDA")
    parser.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "vds2raw"))
    parser.add_argument("--stats-file", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--nlayers", type=int, default=101, choices=[18, 50, 101, 152])
    parser.add_argument("--backbone-checkpoint", default="")
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--l2n", action="store_true")
    parser.add_argument("--fn", action="store_true")
    parser.add_argument("--train-src", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    args.seed = seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit:
        print("Dataset audit:", audit_dataset(data_root))
    stats_file = args.stats_file or str(save_dir / "normalization_stats.json")
    statistics = load_or_compute_statistics(data_root, stats_file)
    print("Normalization:", statistics.to_dict())

    source = RGBView(VDS2RawDataset(data_root, "source_train", statistics))
    target = RGBView(VDS2RawDataset(data_root, "target_train", statistics))
    target_val = RGBView(VDS2RawDataset(data_root, "target_val", statistics))
    pairs = CrossDomainPairDataset(
        source, target, negative_ratio=args.ratio, seed=args.seed
    )
    print(
        f"Official d-SNE pair table: {len(pairs)} pairs "
        f"({pairs.positive_pairs} positive, {pairs.negative_pairs} negative)"
    )
    generator = torch.Generator().manual_seed(args.seed)
    pair_loader = DataLoader(
        pairs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        target_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSNE96(
        embed_size=args.embed_size,
        nlayers=args.nlayers,
        dropout=args.dropout,
        l2_normalize=args.l2n,
    ).to(device)
    if args.backbone_checkpoint:
        model.load_backbone_checkpoint(args.backbone_checkpoint)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    stem = f"vds2raw_dsne_seed{args.seed}"
    checkpoint = save_dir / f"{stem}_best.pth"
    metadata = {
        "method": "d-SNE",
        "seed": args.seed,
        "epochs": args.epochs,
        "data_root": str(data_root),
        "source_partition": "source_train",
        "target_train_partition": "target_train",
        "selection_partition": "target_val",
        "final_partition": "target_test",
        "modalities": ["RGB"],
        "num_classes": 3,
        "backbone": f"ResNetV2-{args.nlayers}",
        "backbone_checkpoint": args.backbone_checkpoint or None,
        "train_source_direction": args.train_src,
        "normalization": statistics.to_dict(),
    }
    history, best_metrics = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(pair_loader, 1):
            source_images, source_labels, target_images, target_labels, _ = batch
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)
            target_labels = target_labels.to(device)
            if args.train_src:
                train_direction(
                    model,
                    optimizer,
                    criterion,
                    source_images,
                    source_labels,
                    target_images,
                    target_labels,
                    args,
                )
            target_ce, adaptation = train_direction(
                model,
                optimizer,
                criterion,
                target_images,
                target_labels,
                source_images,
                source_labels,
                args,
            )
            if batch_index % 10 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(pair_loader)}] "
                    f"target_ce={target_ce.item():.4f} dsne={adaptation.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        val_metrics = evaluate_rgb(model, val_loader, device)
        history.append((epoch, val_metrics))
        print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_val", val_metrics))
        if better_metrics(val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            save_checkpoint(checkpoint, model, epoch, val_metrics, metadata)

    selected = load_selected_checkpoint(checkpoint, model, device)
    del target_val, val_loader
    target_test = RGBView(VDS2RawDataset(data_root, "target_test", statistics))
    test_loader = DataLoader(
        target_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_metrics = evaluate_rgb(model, test_loader, device)
    print(metric_line("FINAL target_test", test_metrics))
    save_experiment_results(
        save_dir,
        stem,
        history,
        selected["selected_epoch"],
        selected["target_val_metrics"],
        test_metrics,
        metadata,
    )


if __name__ == "__main__":
    main()
