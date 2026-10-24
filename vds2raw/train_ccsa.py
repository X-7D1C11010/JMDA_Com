"""CCSA on the reviewed VDS2Raw cross-day supervised-DA protocol."""

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
from vds2raw.models import CCSA96, ccsa_contrastive_loss, euclidean_distance


def main():
    parser = argparse.ArgumentParser(description="CCSA on VDS2Raw cross-day SDA")
    parser.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "vds2raw"))
    parser.add_argument("--stats-file", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Smoke-test only; 0 uses every pair batch.",
    )
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
    pairs = CrossDomainPairDataset(source, target, negative_ratio=3, seed=args.seed)
    print(
        f"Official CCSA pair table: {len(pairs)} pairs "
        f"({pairs.positive_pairs} positive, {pairs.negative_pairs} negative)"
    )
    generator = torch.Generator().manual_seed(args.seed)
    pair_loader = DataLoader(
        pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        generator=generator,
    )
    if len(pair_loader) == 0:
        raise ValueError("pair table is smaller than one full batch")
    val_loader = DataLoader(
        target_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCSA96().to(device)
    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, rho=args.rho, eps=args.eps
    )
    criterion = nn.CrossEntropyLoss()
    stem = f"vds2raw_ccsa_seed{args.seed}"
    checkpoint = save_dir / f"{stem}_best.pth"
    metadata = {
        "method": "CCSA",
        "seed": args.seed,
        "epochs": args.epochs,
        "data_root": str(data_root),
        "source_partition": "source_train",
        "target_train_partition": "target_train",
        "selection_partition": "target_val",
        "final_partition": "target_test",
        "modalities": ["RGB"],
        "num_classes": 3,
        "pair_protocol": "all positives + 3x negatives; source and target updates",
        "normalization": statistics.to_dict(),
    }
    history, best_metrics = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(pair_loader, 1):
            source_images, source_labels, target_images, target_labels, pair_labels = (
                batch
            )
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)
            target_labels = target_labels.to(device)
            pair_labels = pair_labels.to(device)

            source_logits, _, source_features, target_features = model(
                source_images, target_images
            )
            source_ce = criterion(source_logits, source_labels)
            source_csa = ccsa_contrastive_loss(
                pair_labels, euclidean_distance(source_features, target_features)
            )
            source_loss = (1.0 - args.alpha) * source_ce + args.alpha * source_csa
            optimizer.zero_grad()
            source_loss.backward()
            optimizer.step()

            target_logits, _, target_features, source_features = model(
                target_images, source_images
            )
            target_ce = criterion(target_logits, target_labels)
            target_csa = ccsa_contrastive_loss(
                pair_labels, euclidean_distance(target_features, source_features)
            )
            target_loss = (1.0 - args.alpha) * target_ce + args.alpha * target_csa
            optimizer.zero_grad()
            target_loss.backward()
            optimizer.step()
            if batch_index % 10 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(pair_loader)}] source_ce={source_ce.item():.4f} "
                    f"target_ce={target_ce.item():.4f}"
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
