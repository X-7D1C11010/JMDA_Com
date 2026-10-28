"""Paper-faithful natural-image CCSA on the VDS2Raw cross-day protocol."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.domain_pairing import ControlledCrossDomainPairDataset
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
    NaturalImageRGBDataset,
    audit_dataset,
    fixed_samples_per_class,
    load_or_compute_statistics,
)
from vds2raw.models import CCSAOfficeModel, ccsa_contrastive_loss, euclidean_distance


def train_source_initialization(model, loader, device, args):
    """Initialize CCSA's new FC layers using all labelled source samples."""
    model.set_source_initialization_stage(True)
    optimizer = torch.optim.Adadelta(
        model.source_initialization_parameters(),
        lr=args.lr,
        rho=args.rho,
        eps=args.eps,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.source_epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            for start in range(0, images.shape[0], args.micro_batch_size):
                stop = min(start + args.micro_batch_size, images.shape[0])
                logits = model(images[start:stop])
                loss = criterion(logits, labels[start:stop])
                weight = (stop - start) / images.shape[0]
                (loss * weight).backward()
                total_loss += loss.item() * (stop - start)
                total_items += stop - start
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.source_epochs:
            print(
                f"CCSA source FC initialization [{epoch}/{args.source_epochs}] "
                f"CE={total_loss / max(total_items, 1):.4f}"
            )
    model.set_source_initialization_stage(False)


def train_direction(
    model,
    optimizer,
    criterion,
    first_images,
    first_labels,
    second_images,
    pair_labels,
    args,
):
    """One CCSA direction with an effective batch of ``batch_size`` pairs."""
    optimizer.zero_grad()
    count = first_images.shape[0]
    ce_total = 0.0
    csa_total = 0.0
    for start in range(0, count, args.micro_batch_size):
        stop = min(start + args.micro_batch_size, count)
        logits, _, first_features, second_features = model(
            first_images[start:stop], second_images[start:stop]
        )
        ce = criterion(logits, first_labels[start:stop])
        csa = ccsa_contrastive_loss(
            pair_labels[start:stop],
            euclidean_distance(first_features, second_features),
            margin=args.margin,
        )
        loss = (1.0 - args.alpha) * ce + args.alpha * csa
        weight = (stop - start) / count
        (loss * weight).backward()
        ce_total += ce.item() * (stop - start)
        csa_total += csa.item() * (stop - start)
    optimizer.step()
    return ce_total / count, csa_total / count


def load_model_state(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("state_dict", payload))
    model.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser(description="Official natural-image CCSA")
    parser.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "vds2raw"))
    parser.add_argument("--stats-file", default="")  # shared runner compatibility
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--source-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--pairs-per-epoch", type=int, default=768)
    parser.add_argument("--target-samples-per-class", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--pretrained-backbone", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    positive_values = (
        args.epochs,
        args.source_epochs,
        args.batch_size,
        args.source_batch_size,
        args.micro_batch_size,
        args.pairs_per_epoch,
        args.target_samples_per_class,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("epoch, batch, pair and per-class counts must be positive")
    if args.pairs_per_epoch % args.batch_size:
        raise ValueError("pairs-per-epoch must be a multiple of batch-size")
    if args.source_checkpoint and args.pretrained_backbone:
        raise ValueError(
            "source-checkpoint and pretrained-backbone are mutually exclusive"
        )

    args.seed = seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit:
        print("Dataset audit:", audit_dataset(data_root))
    statistics = load_or_compute_statistics(data_root, args.stats_file or None)

    source_train = NaturalImageRGBDataset(
        data_root, "source_train", statistics, augment=True, normalization="imagenet"
    )
    target_full = NaturalImageRGBDataset(
        data_root, "target_train", statistics, augment=True, normalization="imagenet"
    )
    target_train = fixed_samples_per_class(
        target_full, args.target_samples_per_class, args.seed
    )
    target_val = NaturalImageRGBDataset(
        data_root, "target_val", statistics, augment=False, normalization="imagenet"
    )
    print(
        f"CCSA labelled target subset: {len(target_train)} samples "
        f"({args.target_samples_per_class} per class)"
    )

    generator = torch.Generator().manual_seed(args.seed)
    source_loader = DataLoader(
        source_train,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    pairs = ControlledCrossDomainPairDataset(
        source_train,
        target_train,
        pairs_per_epoch=args.pairs_per_epoch,
        negative_ratio=3,
        seed=args.seed,
    )
    pair_loader = DataLoader(
        pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        target_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(
        f"Controlled CCSA pairs/epoch: {len(pairs)} "
        f"({pairs.positive_pairs} positive, {pairs.negative_pairs} negative, "
        f"{len(pair_loader)} effective batches)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCSAOfficeModel(
        pretrained=not args.no_pretrained and not args.source_checkpoint,
        pretrained_backbone_checkpoint=args.pretrained_backbone,
    ).to(device)
    source_checkpoint = save_dir / f"vds2raw_ccsa_seed{args.seed}_source_init.pth"
    if args.source_checkpoint:
        load_model_state(model, args.source_checkpoint)
        print(f"Loaded CCSA source initialization: {args.source_checkpoint}")
    else:
        train_source_initialization(model, source_loader, device, args)
        torch.save({"model_state": model.state_dict()}, source_checkpoint)
        print(f"Saved CCSA source initialization: {source_checkpoint}")

    optimizer = torch.optim.Adadelta(
        model.parameters(), lr=args.lr, rho=args.rho, eps=args.eps
    )
    criterion = nn.CrossEntropyLoss()
    stem = f"vds2raw_ccsa_seed{args.seed}"
    checkpoint = save_dir / f"{stem}_best.pth"
    metadata = {
        "method": "CCSA",
        "implementation": "paper natural-image configuration, PyTorch port",
        "seed": args.seed,
        "epochs": args.epochs,
        "source_initialization_epochs": args.source_epochs,
        "source_initialization_checkpoint_input": args.source_checkpoint,
        "pretrained_backbone_checkpoint_input": args.pretrained_backbone,
        "source_initialization_provenance": (
            "provided source checkpoint"
            if args.source_checkpoint
            else (
                "uploaded ImageNet VGG-16 followed by source_train initialization"
                if args.pretrained_backbone
                else "ImageNet VGG-16 followed by source_train initialization"
            )
        ),
        "data_root": str(data_root),
        "source_partition": "source_train",
        "target_train_partition": "target_train",
        "target_samples_per_class": args.target_samples_per_class,
        "selection_partition": "target_val",
        "final_partition": "target_test",
        "modalities": ["RGB"],
        "raw_dn_scaling": "fixed /255 with sparse high-DN clipping before ImageNet normalization",
        "num_classes": 3,
        "backbone": "ImageNet-pretrained VGG-16 convolutional layers",
        "embedding": [1024, 128],
        "pair_protocol": "epoch-resampled controlled pairs; 1 positive : 3 negative",
        "pairs_per_epoch": args.pairs_per_epoch,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "augmentation": "resize256/random-resized-crop224/H+V flip/color-jitter",
        "normalization": "ImageNet",
    }

    history, best_metrics = [], None
    for epoch in range(1, args.epochs + 1):
        pairs.set_epoch(epoch)
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

            source_ce, source_csa = train_direction(
                model,
                optimizer,
                criterion,
                source_images,
                source_labels,
                target_images,
                pair_labels,
                args,
            )
            target_ce, target_csa = train_direction(
                model,
                optimizer,
                criterion,
                target_images,
                target_labels,
                source_images,
                pair_labels,
                args,
            )
            print(
                f"Epoch [{epoch}/{args.epochs}] batch "
                f"[{batch_index}/{len(pair_loader)}] source_ce={source_ce:.4f} "
                f"target_ce={target_ce:.4f} source_csa={source_csa:.4f} "
                f"target_csa={target_csa:.4f}"
            )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        val_metrics = evaluate_rgb(model, val_loader, device)
        history.append((epoch, val_metrics))
        print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_val", val_metrics))
        if better_metrics(val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            save_checkpoint(checkpoint, model, epoch, val_metrics, metadata)

    last_checkpoint = save_dir / f"{stem}_last.pth"
    save_checkpoint(
        last_checkpoint, model, history[-1][0], history[-1][1], metadata
    )
    selected = load_selected_checkpoint(checkpoint, model, device)
    target_test = NaturalImageRGBDataset(
        data_root, "target_test", statistics, augment=False, normalization="imagenet"
    )
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
