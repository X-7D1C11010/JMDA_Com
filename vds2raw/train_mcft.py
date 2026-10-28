"""Three-modal MCFT adaptation on the VDS2Raw supervised cross-day split."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.reproducibility import seed_everything
from vds2raw.common import (
    better_metrics,
    evaluate_multimodal,
    load_selected_checkpoint,
    metric_line,
    save_checkpoint,
    save_experiment_results,
)
from vds2raw.dataset import VDS2RawDataset, audit_dataset, load_or_compute_statistics
from vds2raw.models import MCFTWithAIS


def main():
    parser = argparse.ArgumentParser(description="MCFT RGB+NIR+AIS on VDS2Raw")
    parser.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "vds2raw"))
    parser.add_argument("--stats-file", default="")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-frequency", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--loss-ita-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_frequency <= 0:
        raise ValueError("epochs, batch-size and eval-frequency must be positive")
    args.seed = seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit:
        print("Dataset audit:", audit_dataset(data_root))
    stats_file = args.stats_file or str(save_dir / "normalization_stats.json")
    statistics = load_or_compute_statistics(data_root, stats_file)
    print("Normalization:", statistics.to_dict())

    source = VDS2RawDataset(
        data_root, "source_train", statistics, image_size=15, augment=args.augment
    )
    target = VDS2RawDataset(
        data_root, "target_train", statistics, image_size=15, augment=args.augment
    )
    train_data = ConcatDataset((source, target))
    target_val = VDS2RawDataset(
        data_root, "target_val", statistics, image_size=15, augment=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
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
    model = MCFTWithAIS().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    stem = f"vds2raw_mcft3_seed{args.seed}"
    checkpoint = save_dir / f"{stem}_best.pth"
    metadata = {
        "method": "MCFT-RGB-NIR-AIS",
        "seed": args.seed,
        "epochs": args.epochs,
        "data_root": str(data_root),
        "source_partition": "source_train",
        "target_train_partition": "target_train",
        "selection_partition": "target_val",
        "final_partition": "target_test",
        "modalities": ["RGB", "NIR", "AIS", "AIS_mask"],
        "num_classes": 3,
        "adaptation_note": "official two-image MCFT trunk plus explicit AIS/mask late-fusion branch",
        "normalization": statistics.to_dict(),
    }
    history, best_metrics = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader, 1):
            labels = batch["label"].to(device)
            logits, loss_ita, loss_itm = model(
                batch["rgb"].to(device),
                batch["nir"].to(device),
                batch["ais"].to(device),
                batch["ais_mask"].to(device),
                labels,
            )
            classification_loss = criterion(logits, labels)
            loss = classification_loss + args.loss_ita_weight * loss_ita + loss_itm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(train_loader)}] ce={classification_loss.item():.4f} "
                    f"ita={loss_ita.item():.4f} itm={loss_itm.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        scheduler.step()

        should_evaluate = (
            epoch == 1 or epoch == args.epochs or epoch % args.eval_frequency == 0
        )
        if should_evaluate:
            val_metrics = evaluate_multimodal(model, val_loader, device)
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
    del target_val, val_loader
    target_test = VDS2RawDataset(
        data_root, "target_test", statistics, image_size=15, augment=False
    )
    test_loader = DataLoader(
        target_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_metrics = evaluate_multimodal(model, test_loader, device)
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
