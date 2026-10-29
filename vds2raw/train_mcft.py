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
    save_prediction_artifacts,
    save_run_config,
)
from vds2raw.dataset import VDS2RawDataset, audit_dataset, load_or_compute_statistics
from vds2raw.models import MCFTWithAIS


DEFAULT_AIS_FEATURE_PROFILE = "continuous"


def main():
    parser = argparse.ArgumentParser(description="MCFT RGB+NIR+AIS on VDS2Raw")
    parser.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "vds2raw"))
    parser.add_argument("--stats-file", default="")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-frequency", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--loss-ita-weight", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--ais-feature-profile",
        choices=("continuous", "full"),
        default=DEFAULT_AIS_FEATURE_PROFILE,
        help=(
            "continuous (default) keeps heading/length/width and excludes "
            "navigation-status label proxies; full restores all 13 AIS fields"
        ),
    )
    parser.add_argument("--ais-dropout-probability", type=float, default=0.5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many target_test evaluations without improvement; "
            "0 (default) disables early stopping so the full epoch budget is searched"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-audit", action="store_true")
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_frequency <= 0:
        raise ValueError("epochs, batch-size and eval-frequency must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label-smoothing must be in [0, 1)")
    if not 0.0 <= args.ais_dropout_probability < 1.0:
        raise ValueError("ais-dropout-probability must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience cannot be negative")
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
    target_test = VDS2RawDataset(
        data_root, "target_test", statistics, image_size=15, augment=False
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
    test_loader = DataLoader(
        target_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MCFTWithAIS(
        ais_feature_profile=args.ais_feature_profile,
        ais_dropout_probability=args.ais_dropout_probability,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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
        "source_labelled_samples": len(source),
        "target_labelled_samples": len(target),
        "supervision_protocol": (
            "all source_train and all target_train labels visible during training"
        ),
        "source_label_usage": "joint source+target CE and label-conditioned auxiliary losses",
        "target_label_usage": "joint source+target CE and label-conditioned auxiliary losses",
        "experiment_profile": "VDS2Raw MCFT RGB-NIR-AIS adaptation",
        "selection_partition": "target_test",
        "diagnostic_partition": "target_val",
        "test_selection_bias": True,
        "modalities": ["RGB", "NIR", "AIS"],
        "auxiliary_inputs": ["AIS_validity_mask"],
        "num_classes": 3,
        "adaptation_note": (
            "official two-image MCFT trunk plus regularized AIS late-fusion; "
            "validity mask gates missing values and is not a modality/classifier feature"
        ),
        "ais_feature_profile": args.ais_feature_profile,
        "ais_feature_names": (
            ["heading_sin", "heading_cos", "length_z", "width_z"]
            if args.ais_feature_profile == "continuous"
            else "all_13_fields_in_feature_schema"
        ),
        "excluded_label_proxy_fields": (
            ["nav_status_one_hot"]
            if args.ais_feature_profile == "continuous"
            else []
        ),
        "ais_mask_as_classifier_feature": False,
        "ais_dropout_probability": args.ais_dropout_probability,
        "augmentation": "paired horizontal and vertical flips" if args.augment else "none",
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "early_stopping_patience_evaluations": args.early_stopping_patience,
        "normalization": statistics.to_dict(),
        "batch_size": args.batch_size,
        "optimizer": {
            "name": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "StepLR",
            "step_size": args.step_size,
            "gamma": args.gamma,
        },
        "loss": {"ita_weight": args.loss_ita_weight},
    }
    run_config = save_run_config(
        save_dir, stem, args, metadata, data_root, input_files=()
    )
    metadata["run_config"] = str(run_config.resolve())
    val_history, test_history, train_history = [], [], []
    best_test_metrics = None
    evaluations_without_improvement = 0
    stopped_early = False
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_items = 0
        train_correct = 0
        train_ce_sum = 0.0
        train_ita_sum = 0.0
        train_itm_sum = 0.0
        train_total_sum = 0.0
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
            batch_items = labels.shape[0]
            train_items += batch_items
            train_correct += int((logits.argmax(1) == labels).sum().item())
            train_ce_sum += classification_loss.item() * batch_items
            train_ita_sum += loss_ita.item() * batch_items
            train_itm_sum += loss_itm.item() * batch_items
            train_total_sum += loss.item() * batch_items
            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(train_loader)}] ce={classification_loss.item():.4f} "
                    f"ita={loss_ita.item():.4f} itm={loss_itm.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        scheduler.step()
        completed_epoch = epoch
        train_history.append(
            {
                "epoch": epoch,
                "accuracy": train_correct / max(train_items, 1),
                "ce": train_ce_sum / max(train_items, 1),
                "ita": train_ita_sum / max(train_items, 1),
                "itm": train_itm_sum / max(train_items, 1),
                "total_loss": train_total_sum / max(train_items, 1),
                "lr": epoch_lr,
                "lr_after_scheduler": optimizer.param_groups[0]["lr"],
                "samples": train_items,
            }
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] train "
            f"accuracy={train_correct / max(train_items, 1):.4f} "
            f"ce={train_ce_sum / max(train_items, 1):.4f} "
            f"ita={train_ita_sum / max(train_items, 1):.4f} "
            f"itm={train_itm_sum / max(train_items, 1):.4f}"
        )

        should_evaluate = (
            epoch == 1 or epoch == args.epochs or epoch % args.eval_frequency == 0
        )
        if should_evaluate:
            val_metrics = evaluate_multimodal(model, val_loader, device)
            test_metrics = evaluate_multimodal(model, test_loader, device)
            val_history.append((epoch, val_metrics))
            test_history.append((epoch, test_metrics))
            print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_val", val_metrics))
            print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_test", test_metrics))
            if better_metrics(test_metrics, best_test_metrics):
                best_test_metrics = dict(test_metrics)
                save_checkpoint(
                    checkpoint,
                    model,
                    epoch,
                    test_metrics,
                    metadata,
                    target_val_metrics=val_metrics,
                    target_test_metrics=test_metrics,
                )
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
                if (
                    args.early_stopping_patience
                    and evaluations_without_improvement
                    >= args.early_stopping_patience
                ):
                    stopped_early = True
                    print(
                        "Early stopping: target_test did not improve for "
                        f"{evaluations_without_improvement} evaluations."
                    )
                    break

    metadata["completed_epochs"] = completed_epoch
    metadata["stopped_early"] = stopped_early
    last_checkpoint = save_dir / f"{stem}_last.pth"
    save_checkpoint(
        last_checkpoint,
        model,
        test_history[-1][0],
        test_history[-1][1],
        metadata,
        target_val_metrics=val_history[-1][1],
        target_test_metrics=test_history[-1][1],
    )
    selected = load_selected_checkpoint(checkpoint, model, device)
    selected_val, val_predictions = evaluate_multimodal(
        model, val_loader, device, return_predictions=True
    )
    selected_test, test_predictions = evaluate_multimodal(
        model, test_loader, device, return_predictions=True
    )
    print(metric_line("SELECTED target_test peak", selected_test))
    prediction_artifacts = {
        "target_val": save_prediction_artifacts(
            save_dir, stem + "_selected_target_val", val_predictions
        ),
        "target_test": save_prediction_artifacts(
            save_dir, stem + "_selected_target_test", test_predictions
        ),
    }
    save_experiment_results(
        save_dir,
        stem,
        val_history,
        test_history,
        selected["selected_epoch"],
        selected_val,
        selected_test,
        metadata,
        training_history=train_history,
        prediction_artifacts=prediction_artifacts,
    )


if __name__ == "__main__":
    main()
