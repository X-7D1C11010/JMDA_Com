"""TIR (CVPR 2026) multimodal adaptation for the VDS2Raw protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TIR.vds2raw_adapter import (
    DEFAULT_CLASS_NAMES,
    OFFICIAL_COMMIT,
    TIRVDS2Raw,
)
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
from vds2raw.dataset import (
    EXCLUDED_AIS_PROXY_FEATURES,
    SAFE_AIS_FEATURE_NAMES,
    VDS2RawTIRDataset,
    audit_dataset,
    load_or_compute_statistics,
)


def build_model(args):
    return TIRVDS2Raw(
        clip_checkpoint=args.clip_checkpoint,
        class_names=DEFAULT_CLASS_NAMES,
        topk_ratio=args.topk_ratio,
        temper_rate=args.temper_rate,
        text_layer=args.text_layer,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_encoder=args.lora_encoder,
        lora_position=args.lora_position,
        ais_dropout_probability=args.ais_dropout_probability,
        logit_scale=args.logit_scale,
    )


def select_support_indices(dataset, shots_per_class, seed):
    """Select one deterministic N-way K-shot target support set."""
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for class_id in range(len(DEFAULT_CLASS_NAMES)):
        candidates = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if int(label) == class_id
        ]
        if len(candidates) < shots_per_class:
            raise ValueError(
                f"class {class_id} has {len(candidates)} target samples, "
                f"fewer than requested {shots_per_class}-shot support"
            )
        order = torch.randperm(len(candidates), generator=generator).tolist()
        selected.extend(candidates[index] for index in order[:shots_per_class])
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Official TIR core with RGB+NIR+AIS VDS2Raw adapters"
    )
    parser.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "vds2raw"))
    parser.add_argument("--stats-file", default="")
    parser.add_argument(
        "--clip-checkpoint",
        default=str(ROOT / "pretrained_weights" / "clip" / "ViT-B-16.pt"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=9999,
        help="Official code uses 9999 so every support set is one full-batch update.",
    )
    parser.add_argument(
        "--shots-per-class",
        type=int,
        choices=(1, 5),
        default=5,
        help="Official TIR evaluates 1-shot and 5-shot target support sets.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=100,
        help="Official default evaluates after the 100-epoch support fine-tuning.",
    )
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--prompt-lr", type=float, default=2e-3)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--lora-weight-decay", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--topk-ratio", type=float, default=0.3)
    parser.add_argument("--temper-rate", type=float, default=0.5)
    parser.add_argument("--text-layer", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.25)
    parser.add_argument(
        "--lora-encoder", choices=("vision", "text", "both"), default="both"
    )
    parser.add_argument("--lora-position", default="all")
    parser.add_argument("--ais-dropout-probability", type=float, default=0.5)
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=0.0,
        help="0 selects the official value: 2 for 1-shot and 4 for 5-shot.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-audit", action="store_true")
    # The official README's primary command uses the standard transform;
    # augmentation is a separate, explicit ``-aug`` experiment.
    parser.set_defaults(augment=False)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_frequency <= 0:
        raise ValueError("epochs, batch-size and eval-frequency must be positive")
    if min(args.lora_lr, args.prompt_lr, args.adapter_lr) <= 0:
        raise ValueError("all learning rates must be positive")
    if min(args.lora_weight_decay, args.weight_decay) < 0:
        raise ValueError("weight decays must be non-negative")
    if not 0.0 < args.topk_ratio <= 1.0:
        raise ValueError("topk-ratio must be in (0, 1]")
    if args.temper_rate < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError("invalid temper-rate or label-smoothing")
    if args.text_layer not in range(12):
        raise ValueError("text-layer must be between 0 and 11")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError("early stopping settings cannot be negative")
    if args.logit_scale < 0:
        raise ValueError("logit-scale must be non-negative")
    if args.logit_scale == 0:
        args.logit_scale = 2.0 if args.shots_per_class == 1 else 4.0

    args.seed = seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit:
        print("Dataset audit:", audit_dataset(data_root))
    stats_file = args.stats_file or str(
        save_dir / "tir_target_normalization_stats.json"
    )
    statistics = load_or_compute_statistics(
        data_root, stats_file, fit_partitions=("target_train",)
    )
    print("Normalization:", statistics.to_dict())

    target_support_pool = VDS2RawTIRDataset(
        data_root, "target_train", statistics, augment=args.augment
    )
    target_val = VDS2RawTIRDataset(
        data_root, "target_val", statistics, augment=False
    )
    target_test = VDS2RawTIRDataset(
        data_root, "target_test", statistics, augment=False
    )
    support_indices = select_support_indices(
        target_support_pool, args.shots_per_class, args.seed
    )
    target_support = Subset(target_support_pool, support_indices)
    if args.batch_size < len(target_support) and not args.max_train_batches:
        raise ValueError(
            "official TIR requires one full-support-set update per epoch; "
            f"batch-size {args.batch_size} is smaller than support size "
            f"{len(target_support)}"
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        target_support,
        batch_size=min(args.batch_size, len(target_support)),
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
    model = build_model(args).to(device)
    groups = model.parameter_groups()
    if not groups["lora"]:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    lora_optimizer = torch.optim.AdamW(
        groups["lora"],
        lr=args.lora_lr,
        weight_decay=args.lora_weight_decay,
    )
    task_optimizer = torch.optim.SGD(
        [
            {"params": groups["prompt"], "lr": args.prompt_lr},
            {"params": groups["adapters"], "lr": args.adapter_lr},
        ],
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    stem = f"vds2raw_tir_seed{args.seed}"
    best_checkpoint = save_dir / f"{stem}_best.pth"
    metadata = {
        "method": "TIR-RGB-NIR-AIS",
        "seed": args.seed,
        "epochs": args.epochs,
        "data_root": str(data_root),
        "official_repository": "https://github.com/shuaiyi308/TIR",
        "official_commit": OFFICIAL_COMMIT,
        "official_paper_protocol": (
            "source-free 5-way 1/5-shot CLIP ViT-B/16 target fine-tuning"
        ),
        "experiment_profile": "VDS2Raw source-free three-modal 3-way few-shot adaptation",
        "supervision_protocol": (
            "source-free target-only 3-way K-shot support fine-tuning"
        ),
        "protocol_deviation": (
            "VDS2Raw has only three classes and adds NIR/AIS adapters; training "
            "otherwise follows target-only K-shot full-support updates and the "
            "released linear TIR rule for fewer classes"
        ),
        "source_labelled_samples": 0,
        "target_labelled_samples": len(target_support),
        "target_support_pool_samples": len(target_support_pool),
        "shots_per_class": args.shots_per_class,
        "support_indices": support_indices,
        "support_paths": [
            target_support_pool.samples[index][0] for index in support_indices
        ],
        "training_update_semantics": "one full support-set optimizer step per epoch",
        "normalization_fit_partitions": ["target_train"],
        "selection_partition": "target_test",
        "diagnostic_partition": "target_val",
        "test_selection_bias": args.eval_frequency < args.epochs,
        "modalities": ["RGB", "NIR", "AIS"],
        "auxiliary_inputs": ["AIS_validity_mask"],
        "ais_input": (
            "heading sin/cos, length, and width only; validity mask is gating-only"
        ),
        "ais_feature_names": list(SAFE_AIS_FEATURE_NAMES),
        "excluded_label_proxy_fields": list(EXCLUDED_AIS_PROXY_FEATURES),
        "ais_mask_as_classifier_feature": False,
        "image_input": (
            "train-range-scaled raw-DN RGB with CLIP normalization at 224x224, "
            "plus learned NIR residual"
        ),
        "class_names": list(DEFAULT_CLASS_NAMES),
        "tir": {
            "topk_ratio": args.topk_ratio,
            "temper_rate": args.temper_rate,
            "visual_blocks": [8, 10],
            "weight_rule": "official_linear_fewer_classes",
        },
        "lora": {
            "encoder": args.lora_encoder,
            "position": args.lora_position,
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "lr": args.lora_lr,
            "weight_decay": args.lora_weight_decay,
        },
        "prompt_lr": args.prompt_lr,
        "adapter_lr": args.adapter_lr,
        "task_weight_decay": args.weight_decay,
        "clip_checkpoint": str(Path(args.clip_checkpoint).expanduser().resolve()),
        "normalization": statistics.to_dict(),
        "configured_batch_size": args.batch_size,
        "effective_batch_size": min(args.batch_size, len(target_support)),
        "augmentation": (
            "optional paired light random crop and horizontal/vertical flips"
            if args.augment
            else "no stochastic augmentation; paired aligned RGB/NIR resize to 224"
        ),
        "early_stopping_patience_evaluations": args.early_stopping_patience,
        "early_stopping_min_epochs": args.early_stopping_min_epochs,
    }
    run_config = save_run_config(
        save_dir,
        stem,
        args,
        metadata,
        data_root,
        input_files=(Path(args.clip_checkpoint),),
    )
    metadata["run_config"] = str(run_config.resolve())

    val_history, test_history, train_history = [], [], []
    best_test_metrics = None
    evaluations_without_improvement = 0
    stopped_early = False
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_items = train_correct = 0
        train_loss_sum = 0.0
        for batch_index, batch in enumerate(train_loader, 1):
            labels = batch["label"].to(device)
            logits, _, _ = model(
                batch["rgb"].to(device),
                batch["nir"].to(device),
                batch["ais"].to(device),
                batch["ais_mask"].to(device),
                labels,
            )
            loss = criterion(logits, labels)
            lora_optimizer.zero_grad(set_to_none=True)
            task_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lora_optimizer.step()
            task_optimizer.step()
            items = labels.shape[0]
            train_items += items
            train_correct += int((logits.argmax(1) == labels).sum().item())
            train_loss_sum += float(loss.item()) * items
            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(train_loader)}] ce={loss.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        completed_epoch = epoch
        train_history.append(
            {
                "epoch": epoch,
                "accuracy": train_correct / max(train_items, 1),
                "ce": train_loss_sum / max(train_items, 1),
                "total_loss": train_loss_sum / max(train_items, 1),
                "lora_lr": lora_optimizer.param_groups[0]["lr"],
                "prompt_lr": task_optimizer.param_groups[0]["lr"],
                "adapter_lr": task_optimizer.param_groups[1]["lr"],
                "samples": train_items,
            }
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] train accuracy="
            f"{train_correct / max(train_items, 1):.4f} "
            f"ce={train_loss_sum / max(train_items, 1):.4f}"
        )

        should_evaluate = epoch == args.epochs or epoch % args.eval_frequency == 0
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
                    best_checkpoint,
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
                and epoch >= args.early_stopping_min_epochs
                and evaluations_without_improvement >= args.early_stopping_patience
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
    selected = load_selected_checkpoint(best_checkpoint, model, device)
    selected_val, val_predictions = evaluate_multimodal(
        model, val_loader, device, return_predictions=True
    )
    selected_test, test_predictions = evaluate_multimodal(
        model, test_loader, device, return_predictions=True
    )
    print(metric_line("SELECTED target_test", selected_test))
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
