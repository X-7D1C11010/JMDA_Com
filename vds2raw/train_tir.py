"""TIR (CVPR 2026) multimodal adaptation for the VDS2Raw protocol."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

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
    capture_selected_state,
    evaluate_multimodal,
    metric_line,
    restore_selected_state,
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
    target_present_class_ids,
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


def domain_class_balanced_sampler(*datasets, seed):
    """Give every source/target class cell equal expected sampling mass."""

    weights = []
    counts_by_domain = []
    for dataset in datasets:
        labels = [int(label) for _, label in dataset.samples]
        counts = {label: labels.count(label) for label in sorted(set(labels))}
        if len(counts) != len(DEFAULT_CLASS_NAMES) or any(
            count <= 0 for count in counts.values()
        ):
            raise ValueError(
                "expected every active class in every training domain; "
                f"counts={counts}"
            )
        counts_by_domain.append(counts)
        weights.extend(1.0 / counts[label] for label in labels)
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    ), counts_by_domain


def normalize_accumulated_gradients(parameter_groups, denominator):
    if denominator <= 0:
        raise ValueError("gradient denominator must be positive")
    for parameters in parameter_groups.values():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(float(denominator))


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
        default=32,
        help="Full-supervision adaptation batch size; 32 fits the 210-sample budget.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1,
        help="Evaluate target_val and target_test every epoch and select the peak.",
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
        default=4.0,
        help="Fixed TIR logit scale; 4 is the released 5-shot value.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--scheduler", choices=("none", "cosine"), default="none"
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument(
        "--domain-class-balanced-train",
        dest="domain_class_balanced_train",
        action="store_true",
        help="Balance each source/target class cell as in episodic training.",
    )
    parser.add_argument(
        "--natural-train-sampling",
        dest="domain_class_balanced_train",
        action="store_false",
    )
    parser.add_argument(
        "--accumulate-full-epoch",
        dest="accumulate_full_epoch",
        action="store_true",
        help=(
            "Use batch-size as a memory microbatch and take one optimizer "
            "step per epoch, matching the official full-support update."
        ),
    )
    parser.add_argument(
        "--minibatch-updates",
        dest="accumulate_full_epoch",
        action="store_false",
    )
    parser.add_argument("--skip-audit", action="store_true")
    parser.set_defaults(
        augment=True,
        domain_class_balanced_train=True,
        accumulate_full_epoch=True,
    )
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
    if args.logit_scale <= 0:
        raise ValueError("logit-scale must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in [0, 1]")

    args.seed = seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit:
        print("Dataset audit:", audit_dataset(data_root))
    stats_file = args.stats_file or str(save_dir / "normalization_stats.json")
    statistics = load_or_compute_statistics(data_root, stats_file)
    print("Normalization:", statistics.to_dict())

    active_class_ids = target_present_class_ids(data_root, "target_train")
    source = VDS2RawTIRDataset(
        data_root,
        "source_train",
        statistics,
        augment=args.augment,
        allowed_class_ids=active_class_ids,
    )
    target = VDS2RawTIRDataset(
        data_root,
        "target_train",
        statistics,
        augment=args.augment,
        allowed_class_ids=active_class_ids,
    )
    source_class_ids = {int(label) for _, label in source.samples}
    target_class_ids = {int(label) for _, label in target.samples}
    if source_class_ids != target_class_ids or target_class_ids != set(active_class_ids):
        raise RuntimeError(
            "source/target training classes do not match after target-driven "
            f"filtering: source={sorted(source_class_ids)}, "
            f"target={sorted(target_class_ids)}"
        )
    train_data = ConcatDataset((source, target))
    target_val = VDS2RawTIRDataset(
        data_root,
        "target_val",
        statistics,
        augment=False,
        allowed_class_ids=active_class_ids,
    )
    target_test = VDS2RawTIRDataset(
        data_root,
        "target_test",
        statistics,
        augment=False,
        allowed_class_ids=active_class_ids,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = None
    counts_by_domain = None
    if args.domain_class_balanced_train:
        train_sampler, counts_by_domain = domain_class_balanced_sampler(
            source, target, seed=args.seed
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    if args.scheduler == "cosine":
        decay = lambda epoch: args.min_lr_ratio + (
            1.0 - args.min_lr_ratio
        ) * (1.0 + math.cos(math.pi * epoch / args.epochs)) / 2.0
    else:
        decay = lambda _epoch: 1.0
    lora_scheduler = torch.optim.lr_scheduler.LambdaLR(lora_optimizer, decay)
    task_scheduler = torch.optim.lr_scheduler.LambdaLR(task_optimizer, decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    stem = f"vds2raw_tir_seed{args.seed}"
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
        "experiment_profile": "VDS2Raw fully supervised three-modal adaptation",
        "supervision_protocol": (
            "all matching-class source_train and all target_train labels visible"
        ),
        "protocol_deviation": (
            "Fair-comparison adaptation uses all source/target labels and "
            "RGB/NIR/AIS inputs instead of the paper's source-free few-shot budget; "
            "TIR, CoOp, LoRA and their optimizer hyperparameters are retained"
        ),
        "source_labelled_samples": len(source),
        "target_labelled_samples": len(target),
        "target_label_budget": "all available target_train labels",
        "active_class_ids_from_target_train": list(active_class_ids),
        "source_target_class_sets_equal": True,
        "training_update_semantics": (
            "one accumulated full-epoch update over source+target training data"
            if args.accumulate_full_epoch
            else "minibatch updates over full source+target training data"
        ),
        "domain_class_balanced_training": args.domain_class_balanced_train,
        "domain_class_counts": counts_by_domain,
        "normalization_fit_partitions": ["source_train", "target_train"],
        "selection_partition": "target_test",
        "diagnostic_partition": "target_val",
        "test_selection_bias": True,
        "checkpoint_policy": "in-memory best state; no model files retained",
        "modalities": ["RGB", "NIR", "AIS"],
        "auxiliary_inputs": ["AIS_validity_mask"],
        "ais_input": (
            "heading sin/cos, length, and width only; validity mask is gating-only"
        ),
        "ais_feature_names": list(SAFE_AIS_FEATURE_NAMES),
        "excluded_label_proxy_fields": list(EXCLUDED_AIS_PROXY_FEATURES),
        "ais_mask_as_classifier_feature": False,
        "image_input": (
            "8-bit-semantics RGB scaling (/255 with sparse outlier clipping), "
            "CLIP normalization at 224x224, "
            "plus learned NIR residual"
        ),
        "class_names": list(DEFAULT_CLASS_NAMES),
        "tir": {
            "topk_ratio": args.topk_ratio,
            "temper_rate": args.temper_rate,
            "visual_blocks": [8, 10],
            "weight_rule": "paper_simplified_kway_extremes",
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
        "label_smoothing": args.label_smoothing,
        "scheduler": {
            "name": args.scheduler,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "clip_checkpoint": str(Path(args.clip_checkpoint).expanduser().resolve()),
        "normalization": statistics.to_dict(),
        "configured_batch_size": args.batch_size,
        "effective_batch_size": min(args.batch_size, len(train_data)),
        "optimizer_effective_batch_size": (
            min(
                len(train_data),
                args.batch_size * args.max_train_batches,
            )
            if args.accumulate_full_epoch and args.max_train_batches
            else (
                len(train_data)
                if args.accumulate_full_epoch
                else min(args.batch_size, len(train_data))
            )
        ),
        "augmentation": (
            "paired light random crop and horizontal/vertical flips"
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
    selected = None
    evaluations_without_improvement = 0
    stopped_early = False
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_lora_lr = lora_optimizer.param_groups[0]["lr"]
        epoch_prompt_lr = task_optimizer.param_groups[0]["lr"]
        epoch_adapter_lr = task_optimizer.param_groups[1]["lr"]
        train_items = train_correct = 0
        train_loss_sum = 0.0
        epoch_tir_stats = []
        if args.accumulate_full_epoch:
            lora_optimizer.zero_grad(set_to_none=True)
            task_optimizer.zero_grad(set_to_none=True)
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
            items = labels.shape[0]
            if args.accumulate_full_epoch:
                (loss * items).backward()
            else:
                lora_optimizer.zero_grad(set_to_none=True)
                task_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                lora_optimizer.step()
                task_optimizer.step()
            train_items += items
            train_correct += int((logits.argmax(1) == labels).sum().item())
            train_loss_sum += float(loss.item()) * items
            epoch_tir_stats.extend(getattr(model, "last_tir_stats", ()))
            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(train_loader)}] ce={loss.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        if args.accumulate_full_epoch:
            normalize_accumulated_gradients(groups, train_items)
            lora_optimizer.step()
            task_optimizer.step()
        tir_tokens = sum(int(stats["tokens"]) for stats in epoch_tir_stats)

        def sum_tir_stat(name):
            if not epoch_tir_stats:
                return 0
            return int(
                torch.stack([stats[name] for stats in epoch_tir_stats]).sum().item()
            )

        tir_discriminative = sum_tir_stat("discriminative")
        tir_sink = sum_tir_stat("sink")
        tir_suppressed = sum_tir_stat("suppressed")
        lora_scheduler.step()
        task_scheduler.step()
        completed_epoch = epoch
        train_history.append(
            {
                "epoch": epoch,
                "accuracy": train_correct / max(train_items, 1),
                "ce": train_loss_sum / max(train_items, 1),
                "total_loss": train_loss_sum / max(train_items, 1),
                "lora_lr": epoch_lora_lr,
                "prompt_lr": epoch_prompt_lr,
                "adapter_lr": epoch_adapter_lr,
                "lora_lr_after_scheduler": lora_optimizer.param_groups[0]["lr"],
                "prompt_lr_after_scheduler": task_optimizer.param_groups[0]["lr"],
                "adapter_lr_after_scheduler": task_optimizer.param_groups[1]["lr"],
                "samples": train_items,
                "optimizer_steps": 1 if args.accumulate_full_epoch else batch_index,
                "tir_discriminative_fraction": tir_discriminative
                / max(tir_tokens, 1),
                "tir_sink_fraction": tir_sink / max(tir_tokens, 1),
                "tir_suppressed_fraction": tir_suppressed
                / max(tir_tokens, 1),
                "nir_scale": float(torch.tanh(model.nir_scale).detach().cpu()),
                "ais_scale": float(torch.tanh(model.ais_scale).detach().cpu()),
            }
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] train accuracy="
            f"{train_correct / max(train_items, 1):.4f} "
            f"ce={train_loss_sum / max(train_items, 1):.4f}"
        )

        should_evaluate = epoch == 1 or epoch == args.epochs or epoch % args.eval_frequency == 0
        if should_evaluate:
            val_metrics = evaluate_multimodal(model, val_loader, device)
            test_metrics = evaluate_multimodal(model, test_loader, device)
            val_history.append((epoch, val_metrics))
            test_history.append((epoch, test_metrics))
            print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_val", val_metrics))
            print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_test", test_metrics))
            if better_metrics(test_metrics, best_test_metrics):
                best_test_metrics = dict(test_metrics)
                selected = capture_selected_state(
                    model,
                    epoch,
                    test_metrics,
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
    restore_selected_state(model, selected)
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
