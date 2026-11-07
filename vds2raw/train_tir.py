"""TIR (CVPR 2026) multimodal adaptation for the VDS2Raw protocol."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

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
        tir_min_weight=args.tir_min_weight,
        tir_max_weight=args.tir_max_weight,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_encoder=args.lora_encoder,
        lora_position=args.lora_position,
        lora_params=args.lora_params,
        ais_dropout_probability=args.ais_dropout_probability,
        nir_initial_scale=args.nir_initial_scale,
        ais_initial_scale=args.ais_initial_scale,
        logit_scale=args.logit_scale,
    )


class DomainTaggedDataset(Dataset):
    """Attach a domain id without changing the shared dataset interface."""

    def __init__(self, dataset, domain_id):
        self.dataset = dataset
        self.domain_id = int(domain_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["domain"] = self.domain_id
        return item


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


def mild_class_weights(*datasets, power=0.5):
    """Return inverse-frequency weights without replacement oversampling.

    Combining source and target counts keeps the supervised budget intact.
    The square-root default is deliberately milder than exact inverse class
    frequency and therefore does not repeatedly expose minority samples.
    """

    if not 0.0 <= power <= 1.0:
        raise ValueError("class weight power must be in [0, 1]")
    counts = torch.zeros(len(DEFAULT_CLASS_NAMES), dtype=torch.float64)
    for dataset in datasets:
        for _, label in dataset.samples:
            counts[int(label)] += 1
    if bool((counts <= 0).any()):
        raise ValueError(f"every active class needs training samples; counts={counts}")
    weights = counts.pow(-float(power))
    # Keep the expected per-sample loss scale at one inside each domain. This
    # makes source_loss_weight/target_loss_weight reflect the requested domain
    # contribution instead of being distorted by different class histograms.
    weights = weights * counts.sum() / (weights * counts).sum()
    return weights.float(), counts.to(torch.int64).tolist()


def warmup_cosine_multiplier(epoch_index, total_epochs, warmup_epochs, min_lr_ratio):
    """Epoch scheduler with an explicit low-LR warmup and cosine decay."""

    if total_epochs <= 0 or warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup must satisfy 0 <= warmup_epochs < total_epochs")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    epoch_index = min(max(int(epoch_index), 0), int(total_epochs))
    if warmup_epochs and epoch_index < warmup_epochs:
        return float(epoch_index + 1) / float(warmup_epochs)
    progress = (epoch_index - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * (
        1.0 + math.cos(math.pi * progress)
    ) / 2.0


def better_validation_metrics(candidate, incumbent):
    """Prefer macro F1 for early stopping, with accuracy as the tie-breaker."""

    if incumbent is None:
        return True
    return (candidate["f1_macro"], candidate["accuracy"]) > (
        incumbent["f1_macro"],
        incumbent["accuracy"],
    )


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
    parser.add_argument("--tir-min-weight", type=float, default=0.0)
    parser.add_argument("--tir-max-weight", type=float, default=2.0)
    parser.add_argument("--text-layer", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.25)
    parser.add_argument(
        "--lora-encoder", choices=("vision", "text", "both"), default="both"
    )
    parser.add_argument("--lora-position", default="all")
    parser.add_argument(
        "--lora-params",
        nargs="+",
        choices=("q", "k", "v"),
        default=("q", "k", "v"),
    )
    parser.add_argument("--ais-dropout-probability", type=float, default=0.3)
    parser.add_argument("--nir-initial-scale", type=float, default=0.01)
    parser.add_argument("--ais-initial-scale", type=float, default=0.80)
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
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--source-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-loss-weight", type=float, default=0.5)
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
        domain_class_balanced_train=False,
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
    if not 0.0 <= args.tir_min_weight <= args.tir_max_weight:
        raise ValueError("TIR weight bounds must satisfy 0 <= min <= max")
    if not -1.0 < args.nir_initial_scale < 1.0:
        raise ValueError("nir-initial-scale must be in (-1, 1)")
    if not 0.0 < args.ais_initial_scale < 1.0:
        raise ValueError("ais-initial-scale must be in (0, 1)")
    if args.text_layer not in range(12):
        raise ValueError("text-layer must be between 0 and 11")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError("early stopping settings cannot be negative")
    if args.logit_scale <= 0:
        raise ValueError("logit-scale must be positive")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be in [0, 1]")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs must be in [0, epochs)")
    if not 0.0 <= args.class_weight_power <= 1.0:
        raise ValueError("class-weight-power must be in [0, 1]")
    if args.source_loss_weight < 0 or args.target_loss_weight < 0:
        raise ValueError("domain loss weights must be non-negative")
    domain_loss_weight_sum = args.source_loss_weight + args.target_loss_weight
    if domain_loss_weight_sum <= 0:
        raise ValueError("at least one domain loss weight must be positive")
    args.source_loss_weight /= domain_loss_weight_sum
    args.target_loss_weight /= domain_loss_weight_sum

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
    train_data = ConcatDataset(
        (DomainTaggedDataset(source, 0), DomainTaggedDataset(target, 1))
    )
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
    source_class_weights, source_class_counts = mild_class_weights(
        source, power=args.class_weight_power
    )
    target_class_weights, target_class_counts = mild_class_weights(
        target, power=args.class_weight_power
    )
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
        decay = lambda epoch: warmup_cosine_multiplier(
            epoch,
            args.epochs,
            args.warmup_epochs,
            args.min_lr_ratio,
        )
    else:
        decay = lambda _epoch: 1.0
    lora_scheduler = torch.optim.lr_scheduler.LambdaLR(lora_optimizer, decay)
    task_scheduler = torch.optim.lr_scheduler.LambdaLR(task_optimizer, decay)
    source_criterion = nn.CrossEntropyLoss(
        weight=source_class_weights.to(device),
        label_smoothing=args.label_smoothing,
        reduction="sum",
    )
    target_criterion = nn.CrossEntropyLoss(
        weight=target_class_weights.to(device),
        label_smoothing=args.label_smoothing,
        reduction="sum",
    )

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
            "the TIR, CoOp and LoRA mechanisms and the previously validated "
            "full-support optimizer scale are retained"
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
        "sampling": (
            "legacy replacement domain-class balancing"
            if args.domain_class_balanced_train
            else "natural shuffle without replacement"
        ),
        "domain_separated_supervised_loss": True,
        "source_class_counts": source_class_counts,
        "target_class_counts": target_class_counts,
        "class_weight_power": args.class_weight_power,
        "source_class_weights": source_class_weights.tolist(),
        "target_class_weights": target_class_weights.tolist(),
        "source_loss_weight": args.source_loss_weight,
        "target_loss_weight": args.target_loss_weight,
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
            "weight_rule": "paper_simplified_kway_extremes_bounded",
            "min_weight": args.tir_min_weight,
            "max_weight": args.tir_max_weight,
        },
        "lora": {
            "encoder": args.lora_encoder,
            "position": args.lora_position,
            "params": list(args.lora_params),
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
            "warmup_epochs": args.warmup_epochs,
        },
        "initial_fusion_scales": {
            "nir": args.nir_initial_scale,
            "ais": args.ais_initial_scale,
        },
        "ais_fusion_constraint": "sigmoid_positive_(0,1)",
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
        "early_stopping_monitor": (
            "disabled"
            if args.early_stopping_patience == 0
            else "target_val macro F1, accuracy tie-breaker"
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
    best_val_metrics = None
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
        source_items = source_correct = target_items = target_correct = 0
        source_loss_sum = target_loss_sum = 0.0
        epoch_tir_stats = []
        if args.accumulate_full_epoch:
            lora_optimizer.zero_grad(set_to_none=True)
            task_optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, 1):
            labels = batch["label"].to(device)
            domains = batch["domain"].to(device)
            logits, _, _ = model(
                batch["rgb"].to(device),
                batch["nir"].to(device),
                batch["ais"].to(device),
                batch["ais_mask"].to(device),
                labels,
            )
            items = labels.shape[0]
            source_mask = domains.eq(0)
            target_mask = domains.eq(1)
            if not bool((source_mask | target_mask).all()):
                raise RuntimeError("unexpected domain id in TIR training batch")
            source_batch_items = int(source_mask.sum().item())
            target_batch_items = int(target_mask.sum().item())
            source_loss = (
                source_criterion(logits[source_mask], labels[source_mask])
                if source_batch_items
                else logits.new_zeros(())
            )
            target_loss = (
                target_criterion(logits[target_mask], labels[target_mask])
                if target_batch_items
                else logits.new_zeros(())
            )
            if args.accumulate_full_epoch:
                loss = (
                    args.source_loss_weight * source_loss / len(source)
                    + args.target_loss_weight * target_loss / len(target)
                )
            else:
                loss = (
                    args.source_loss_weight
                    * source_loss
                    / max(source_batch_items, 1)
                    + args.target_loss_weight
                    * target_loss
                    / max(target_batch_items, 1)
                )
            if args.accumulate_full_epoch:
                loss.backward()
            else:
                lora_optimizer.zero_grad(set_to_none=True)
                task_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                lora_optimizer.step()
                task_optimizer.step()
            train_items += items
            train_correct += int((logits.argmax(1) == labels).sum().item())
            source_items += source_batch_items
            target_items += target_batch_items
            if source_batch_items:
                source_correct += int(
                    (logits[source_mask].argmax(1) == labels[source_mask])
                    .sum()
                    .item()
                )
            if target_batch_items:
                target_correct += int(
                    (logits[target_mask].argmax(1) == labels[target_mask])
                    .sum()
                    .item()
                )
            source_loss_sum += float(source_loss.detach().item())
            target_loss_sum += float(target_loss.detach().item())
            epoch_tir_stats.extend(getattr(model, "last_tir_stats", ()))
            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(train_loader)}] "
                    f"domain_loss={loss.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        if args.accumulate_full_epoch:
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
        tir_clipped = sum_tir_stat("clipped")
        lora_scheduler.step()
        task_scheduler.step()
        completed_epoch = epoch
        source_epoch_ce = source_loss_sum / max(source_items, 1)
        target_epoch_ce = target_loss_sum / max(target_items, 1)
        weighted_epoch_ce = (
            args.source_loss_weight * source_epoch_ce
            + args.target_loss_weight * target_epoch_ce
        )
        train_history.append(
            {
                "epoch": epoch,
                "accuracy": train_correct / max(train_items, 1),
                "source_accuracy": source_correct / max(source_items, 1),
                "target_accuracy": target_correct / max(target_items, 1),
                "ce": weighted_epoch_ce,
                "source_weighted_ce": source_epoch_ce,
                "target_weighted_ce": target_epoch_ce,
                "total_loss": weighted_epoch_ce,
                "lora_lr": epoch_lora_lr,
                "prompt_lr": epoch_prompt_lr,
                "adapter_lr": epoch_adapter_lr,
                "lora_lr_after_scheduler": lora_optimizer.param_groups[0]["lr"],
                "prompt_lr_after_scheduler": task_optimizer.param_groups[0]["lr"],
                "adapter_lr_after_scheduler": task_optimizer.param_groups[1]["lr"],
                "samples": train_items,
                "source_samples": source_items,
                "target_samples": target_items,
                "optimizer_steps": 1 if args.accumulate_full_epoch else batch_index,
                "tir_discriminative_fraction": tir_discriminative
                / max(tir_tokens, 1),
                "tir_sink_fraction": tir_sink / max(tir_tokens, 1),
                "tir_suppressed_fraction": tir_suppressed
                / max(tir_tokens, 1),
                "tir_clipped_fraction": tir_clipped / max(tir_tokens, 1),
                "nir_scale": float(torch.tanh(model.nir_scale).detach().cpu()),
                "ais_scale": float(model.effective_ais_scale().detach().cpu()),
            }
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] train accuracy="
            f"{train_correct / max(train_items, 1):.4f} "
            f"source_accuracy={source_correct / max(source_items, 1):.4f} "
            f"target_accuracy={target_correct / max(target_items, 1):.4f} "
            f"source_ce={source_epoch_ce:.4f} target_ce={target_epoch_ce:.4f} "
            f"weighted_ce={weighted_epoch_ce:.4f}"
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
            if better_validation_metrics(val_metrics, best_val_metrics):
                best_val_metrics = dict(val_metrics)
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
                    "Early stopping: target_val did not improve for "
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
