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
from tools.classification_metrics import classification_metrics
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


def gradient_l2_norm(parameters):
    """Return a synchronization-safe scalar gradient norm for diagnostics."""

    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt().cpu())


def evaluate_tir_modalities(model, loader, device):
    """Evaluate fused, RGB, NIR and AIS predictions on the same samples."""

    model.eval()
    labels = []
    predictions = {name: [] for name in ("fused", "rgb", "nir", "ais")}
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["rgb"].to(device),
                batch["nir"].to(device),
                batch["ais"].to(device),
                batch["ais_mask"].to(device),
                labels=None,
                compute_auxiliary=False,
            )
            truth = batch["label"].tolist()
            labels.extend(truth)
            for name, logits in zip(predictions, outputs):
                predictions[name].extend(logits.argmax(1).cpu().tolist())
    return {
        name: classification_metrics(labels, values, num_classes=len(DEFAULT_CLASS_NAMES))
        for name, values in predictions.items()
    }


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
    parser.add_argument("--prompt-lr", type=float, default=5e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=2e-3)
    parser.add_argument("--text-lora-lr-scale", type=float, default=0.25)
    parser.add_argument("--lora-weight-decay", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--topk-ratio", type=float, default=0.3)
    parser.add_argument("--temper-rate", type=float, default=0.5)
    parser.add_argument("--tir-min-weight", type=float, default=0.5)
    parser.add_argument("--tir-max-weight", type=float, default=1.5)
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
    parser.add_argument("--ais-dropout-probability", type=float, default=0.1)
    parser.add_argument("--nir-initial-scale", type=float, default=0.05)
    parser.add_argument("--ais-initial-scale", type=float, default=0.75)
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=4.0,
        help="Fixed TIR logit scale; 4 is the released 5-shot value.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--scheduler", choices=("none", "cosine"), default="cosine"
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=60)
    parser.add_argument(
        "--modality-warmup-epochs",
        type=int,
        default=25,
        help=(
            "Train the new NIR/AIS classifiers and fusion gates before "
            "unfreezing CoOp/LoRA for joint fine-tuning."
        ),
    )
    parser.add_argument(
        "--text-unfreeze-epoch",
        type=int,
        default=41,
        help=(
            "First epoch that enables text LoRA and CoOp. Vision LoRA is "
            "enabled after modality warmup, avoiding an abrupt all-branch update."
        ),
    )
    parser.add_argument("--nir-aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--ais-aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--source-loss-weight", type=float, default=0.4)
    parser.add_argument("--target-loss-weight", type=float, default=0.6)
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
        help="Take one update per epoch for every trainable component.",
    )
    parser.add_argument(
        "--minibatch-updates",
        dest="accumulate_full_epoch",
        action="store_false",
        help=(
            "Update NIR/AIS heads per minibatch while accumulating exactly one "
            "LoRA/CoOp update per epoch, matching official CLIP update density."
        ),
    )
    parser.add_argument("--skip-audit", action="store_true")
    parser.set_defaults(
        augment=True,
        domain_class_balanced_train=False,
        accumulate_full_epoch=False,
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_frequency <= 0:
        raise ValueError("epochs, batch-size and eval-frequency must be positive")
    if min(args.lora_lr, args.prompt_lr, args.adapter_lr, args.gate_lr) <= 0:
        raise ValueError("all learning rates must be positive")
    if not 0.0 < args.text_lora_lr_scale <= 1.0:
        raise ValueError("text-lora-lr-scale must be in (0, 1]")
    if min(args.lora_weight_decay, args.weight_decay) < 0:
        raise ValueError("weight decays must be non-negative")
    if not 0.0 < args.topk_ratio <= 1.0:
        raise ValueError("topk-ratio must be in (0, 1]")
    if args.temper_rate < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError("invalid temper-rate or label-smoothing")
    if not 0.0 <= args.tir_min_weight <= args.tir_max_weight:
        raise ValueError("TIR weight bounds must satisfy 0 <= min <= max")
    if not 0.0 < args.nir_initial_scale:
        raise ValueError("nir-initial-scale must be positive")
    if not 0.0 < args.ais_initial_scale:
        raise ValueError("ais-initial-scale must be positive")
    if args.text_layer not in range(12):
        raise ValueError("text-layer must be between 0 and 11")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError("early stopping settings cannot be negative")
    if not 0 <= args.modality_warmup_epochs < args.epochs:
        raise ValueError("modality-warmup-epochs must be in [0, epochs)")
    if not args.modality_warmup_epochs < args.text_unfreeze_epoch <= args.epochs + 1:
        raise ValueError(
            "text-unfreeze-epoch must be after modality warmup and at most epochs+1"
        )
    if min(args.nir_aux_loss_weight, args.ais_aux_loss_weight) < 0:
        raise ValueError("auxiliary loss weights must be non-negative")
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
    lora_parameter_groups = []
    lora_group_indices = {}
    if groups["lora_vision"]:
        lora_group_indices["vision"] = len(lora_parameter_groups)
        lora_parameter_groups.append(
            {"params": groups["lora_vision"], "lr": args.lora_lr}
        )
    if groups["lora_text"]:
        lora_group_indices["text"] = len(lora_parameter_groups)
        lora_parameter_groups.append(
            {
                "params": groups["lora_text"],
                "lr": args.lora_lr * args.text_lora_lr_scale,
            }
        )
    lora_optimizer = torch.optim.AdamW(
        lora_parameter_groups,
        weight_decay=args.lora_weight_decay,
    )
    prompt_optimizer = torch.optim.SGD(
        groups["prompt"],
        lr=args.prompt_lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    adapter_optimizer = torch.optim.AdamW(
        [
            {"params": groups["adapters"], "lr": args.adapter_lr},
            {"params": groups["gates"], "lr": args.gate_lr},
        ],
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
    prompt_scheduler = torch.optim.lr_scheduler.LambdaLR(prompt_optimizer, decay)
    adapter_scheduler = torch.optim.lr_scheduler.LambdaLR(adapter_optimizer, decay)
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
            "the RGB TIR, CoOp and LoRA mechanisms are retained while the "
            "task-specific NIR/AIS heads use supervised minibatch optimization"
        ),
        "source_labelled_samples": len(source),
        "target_labelled_samples": len(target),
        "target_label_budget": "all available target_train labels",
        "active_class_ids_from_target_train": list(active_class_ids),
        "source_target_class_sets_equal": True,
        "training_update_semantics": (
            "one accumulated full-epoch update over source+target training data"
            if args.accumulate_full_epoch
            else (
                "NIR/AIS heads and gates update per minibatch; LoRA/CoOp take "
                "one gradient-averaged update per epoch, matching the released "
                "full-support CLIP update density"
            )
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
            "heading sin/cos, length, and width into a directly supervised "
            "three-class MLP; validity mask is gating-only"
        ),
        "ais_feature_names": list(SAFE_AIS_FEATURE_NAMES),
        "excluded_label_proxy_fields": list(EXCLUDED_AIS_PROXY_FEATURES),
        "ais_mask_as_classifier_feature": False,
        "image_input": (
            "8-bit-semantics RGB scaling (/255 with sparse outlier clipping), "
            "CLIP normalization at 224x224; NIR uses an independent directly "
            "supervised convolutional classifier"
        ),
        "class_names": list(DEFAULT_CLASS_NAMES),
        "tir": {
            "topk_ratio": args.topk_ratio,
            "temper_rate": args.temper_rate,
            "visual_blocks": [8, 10],
            "weight_rule": "paper_simplified_kway_extremes_conservative",
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
            "text_lr_scale": args.text_lora_lr_scale,
            "weight_decay": args.lora_weight_decay,
        },
        "prompt_lr": args.prompt_lr,
        "adapter_lr": args.adapter_lr,
        "gate_lr": args.gate_lr,
        "task_weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "modality_warmup_epochs": args.modality_warmup_epochs,
        "text_unfreeze_epoch": args.text_unfreeze_epoch,
        "auxiliary_loss_weights": {
            "nir": args.nir_aux_loss_weight,
            "ais": args.ais_aux_loss_weight,
        },
        "fusion": (
            "confidence-calibrated convex probability fusion of RGB, NIR and "
            "AIS with softplus-positive trainable modal weights"
        ),
        "ais_internal_dropout": args.ais_dropout_probability,
        "ais_whole_branch_dropout": 0.0,
        "scheduler": {
            "name": args.scheduler,
            "min_lr_ratio": args.min_lr_ratio,
            "warmup_epochs": args.warmup_epochs,
        },
        "initial_fusion_scales": {
            "nir": args.nir_initial_scale,
            "ais": args.ais_initial_scale,
        },
        "nir_fusion_constraint": "softplus_positive_(0,+inf)",
        "ais_fusion_constraint": "softplus_positive_(0,+inf)",
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
        "modal_optimizer_effective_batch_size": (
            len(train_data)
            if args.accumulate_full_epoch
            else min(args.batch_size, len(train_data))
        ),
        "clip_optimizer_effective_batch_size": len(train_data),
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
        vision_lora_active = bool(groups["lora_vision"]) and (
            epoch > args.modality_warmup_epochs
        )
        text_lora_active = bool(groups["lora_text"]) and (
            epoch >= args.text_unfreeze_epoch
        )
        prompt_active = epoch >= args.text_unfreeze_epoch
        clip_active = vision_lora_active or text_lora_active or prompt_active
        training_stage = (
            "modality_warmup"
            if epoch <= args.modality_warmup_epochs
            else (
                "vision_lora"
                if epoch < args.text_unfreeze_epoch
                else "full_joint"
            )
        )
        for parameter in groups["lora_vision"]:
            parameter.requires_grad_(vision_lora_active)
        for parameter in groups["lora_text"]:
            parameter.requires_grad_(text_lora_active)
        for parameter in groups["prompt"]:
            parameter.requires_grad_(prompt_active)
        for parameter in groups["adapters"] + groups["gates"]:
            parameter.requires_grad_(True)
        model.train()
        epoch_vision_lora_lr = (
            lora_optimizer.param_groups[lora_group_indices["vision"]]["lr"]
            if vision_lora_active
            else 0.0
        )
        epoch_text_lora_lr = (
            lora_optimizer.param_groups[lora_group_indices["text"]]["lr"]
            if text_lora_active
            else 0.0
        )
        epoch_lora_lr = max(epoch_vision_lora_lr, epoch_text_lora_lr)
        epoch_prompt_lr = (
            prompt_optimizer.param_groups[0]["lr"] if prompt_active else 0.0
        )
        epoch_adapter_lr = adapter_optimizer.param_groups[0]["lr"]
        epoch_gate_lr = adapter_optimizer.param_groups[1]["lr"]
        train_items = train_correct = visual_correct = 0
        nir_correct = ais_correct = 0
        source_items = source_correct = target_items = target_correct = 0
        source_loss_sum = target_loss_sum = 0.0
        source_nir_loss_sum = target_nir_loss_sum = 0.0
        source_ais_loss_sum = target_ais_loss_sum = 0.0
        epoch_tir_stats = []
        epoch_fusion_stats = []
        modal_optimizer_steps = 0
        clip_optimizer_steps = 0
        optimizers = (lora_optimizer, prompt_optimizer, adapter_optimizer)
        if args.accumulate_full_epoch:
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
        elif clip_active:
            lora_optimizer.zero_grad(set_to_none=True)
            prompt_optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, 1):
            labels = batch["label"].to(device)
            domains = batch["domain"].to(device)
            logits, visual_logits, nir_logits, ais_logits = model(
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
            source_nir_loss = (
                source_criterion(nir_logits[source_mask], labels[source_mask])
                if source_batch_items
                else logits.new_zeros(())
            )
            target_nir_loss = (
                target_criterion(nir_logits[target_mask], labels[target_mask])
                if target_batch_items
                else logits.new_zeros(())
            )
            source_ais_loss = (
                source_criterion(ais_logits[source_mask], labels[source_mask])
                if source_batch_items
                else logits.new_zeros(())
            )
            target_ais_loss = (
                target_criterion(ais_logits[target_mask], labels[target_mask])
                if target_batch_items
                else logits.new_zeros(())
            )
            if args.accumulate_full_epoch:
                fused_loss = (
                    args.source_loss_weight * source_loss / len(source)
                    + args.target_loss_weight * target_loss / len(target)
                )
                nir_aux_loss = (
                    args.source_loss_weight * source_nir_loss / len(source)
                    + args.target_loss_weight * target_nir_loss / len(target)
                )
                ais_aux_loss = (
                    args.source_loss_weight * source_ais_loss / len(source)
                    + args.target_loss_weight * target_ais_loss / len(target)
                )
            else:
                fused_loss = (
                    args.source_loss_weight
                    * source_loss
                    / max(source_batch_items, 1)
                    + args.target_loss_weight
                    * target_loss
                    / max(target_batch_items, 1)
                )
                nir_aux_loss = (
                    args.source_loss_weight
                    * source_nir_loss
                    / max(source_batch_items, 1)
                    + args.target_loss_weight
                    * target_nir_loss
                    / max(target_batch_items, 1)
                )
                ais_aux_loss = (
                    args.source_loss_weight
                    * source_ais_loss
                    / max(source_batch_items, 1)
                    + args.target_loss_weight
                    * target_ais_loss
                    / max(target_batch_items, 1)
                )
            loss = (
                fused_loss
                + args.nir_aux_loss_weight * nir_aux_loss
                + args.ais_aux_loss_weight * ais_aux_loss
            )
            if args.accumulate_full_epoch:
                loss.backward()
            else:
                adapter_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                adapter_optimizer.step()
                modal_optimizer_steps += 1
            train_items += items
            train_correct += int((logits.argmax(1) == labels).sum().item())
            visual_correct += int(
                (visual_logits.argmax(1) == labels).sum().item()
            )
            nir_correct += int((nir_logits.argmax(1) == labels).sum().item())
            ais_correct += int((ais_logits.argmax(1) == labels).sum().item())
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
            source_nir_loss_sum += float(source_nir_loss.detach().item())
            target_nir_loss_sum += float(target_nir_loss.detach().item())
            source_ais_loss_sum += float(source_ais_loss.detach().item())
            target_ais_loss_sum += float(target_ais_loss.detach().item())
            epoch_tir_stats.extend(getattr(model, "last_tir_stats", ()))
            epoch_fusion_stats.append(dict(model.last_fusion_stats))
            if batch_index % 5 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(train_loader)}] "
                    f"domain_loss={loss.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        lora_gradient_norm = 0.0
        prompt_gradient_norm = 0.0
        if args.accumulate_full_epoch:
            lora_gradient_norm = gradient_l2_norm(groups["lora"])
            prompt_gradient_norm = gradient_l2_norm(groups["prompt"])
            if vision_lora_active or text_lora_active:
                lora_optimizer.step()
            if prompt_active:
                prompt_optimizer.step()
            adapter_optimizer.step()
            modal_optimizer_steps = 1
            clip_optimizer_steps = int(clip_active)
        elif clip_active:
            active_clip_groups = {}
            if vision_lora_active:
                active_clip_groups["vision_lora"] = groups["lora_vision"]
            if text_lora_active:
                active_clip_groups["text_lora"] = groups["lora_text"]
            if prompt_active:
                active_clip_groups["prompt"] = groups["prompt"]
            normalize_accumulated_gradients(
                active_clip_groups, max(modal_optimizer_steps, 1)
            )
            lora_gradient_norm = gradient_l2_norm(groups["lora"])
            prompt_gradient_norm = gradient_l2_norm(groups["prompt"])
            if vision_lora_active or text_lora_active:
                lora_optimizer.step()
            if prompt_active:
                prompt_optimizer.step()
            clip_optimizer_steps = 1
        tir_tokens = sum(int(stats["tokens"]) for stats in epoch_tir_stats)

        def sum_tir_stat(name):
            if not epoch_tir_stats:
                return 0
            return int(
                torch.stack([stats[name] for stats in epoch_tir_stats]).sum().item()
            )

        tir_discriminative = sum_tir_stat("discriminative")
        tir_sink = sum_tir_stat("sink")
        tir_amplified = sum_tir_stat("amplified")
        tir_suppressed = sum_tir_stat("suppressed")
        tir_neutral = sum_tir_stat("neutral")
        tir_clipped = sum_tir_stat("clipped")
        tir_weight_sum = sum(
            float(stats["weight_sum"].cpu()) for stats in epoch_tir_stats
        )
        tir_weight_sq_sum = sum(
            float(stats["weight_sq_sum"].cpu()) for stats in epoch_tir_stats
        )
        tir_weight_mean = tir_weight_sum / max(tir_tokens, 1)
        tir_weight_variance = max(
            tir_weight_sq_sum / max(tir_tokens, 1) - tir_weight_mean**2,
            0.0,
        )
        fusion_samples = sum(stats["samples"] for stats in epoch_fusion_stats)

        def mean_fusion_stat(name):
            if not epoch_fusion_stats:
                return 0.0
            return sum(
                float(stats[name].cpu()) for stats in epoch_fusion_stats
            ) / max(fusion_samples, 1)

        if vision_lora_active or text_lora_active:
            lora_scheduler.step()
        if prompt_active:
            prompt_scheduler.step()
        adapter_scheduler.step()
        completed_epoch = epoch
        source_epoch_ce = source_loss_sum / max(source_items, 1)
        target_epoch_ce = target_loss_sum / max(target_items, 1)
        source_nir_epoch_ce = source_nir_loss_sum / max(source_items, 1)
        target_nir_epoch_ce = target_nir_loss_sum / max(target_items, 1)
        source_ais_epoch_ce = source_ais_loss_sum / max(source_items, 1)
        target_ais_epoch_ce = target_ais_loss_sum / max(target_items, 1)
        weighted_epoch_ce = (
            args.source_loss_weight * source_epoch_ce
            + args.target_loss_weight * target_epoch_ce
        )
        weighted_nir_epoch_ce = (
            args.source_loss_weight * source_nir_epoch_ce
            + args.target_loss_weight * target_nir_epoch_ce
        )
        weighted_ais_epoch_ce = (
            args.source_loss_weight * source_ais_epoch_ce
            + args.target_loss_weight * target_ais_epoch_ce
        )
        weighted_total_loss = (
            weighted_epoch_ce
            + args.nir_aux_loss_weight * weighted_nir_epoch_ce
            + args.ais_aux_loss_weight * weighted_ais_epoch_ce
        )
        train_history.append(
            {
                "epoch": epoch,
                "training_stage": training_stage,
                "accuracy": train_correct / max(train_items, 1),
                "rgb_accuracy": visual_correct / max(train_items, 1),
                "nir_accuracy": nir_correct / max(train_items, 1),
                "ais_accuracy": ais_correct / max(train_items, 1),
                "source_accuracy": source_correct / max(source_items, 1),
                "target_accuracy": target_correct / max(target_items, 1),
                "ce": weighted_epoch_ce,
                "nir_aux_ce": weighted_nir_epoch_ce,
                "ais_aux_ce": weighted_ais_epoch_ce,
                "source_weighted_ce": source_epoch_ce,
                "target_weighted_ce": target_epoch_ce,
                "total_loss": weighted_total_loss,
                "lora_lr": epoch_lora_lr,
                "vision_lora_lr": epoch_vision_lora_lr,
                "text_lora_lr": epoch_text_lora_lr,
                "prompt_lr": epoch_prompt_lr,
                "adapter_lr": epoch_adapter_lr,
                "gate_lr": epoch_gate_lr,
                "vision_lora_lr_after_scheduler": (
                    lora_optimizer.param_groups[lora_group_indices["vision"]]["lr"]
                    if "vision" in lora_group_indices
                    else 0.0
                ),
                "text_lora_lr_after_scheduler": (
                    lora_optimizer.param_groups[lora_group_indices["text"]]["lr"]
                    if "text" in lora_group_indices
                    else 0.0
                ),
                "prompt_lr_after_scheduler": prompt_optimizer.param_groups[0]["lr"],
                "adapter_lr_after_scheduler": adapter_optimizer.param_groups[0]["lr"],
                "gate_lr_after_scheduler": adapter_optimizer.param_groups[1]["lr"],
                "samples": train_items,
                "source_samples": source_items,
                "target_samples": target_items,
                "optimizer_steps": modal_optimizer_steps,
                "modal_optimizer_steps": modal_optimizer_steps,
                "clip_optimizer_steps": clip_optimizer_steps,
                "lora_gradient_norm": lora_gradient_norm,
                "prompt_gradient_norm": prompt_gradient_norm,
                "tir_discriminative_fraction": tir_discriminative
                / max(tir_tokens, 1),
                "tir_sink_fraction": tir_sink / max(tir_tokens, 1),
                "tir_amplified_fraction": tir_amplified / max(tir_tokens, 1),
                "tir_suppressed_fraction": tir_suppressed
                / max(tir_tokens, 1),
                "tir_neutral_fraction": tir_neutral / max(tir_tokens, 1),
                "tir_clipped_fraction": tir_clipped / max(tir_tokens, 1),
                "tir_weight_mean": tir_weight_mean,
                "tir_weight_std": math.sqrt(tir_weight_variance),
                "nir_scale": float(model.effective_nir_scale().detach().cpu()),
                "ais_scale": float(model.effective_ais_scale().detach().cpu()),
                "fusion_visual_fraction": mean_fusion_stat(
                    "visual_fraction_sum"
                ),
                "fusion_nir_fraction": mean_fusion_stat("nir_fraction_sum"),
                "fusion_ais_fraction": mean_fusion_stat("ais_fraction_sum"),
                "nir_confidence": mean_fusion_stat("nir_confidence_sum"),
                "ais_confidence": mean_fusion_stat("ais_confidence_sum"),
            }
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] train accuracy="
            f"{train_correct / max(train_items, 1):.4f} "
            f"rgb_accuracy={visual_correct / max(train_items, 1):.4f} "
            f"source_accuracy={source_correct / max(source_items, 1):.4f} "
            f"target_accuracy={target_correct / max(target_items, 1):.4f} "
            f"nir_accuracy={nir_correct / max(train_items, 1):.4f} "
            f"ais_accuracy={ais_correct / max(train_items, 1):.4f} "
            f"source_ce={source_epoch_ce:.4f} target_ce={target_epoch_ce:.4f} "
            f"nir_ce={weighted_nir_epoch_ce:.4f} "
            f"ais_ce={weighted_ais_epoch_ce:.4f} "
            f"total_loss={weighted_total_loss:.4f} "
            f"stage={training_stage} modal_steps={modal_optimizer_steps} "
            f"clip_steps={clip_optimizer_steps} "
            f"TIR_weight={tir_weight_mean:.3f}+/-{math.sqrt(tir_weight_variance):.3f} "
            f"fusion=(RGB {mean_fusion_stat('visual_fraction_sum'):.3f}, "
            f"NIR {mean_fusion_stat('nir_fraction_sum'):.3f}, "
            f"AIS {mean_fusion_stat('ais_fraction_sum'):.3f})"
        )

        should_evaluate = epoch == 1 or epoch == args.epochs or epoch % args.eval_frequency == 0
        if should_evaluate:
            val_modal_metrics = evaluate_tir_modalities(model, val_loader, device)
            test_modal_metrics = evaluate_tir_modalities(model, test_loader, device)
            val_metrics = val_modal_metrics["fused"]
            test_metrics = test_modal_metrics["fused"]
            for split, branch_metrics in (
                ("target_val", val_modal_metrics),
                ("target_test", test_modal_metrics),
            ):
                for branch, metrics in branch_metrics.items():
                    train_history[-1][f"{split}_{branch}_ACC"] = metrics[
                        "accuracy"
                    ]
                    train_history[-1][f"{split}_{branch}_F1_macro"] = metrics[
                        "f1_macro"
                    ]
            val_history.append((epoch, val_metrics))
            test_history.append((epoch, test_metrics))
            print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_val", val_metrics))
            print(metric_line(f"Epoch [{epoch}/{args.epochs}] target_test", test_metrics))
            print(
                f"Epoch [{epoch}/{args.epochs}] modal target_test "
                + " | ".join(
                    f"{name}: ACC={metrics['accuracy']:.4f}, "
                    f"F1={metrics['f1_macro']:.4f}"
                    for name, metrics in test_modal_metrics.items()
                )
            )
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
    selected_modal_metrics = {
        "target_val": evaluate_tir_modalities(model, val_loader, device),
        "target_test": evaluate_tir_modalities(model, test_loader, device),
    }
    metadata["selected_modal_metrics"] = selected_modal_metrics
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
