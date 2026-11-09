"""Train TIR on paired VIS+IR weather domains with full target supervision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TIR.vds2raw_adapter import OFFICIAL_COMMIT
from TIR.weather_adapter import TIRWeather
from tools.class_names import load_class_names
from tools.classification_metrics import classification_metrics
from tools.metrics_io import save_metric_history
from tools.paired_transforms import PairedNaturalImageTransform
from tools.reproducibility import seed_everything
from tools.uda_data import (
    PairedUdaImageListDataset,
    assert_training_label_space,
    uda_label_space,
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TARGETS = ("backlight", "rainy", "foggy", "dark")


class DomainTaggedDataset(Dataset):
    """Attach domain and path metadata to the shared paired dataset."""

    def __init__(self, dataset, domain_id):
        self.dataset = dataset
        self.domain_id = int(domain_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs, label = self.dataset[index]
        (vis_path, ir_path), _ = self.dataset.samples[index]
        return {
            "vis": inputs["rgb"],
            "ir": inputs["nir"],
            "label": int(label),
            "domain": self.domain_id,
            "vis_path": vis_path,
            "ir_path": ir_path,
        }


def mild_class_weights(dataset, class_count, power):
    if not 0.0 <= power <= 1.0:
        raise ValueError("class weight power must be in [0, 1]")
    counts = torch.zeros(class_count, dtype=torch.float64)
    for _, label in dataset.samples:
        counts[int(label)] += 1
    if bool((counts <= 0).any()):
        raise ValueError(f"every active class needs samples; counts={counts.tolist()}")
    weights = counts.pow(-float(power))
    weights = weights * counts.sum() / (weights * counts).sum()
    return weights.float(), counts.to(torch.int64).tolist()


def scheduled_multiplier(step, total_steps, warmup_steps, min_ratio):
    """Warm up and decay a parameter group only while that group is active."""

    if step <= 0:
        return 0.0
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("schedule needs 0 <= warmup_steps < total_steps")
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = (step - warmup_steps - 1) / max(decay_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0


def normalize_gradients(parameters, denominator):
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(float(max(denominator, 1)))


def gradient_norm(parameters):
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt().cpu())


def evaluate(model, loader, device, class_count):
    model.eval()
    labels = []
    predictions = {"fused": [], "vis": [], "ir": []}
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["vis"].to(device), batch["ir"].to(device))
            labels.extend(batch["label"].tolist())
            for name, logits in zip(predictions, outputs):
                predictions[name].extend(logits.argmax(1).cpu().tolist())
    return {
        name: classification_metrics(labels, values, num_classes=class_count)
        for name, values in predictions.items()
    }


def better(candidate, incumbent):
    if incumbent is None:
        return True
    return (candidate["accuracy"], candidate["f1_macro"]) > (
        incumbent["accuracy"],
        incumbent["f1_macro"],
    )


def write_records(save_dir, stem, records, metadata):
    if not records:
        raise ValueError("training history cannot be empty")
    save_dir.mkdir(parents=True, exist_ok=True)
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    csv_path = save_dir / f"{stem}.csv"
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(csv_path)
    json_path = save_dir / f"{stem}.json"
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"metadata": metadata, "history": records}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(json_path)
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="TIR CLIP/CoOp/LoRA/TIR core on paired weather VIS+IR data"
    )
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data_lists"))
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "tir_weather"))
    parser.add_argument("--class-names-file", default=str(ROOT / "class.txt"))
    parser.add_argument(
        "--clip-checkpoint",
        default=str(ROOT / "pretrained_weights" / "clip" / "ViT-B-16.pt"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-frequency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--text-lora-lr-scale", type=float, default=0.25)
    parser.add_argument("--prompt-lr", type=float, default=5e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=2e-3)
    parser.add_argument("--lora-weight-decay", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.25)
    parser.add_argument(
        "--lora-encoder", choices=("vision", "text", "both"), default="both"
    )
    parser.add_argument("--lora-position", default="all")
    parser.add_argument(
        "--lora-params", nargs="+", choices=("q", "k", "v"), default=("q", "k", "v")
    )
    parser.add_argument("--topk-ratio", type=float, default=0.3)
    parser.add_argument("--temper-rate", type=float, default=0.5)
    parser.add_argument("--tir-min-weight", type=float, default=0.5)
    parser.add_argument("--tir-max-weight", type=float, default=1.5)
    parser.add_argument("--text-layer", type=int, default=10)
    parser.add_argument("--logit-scale", type=float, default=4.0)
    parser.add_argument("--ir-initial-scale", type=float, default=0.1)
    parser.add_argument("--ir-dropout", type=float, default=0.1)
    parser.add_argument("--ir-aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--source-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-loss-weight", type=float, default=0.5)
    parser.add_argument("--modality-warmup-epochs", type=int, default=10)
    parser.add_argument("--text-unfreeze-epoch", type=int, default=31)
    parser.add_argument("--lr-warmup-steps", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=50)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.eval_frequency <= 0 or args.num_workers < 0 or args.max_train_batches < 0:
        raise ValueError("invalid evaluation, worker, or smoke-batch setting")
    if min(args.lora_lr, args.prompt_lr, args.adapter_lr, args.gate_lr) <= 0:
        raise ValueError("all learning rates must be positive")
    if min(args.lora_weight_decay, args.weight_decay) < 0:
        raise ValueError("weight decays must be non-negative")
    if not 0.0 < args.text_lora_lr_scale <= 1.0:
        raise ValueError("text-lora-lr-scale must be in (0, 1]")
    if not 0.0 < args.topk_ratio <= 1.0 or args.temper_rate < 0:
        raise ValueError("invalid TIR top-k or temperature setting")
    if not 0 <= args.tir_min_weight <= args.tir_max_weight:
        raise ValueError("TIR weight bounds must satisfy 0 <= min <= max")
    if args.text_layer not in range(12):
        raise ValueError("text-layer must be in 0..11")
    if not 0 <= args.label_smoothing < 1 or not 0 <= args.ir_dropout < 1:
        raise ValueError("invalid label smoothing or IR dropout")
    if args.ir_initial_scale <= 0 or args.ir_aux_loss_weight < 0:
        raise ValueError("IR scale must be positive and auxiliary weight non-negative")
    if not 0 <= args.class_weight_power <= 1:
        raise ValueError("class-weight-power must be in [0, 1]")
    if args.source_loss_weight < 0 or args.target_loss_weight < 0:
        raise ValueError("domain loss weights must be non-negative")
    domain_weight_sum = args.source_loss_weight + args.target_loss_weight
    if domain_weight_sum <= 0:
        raise ValueError("at least one domain loss weight must be positive")
    args.source_loss_weight /= domain_weight_sum
    args.target_loss_weight /= domain_weight_sum
    if not 0 <= args.modality_warmup_epochs < args.epochs:
        raise ValueError("modality-warmup-epochs must be in [0, epochs)")
    if not args.modality_warmup_epochs < args.text_unfreeze_epoch <= args.epochs:
        raise ValueError(
            "text-unfreeze-epoch must follow modality warmup and not exceed epochs"
        )
    active_vision_steps = args.epochs - args.modality_warmup_epochs
    active_text_steps = args.epochs - args.text_unfreeze_epoch + 1
    if (
        not 0
        <= args.lr_warmup_steps
        < min(args.epochs, active_vision_steps, active_text_steps)
    ):
        raise ValueError("lr-warmup-steps must be smaller than every active schedule")
    if not 0 <= args.min_lr_ratio <= 1 or args.grad_clip < 0:
        raise ValueError("invalid LR floor or gradient clipping")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 0:
        raise ValueError("early stopping settings cannot be negative")

    args.seed = seed_everything(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    clip_checkpoint = Path(args.clip_checkpoint).expanduser().resolve()
    if not clip_checkpoint.is_file():
        raise FileNotFoundError(f"missing strict CLIP checkpoint: {clip_checkpoint}")

    class_ids, label_map = uda_label_space(data_dir, args.target)
    all_class_names = load_class_names(args.class_names_file, expected_count=14)
    class_names = [all_class_names[class_id] for class_id in class_ids]
    class_count = len(class_names)
    print(
        f"Target-driven class space: original_ids={class_ids}, "
        f"names={class_names}, remapped=0..{class_count - 1}"
    )

    train_transform = PairedNaturalImageTransform(
        size=224,
        resize=256,
        rgb_mean=CLIP_MEAN,
        rgb_std=CLIP_STD,
        random_crop=True,
        horizontal_flip=True,
        vertical_flip=False,
        color_jitter=0.1,
    )
    eval_transform = PairedNaturalImageTransform(
        size=224,
        resize=256,
        rgb_mean=CLIP_MEAN,
        rgb_std=CLIP_STD,
        random_crop=False,
        horizontal_flip=False,
        vertical_flip=False,
        color_jitter=0.0,
    )
    source = PairedUdaImageListDataset(
        data_dir / "sunny_train.txt", transform=train_transform, label_map=label_map
    )
    target_train = PairedUdaImageListDataset(
        data_dir / f"{args.target}_train.txt",
        transform=train_transform,
        label_map=label_map,
    )
    target_val = PairedUdaImageListDataset(
        data_dir / f"{args.target}_val.txt",
        transform=eval_transform,
        label_map=label_map,
    )
    assert_training_label_space(source, target_train, class_ids)
    print(
        f"Source sunny train: kept={len(source)}, label_dropped={source.skipped}, "
        f"unpaired_dropped={source.dropped_unpaired}"
    )
    print(
        f"Target {args.target} train (all labels visible): kept={len(target_train)}, "
        f"label_dropped={target_train.skipped}, "
        f"unpaired_dropped={target_train.dropped_unpaired}"
    )
    print(
        f"Target {args.target} val: kept={len(target_val)}, "
        f"unpaired_dropped={target_val.dropped_unpaired}"
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    train_data = ConcatDataset(
        (DomainTaggedDataset(source, 0), DomainTaggedDataset(target_train, 1))
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        DomainTaggedDataset(target_val, 1),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; fresh model initialization for seed {args.seed}")
    model = TIRWeather(
        clip_checkpoint=clip_checkpoint,
        class_names=class_names,
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
        ir_initial_scale=args.ir_initial_scale,
        ir_dropout=args.ir_dropout,
        logit_scale=args.logit_scale,
    ).to(device)
    groups = model.parameter_groups()
    if not groups["lora"] or not groups["adapters"] or not groups["prompt"]:
        raise RuntimeError("TIR parameter injection/groups are incomplete")

    lora_groups = []
    lora_indices = {}
    if groups["lora_vision"]:
        lora_indices["vision"] = len(lora_groups)
        lora_groups.append({"params": groups["lora_vision"], "lr": args.lora_lr})
    if groups["lora_text"]:
        lora_indices["text"] = len(lora_groups)
        lora_groups.append(
            {
                "params": groups["lora_text"],
                "lr": args.lora_lr * args.text_lora_lr_scale,
            }
        )
    lora_optimizer = torch.optim.AdamW(lora_groups, weight_decay=args.lora_weight_decay)
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

    source_weights, source_counts = mild_class_weights(
        source, class_count, args.class_weight_power
    )
    target_weights, target_counts = mild_class_weights(
        target_train, class_count, args.class_weight_power
    )
    source_criterion = nn.CrossEntropyLoss(
        weight=source_weights.to(device),
        label_smoothing=args.label_smoothing,
        reduction="sum",
    )
    target_criterion = nn.CrossEntropyLoss(
        weight=target_weights.to(device),
        label_smoothing=args.label_smoothing,
        reduction="sum",
    )

    metadata = {
        "method": "TIR-VIS-IR",
        "official_repository": "https://github.com/shuaiyi308/TIR",
        "official_commit": OFFICIAL_COMMIT,
        "official_core": "CLIP ViT-B/16 + CoOp + LoRA + TIR",
        "weather_adaptation": "independent supervised IR CNN with positive confidence fusion",
        "target": args.target,
        "seed": args.seed,
        "epochs": args.epochs,
        "selection_partition": f"{args.target}_val",
        "selection_rule": "highest ACC, macro-F1 tie-breaker",
        "source_partition": "sunny_train",
        "target_train_partition": f"{args.target}_train",
        "target_label_budget": "all available target_train labels",
        "source_target_class_sets_equal": True,
        "original_class_ids": class_ids,
        "class_names": class_names,
        "source_samples": len(source),
        "target_train_samples": len(target_train),
        "target_val_samples": len(target_val),
        "source_class_counts": source_counts,
        "target_class_counts": target_counts,
        "source_class_weights": source_weights.tolist(),
        "target_class_weights": target_weights.tolist(),
        "source_loss_weight": args.source_loss_weight,
        "target_loss_weight": args.target_loss_weight,
        "modalities": ["VIS_RGB", "IR_grayscale"],
        "augmentation": "paired random resized crop + horizontal flip; VIS-only mild color jitter",
        "normalization": {"VIS": "CLIP ViT-B/16", "IR": "mean=0.5,std=0.5"},
        "training_update_semantics": (
            "IR head/gate update per minibatch; LoRA/CoOp gradients are averaged "
            "into one update per epoch"
        ),
        "checkpoint_policy": "no model file; complete histories and best metrics only",
        "clip_checkpoint": str(clip_checkpoint),
        "configuration": vars(args),
    }

    metric_history = []
    train_history = []
    best_metrics = None
    best_epoch = None
    evaluations_without_improvement = 0
    stopped_early = False
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        vision_active = (
            bool(groups["lora_vision"]) and epoch > args.modality_warmup_epochs
        )
        text_active = bool(groups["lora_text"]) and epoch >= args.text_unfreeze_epoch
        prompt_active = epoch >= args.text_unfreeze_epoch
        clip_active = vision_active or text_active or prompt_active
        stage = (
            "ir_warmup"
            if epoch <= args.modality_warmup_epochs
            else ("vision_lora" if epoch < args.text_unfreeze_epoch else "full_joint")
        )
        for parameter in groups["lora_vision"]:
            parameter.requires_grad_(vision_active)
        for parameter in groups["lora_text"]:
            parameter.requires_grad_(text_active)
        for parameter in groups["prompt"]:
            parameter.requires_grad_(prompt_active)
        for parameter in groups["adapters"] + groups["gates"]:
            parameter.requires_grad_(True)

        adapter_multiplier = scheduled_multiplier(
            epoch, args.epochs, args.lr_warmup_steps, args.min_lr_ratio
        )
        vision_step = epoch - args.modality_warmup_epochs if vision_active else 0
        vision_multiplier = (
            scheduled_multiplier(
                vision_step,
                active_vision_steps,
                args.lr_warmup_steps,
                args.min_lr_ratio,
            )
            if vision_active
            else 0.0
        )
        text_step = epoch - args.text_unfreeze_epoch + 1 if text_active else 0
        text_multiplier = (
            scheduled_multiplier(
                text_step,
                active_text_steps,
                args.lr_warmup_steps,
                args.min_lr_ratio,
            )
            if text_active
            else 0.0
        )
        if "vision" in lora_indices:
            lora_optimizer.param_groups[lora_indices["vision"]]["lr"] = (
                args.lora_lr * vision_multiplier if vision_active else 0.0
            )
        if "text" in lora_indices:
            lora_optimizer.param_groups[lora_indices["text"]]["lr"] = (
                args.lora_lr * args.text_lora_lr_scale * text_multiplier
                if text_active
                else 0.0
            )
        prompt_optimizer.param_groups[0]["lr"] = (
            args.prompt_lr * text_multiplier if prompt_active else 0.0
        )
        adapter_optimizer.param_groups[0]["lr"] = args.adapter_lr * adapter_multiplier
        adapter_optimizer.param_groups[1]["lr"] = args.gate_lr * adapter_multiplier

        model.train()
        lora_optimizer.zero_grad(set_to_none=True)
        prompt_optimizer.zero_grad(set_to_none=True)
        train_items = fused_correct = vis_correct = ir_correct = 0
        source_items = target_items = source_correct = target_correct = 0
        source_loss_sum = target_loss_sum = 0.0
        source_ir_loss_sum = target_ir_loss_sum = 0.0
        clip_batches = 0
        tir_tokens = tir_amplified = tir_suppressed = tir_neutral = 0
        visual_fraction_sum = ir_fraction_sum = ir_confidence_sum = 0.0
        fusion_samples = 0
        for batch_index, batch in enumerate(train_loader, 1):
            labels = batch["label"].to(device)
            domains = batch["domain"].to(device)
            fused_logits, visual_logits, ir_logits = model(
                batch["vis"].to(device, non_blocking=args.pin_memory),
                batch["ir"].to(device, non_blocking=args.pin_memory),
            )
            source_mask = domains.eq(0)
            target_mask = domains.eq(1)
            source_batch_items = int(source_mask.sum().item())
            target_batch_items = int(target_mask.sum().item())
            source_loss = (
                source_criterion(fused_logits[source_mask], labels[source_mask])
                if source_batch_items
                else fused_logits.new_zeros(())
            )
            target_loss = (
                target_criterion(fused_logits[target_mask], labels[target_mask])
                if target_batch_items
                else fused_logits.new_zeros(())
            )
            source_ir_loss = (
                source_criterion(ir_logits[source_mask], labels[source_mask])
                if source_batch_items
                else fused_logits.new_zeros(())
            )
            target_ir_loss = (
                target_criterion(ir_logits[target_mask], labels[target_mask])
                if target_batch_items
                else fused_logits.new_zeros(())
            )
            active_weight = (args.source_loss_weight if source_batch_items else 0.0) + (
                args.target_loss_weight if target_batch_items else 0.0
            )
            fused_loss = (
                args.source_loss_weight * source_loss / max(source_batch_items, 1)
                + args.target_loss_weight * target_loss / max(target_batch_items, 1)
            ) / active_weight
            ir_aux_loss = (
                args.source_loss_weight * source_ir_loss / max(source_batch_items, 1)
                + args.target_loss_weight * target_ir_loss / max(target_batch_items, 1)
            ) / active_weight
            loss = fused_loss + args.ir_aux_loss_weight * ir_aux_loss

            adapter_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    groups["adapters"] + groups["gates"], args.grad_clip
                )
            adapter_optimizer.step()
            clip_batches += int(clip_active)

            items = int(labels.shape[0])
            fused_prediction = fused_logits.argmax(1)
            train_items += items
            fused_correct += int((fused_prediction == labels).sum().item())
            vis_correct += int((visual_logits.argmax(1) == labels).sum().item())
            ir_correct += int((ir_logits.argmax(1) == labels).sum().item())
            source_items += source_batch_items
            target_items += target_batch_items
            if source_batch_items:
                source_correct += int(
                    (fused_prediction[source_mask] == labels[source_mask]).sum().item()
                )
            if target_batch_items:
                target_correct += int(
                    (fused_prediction[target_mask] == labels[target_mask]).sum().item()
                )
            source_loss_sum += float(source_loss.detach().cpu())
            target_loss_sum += float(target_loss.detach().cpu())
            source_ir_loss_sum += float(source_ir_loss.detach().cpu())
            target_ir_loss_sum += float(target_ir_loss.detach().cpu())
            for stats in getattr(model, "last_tir_stats", ()):
                tir_tokens += int(stats["tokens"])
                tir_amplified += int(stats["amplified"].cpu())
                tir_suppressed += int(stats["suppressed"].cpu())
                tir_neutral += int(stats["neutral"].cpu())
            fusion = model.last_fusion_stats
            fusion_samples += fusion["samples"]
            visual_fraction_sum += float(fusion["visual_fraction_sum"].cpu())
            ir_fraction_sum += float(fusion["ir_fraction_sum"].cpu())
            ir_confidence_sum += float(fusion["ir_confidence_sum"].cpu())
            if batch_index % 10 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] batch "
                    f"[{batch_index}/{len(train_loader)}] loss={loss.item():.4f}"
                )
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

        clip_parameters = []
        if vision_active:
            clip_parameters.extend(groups["lora_vision"])
        if text_active:
            clip_parameters.extend(groups["lora_text"])
        if prompt_active:
            clip_parameters.extend(groups["prompt"])
        normalize_gradients(clip_parameters, clip_batches)
        clip_gradient_norm = gradient_norm(clip_parameters)
        if args.grad_clip and clip_parameters:
            torch.nn.utils.clip_grad_norm_(clip_parameters, args.grad_clip)
        if vision_active or text_active:
            lora_optimizer.step()
        if prompt_active:
            prompt_optimizer.step()

        completed_epoch = epoch
        source_epoch_ce = source_loss_sum / max(source_items, 1)
        target_epoch_ce = target_loss_sum / max(target_items, 1)
        source_ir_epoch_ce = source_ir_loss_sum / max(source_items, 1)
        target_ir_epoch_ce = target_ir_loss_sum / max(target_items, 1)
        train_record = {
            "epoch": epoch,
            "stage": stage,
            "train_ACC": fused_correct / max(train_items, 1),
            "train_VIS_ACC": vis_correct / max(train_items, 1),
            "train_IR_ACC": ir_correct / max(train_items, 1),
            "source_ACC": source_correct / max(source_items, 1),
            "target_train_ACC": target_correct / max(target_items, 1),
            "source_CE": source_epoch_ce,
            "target_CE": target_epoch_ce,
            "source_IR_CE": source_ir_epoch_ce,
            "target_IR_CE": target_ir_epoch_ce,
            "effective_IR_scale": float(model.effective_ir_scale().detach().cpu()),
            "VIS_fusion_fraction": visual_fraction_sum / max(fusion_samples, 1),
            "IR_fusion_fraction": ir_fraction_sum / max(fusion_samples, 1),
            "IR_confidence": ir_confidence_sum / max(fusion_samples, 1),
            "TIR_amplified_fraction": tir_amplified / max(tir_tokens, 1),
            "TIR_suppressed_fraction": tir_suppressed / max(tir_tokens, 1),
            "TIR_neutral_fraction": tir_neutral / max(tir_tokens, 1),
            "vision_lora_lr": (
                lora_optimizer.param_groups[lora_indices["vision"]]["lr"]
                if "vision" in lora_indices
                else 0.0
            ),
            "text_lora_lr": (
                lora_optimizer.param_groups[lora_indices["text"]]["lr"]
                if "text" in lora_indices
                else 0.0
            ),
            "prompt_lr": prompt_optimizer.param_groups[0]["lr"],
            "adapter_lr": adapter_optimizer.param_groups[0]["lr"],
            "gate_lr": adapter_optimizer.param_groups[1]["lr"],
            "clip_gradient_norm": clip_gradient_norm,
            "modal_optimizer_steps": batch_index,
            "clip_optimizer_steps": int(clip_active),
        }

        should_evaluate = epoch % args.eval_frequency == 0 or epoch == args.epochs
        if should_evaluate:
            evaluation = evaluate(model, val_loader, device, class_count)
            fused_metrics = evaluation["fused"]
            metric_history.append((epoch, fused_metrics))
            train_record.update(
                {
                    "target_val_ACC": fused_metrics["accuracy"],
                    "target_val_Precision_macro": fused_metrics["precision_macro"],
                    "target_val_Recall_macro": fused_metrics["recall_macro"],
                    "target_val_F1_macro": fused_metrics["f1_macro"],
                    "target_val_VIS_ACC": evaluation["vis"]["accuracy"],
                    "target_val_IR_ACC": evaluation["ir"]["accuracy"],
                }
            )
            print(
                f"Epoch [{epoch}/{args.epochs}] target_val "
                f"ACC={fused_metrics['accuracy']:.4f} "
                f"P={fused_metrics['precision_macro']:.4f} "
                f"R={fused_metrics['recall_macro']:.4f} "
                f"F1={fused_metrics['f1_macro']:.4f} "
                f"VIS_ACC={evaluation['vis']['accuracy']:.4f} "
                f"IR_ACC={evaluation['ir']['accuracy']:.4f}"
            )
            if better(fused_metrics, best_metrics):
                best_metrics = dict(fused_metrics)
                best_epoch = epoch
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
        train_history.append(train_record)
        if (
            should_evaluate
            and args.early_stopping_patience
            and epoch >= args.early_stopping_min_epochs
            and evaluations_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print("Early stopping triggered on target_val ACC/F1.")
            break

    if best_metrics is None or best_epoch is None:
        raise RuntimeError("training completed without a target validation evaluation")
    metadata.update(
        {
            "completed_epochs": completed_epoch,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        }
    )
    stem = f"tir_weather_{args.target}_seed{args.seed}"
    metrics_csv, metrics_json = save_metric_history(
        save_dir, stem, metric_history, metadata=metadata
    )
    train_csv, train_json = write_records(
        save_dir, f"{stem}_train_history", train_history, metadata
    )
    print(
        f"FINAL_BEST target={args.target} seed={args.seed} epoch={best_epoch} "
        f"ACC={best_metrics['accuracy']:.6f} "
        f"Precision_macro={best_metrics['precision_macro']:.6f} "
        f"Recall_macro={best_metrics['recall_macro']:.6f} "
        f"F1_macro={best_metrics['f1_macro']:.6f}"
    )
    print(f"Metrics: {metrics_csv}; {metrics_json}")
    print(f"Training diagnostics: {train_csv}; {train_json}")
    print("No model/checkpoint file was retained.")


if __name__ == "__main__":
    main()
