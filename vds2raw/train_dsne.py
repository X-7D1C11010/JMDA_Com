"""Official natural-image d-SNE recipe on the VDS2Raw cross-day protocol."""

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
    NaturalImageRGBDataset,
    audit_dataset,
    fixed_samples_per_class,
    load_or_compute_statistics,
)
from vds2raw.models import DSNEOfficeModel, dsne_loss


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
    """Match the released d-SNE target-direction update exactly."""
    first_logits, first_features = model(first_images, return_embedding=True)
    _, second_features = model(second_images, return_embedding=True)
    cross_entropy = criterion(first_logits, first_labels)
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


def train_source_model(model, loader, device, args):
    """Official ImageNet -> labelled-source supervised fine-tuning stage."""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.source_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_index, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            sample_count += images.shape[0]
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
        if epoch == 1 or epoch % 10 == 0 or epoch == args.source_epochs:
            print(
                f"d-SNE source fine-tune [{epoch}/{args.source_epochs}] "
                f"CE={loss_sum / max(sample_count, 1):.4f}"
            )


def load_model_state(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("state_dict", payload))
    model.load_state_dict(state)


def evaluate_budget(
    model,
    checkpoint,
    budget,
    history,
    metadata,
    data_root,
    statistics,
    save_dir,
    device,
    args,
):
    selected = load_selected_checkpoint(checkpoint, model, device)
    target_test = NaturalImageRGBDataset(
        data_root, "target_test", statistics, augment=False, normalization="half"
    )
    test_loader = DataLoader(
        target_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_metrics = evaluate_rgb(model, test_loader, device)
    print(metric_line(f"FINAL target_test (budget={budget})", test_metrics))
    budget_metadata = dict(metadata)
    budget_metadata["method"] = f"d-SNE@{budget}"
    budget_metadata["adaptation_budget_epochs"] = budget
    stem = f"vds2raw_dsne_e{budget}_seed{args.seed}"
    save_experiment_results(
        save_dir,
        stem,
        history[:budget],
        selected["selected_epoch"],
        selected["target_val_metrics"],
        test_metrics,
        budget_metadata,
    )


def main():
    parser = argparse.ArgumentParser(description="Official natural-image d-SNE")
    parser.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "vds2raw"))
    parser.add_argument("--stats-file", default="")  # shared runner compatibility
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--source-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--nlayers", type=int, default=101)
    parser.add_argument("--target-samples-per-class", type=int, default=10)
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--pretrained-backbone", default="")
    parser.add_argument("--backbone-checkpoint", default="", help=argparse.SUPPRESS)
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--l2n", action="store_true")
    parser.add_argument("--fn", dest="fn", action="store_true", default=True)
    parser.add_argument("--no-fn", dest="fn", action="store_false")
    parser.add_argument("--train-src", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.source_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, source-epochs and batch-size must be positive")
    if args.nlayers != 101:
        raise ValueError("formal d-SNE reproduction requires ResNet-101")
    if args.embed_size != 512:
        raise ValueError("official natural-image d-SNE uses a 512-D latent space")
    if args.backbone_checkpoint:
        if args.pretrained_backbone:
            raise ValueError(
                "backbone-checkpoint and pretrained-backbone refer to the same input; "
                "provide only one"
            )
        args.pretrained_backbone = args.backbone_checkpoint
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
        data_root, "source_train", statistics, augment=True, normalization="half"
    )
    target_full = NaturalImageRGBDataset(
        data_root, "target_train", statistics, augment=True, normalization="half"
    )
    target_train = fixed_samples_per_class(
        target_full, args.target_samples_per_class, args.seed
    )
    target_val = NaturalImageRGBDataset(
        data_root, "target_val", statistics, augment=False, normalization="half"
    )
    print(
        f"d-SNE labelled target subset: {len(target_train)} samples "
        f"({args.target_samples_per_class} per class)"
    )

    generator = torch.Generator().manual_seed(args.seed)
    source_loader = DataLoader(
        source_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    pairs = CrossDomainPairDataset(
        source_train, target_train, negative_ratio=args.ratio, seed=args.seed
    )
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
    print(
        f"Official d-SNE pair table: {len(pairs)} pairs "
        f"({pairs.positive_pairs} positive, {pairs.negative_pairs} negative, "
        f"batch={args.batch_size})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSNEOfficeModel(
        embed_size=args.embed_size,
        pretrained=not args.no_pretrained and not args.source_checkpoint,
        pretrained_backbone_checkpoint=args.pretrained_backbone,
        dropout=args.dropout,
        l2_normalize=args.l2n,
    ).to(device)
    source_checkpoint = save_dir / f"vds2raw_dsne_seed{args.seed}_source_tuned.pth"
    if args.source_checkpoint:
        load_model_state(model, args.source_checkpoint)
        print(f"Loaded d-SNE source-tuned model: {args.source_checkpoint}")
    else:
        train_source_model(model, source_loader, device, args)
        torch.save({"model_state": model.state_dict()}, source_checkpoint)
        print(f"Saved d-SNE source-tuned model: {source_checkpoint}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    report_budgets = sorted({min(10, args.epochs), args.epochs})
    checkpoints = {
        budget: save_dir / f"vds2raw_dsne_e{budget}_seed{args.seed}_best.pth"
        for budget in report_budgets
    }
    best_by_budget = {budget: None for budget in report_budgets}
    metadata = {
        "method": "d-SNE",
        "implementation": "official natural-image configuration, PyTorch port",
        "seed": args.seed,
        "epochs": args.epochs,
        "source_fine_tune_epochs": args.source_epochs,
        "source_checkpoint_input": args.source_checkpoint,
        "pretrained_backbone_checkpoint_input": args.pretrained_backbone,
        "source_initialization_provenance": (
            "provided source checkpoint"
            if args.source_checkpoint
            else (
                "uploaded ImageNet ResNetV2-101 followed by source_train fine-tuning"
                if args.pretrained_backbone
                else "ImageNet ResNetV2-101 followed by source_train fine-tuning"
            )
        ),
        "data_root": str(data_root),
        "source_partition": "source_train",
        "target_train_partition": "target_train",
        "target_samples_per_class": args.target_samples_per_class,
        "selection_partition": "target_val",
        "final_partition": "target_test",
        "modalities": ["RGB"],
        "num_classes": 3,
        "backbone": "ImageNet-pretrained ResNetV2-101",
        "embedding": [4096, 512],
        "feature_normalization_loss": args.fn,
        "output_l2_normalization": args.l2n,
        "train_source_direction_during_adaptation": args.train_src,
        "pair_protocol": "all positives + equal-count sampled negatives",
        "batch_size": args.batch_size,
        "augmentation": "resize256/random-resized-crop224/H+V flip/color-jitter",
        "normalization": "0.5 mean / 0.5 std, as released d-SNE defaults",
        "raw_dn_scaling": "fixed /255 with sparse high-DN clipping before d-SNE normalization",
    }

    history = []
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
        for budget in report_budgets:
            if epoch <= budget and better_metrics(
                val_metrics, best_by_budget[budget]
            ):
                best_by_budget[budget] = dict(val_metrics)
                save_checkpoint(
                    checkpoints[budget], model, epoch, val_metrics, metadata
                )

    last_checkpoint = save_dir / f"vds2raw_dsne_seed{args.seed}_last.pth"
    save_checkpoint(
        last_checkpoint, model, history[-1][0], history[-1][1], metadata
    )
    for budget in report_budgets:
        evaluate_budget(
            model,
            checkpoints[budget],
            budget,
            history,
            metadata,
            data_root,
            statistics,
            save_dir,
            device,
            args,
        )


if __name__ == "__main__":
    main()
