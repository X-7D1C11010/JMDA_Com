"""Apply narrowly scoped changes to a disposable, pinned trainer snapshot."""

import argparse
from pathlib import Path
from shutil import copyfile


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one pinned source anchor, found {count}: {old!r}")
    return source.replace(old, new, 1)


def import_extension(source, name):
    anchor = "from uda_multimodal import "
    return replace_once(source, anchor, f"from historical_uda_extensions import {name}\n{anchor}")


def cap_dann_grl(source):
    anchor = "alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0"
    return replace_once(
        source, anchor,
        "alpha = min(2.0 / (1.0 + np.exp(-10 * p)) - 1.0, 0.45)",
    )


def mcc_source_only(source):
    # The historical target loader consumes RNG when creating its iterator
    # before the model is initialized. Iterate dummy indices on the same
    # schedule to preserve that RNG behavior without opening target images.
    source = replace_once(
        source, "tgt_loader = DataLoader(tgt_ds,",
        "tgt_loader = DataLoader(range(len(tgt_ds)),",
    )
    source = replace_once(
        source,
        "        x_t, = next(train_target_iter)[:1]\n"
        "        x_s, x_t = x_s.to(device), x_t.to(device)\n"
        "        labels_s = labels_s.to(device)",
        "        next(train_target_iter)  # Advance the dummy index sampler.\n"
        "        x_s, labels_s = x_s.to(device), labels_s.to(device)",
    )
    start = source.index("        source_warmup = epoch <= args.source_warmup_epochs")
    stop = source.index("        loss.backward()", start)
    return replace_once(
        source, source[start:stop],
        "        y_s, _ = classifier(x_s)\n"
        "        y_t = None\n"
        "        cls_loss = F.cross_entropy(y_s, labels_s, label_smoothing=args.label_smoothing)\n"
        "        transfer_loss = cls_loss.new_zeros(())\n"
        "        transfer_weight = 0.0\n"
        "        loss = cls_loss\n",
    )


def continuous_coral_sampling(source):
    source = import_extension(source, "ContinuousShuffleSampler")
    anchor = "    planned_updates = args.max_iterations or args.epochs * n_batches\n"
    source = replace_once(
        source, anchor, anchor +
        "    # Keep the historical reporting cadence and CORAL schedule.\n"
        "    src_loader = DataLoader(src_ds, batch_size=args.source_batch_size,\n"
        "                            sampler=ContinuousShuffleSampler(len(src_ds), args.seed),\n"
        "                            **loader_kwargs)\n"
        "    tgt_loader = DataLoader(tgt_ds, batch_size=args.target_batch_size,\n"
        "                            sampler=ContinuousShuffleSampler(len(tgt_ds), args.seed + 1),\n"
        "                            **loader_kwargs)\n"
        "    src_iter, tgt_iter = iter(src_loader), iter(tgt_loader)\n"
        "    print('CORAL sampling: continuous independent source and target passes')\n",
    )
    return replace_once(
        source, "        src_iter = iter(src_loader)\n        tgt_iter = iter(tgt_loader)\n", "",
    )


def full_frame_augmentation(source, multimodal):
    source = import_extension(source, "full_frame_training_transform")
    anchor = "    src_ds = dataset_factory("
    if anchor not in source:
        anchor = "        src_ds = paired_dataset("
    indent = anchor[:len(anchor) - len(anchor.lstrip())]
    source = replace_once(
        source, anchor, indent + "train_tf = full_frame_training_transform(train_tf)\n" + anchor,
    )
    multimodal = replace_once(
        multimodal, "from uda_data import read_image_list",
        "from historical_uda_extensions import VisibleJitter\nfrom uda_data import read_image_list",
    )
    multimodal = replace_once(
        multimodal,
        "            operations.append(copy.deepcopy(operation))",
        "            operations.append(transforms.ColorJitter(contrast=0.10)\n"
        "                              if isinstance(operation, VisibleJitter)\n"
        "                              else copy.deepcopy(operation))",
    )
    return source, multimodal


def infrared_pretraining(source, method):
    source = import_extension(source, "pretrain_ir_encoder")
    if method == "dann":
        anchor = "        parameter_groups = ["
        line = "        pretrain_ir_encoder(model.ir_backbone, src_ds, num_classes, args, device)\n"
    else:
        anchor = "    classifier = ImageClassifier("
        line = "    pretrain_ir_encoder(backbone.ir_encoder.to(device), src_ds, num_classes, args, device)\n"
    return replace_once(source, anchor, line + anchor)


def ema_consistency(source, method):
    source = import_extension(source, "EMATargetTeacher")
    if method in {"mdd", "mcc"}:
        init = "    best_acc = -float('inf')"
        student = "classifier"
        predict = "lambda model, x: model(x)"
        source = replace_once(source, "epoch, args, eff=None):", "epoch, args, eff=None, teacher=None):")
        source = replace_once(source, "epoch, args, eff=eff)", "epoch, args, eff=eff, teacher=teacher)")
        step = "        optimizer.step()"
        loss = "        loss = cls_loss + transfer_loss * transfer_weight"
        consistency = "teacher.loss(y_t, x_t)"
        if method == "mcc":
            consistency += " if y_t is not None else loss.new_zeros(())"
    elif method == "cdan":
        init = "    criterion = nn.CrossEntropyLoss()"
        student, predict = "model", "lambda model, x: model(x)[1]"
        step = "        optimizer.step()"
        loss = "        loss = loss_cls + args.trade_off * loss_transfer"
        consistency = "teacher.loss(outputs_target, tgt_imgs)"
    elif method == "deepcoral":
        init = "    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)"
        student, predict = "model", "lambda model, x: model.predict_components(x)['fused']"
        step = "            optimizer.step()"
        loss = "            loss = loss_cls + _lambda * loss_coral"
        consistency = "teacher.loss(tgt_outputs['fused'], tgt_imgs)"
    else:
        init = "    criterion = torch.nn.NLLLoss()"
        student, predict = "model", "lambda model, x: model(x, 0.0)[0]"
        step = "            optimizer.step()"
        loss = "            loss = loss_cls + args.domain_loss_weight * loss_domain"
        consistency = "teacher.loss(tgt_class_out, tgt_imgs)"
        source = replace_once(
            source, "_, tgt_domain_out = model(tgt_imgs, alpha)",
            "tgt_class_out, tgt_domain_out = model(tgt_imgs, alpha)",
        )
    indent = step[:len(step) - len(step.lstrip())]
    source = replace_once(source, init, f"    teacher = EMATargetTeacher({student}, {predict})\n" + init)
    source = replace_once(source, loss, f"{loss}\n{indent}loss = loss + ({consistency})")
    source = replace_once(source, step, f"{step}\n{indent}teacher.update({student})")
    anchor = "    if best_metrics is None or last_metrics is None:"
    return replace_once(source, anchor, "    teacher.report()\n" + anchor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer", type=Path)
    parser.add_argument("method", choices=("dann", "deepcoral", "cdan", "mdd", "mcc"))
    parser.add_argument("variant", choices=(
        "tuned", "mcc_source_only", "sampling", "augmentation", "ir_pretrain", "ema_teacher",
    ))
    args = parser.parse_args()
    trainer = args.trainer
    tools_dir = trainer.parent.parent / "tools"
    source = trainer.read_text(encoding="utf-8")
    multimodal_path = tools_dir / "uda_multimodal.py"
    updates = {}
    if args.variant == "tuned" and args.method == "dann":
        source = cap_dann_grl(source)
    elif args.variant == "mcc_source_only" and args.method == "mcc":
        source = mcc_source_only(source)
    elif args.variant == "sampling" and args.method == "deepcoral":
        source = continuous_coral_sampling(source)
    elif args.variant == "augmentation":
        source, multimodal = full_frame_augmentation(
            source, multimodal_path.read_text(encoding="utf-8"),
        )
        updates[multimodal_path] = multimodal
    elif args.variant == "ir_pretrain" and args.method in {"dann", "mdd", "mcc"}:
        source = infrared_pretraining(source, args.method)
    elif args.variant == "ema_teacher":
        source = ema_consistency(source, args.method)
    else:
        parser.error(f"{args.variant} has no source overlay for {args.method}")

    updates[trainer] = source
    for path, text in updates.items():
        compile(text, str(path), "exec")
    if args.variant in {"sampling", "augmentation", "ir_pretrain", "ema_teacher"}:
        copyfile(Path(__file__).with_name("historical_uda_extensions.py"),
                 tools_dir / "historical_uda_extensions.py")
    for path, text in updates.items():
        path.write_text(text, encoding="utf-8")
    print(f"[historical variant] {args.method} {args.variant}: {trainer}")


if __name__ == "__main__":
    main()
