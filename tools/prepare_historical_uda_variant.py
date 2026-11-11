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


def cap_cdan_grl(source, network):
    # With the pinned 10,000-step ramp this cap takes effect before step 700.
    cap = "0.30"
    network = replace_once(
        network,
        "coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)",
        f"coeff = min(calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter), {cap})",
    )
    source = replace_once(
        source,
        "calc_coeff(iter_num, alpha=args.coeff_alpha,\n                                            max_iter=coeff_max_iter), random_layer)",
        "min(calc_coeff(iter_num, alpha=args.coeff_alpha,\n"
        f"                                                max_iter=coeff_max_iter), {cap}), random_layer)",
    )
    return replace_once(
        source,
        "coeff={calc_coeff(iter_num, alpha=args.coeff_alpha, max_iter=coeff_max_iter):.4f}",
        f"coeff={{min(calc_coeff(iter_num, alpha=args.coeff_alpha, max_iter=coeff_max_iter), {cap}):.4f}}",
    ), network


def cdan_source_only(source):
    # Keep the target sampler's RNG draws and batch cadence, with dummy indices.
    source = replace_once(source, "tgt_loader = DataLoader(tgt_ds,",
                          "tgt_loader = DataLoader(range(len(tgt_ds)),")
    start = source.index("        tgt_imgs, _ = next(iter_target)")
    stop = source.index("        optimizer.zero_grad()", start)
    source = replace_once(
        source, source[start:stop],
        "        next(iter_target)\n"
        "        src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)\n"
        "        if eff:\n"
        "            eff.dataloader_done()\n"
        "        _, outputs_source = model(src_imgs)\n"
        "        loss_cls = criterion(outputs_source, src_labels)\n"
        "        loss_transfer = loss_cls.new_zeros(())\n"
        "        loss = loss_cls\n",
    )
    source = replace_once(source, "    ad_net.alpha = float(args.coeff_alpha)\n",
                          "    ad_net.alpha = float(args.coeff_alpha)\n"
                          "    ad_net.requires_grad_(False)\n"
                          "    print('CDAN source-only control: target image forward and transfer loss disabled')\n")
    return replace_once(
        source, "coeff={calc_coeff(iter_num, alpha=args.coeff_alpha, max_iter=coeff_max_iter):.4f}",
        "coeff=0.0000",
    )


def add_dann_modality_diagnostics(source):
    source = replace_once(
        source, "    all_labels, all_predictions = [], []\n",
        "    all_labels, all_predictions = [], []\n"
        "    vis_zeroed_correct = ir_zeroed_correct = 0\n",
    )
    source = replace_once(
        source, "            all_predictions.extend(pred.cpu().tolist())\n",
        "            all_predictions.extend(pred.cpu().tolist())\n"
        "            vis_zeroed = imgs.clone()\n"
        "            vis_zeroed[:, :3].zero_()\n"
        "            ir_zeroed = imgs.clone()\n"
        "            ir_zeroed[:, 3:].zero_()\n"
        "            vis_zeroed_correct += int((classify_dann(model, vis_zeroed).argmax(1) == labels).sum())\n"
        "            ir_zeroed_correct += int((classify_dann(model, ir_zeroed).argmax(1) == labels).sum())\n",
    )
    return replace_once(
        source, "    return classification_metrics(all_labels, all_predictions, num_classes=num_classes)\n",
        "    metrics = classification_metrics(all_labels, all_predictions, num_classes=num_classes)\n"
        "    metrics['vis_zeroed_acc'] = vis_zeroed_correct / len(all_labels)\n"
        "    metrics['ir_zeroed_acc'] = ir_zeroed_correct / len(all_labels)\n"
        "    return metrics\n",
    )


def target_bn_probe(source, method):
    source = import_extension(source, "TargetBNProbe, center_view_transform")
    model = "classifier" if method in {"mdd", "mcc"} else "model"
    forward = {
        "dann": "classify_dann",
        "deepcoral": "lambda m, x: m.predict_components(x)['fused']",
        "cdan": "lambda m, x: m(x)[1]",
        "mdd": "lambda m, x: m.head(m.bottleneck(m.backbone(x)))",
        "mcc": "lambda m, x: m(x)",
    }[method]
    transform = "center_view_transform(train_tf)" if method == "cdan" else "val_tf"
    batch_size = "args.target_batch_size" if method == "deepcoral" else "args.batch_size"
    anchor = "        print(f'Efficiency logging -> {eff.run_dir}')\n"
    source = replace_once(
        source, anchor, anchor +
        f"    bn_probe = TargetBNProbe({model}, tgt_ds, {transform},\n"
        f"                             {forward}, device, {batch_size})\n"
        "    print('Prediction protocol: target_train BN calibration; paired raw evaluation is logged')\n",
    )
    if method == "cdan":
        start = source.index("            val_metrics = evaluate_10crop(model, tgt_val_list")
        end = source.index("            acc = val_metrics['accuracy']", start)
        anchor = source[start:end]
        call = anchor.strip().split(" = ", 1)[1]
        position = "iter_num + 1"
    else:
        indent = "        " if method in {"mdd", "mcc"} else "            "
        call = ("validate(tgt_val_loader, classifier, num_classes)" if method in {"mdd", "mcc"}
                else "evaluate(model, tgt_val_loader, device, num_classes)")
        anchor = indent + "val_metrics = " + call + "\n"
        position = "epoch" if method in {"mdd", "mcc"} else "epoch + 1"
    indent = anchor[:len(anchor) - len(anchor.lstrip())]
    source = replace_once(
        source, anchor, anchor + indent + "val_metrics = bn_probe.evaluate(\n"
        + indent + f"    val_metrics, lambda: {call}, {position})\n",
    )
    return replace_once(
        source, "        eff.finalize(",
        "        eff.record_calibration(bn_probe.summary())\n        eff.finalize(",
    )


def instrument_run(source, method, variant):
    symbols = "classification_metrics, report_validation"
    if method == "dann":
        symbols += ", classify_dann"
    elif method == "deepcoral":
        symbols += ", report_coral_gradients"
    elif method == "cdan" and variant != "cdan_source_only":
        symbols += ", target_prediction_stats"
    if method == "mcc" and variant in {"bn_probe", "mcc_stronger", "mcc_stronger_bn"}:
        symbols += ", report_mcc_gradients"
    if variant in {"ir_pretrain", "dann_ir_pretrain"}:
        symbols += ", start_stage_timer, finish_stage_timer"
    source = replace_once(
        source, "from classification_metrics import classification_metrics, format_classification_metrics",
        "from classification_metrics import format_classification_metrics\n"
        f"from historical_uda_metrics import {symbols}",
    )
    source = replace_once(
        source, "from efficiency_common import EfficiencyLogger",
        "from historical_uda_metrics import HistoricalEfficiencyLogger as EfficiencyLogger",
    )
    if method == "cdan":
        source = replace_once(source, "        if iter_num == 0 and eff:\n",
                              "        if iter_num % args.test_interval == 0 and eff:\n"
                              "            block += 1\n")
        source = replace_once(source, "            if eff and iter_num == args.num_iterations - 1:\n"
                                      "                eff.epoch_end(block)\n", "")
        anchor = "        if should_evaluate:\n"
        additions = "            if eff:\n                eff.epoch_end(block)\n"
        if variant != "cdan_source_only":
            additions += (
                "            print(f'CDAN target batch predictions at update {iter_num + 1}: '\n"
                "                  f'{target_prediction_stats(outputs_target)}')\n"
            )
        source = replace_once(source, anchor, anchor + additions)
    else:
        # Move the existing timing boundary ahead of source/target evaluation.
        indent = "        "
        epoch_end = f"{indent}if eff:\n{indent}    eff.epoch_end(epoch)\n"
        source = replace_once(source, epoch_end, "")
        starts = {
            "dann": "        is_final = ",
            "deepcoral": "        source_metrics = evaluate(",
            "mdd": "        val_metrics = validate(",
            "mcc": "        val_metrics = validate(",
        }
        anchor = starts[method]
        source = replace_once(source, anchor, epoch_end + anchor)

    metrics = "fused_metrics" if method == "deepcoral" else "val_metrics"
    line = (f"print(format_classification_metrics({metrics}), flush=True)"
            if method == "mcc" else f"print(format_classification_metrics({metrics}))")
    indent = "        " if method in {"mdd", "mcc"} else "            "
    source = replace_once(source, indent + line + "\n",
                          indent + line + "\n" + indent + f"report_validation({metrics}, class_ids)\n")
    forward_calls = {
        "dann": "model, classify_dann, method='dann'",
        "deepcoral": "model, lambda m, x: m.predict_components(x)['fused'], method='deepcoral'",
        "cdan": ("model, lambda m, x: m(x)[1], method='cdan', training_module=ad_net, "
                 "training_unit_updates=args.test_interval"),
        "mdd": "classifier, lambda m, x: m.head(m.bottleneck(m.backbone(x))), method='mdd'",
        "mcc": "classifier, lambda m, x: m(x), method='mcc'",
    }
    anchor = "        eff.finalize("
    source = replace_once(source, anchor, f"        eff.benchmark({forward_calls[method]})\n" + anchor)
    if method == "dann" and variant in {"dann_capped", "dann_ir_pretrain", "bn_probe"}:
        source = add_dann_modality_diagnostics(source)
    if method == "deepcoral" and variant in {"coral_lr5", "sampling", "bn_probe"}:
        anchor = "            loss.backward()"
        source = replace_once(
            source, anchor,
            "            if i + 1 == batches_this_epoch:\n"
            "                report_coral_gradients(global_step + 1, model.fc, loss_cls, _lambda * loss_coral)\n"
            + anchor,
        )
    if method == "mdd":
        source = replace_once(
            source, "        if (i + 1) % args.print_freq == 0:",
            "        if (i + 1) % args.print_freq == 0 or i + 1 == args.iters_per_epoch:",
        )
    if method == "mcc" and variant in {"bn_probe", "mcc_stronger", "mcc_stronger_bn"}:
        anchor = "        loss.backward()"
        source = replace_once(
            source, anchor,
            "        if transfer_weight > 0 and i + 1 == args.iters_per_epoch:\n"
            "            report_mcc_gradients(epoch, classifier.head, cls_loss, transfer_weight * transfer_loss)\n"
            + anchor,
        )
    return source


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
        "    src_iter = iter(src_loader)\n"
        "    print('CORAL sampling: continuous source passes, target reshuffles each epoch')\n",
    )
    return replace_once(
        source, "        src_iter = iter(src_loader)\n", "",
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
    indent = line[:len(line) - len(line.lstrip())]
    source = replace_once(source, anchor,
                          indent + "ir_started = start_stage_timer(device)\n"
                          + line + indent + "ir_timing = finish_stage_timer(device, ir_started)\n"
                          + anchor)
    anchor = "        print(f'Efficiency logging -> {eff.run_dir}')"
    return replace_once(source, anchor, "        eff.record_pretraining(ir_timing)\n" + anchor)


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
        "historical_best", "tuned", "dann_capped", "dann_ir_pretrain", "coral_lr5",
        "sampling", "cdan_source_only", "cdan_grl_cap", "mdd_ramp8", "mdd_ramp4",
        "mcc_source_only", "mcc_joint_zero", "augmentation", "ir_pretrain", "ema_teacher",
        "bn_probe", "mcc_stronger", "mcc_stronger_bn",
    ))
    args = parser.parse_args()
    trainer = args.trainer
    tools_dir = trainer.parent.parent / "tools"
    source = trainer.read_text(encoding="utf-8")
    multimodal_path = tools_dir / "uda_multimodal.py"
    updates = {}
    if args.variant in {"tuned", "dann_capped", "dann_ir_pretrain", "bn_probe"} and args.method == "dann":
        source = cap_dann_grl(source)
        if args.variant == "dann_ir_pretrain":
            source = infrared_pretraining(source, args.method)
    elif args.variant == "cdan_source_only" and args.method == "cdan":
        source = cdan_source_only(source)
    elif args.variant == "cdan_grl_cap" and args.method == "cdan":
        network = trainer.parent / "pytorch" / "network.py"
        source, updates[network] = cap_cdan_grl(source, network.read_text(encoding="utf-8"))
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

    if args.variant in {"bn_probe", "mcc_stronger_bn"}:
        source = target_bn_probe(source, args.method)
    source = instrument_run(source, args.method, args.variant)
    updates[trainer] = source
    for path, text in updates.items():
        compile(text, str(path), "exec")
    shared_files = ["historical_uda_metrics.py", "efficiency_common.py"]
    if args.variant in {"sampling", "augmentation", "ir_pretrain", "dann_ir_pretrain",
                        "ema_teacher", "bn_probe", "mcc_stronger_bn"}:
        shared_files.append("historical_uda_extensions.py")
    for filename in shared_files:
        copyfile(Path(__file__).with_name(filename), tools_dir / filename)
    for path, text in updates.items():
        path.write_text(text, encoding="utf-8")
    print(f"[historical variant] {args.method} {args.variant}: {trainer}")


if __name__ == "__main__":
    main()
