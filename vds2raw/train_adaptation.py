"""DANN / Deep CORAL / CDAN+E / MDD / MCC on unsupervised VDS2Raw.

Strict split: labelled source_train + unlabelled target_train, select on target_val,
restore the selected state in memory, then evaluate target_test exactly once.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from tools.classification_metrics import classification_metrics
from tools.efficiency_common import measure_inference_latency
from tools.reproducibility import seed_everything
from vds2raw.adaptation import (
    METHODS, DEFAULTS, AdaptationDataset, AdaptationModel, PairedEpochSampler,
    prepare_statistics, optimizer_groups, resolve_training_settings, source_classification_weights,
)
from vds2raw.common import better_metrics, build_run_provenance, save_prediction_artifacts
from vds2raw.dataset import DISPLAY_CLASS_NAMES, audit_dataset


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def event(directory, filename, payload):
    with (Path(directory)/filename).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, allow_nan=False)+"\n")


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def feature_norms(branches, source_count, names=("RGB", "NIR", "AIS")):
    result = {}
    for domain, section in (("source", slice(None, source_count)), ("target", slice(source_count, None))):
        result[domain] = {
            name: float(values[section].norm(dim=1).mean())
            for name, values in zip(names, branches)
        }
    return result


def gradient_diagnostics(model, classification, transfer):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    ce = torch.autograd.grad(classification, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    da = torch.autograd.grad(transfer, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    result = {}
    for group, prefix in (("RGB", "rgb_encoder."), ("NIR", "nir_encoder."),
                          ("AIS", "ais_encoder."), ("head", "head."),
                          ("gate", "modality_logits"), ("bottleneck", "bottleneck.")):
        norm_ce = classification.new_zeros(())
        norm_da = classification.new_zeros(())
        dot = classification.new_zeros(())
        for (name, _), gc, gd in zip(parameters, ce, da):
            if name.startswith(prefix):
                if gc is not None:
                    norm_ce += gc.detach().square().sum()
                if gd is not None:
                    norm_da += gd.detach().square().sum()
                if gc is not None and gd is not None:
                    dot += (gc.detach()*gd.detach()).sum()
        nc, nd, dp = float(norm_ce.sqrt()), float(norm_da.sqrt()), float(dot)
        result[group] = {"CE_norm": nc, "weighted_DA_norm": nd,
                         "DA_to_CE_ratio": nd/max(nc, 1e-12),
                         "cosine": dp/max(nc*nd, 1e-12)}
    return result


def evaluate(model, dataset, device, epoch, directory, batch_size=64, event_prefix="VDS_EVALUATION", return_details=False):
    """Deterministic evaluation; every sample receives a probability record."""
    model.eval()
    truth, predicted, records = [], [], []
    num_classes = getattr(model, "num_classes", 3)
    names = getattr(model, "feature_names", ("RGB", "NIR", "AIS"))
    feature_sums = [0.0]*len(names)
    normalized_sums = [0.0]*len(names)
    losses = []
    labelled = dataset.labelled
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0)):
            labels = batch["label"] if labelled else None
            logits = model(move(batch, device))
            probabilities = logits.softmax(1).cpu()
            guesses = probabilities.argmax(1)
            if labelled:
                truth.extend(labels.tolist())
                losses.extend(nn.functional.cross_entropy(logits, labels.to(device), reduction="none").cpu().tolist())
            predicted.extend(guesses.tolist())
            for index, values in enumerate(model.last_features):
                feature_sums[index] += float(values.norm(dim=1).sum())
            for index, values in enumerate(getattr(model, "last_normalized_features", model.last_features)):
                normalized_sums[index] += float(values.norm(dim=1).sum())
            for index, (path, guess, scores, raw_logits) in enumerate(zip(
                    batch["path"], guesses.tolist(), probabilities.tolist(), logits.cpu().tolist())):
                row = {"epoch": epoch, "partition": dataset.partition, "sample_index": len(records),
                       "sample_id": path, "y_pred": guess, "labels_used": labelled,
                       "probabilities": scores, "logits": raw_logits}
                if labelled:
                    label = int(labels[index])
                    row.update(y_true=label, correct=label == guess)
                event(directory, "predictions.jsonl", row)
                records.append(row)
    metrics = classification_metrics(truth, predicted, num_classes=num_classes) if labelled else {"prediction_samples": len(records)}
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for expected, guessed in zip(truth, predicted):
        matrix[expected][guessed] += 1
    if labelled:
        metrics["confusion_matrix"] = matrix
    details = {"epoch": epoch, "partition": dataset.partition, "metrics": metrics,
               "labels_used": labelled,
               "mean_max_probability": statistics.mean(max(row["probabilities"]) for row in records),
               "mean_entropy": statistics.mean(-sum(p*math.log(max(p, 1e-12)) for p in row["probabilities"]) for row in records),
               "predicted_class_counts": [predicted.count(i) for i in range(num_classes)],
               "feature_norms": dict(zip(names, (x/len(dataset) for x in feature_sums))),
               "normalized_feature_norms": dict(zip(names, (x/len(dataset) for x in normalized_sums))),
               "modality_weights": model.modality_weights().detach().cpu().tolist()}
    if labelled:
        details.update(cross_entropy=statistics.mean(losses),
                       cross_entropy_by_class=[statistics.mean([loss for loss, label in zip(losses, truth) if label == c])
                                               if c in truth else None for c in range(num_classes)],
                       mean_confidence_wrong=(statistics.mean(max(row["probabilities"]) for row in records if not row["correct"])
                                              if any(not row["correct"] for row in records) else None),
                       recall_by_class=[row[i]/max(sum(row), 1) for i, row in enumerate(matrix)])
    event(directory, "evaluation.jsonl", details)
    print(event_prefix+": "+json.dumps(details), flush=True)
    return (metrics, records, details) if return_details else (metrics, records)


def profile(model, dataset, device, train_times, peak_reserved):
    model.eval()
    example = move(next(iter(DataLoader(dataset, batch_size=1, num_workers=0))), device)
    macs = 0

    def hook(layer, _input, output):
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            macs += output.numel()*(layer.in_channels//layer.groups)*math.prod(layer.kernel_size)
        else:
            macs += output.numel()*layer.in_features

    handles = [layer.register_forward_hook(hook) for layer in model.modules()
               if isinstance(layer, (nn.Conv2d, nn.Linear))]
    try:
        with torch.inference_mode():
            model(example)
    finally:
        for handle in handles:
            handle.remove()
    result = measure_inference_latency(model, example, device, warmup=10, iters=50)
    result.update(total_params=sum(p.numel() for p in model.parameters()),
                  trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  inference_GFLOPs=2*macs/1e9, total_train_time_s=sum(train_times),
                  train_epoch_mean_s=statistics.mean(train_times),
                  peak_train_reserved_gb=peak_reserved,
                  inference_input={key: list(example[key].shape) for key in ("rgb", "nir", "ais", "ais_mask") if key in example},
                  inference_batch_size=1,
                  flops_scope="executed Conv2d/Linear only; 1 MAC=2 FLOPs",
                  training_time_scope="optimizer windows including gradient diagnostics, excluding evaluation and GPU queue",
                  total_parameter_scope="all training modules, including any discriminator/adversarial head")
    return result


def validate_diagnostics(directory, datasets, epochs, selected_epoch, num_classes=3, modalities=("RGB", "NIR", "AIS")):
    """Fail on missing persisted diagnostics, duplicate IDs or test reuse."""
    expected_epochs = set(range(1, epochs+1))
    for filename in ("training.jsonl", "gradients.jsonl"):
        records = [json.loads(row) for row in (directory/filename).read_text(encoding="utf-8").splitlines()]
        if len(records) != epochs or {row["epoch"] for row in records} != expected_epochs:
            raise RuntimeError(f"Incomplete diagnostic windows: {filename}")
        if filename == "gradients.jsonl" and any(
                not (set(modalities) | {"head", "gate"}) <= set(row["gradients"]) for row in records):
            raise RuntimeError("Missing modality/head gradient diagnostics")
    groups = {}
    with (directory/"predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["partition"] == "target_train" and (
                    row.get("labels_used") is not False or {"y_true", "correct", "label"} & row.keys()):
                raise RuntimeError("Target training labels leaked into diagnostics")
            probabilities = row["probabilities"]
            if (len(probabilities) != num_classes or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
                    or abs(sum(probabilities)-1) > 1e-5):
                raise RuntimeError("Invalid prediction probabilities")
            key = (row["partition"], row["epoch"])
            ids = groups.setdefault(key, set())
            if row["sample_id"] in ids:
                raise RuntimeError(f"Duplicate evaluation sample: {key}")
            ids.add(row["sample_id"])
    checked = {}
    for dataset in datasets:
        partition = dataset.partition
        positions = {e for p, e in groups if p == partition}
        expected = {selected_epoch} if partition == "target_test" else expected_epochs
        if positions != expected:
            raise RuntimeError(f"Missing or extra prediction windows: {partition}")
        actual_ids = set(map(str, dataset.paths))
        if any(groups[(partition, e)] != actual_ids for e in expected):
            raise RuntimeError(f"Incomplete prediction sample coverage: {partition}")
        checked[partition] = len(actual_ids)*len(expected)
    return checked


def make_optimizer(model, args):
    groups = optimizer_groups(model, args.lr, getattr(args, "ais_lr_multiplier", None) or 1.0)
    optimizer = (torch.optim.Adam(groups, weight_decay=args.weight_decay) if args.method == "dann"
                 else torch.optim.SGD(groups, momentum=0.9, nesterov=args.method != "deepcoral",
                                      weight_decay=args.weight_decay*(2 if args.method == "cdan" else 1)))
    # The released MDD example uses gamma=.0002; CDAN/MCC use .001.
    # These clocks must not be collapsed into a single shared constant.
    if args.lr_gamma is None:
        args.lr_gamma = 0.0002 if args.method == "mdd" else 0.001
    schedule_per_update = args.lr_scheduler == "reference" and args.method in {"cdan", "mdd", "mcc"}
    scheduler = (torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1+args.lr_gamma*step)**(-args.lr_decay))
        if schedule_per_update else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        if args.lr_scheduler == "cosine" else None)
    return optimizer, scheduler, schedule_per_update


class ValidationLossControl:
    """Validation-only LR plateau control, independent of ACC checkpoint selection."""

    def __init__(self, args):
        self.patience = args.val_plateau_patience
        self.factor, self.min_scale = args.val_plateau_factor, args.val_plateau_min_scale
        self.min_delta = args.validation_loss_min_delta
        self.start_epoch = getattr(args, "validation_control_start_epoch", 1)
        self.best_loss, self.best_loss_epoch = math.inf, 0
        self.bad_epochs = self.plateau_bad_epochs = 0
        self.lr_scale = 1.0

    def observe(self, loss, epoch, optimizer):
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite target_val cross entropy")
        if epoch < self.start_epoch:
            return {"validation_cross_entropy": loss, "monitor_active": False,
                    "CE_improved": False, "best_validation_CE": None, "best_validation_CE_epoch": None,
                    "CE_bad_epochs": 0, "plateau_bad_epochs": 0, "lr_scale": self.lr_scale, "lr_reduced": False}
        improved = loss < self.best_loss-self.min_delta
        if improved:
            self.best_loss, self.best_loss_epoch = loss, epoch
            self.bad_epochs = self.plateau_bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.plateau_bad_epochs += 1
        reduced = False
        if self.patience and self.plateau_bad_epochs >= self.patience:
            old_scale = self.lr_scale
            self.lr_scale = max(self.min_scale, old_scale*self.factor)
            for group in optimizer.param_groups:
                group["lr"] *= self.lr_scale/old_scale
            reduced = self.lr_scale < old_scale
            self.plateau_bad_epochs = 0
        return {"validation_cross_entropy": loss, "monitor_active": True, "CE_improved": improved,
                "best_validation_CE": self.best_loss, "best_validation_CE_epoch": self.best_loss_epoch,
                "CE_bad_epochs": self.bad_epochs, "plateau_bad_epochs": self.plateau_bad_epochs,
                "lr_scale": self.lr_scale, "lr_reduced": reduced}


def step_scaled_scheduler(scheduler, optimizer, scale):
    """Compose plateau scaling with the reference schedule, including recursive schedulers.

    The scheduler sees unscaled rates; its base rates and update clock stay
    intact. Applying only a second scheduler would let LambdaLR overwrite cuts.
    """
    if scheduler is None:
        return
    if scale != 1.0:
        for group in optimizer.param_groups:
            group["lr"] /= scale
    scheduler.step()
    if scale != 1.0:
        for group in optimizer.param_groups:
            group["lr"] *= scale


def validation_modality_diagnostics(model, dataset, device, selected_epoch, directory, batch_size=64):
    """Validation-only feature sensitivity and label-metadata independence checks.

    Zeroed features are an ablation of this trained fusion network, not an
    independently trained single-modality model. No test data are evaluated.
    """
    model.eval()
    names = getattr(model, "feature_names", ("RGB", "NIR", "AIS"))
    previous = getattr(model, "eval_disabled_modalities", ())
    result = {"partition": dataset.partition, "selected_epoch": selected_epoch,
              "kind": "zeroed_feature_sensitivity", "independent_single_modality_models": False}
    if dataset.partition != "target_val":
        raise ValueError("Modality diagnostics may only use target_val")
    (directory/"modality_diagnostics/full").mkdir(parents=True, exist_ok=True)
    try:
        model.eval_disabled_modalities = ()
        sample = move(next(iter(DataLoader(dataset, batch_size=min(8, len(dataset)), num_workers=0,
                                          generator=torch.Generator().manual_seed(0)))), device)
        with torch.no_grad():
            original = model(sample)
            altered = dict(sample)
            altered["label"] = (sample["label"]+1) % getattr(model, "num_classes", 3)
            altered["path"] = ["metadata_changed"]*len(sample["path"])
            independent = torch.equal(original, model(altered))
        if not independent:
            raise RuntimeError("Inference depends on label/path metadata")
        result["inference_independent_of_label_and_path"] = independent
        full, _ = evaluate(model, dataset, device, selected_epoch, directory/"modality_diagnostics/full",
                           batch_size, event_prefix="MODALITY_DIAGNOSTIC")
        result["full"] = full
        result["zeroed"] = {}
        for index, name in enumerate(names):
            model.eval_disabled_modalities = (index,)
            folder = directory/"modality_diagnostics"/f"zero_{name}"
            folder.mkdir(parents=True, exist_ok=True)
            metrics, _ = evaluate(model, dataset, device, selected_epoch, folder, batch_size,
                                  event_prefix="MODALITY_DIAGNOSTIC")
            result["zeroed"][name] = {"metrics": metrics,
                                      "ACC_drop": full["accuracy"]-metrics["accuracy"]}
    finally:
        model.eval_disabled_modalities = previous
    write_json(directory/"modality_diagnostics.json", result)
    return result


def run(args):
    args.seed = seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    root, output = Path(args.data_root).expanduser().resolve(), Path(args.save_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_file) if args.stats_file else output/"normalization_stats.json"
    if not args.skip_audit:
        report = audit_dataset(root)
        # The old MCFT/TIR audit annotates a four-field model; this runner uses
        # the full schema requested for this separate experiment.
        report.update(model_ais_feature_names=json.loads((root/"feature_schema.json").read_text(encoding="utf-8"))["ais_feature_names"],
                      excluded_ais_proxy_features=[], ais_mask_usage="13 validity values plus 13 gated features")
        write_json(output/"dataset_audit.json", report)
        print("VDS_DATA_AUDIT: "+json.dumps(report), flush=True)
    images, ais = prepare_statistics(root, stats_path)
    if args.audit_only:
        return
    if args.method is None:
        raise ValueError("--method is required unless --audit-only is used")
    run_dir = output/f"{args.method}_seed{args.seed}"
    run_dir.mkdir(exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty; use a new output directory: {run_dir}")
    resolve_training_settings(args)
    if args.adaptation_ramp_epochs is None:
        args.adaptation_ramp_epochs = args.epochs if args.method == "deepcoral" else 0
    source = AdaptationDataset(root, "source_train", images, ais, augment=args.augment)
    target = AdaptationDataset(root, "target_train", images, ais, augment=args.augment)
    source_eval = AdaptationDataset(root, "source_train", images, ais)
    target_eval = AdaptationDataset(root, "target_train", images, ais)
    validation = AdaptationDataset(root, "target_val", images, ais)
    if {y for _, y in source.samples} != {0, 1, 2}:
        raise ValueError("Source must contain the three declared dataset classes")
    args.steps_per_epoch = math.ceil(max(len(source), len(target))/args.batch_size)
    samplers = [PairedEpochSampler(ds, args.batch_size, args.steps_per_epoch, args.seed+offset)
                for ds, offset in ((source, 0), (target, 10000))]
    loader_args = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    if args.num_workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2, timeout=300)
    loaders = [DataLoader(ds, batch_sampler=sampler,
                          generator=torch.Generator().manual_seed(args.seed+20000+index), **loader_args)
               for index, (ds, sampler) in enumerate(zip((source, target), samplers))]
    device = torch.device(args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptationModel(args.method, args.weights, not args.no_pretrained).to(device)
    model.feature_normalization = args.feature_normalization
    source_counts, source_weights = source_classification_weights(source.samples, model.num_classes if hasattr(model, "num_classes") else 3)
    if args.source_class_balance:
        model.source_class_weights = source_weights.to(device)
    optimizer, scheduler, schedule_per_update = make_optimizer(model, args)
    validation_control = ValidationLossControl(args)
    config = {
        "method": args.method, "protocol": "UDA", "modalities": ["RGB", "NIR", "AIS"],
        "optimization_profile": args.profile,
        "planned_optimizer_updates": args.epochs*args.steps_per_epoch,
        "source_classification": {"class_counts": source_counts.tolist(),
                                  "weighted_CE": args.source_class_balance,
                                  "weights": source_weights.tolist() if args.source_class_balance else None,
                                  "counts_fit_partition": "source_train only"},
        "training_partitions": ["source_train", "target_train"],
        "selection_partition": "target_val", "test_partition": "target_test",
        "target_training_labels_used": False, "test_selection_bias": False,
        "classification_supervision": "source_train only; target_train label field is never read",
        "target_validation_labels_used_for_selection": True,
        "test_evaluations_for_selection": 0, "test_evaluation_policy": "once after restoring best target_val state",
        "class_names": DISPLAY_CLASS_NAMES, "statistics": {**images.to_dict(), "ais": ais},
        "architecture": "RGB ResNet18 + NIR ResNet18 (mean RGB conv1 initialization) + AIS/mask MLP; weighted concatenation",
        "adaptation_note": "source CE plus named core DA loss; multimodal feature interface; target_val selected and test held out",
        "augmentation": "paired padded crop 96, horizontal/vertical flip with AIS heading correction, RGB gain 0.95..1.05; eval deterministic",
        "model": repr(model), "gradient_sampling": "last update of each epoch, CE versus weighted adaptation",
        "optimizer": type(optimizer).__name__, "lr_schedule_per_update": schedule_per_update,
        "validation_control": {"plateau_monitor": "target_val cross entropy",
                               "plateau_patience": args.val_plateau_patience,
                               "plateau_factor": args.val_plateau_factor,
                               "plateau_min_scale": args.val_plateau_min_scale,
                               "control_start_epoch": args.validation_control_start_epoch,
                               "early_stopping_monitor": args.early_stopping_monitor,
                               "early_stopping_patience": args.early_stopping_patience,
                               "early_stopping_min_epochs": args.early_stopping_min_epochs,
                               "best_state_selection": "target_val ACC, then macro F1",
                               "test_metrics_used_for_control": False},
        "optimizer_groups": [{"lr": group["lr"], "weight_decay": group["weight_decay"],
                              "role": group.get("role"),
                              "parameters": sum(p.numel() for p in group["params"]),
                              "contains_modality_weights": any(p is model.modality_logits for p in group["params"])}
                             for group in optimizer.param_groups],
        "provenance": build_run_provenance(args, root, input_files=(() if args.no_pretrained else (args.weights,))),
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), ROOT/"vds2raw/adaptation.py", ROOT/"vds2raw/dataset.py", ROOT/"vds2raw/common.py")},
    }
    write_json(run_dir/"run_config.json", config)
    print("VDS_PROTOCOL: "+json.dumps({key: config[key] for key in ("protocol", "training_partitions", "selection_partition", "test_evaluation_policy", "architecture")}), flush=True)
    best_state = best_metrics = None
    best_epoch, stale, update = 0, 0, 0
    stop_reason = "max_epochs"
    history, times, peak_reserved = [], [], None
    for epoch in range(1, args.epochs+1):
        for sampler in samplers:
            sampler.epoch = epoch-1
        model.train()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        totals = {key: 0.0 for key in ("classification", "source_ce", "transfer", "weighted_transfer")}
        seen_source, seen_target = set(), set()
        source_correct = source_seen = target_seen = steps = 0
        for step, (source_batch, target_batch) in enumerate(zip(*loaders), 1):
            seen_source.update(source_batch["path"])
            seen_target.update(target_batch["path"])
            if "label" in target_batch:
                raise RuntimeError("UDA target-training batch must not contain labels")
            source_batch, target_batch = move(source_batch, device), move(target_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.objective(source_batch, target_batch, update, args.epochs*args.steps_per_epoch, args)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, update {update}")
            sample_gradients = step == len(loaders[0]) or (args.max_train_batches and step == args.max_train_batches)
            if sample_gradients:
                event(run_dir, "gradients.jsonl", {"epoch": epoch, "update": update+1,
                      "gradients": gradient_diagnostics(model, details["classification"], details["weighted_transfer"])})
            norms = feature_norms(model.last_features, len(source_batch["label"]))
            loss.backward()
            optimizer.step()
            source_correct += int((details["source_logits"].argmax(1) == source_batch["label"]).sum())
            source_seen += len(source_batch["label"])
            target_seen += len(target_batch["rgb"])
            update += 1
            steps += 1
            values = {key: float(details[key].detach()) for key in totals}
            for key in totals:
                totals[key] += values[key]
            record = {"epoch": epoch, "update": update, **values,
                      "transfer_weight": details["transfer_weight"], "grl_coefficient": details["grl_coefficient"],
                      "entropy_grl_coefficient": details["entropy_grl_coefficient"],
                      "domain_accuracy": details["domain_accuracy"], "feature_norms": norms,
                      "source_ce_unweighted": float(details["source_ce_unweighted"]),
                      "domain_feature_moments": details["domain_feature_moments"],
                      "normalized_feature_norms": feature_norms(model.last_normalized_features, len(source_batch["label"])),
                      "modality_weights": model.modality_weights().detach().cpu().tolist(),
                      "learning_rates": [g["lr"] for g in optimizer.param_groups]}
            event(run_dir, "updates.jsonl", record)
            if scheduler is not None and schedule_per_update:
                step_scaled_scheduler(scheduler, optimizer, validation_control.lr_scale)
            if step == 1 or step == len(loaders[0]) or step % 5 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] batch [{step}/{len(loaders[0])}] "
                      f"source_CE={values['source_ce']:.4f} target_labels_used=False "
                      f"DA={values['transfer']:.4f} weight={details['transfer_weight']:.4f}", flush=True)
            if args.max_train_batches and step >= args.max_train_batches:
                break
        synchronize(device)
        times.append(time.perf_counter()-started)
        if device.type == "cuda":
            peak_reserved = max(peak_reserved or 0, torch.cuda.max_memory_reserved(device)/1024**3)
        if not args.max_train_batches and (len(seen_source) != len(source) or len(seen_target) != len(target)):
            raise RuntimeError("Training sampler failed complete source/target coverage")
        training = {"epoch": epoch, **{key: value/steps for key, value in totals.items()},
                    "updates": steps, "source_sample_exposures": source_seen, "target_sample_exposures": target_seen,
                    "source_unique_samples": len(seen_source), "target_unique_samples": len(seen_target),
                    "source_train_accuracy": source_correct/source_seen, "target_training_labels_used": False,
                    "training_time_s": times[-1]}
        event(run_dir, "training.jsonl", training)
        for ds in (source_eval, target_eval):
            evaluate(model, ds, device, epoch, run_dir)
        metrics, _, validation_details = evaluate(model, validation, device, epoch, run_dir, return_details=True)
        history.append({"epoch": epoch, **metrics})
        if better_metrics(metrics, best_metrics):
            best_metrics, best_epoch, stale = dict(metrics), epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        control = validation_control.observe(validation_details["cross_entropy"], epoch, optimizer)
        if scheduler is not None and not schedule_per_update:
            step_scaled_scheduler(scheduler, optimizer, validation_control.lr_scale)
        stopping_stale = validation_control.bad_epochs if args.early_stopping_monitor == "cross_entropy" else stale
        should_stop = bool(args.early_stopping_patience and epoch >= args.early_stopping_min_epochs
                           and stopping_stale >= args.early_stopping_patience)
        control.update(epoch=epoch, selected_ACC_epoch=best_epoch, ACC_F1_bad_epochs=stale,
                       early_stopping_monitor=args.early_stopping_monitor,
                       early_stopping_bad_epochs=stopping_stale, early_stop=should_stop,
                       next_learning_rates=[g["lr"] for g in optimizer.param_groups])
        event(run_dir, "validation_control.jsonl", control)
        print("VDS_VALIDATION_CONTROL: "+json.dumps(control), flush=True)
        if should_stop:
            stop_reason = "validation_"+args.early_stopping_monitor+"_patience"
            break
    # No test loader or test predictions exist in the optimization loop.
    model.load_state_dict(best_state, strict=True)
    sensitivity = None
    if args.modality_diagnostics:
        (run_dir/"modality_diagnostics/full").mkdir(parents=True, exist_ok=True)
        sensitivity = validation_modality_diagnostics(model, validation, device, best_epoch, run_dir)
        if sensitivity["full"] != best_metrics:
            raise RuntimeError("Restored validation metrics differ from selected metrics")
        model.load_state_dict(best_state, strict=True)
    test = AdaptationDataset(root, "target_test", images, ais)
    test_metrics, test_records = evaluate(model, test, device, best_epoch, run_dir)
    artifacts = save_prediction_artifacts(run_dir, "target_test", test_records, DISPLAY_CLASS_NAMES)
    complexity = profile(model, validation, device, times, peak_reserved)
    for key in ("total_params", "trainable_params", "inference_GFLOPs", "total_train_time_s", "train_epoch_mean_s", "latency_mean_ms", "throughput_samples_s"):
        if not math.isfinite(complexity[key]) or complexity[key] <= 0:
            raise RuntimeError(f"Invalid complexity metric: {key}")
    if device.type == "cuda" and not peak_reserved:
        raise RuntimeError("Missing CUDA training memory measurement")
    # Check persisted events, not just counters in memory.
    evaluations = [json.loads(row) for row in (run_dir/"evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    for partition in ("source_train", "target_train", "target_val"):
        if len([e for e in evaluations if e["partition"] == partition]) != len(history):
            raise RuntimeError(f"Missing evaluation events: {partition}")
    if len([e for e in evaluations if e["partition"] == "target_test"]) != 1:
        raise RuntimeError("target_test must be evaluated exactly once")
    checked = validate_diagnostics(run_dir, (source_eval, target_eval, validation, test), len(history), best_epoch)
    control_records = [json.loads(row) for row in (run_dir/"validation_control.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(control_records) != len(history) or [r["epoch"] for r in control_records] != list(range(1, len(history)+1)):
        raise RuntimeError("Missing validation control diagnostics")
    completeness = {"status": "PASS", "epochs": len(history), "selected_epoch": best_epoch,
                    "protocol": "UDA", "target_training_labels_used": False,
                    "selection_partition": "target_val", "test_evaluations": 1,
                    "test_evaluations_for_selection": 0, "complexity_status": "PASS",
                    "ais_dimensions": 13, "ais_mask_dimensions": 13,
                    "prediction_rows_checked": checked,
                    "partial_training_smoke": bool(args.max_train_batches)}
    write_json(run_dir/"completeness.json", completeness)
    result = {"method": args.method, "seed": args.seed, "protocol": "UDA", "target_training_labels_used": False,
              "optimization_profile": args.profile,
              "planned_optimizer_updates": args.epochs*args.steps_per_epoch,
              "selected_epoch": best_epoch, "selection_partition": "target_val",
              "test_selection_bias": False, "test_evaluations_for_selection": 0,
              "test_evaluations": 1, "completed_epochs": len(history),
              "stop_reason": stop_reason, "stopped_early": len(history) < args.epochs,
              "validation_control": {**config["validation_control"],
                                     "final_lr_scale": validation_control.lr_scale,
                                     "best_validation_CE": validation_control.best_loss if math.isfinite(validation_control.best_loss) else None,
                                     "best_validation_CE_epoch": validation_control.best_loss_epoch},
              "feature_normalization": args.feature_normalization,
              "modality_diagnostics": sensitivity,
              "target_val": best_metrics, "target_test": test_metrics, "complexity": complexity,
              "prediction_artifacts": artifacts, "diagnostics_dir": str(run_dir),
              "modality_weights_at_selected_epoch": model.modality_weights().detach().cpu().tolist(),
              "smoke": bool(args.max_train_batches)}
    write_json(run_dir/"result.json", result)
    print("VDS_COMPLETENESS: "+json.dumps(completeness), flush=True)
    print("FINAL_TEST_METRICS_JSON: "+json.dumps(result), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--data-root", default=r"D:\Downloads\VDS2Raw\threeclass_ready")
    p.add_argument("--save-dir", default=str(ROOT/"runs/vds2raw_da"))
    p.add_argument("--stats-file", default="")
    p.add_argument("--weights", default=str(ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--profile", choices=("baseline", "optimized", "mcc_v2", "mcc_v3"), default="baseline")
    p.add_argument("--feature-normalization", choices=("none", "l2"), default=None)
    p.add_argument("--modality-diagnostics", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--source-class-balance", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--domain-forward", choices=("reference", "joint"), default=None)
    p.add_argument("--log-feature-moments", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ais-lr-multiplier", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float)
    p.add_argument("--trade-off", type=float)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--lr-scheduler", choices=("reference", "constant", "cosine"), default="reference")
    p.add_argument("--lr-gamma", type=float, default=None,
                   help="reference defaults: MDD .0002; CDAN/MCC .001")
    p.add_argument("--lr-decay", type=float, default=0.75)
    p.add_argument("--adaptation-warmup-epochs", type=int, default=None)
    p.add_argument("--adaptation-ramp-epochs", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--early-stopping-monitor", choices=("accuracy_f1", "cross_entropy"), default=None)
    p.add_argument("--early-stopping-min-epochs", type=int, default=None)
    p.add_argument("--val-plateau-patience", type=int, default=None)
    p.add_argument("--val-plateau-factor", type=float, default=None)
    p.add_argument("--val-plateau-min-scale", type=float, default=None)
    p.add_argument("--validation-loss-min-delta", type=float, default=None)
    p.add_argument("--validation-control-start-epoch", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if (args.epochs is not None and args.epochs < 1) or args.batch_size < 2 or args.num_workers < 0 or args.cpu_threads < 1:
        raise ValueError("Invalid epochs/batch/workers/CPU threads")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("Invalid label smoothing")
    if (args.lr is not None and args.lr <= 0) or (args.trade_off is not None and args.trade_off < 0):
        raise ValueError("Invalid learning rate/transfer weight")
    if (args.lr_gamma is not None and args.lr_gamma < 0) or args.lr_decay <= 0:
        raise ValueError("Invalid inverse learning-rate schedule")
    if args.ais_lr_multiplier is not None and args.ais_lr_multiplier <= 0:
        raise ValueError("AIS learning-rate multiplier must be positive")
    if args.val_plateau_factor is not None and not 0 < args.val_plateau_factor < 1:
        raise ValueError("Validation plateau factor must be in (0,1)")
    if args.val_plateau_min_scale is not None and not 0 < args.val_plateau_min_scale <= 1:
        raise ValueError("Validation plateau minimum scale must be in (0,1]")
    if min(args.adaptation_warmup_epochs or 0, args.adaptation_ramp_epochs or 0, args.early_stopping_patience or 0,
           args.early_stopping_min_epochs or 0, args.val_plateau_patience or 0, args.validation_loss_min_delta or 0,
           args.validation_control_start_epoch or 0, args.max_train_batches, args.weight_decay) < 0:
        raise ValueError("Negative schedule or regularization argument")
    run(args)
