"""DANN / Deep CORAL / CDAN+E / MDD / MCC on supervised VDS2Raw.

Strict split: labelled source_train + target_train, select on target_val,
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
    prepare_statistics,
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


def feature_norms(branches, source_count):
    result = {}
    for domain, section in (("source", slice(None, source_count)), ("target", slice(source_count, None))):
        result[domain] = {
            name: float(values[section].norm(dim=1).mean())
            for name, values in zip(("RGB", "NIR", "AIS"), branches)
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


def evaluate(model, dataset, device, epoch, directory, batch_size=64):
    """Deterministic evaluation; every sample receives a probability record."""
    model.eval()
    truth, predicted, records = [], [], []
    feature_sums = [0.0, 0.0, 0.0]
    losses = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0)):
            labels = batch["label"]
            logits = model(move(batch, device))
            probabilities = logits.softmax(1).cpu()
            guesses = probabilities.argmax(1)
            truth.extend(labels.tolist())
            predicted.extend(guesses.tolist())
            losses.extend(nn.functional.cross_entropy(logits, labels.to(device), reduction="none").cpu().tolist())
            for index, values in enumerate(model.last_features):
                feature_sums[index] += float(values.norm(dim=1).sum())
            for path, label, guess, scores, raw_logits in zip(
                    batch["path"], labels.tolist(), guesses.tolist(), probabilities.tolist(), logits.cpu().tolist()):
                row = {"epoch": epoch, "partition": dataset.partition, "sample_index": len(records),
                       "sample_id": path, "y_true": label, "y_pred": guess,
                       "correct": label == guess, "probabilities": scores, "logits": raw_logits}
                event(directory, "predictions.jsonl", row)
                records.append(row)
    metrics = classification_metrics(truth, predicted, num_classes=3)
    matrix = [[0 for _ in range(3)] for _ in range(3)]
    for expected, guessed in zip(truth, predicted):
        matrix[expected][guessed] += 1
    metrics["confusion_matrix"] = matrix
    details = {"epoch": epoch, "partition": dataset.partition, "metrics": metrics,
               "cross_entropy": statistics.mean(losses),
               "cross_entropy_by_class": [statistics.mean([loss for loss, label in zip(losses, truth) if label == c])
                                          if c in truth else None for c in range(3)],
               "mean_max_probability": statistics.mean(max(row["probabilities"]) for row in records),
               "mean_confidence_wrong": (statistics.mean(max(row["probabilities"]) for row in records if not row["correct"])
                                         if any(not row["correct"] for row in records) else None),
               "recall_by_class": [row[i]/max(sum(row), 1) for i, row in enumerate(matrix)],
               "predicted_class_counts": [sum(row[i] for row in matrix) for i in range(3)],
               "feature_norms": dict(zip(("RGB", "NIR", "AIS"), (x/len(dataset) for x in feature_sums))),
               "modality_weights": model.modality_weights().detach().cpu().tolist()}
    event(directory, "evaluation.jsonl", details)
    print("VDS_EVALUATION: "+json.dumps(details), flush=True)
    return metrics, records


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
                  inference_input="RGB 3x96x96 + NIR 1x96x96 + AIS 13 + mask 13",
                  inference_batch_size=1,
                  flops_scope="executed Conv2d/Linear only; 1 MAC=2 FLOPs",
                  training_time_scope="optimizer windows including gradient diagnostics, excluding evaluation and GPU queue",
                  total_parameter_scope="all training modules, including any discriminator/adversarial head")
    return result


def validate_diagnostics(directory, datasets, epochs, selected_epoch):
    """Fail on missing persisted diagnostics, duplicate IDs or test reuse."""
    expected_epochs = set(range(1, epochs+1))
    for filename in ("training.jsonl", "gradients.jsonl"):
        records = [json.loads(row) for row in (directory/filename).read_text(encoding="utf-8").splitlines()]
        if len(records) != epochs or {row["epoch"] for row in records} != expected_epochs:
            raise RuntimeError(f"Incomplete diagnostic windows: {filename}")
        if filename == "gradients.jsonl" and any(
                not {"RGB", "NIR", "AIS", "head", "gate"} <= set(row["gradients"]) for row in records):
            raise RuntimeError("Missing modality/head gradient diagnostics")
    groups = {}
    with (directory/"predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            probabilities = row["probabilities"]
            if (len(probabilities) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
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
    args.lr = DEFAULTS[args.method]["lr"] if args.lr is None else args.lr
    args.trade_off = DEFAULTS[args.method]["trade_off"] if args.trade_off is None else args.trade_off
    source = AdaptationDataset(root, "source_train", images, ais, augment=args.augment)
    target = AdaptationDataset(root, "target_train", images, ais, augment=args.augment)
    source_eval = AdaptationDataset(root, "source_train", images, ais)
    target_eval = AdaptationDataset(root, "target_train", images, ais)
    validation = AdaptationDataset(root, "target_val", images, ais)
    if {y for _, y in source.samples} != {0, 1, 2} or {y for _, y in target.samples} != {0, 1, 2}:
        raise ValueError("Both training domains must contain all three classes")
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
    trunk = list(model.rgb_encoder.parameters())+list(model.nir_encoder.parameters())
    trunk_ids = {id(p) for p in trunk}
    task = [p for p in model.parameters() if id(p) not in trunk_ids]
    groups = [{"params": trunk, "lr": args.lr*0.1}, {"params": task, "lr": args.lr}]
    optimizer = (torch.optim.AdamW(groups, weight_decay=args.weight_decay) if args.method == "dann"
                 else torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    config = {
        "method": args.method, "protocol": "SDA", "modalities": ["RGB", "NIR", "AIS"],
        "training_partitions": ["source_train", "target_train"],
        "selection_partition": "target_val", "test_partition": "target_test",
        "target_training_labels_used": True, "test_selection_bias": False,
        "test_evaluations_for_selection": 0, "test_evaluation_policy": "once after restoring best target_val state",
        "class_names": DISPLAY_CLASS_NAMES, "statistics": {**images.to_dict(), "ais": ais},
        "architecture": "RGB ResNet18 + NIR ResNet18 (mean RGB conv1 initialization) + AIS/mask MLP; weighted concatenation",
        "adaptation_note": "supervised three-modal extensions; source and target CE plus named core DA loss; not original UDA benchmark",
        "augmentation": "paired padded crop 96, horizontal/vertical flip with AIS heading correction, RGB gain 0.95..1.05; eval deterministic",
        "model": repr(model), "gradient_sampling": "last update of each epoch, CE versus weighted adaptation",
        "optimizer": type(optimizer).__name__,
        "optimizer_groups": [{"lr": group["lr"], "weight_decay": group["weight_decay"],
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
    history, times, peak_reserved = [], [], None
    for epoch in range(1, args.epochs+1):
        for sampler in samplers:
            sampler.epoch = epoch-1
        model.train()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        totals = {key: 0.0 for key in ("classification", "source_ce", "target_ce", "transfer", "weighted_transfer")}
        seen_source, seen_target = set(), set()
        source_correct = target_correct = source_seen = target_seen = steps = 0
        for step, (source_batch, target_batch) in enumerate(zip(*loaders), 1):
            seen_source.update(source_batch["path"])
            seen_target.update(target_batch["path"])
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
            target_correct += int((details["target_logits"].argmax(1) == target_batch["label"]).sum())
            source_seen += len(source_batch["label"])
            target_seen += len(target_batch["label"])
            update += 1
            steps += 1
            values = {key: float(details[key].detach()) for key in totals}
            for key in totals:
                totals[key] += values[key]
            record = {"epoch": epoch, "update": update, **values,
                      "transfer_weight": details["transfer_weight"], "grl_coefficient": details["grl_coefficient"],
                      "domain_accuracy": details["domain_accuracy"], "feature_norms": norms,
                      "modality_weights": model.modality_weights().detach().cpu().tolist(),
                      "learning_rates": [g["lr"] for g in optimizer.param_groups]}
            event(run_dir, "updates.jsonl", record)
            if step == 1 or step == len(loaders[0]) or step % 5 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] batch [{step}/{len(loaders[0])}] "
                      f"source_CE={values['source_ce']:.4f} target_CE={values['target_ce']:.4f} "
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
                    "source_train_accuracy": source_correct/source_seen, "target_train_accuracy": target_correct/target_seen,
                    "training_time_s": times[-1]}
        event(run_dir, "training.jsonl", training)
        for ds in (source_eval, target_eval):
            evaluate(model, ds, device, epoch, run_dir)
        metrics, _ = evaluate(model, validation, device, epoch, run_dir)
        history.append({"epoch": epoch, **metrics})
        if better_metrics(metrics, best_metrics):
            best_metrics, best_epoch, stale = dict(metrics), epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            break
    # No test loader or test predictions exist in the optimization loop.
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
    completeness = {"status": "PASS", "epochs": len(history), "selected_epoch": best_epoch,
                    "selection_partition": "target_val", "test_evaluations": 1,
                    "test_evaluations_for_selection": 0, "complexity_status": "PASS",
                    "ais_dimensions": 13, "ais_mask_dimensions": 13,
                    "prediction_rows_checked": checked,
                    "partial_training_smoke": bool(args.max_train_batches)}
    write_json(run_dir/"completeness.json", completeness)
    result = {"method": args.method, "seed": args.seed, "protocol": "SDA",
              "selected_epoch": best_epoch, "selection_partition": "target_val",
              "test_selection_bias": False, "test_evaluations_for_selection": 0,
              "test_evaluations": 1, "completed_epochs": len(history),
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float)
    p.add_argument("--trade-off", type=float)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--source-weight", type=float, default=1)
    p.add_argument("--target-weight", type=float, default=1)
    p.add_argument("--adaptation-warmup-epochs", type=int, default=5)
    p.add_argument("--adaptation-ramp-epochs", type=int, default=10)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0 or args.cpu_threads < 1:
        raise ValueError("Invalid epochs/batch/workers/CPU threads")
    if not 0 <= args.label_smoothing < 1 or args.source_weight <= 0 or args.target_weight <= 0:
        raise ValueError("Invalid classification supervision weights")
    if (args.lr is not None and args.lr <= 0) or (args.trade_off is not None and args.trade_off < 0):
        raise ValueError("Invalid learning rate/transfer weight")
    if min(args.adaptation_warmup_epochs, args.adaptation_ramp_epochs, args.early_stopping_patience, args.max_train_batches, args.weight_decay) < 0:
        raise ValueError("Negative schedule or regularization argument")
    run(args)
