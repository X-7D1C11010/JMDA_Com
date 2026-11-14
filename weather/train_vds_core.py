"""Weather VIS+IR UDA using the same five audited method cores as VDS2Raw.

Only data/metadata and the number of modalities/classes change. No target
training label enters a batch, loss, gradient diagnostic, or prediction log.
Weather has no separate test partition: report the best target_val epoch.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tools"))

import torch
from torch.utils.data import DataLoader, Dataset

from tools.reproducibility import seed_everything
from tools.uda_data import uda_label_space, assert_training_label_space, read_image_list
from tools.uda_multimodal import (
    paired_dataset, label_free_target_paired_dataset,
    jmda_train_transform, jmda_val_transform,
)
from vds2raw.adaptation import AdaptationModel, PairedEpochSampler, resolve_training_settings, METHODS, source_classification_weights
from vds2raw.common import better_metrics, build_run_provenance
from vds2raw.train_adaptation import (
    make_optimizer, move, synchronize, write_json,
    event, evaluate, profile, gradient_diagnostics, feature_norms, validate_diagnostics,
    ValidationLossControl, step_scaled_scheduler,
    validation_modality_diagnostics,
)

TARGETS = ("backlight", "rainy", "foggy", "dark")


def resolve_weather_settings(args):
    defaults = {"epochs": 100, "lr_scheduler": "reference", "adaptation_warmup_epochs": 0,
                "early_stopping_patience": 0, "early_stopping_min_epochs": 0,
                "domain_forward": "reference", "log_feature_moments": False,
                "freeze_bn_after_epochs": -1, "freeze_bn_scope": "all",
                "feature_normalization": "none", "auxiliary_source_weight": 0.,
                "source_class_balance": False, "label_smoothing": 0., "gradient_clip_norm": 0.}
    defaults["domain_running_stats"] = False
    if args.weather_profile == "timing_v2":
        defaults.update(epochs=40, lr_scheduler="cosine", domain_forward="joint", log_feature_moments=True,
                        adaptation_ramp_epochs=5, early_stopping_patience=15, early_stopping_min_epochs=20)
        defaults.update(lr=.00015, trade_off=.20) if args.method == "dann" else defaults.update(lr=.0001, trade_off=.10)
    elif args.weather_profile == "bn_stable":
        # Single-factor change relative to the baseline: after three epochs
        # keep BN running statistics fixed, with affine/conv weights trainable.
        defaults.update(freeze_bn_after_epochs=3, log_feature_moments=True, modality_diagnostics=True)
    elif args.weather_profile == "balanced_v4":
        # Keep each named DA formula. Correct the fusion scale and give both
        # modalities source supervision; use method-specific adaptation timing.
        defaults.update(feature_normalization="l2", auxiliary_source_weight=.20,
                        source_class_balance=True, label_smoothing=.05, gradient_clip_norm=5.,
                        log_feature_moments=True, modality_diagnostics=True, freeze_bn_scope="encoders",
                        val_plateau_patience=15, early_stopping_patience=30,
                        early_stopping_min_epochs=60, early_stopping_monitor="cross_entropy")
        tuned = {
            "dann": dict(lr=.0002, trade_off=.10, adaptation_warmup_epochs=3,
                         adaptation_ramp_epochs=10, lr_scheduler="cosine",
                         freeze_bn_after_epochs=-1 if args.target == "backlight" else 3),
            "deepcoral": dict(lr=.0001, trade_off=.05, adaptation_warmup_epochs=3,
                              adaptation_ramp_epochs=15, lr_scheduler="cosine"),
            "cdan": dict(lr=.0002, trade_off=.50, adaptation_warmup_epochs=3,
                         adaptation_ramp_epochs=10, freeze_bn_after_epochs=3),
            "mdd": dict(lr=.0015, trade_off=.20, adaptation_warmup_epochs=3, adaptation_ramp_epochs=10),
            "mcc": dict(lr=.0015, trade_off=.20, adaptation_warmup_epochs=5, adaptation_ramp_epochs=15),
        }[args.method]
        defaults.update(tuned)
        defaults["validation_control_start_epoch"] = tuned["adaptation_warmup_epochs"]+tuned["adaptation_ramp_epochs"]
    elif args.weather_profile == "domain_stats_v5":
        # Keep baseline LR, loss, epoch and augmentation settings. Change only
        # domain-specific normalization: no post-hoc target recalibration.
        defaults.update(domain_running_stats=True, log_feature_moments=True, modality_diagnostics=True)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    resolve_training_settings(args)
    if args.adaptation_ramp_epochs is None:
        args.adaptation_ramp_epochs = args.epochs if args.method == "deepcoral" else 0
    return args


def apply_batchnorm_policy(model, epoch, freeze_after, scope="all"):
    if scope not in {"all", "encoders"}:
        raise ValueError("BN freeze scope must be all or encoders")
    frozen = freeze_after >= 0 and epoch > freeze_after
    count = 0
    if frozen:
        modules = model.modules() if scope == "all" else (
            module for encoder in (model.rgb_encoder, model.nir_encoder) for module in encoder.modules())
        for module in modules:
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                count += 1
    return {"BN_running_stats_frozen": frozen, "frozen_BN_modules": count,
            "scope": scope,
            "affine_parameters_trainable": True}


class WeatherView(Dataset):
    """Adapt existing exact-pair image loaders to the common core dictionary."""

    def __init__(self, dataset, partition):
        self.dataset, self.partition = dataset, partition
        self.labelled = partition != "target_train"
        if self.labelled == bool(getattr(dataset, "label_free", False)):
            raise ValueError("Target training must use the label-free pair loader")
        self.paths = [str(row[0]) for row in dataset.paired_samples]
        self._evaluation_cache = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self._evaluation_cache is not None:
            rgb, nir, labels = self._evaluation_cache
            item = {"rgb": rgb[index], "nir": nir[index], "path": self.paths[index]}
            if self.labelled:
                item["label"] = labels[index]
            return item
        images, label = self.dataset[index]
        result = {"rgb": images[:3], "nir": images[3:], "path": self.paths[index]}
        if self.labelled:
            result["label"] = label
        elif label != -1:
            raise RuntimeError("Target loader returned a real label")
        return result

    def cache_evaluation_inputs(self, workers=0):
        """Cache deterministic CPU image tensors, never model features/statistics."""
        operations = getattr(self.dataset.transform, "transforms", ())
        if any(type(operation).__name__.startswith("Random") for operation in operations):
            raise ValueError("Do not cache stochastic training transforms")
        if self._evaluation_cache is not None:
            raise ValueError("Evaluation inputs already cached")
        cached_rgb = torch.empty(len(self), 3, 224, 224)
        cached_nir = torch.empty_like(cached_rgb)
        labels, offset = [], 0
        for batch in DataLoader(self, batch_size=32, num_workers=workers, shuffle=False,
                                generator=torch.Generator().manual_seed(0)):
            count = len(batch["path"])
            if batch["path"] != self.paths[offset:offset+count]:
                raise RuntimeError("Evaluation cache changed cohort order")
            cached_rgb[offset:offset+count] = batch["rgb"]
            cached_nir[offset:offset+count] = batch["nir"]
            if self.labelled:
                labels.extend(batch["label"].tolist())
            offset += count
        if offset != len(self):
            raise RuntimeError("Evaluation cache is incomplete")
        self._evaluation_cache = cached_rgb, cached_nir, labels
        return {"samples": offset, "bytes": (cached_rgb.numel()+cached_nir.numel())*cached_rgb.element_size(),
                "model_features_cached": False, "statistics_fitted": False}


def prepare_datasets(data_dir, target_name, augment=True):
    data_dir = Path(data_dir).resolve()
    class_ids, label_map = uda_label_space(str(data_dir), target_name)
    train_transform = jmda_train_transform() if augment else jmda_val_transform()
    val_transform = jmda_val_transform()
    source_path = str(data_dir/"sunny_train.txt")
    target_path = str(data_dir/f"{target_name}_train.txt")
    source = paired_dataset(source_path, train_transform, label_map)
    target = label_free_target_paired_dataset(target_path, train_transform, label_map)
    source_eval = paired_dataset(source_path, val_transform, label_map)
    target_eval = label_free_target_paired_dataset(target_path, val_transform, label_map)
    validation = paired_dataset(str(data_dir/f"{target_name}_val.txt"), val_transform, label_map)
    assert_training_label_space(source, target, class_ids)
    # Target metadata is only used to check the declared cohort/category set.
    # Its labels are never stored in either target-training dataset/view.
    from tools.data_paths import remap_data_path
    expected_target = {str(Path(remap_data_path(path))) for path, _ in read_image_list(target_path)}
    actual_target = {str(Path(path)) for path, _ in target.paired_samples}
    if expected_target != actual_target or len(actual_target) != len(target):
        raise ValueError("Target label-free cohort differs from prepared target metadata")
    groups = (source, target, validation)
    modality_ids = []
    for dataset in groups:
        if dataset.missing_ir:
            raise ValueError("Prepared list has missing exact infrared mates")
        paths = [str(Path(p).resolve()) for row in dataset.paired_samples for p in row[:2]]
        if len(set(paths)) != len(paths):
            raise ValueError("Repeated sample/modality path in a partition")
        if any(not Path(p).is_file() for p in paths):
            raise FileNotFoundError("Prepared image path does not exist")
        modality_ids.append(set(paths))
    if any(modality_ids[i] & modality_ids[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Image leakage between source_train, target_train, target_val")
    if {label for _, label in validation.samples} != set(range(len(class_ids))):
        raise ValueError("Validation class support differs after exact pairing")
    datasets = [WeatherView(dataset, partition) for dataset, partition in (
        (source, "source_train"), (target, "target_train"), (source_eval, "source_train"),
        (target_eval, "target_train"), (validation, "target_val"))]
    audit = {"status": "PASS", "source": "sunny", "target": target_name,
             "original_class_ids": class_ids, "compact_label_map": label_map,
             "source_classes_equal_target_classes": True,
             "source_samples_excluded_outside_target_support": source.skipped,
             "counts": {ds.partition: len(ds) for ds in datasets},
             "pairing": "exact VIS/IR filename stem", "cross_partition_path_overlap": 0,
             "target_training_labels_used": False,
             "category_support_policy": "known target category set; shared compact mapping; no target instance labels in training"}
    return datasets, audit


def run(args):
    if args.method not in METHODS:
        raise ValueError("Unknown weather adaptation method")
    if args.weather_profile in {"timing_v2", "bn_stable"} and args.method not in {"dann", "deepcoral"}:
        raise ValueError("Historical timing/BN comparison is only defined for DANN and Deep CORAL")
    args.seed = seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    resolve_weather_settings(args)
    datasets, audit = prepare_datasets(args.data_dir, args.target, args.augment)
    source, target, source_eval, target_eval, validation = datasets
    cache_record = {"enabled": args.cache_evaluation, "partitions": {}}
    if args.cache_evaluation:
        cache_started = time.perf_counter()
        for dataset in (source_eval, target_eval, validation):
            cache_record["partitions"][dataset.partition] = dataset.cache_evaluation_inputs(args.num_workers)
        cache_record["build_time_s"] = time.perf_counter()-cache_started
    output = Path(args.save_dir).resolve()/f"{args.method}_{args.target}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    write_json(output/"dataset_audit.json", audit)
    write_json(output/"evaluation_cache.json", cache_record)
    args.steps_per_epoch = math.ceil(max(len(source), len(target))/args.batch_size)
    samplers = [PairedEpochSampler(ds, args.batch_size, args.steps_per_epoch, args.seed+offset)
                for ds, offset in ((source, 0), (target, 10000))]
    loader_args = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    if args.num_workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2, timeout=300)
    loaders = [DataLoader(ds, batch_sampler=sampler,
                          generator=torch.Generator().manual_seed(args.seed+20000+i), **loader_args)
               for i, (ds, sampler) in enumerate(zip((source, target), samplers))]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = len(audit["original_class_ids"])
    model = AdaptationModel(args.method, args.weights, not args.no_pretrained,
                            num_classes=classes, use_ais=False, ir_channels=3,
                            feature_normalization=args.feature_normalization,
                            auxiliary_source_weight=args.auxiliary_source_weight,
                            image_backbone=args.image_backbone,
                            freeze_image_encoders=args.freeze_image_encoders,
                            unit_method_features=args.unit_method_features,
                            encoder_training_scope=args.encoder_training_scope,
                            classifier_layout=args.classifier_layout).to(device)
    if args.initial_vis_weight is not None:
        desired = args.initial_vis_weight
        if not .05 < desired < .95:
            raise ValueError("Initial VIS weight must lie strictly between .05 and .95")
        probabilities = model.modality_logits.new_tensor([(desired-.05)/.90, (.95-desired)/.90])
        with torch.no_grad():
            model.modality_logits.copy_(probabilities.log())
    if args.domain_running_stats:
        from weather.domain_running_stats import install_domain_statistics
        normalization_modules = install_domain_statistics(model)
        if normalization_modules == 0:
            raise ValueError("No BatchNorm layers found for the domain-statistics variant")
        model.domain_running_stats = True
    source_counts, source_weights = source_classification_weights(source.dataset.samples, classes)
    if args.source_class_balance:
        model.source_class_weights = source_weights.to(device)
    optimizer, scheduler, schedule_per_update = make_optimizer(model, args)
    fda_generator = torch.Generator(device=device).manual_seed(args.seed+33000) if args.fda_beta else None
    zoom_generator = (torch.Generator(device=device).manual_seed(args.seed+34000)
                      if args.source_zoom_min_area < 1 and args.source_zoom_probability else None)
    validation_control = ValidationLossControl(args)
    inputs = list(Path(args.data_dir).glob("*.txt"))+list(Path(args.data_dir).glob("*.tsv"))
    if not args.no_pretrained:
        inputs.append(Path(args.weights))
    config = {"method": args.method, "protocol": "UDA", "source": "sunny", "target": args.target,
              "modalities": ["VIS", "IR"], "core": "vds2raw.adaptation.AdaptationModel",
              "weather_profile": args.weather_profile, "domain_forward": args.domain_forward,
              "target_training_labels_used": False, "target_test_available": False,
              "selection_partition": "target_val", "evaluation_protocol": "best_target_val_within_budget",
              "architecture": f"two independent ImageNet {args.image_backbone} encoders; learned global weights; feature concatenation",
              "image_backbone": args.image_backbone, "freeze_image_encoders": getattr(model, "freeze_image_encoders", args.freeze_image_encoders),
              "encoder_training_scope": getattr(model, "encoder_training_scope", "all"),
              "unit_method_features": args.unit_method_features,
              "classifier_layout": args.classifier_layout,
              "initial_vis_weight": args.initial_vis_weight,
              "source_classification": {"class_counts": source_counts.tolist(), "weighted_CE": args.source_class_balance,
                                        "weights": source_weights.tolist() if args.source_class_balance else None,
                                        "auxiliary_source_weight": args.auxiliary_source_weight},
              "feature_normalization": args.feature_normalization,
              "domain_running_stats": args.domain_running_stats,
              "evaluation_cache": cache_record,
              "augmentation": "JMDA paired Resize256x256/RandomCrop224/HFlip; eval Resize224x224; VIS ImageNet norm, IR .5/.5",
              "extra_training_augmentation": {"FDA_beta": args.fda_beta, "FDA_probability": args.fda_probability,
                                               "source_zoom_min_area": args.source_zoom_min_area,
                                               "source_zoom_probability": args.source_zoom_probability},
              "planned_optimizer_updates": args.epochs*args.steps_per_epoch,
              "sampler": "complete larger-domain coverage; cycle shorter domain; fixed full per-domain batches",
              "optimizer": type(optimizer).__name__, "dataset_audit": audit,
              "optimizer_groups": [{"lr": g["lr"], "role": g["role"], "weight_decay": g["weight_decay"],
                                    "parameters": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups],
              "arguments": vars(args), "model": repr(model),
              "provenance": build_run_provenance(args, args.data_dir, inputs),
              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (Path(__file__), ROOT/"vds2raw/adaptation.py", ROOT/"vds2raw/train_adaptation.py",
                               ROOT/"tools/uda_multimodal.py", ROOT/"tools/uda_data.py")}}
    write_json(output/"run_config.json", config)
    print("WEATHER_PROTOCOL: "+json.dumps({**audit, "arguments": vars(args)}), flush=True)
    best_state = best_metrics = best_records = None
    best_epoch = update = stale = 0
    screening = bool(args.stop_after_epochs and args.stop_after_epochs < args.epochs)
    epoch_limit = min(args.epochs, args.stop_after_epochs) if screening else args.epochs
    stop_reason = "screening_epoch_limit" if screening else "max_epochs"
    times, history, peak_reserved = [], [], None
    for epoch in range(1, epoch_limit+1):
        for sampler in samplers:
            sampler.epoch = epoch-1
        model.train()
        bn_policy = apply_batchnorm_policy(model, epoch, args.freeze_bn_after_epochs, args.freeze_bn_scope)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        totals = {key: 0.0 for key in ("classification", "source_ce", "transfer", "weighted_transfer")}
        seen_source, seen_target = set(), set()
        source_correct = source_seen = target_seen = steps = 0
        for step, (sb, tb) in enumerate(zip(*loaders), 1):
            seen_source.update(sb["path"])
            seen_target.update(tb["path"])
            if "label" in tb:
                raise RuntimeError("Target batch must not contain labels")
            sb, tb = move(sb, device), move(tb, device)
            from weather.fourier_style import augment_source_style
            sb, input_augmentation = augment_source_style(sb, tb, args.fda_beta, args.fda_probability, fda_generator)
            from weather.source_zoom import augment_source_zoom
            sb, zoom_record = augment_source_zoom(sb, args.source_zoom_min_area, args.source_zoom_probability, zoom_generator)
            input_augmentation.update(zoom_record)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.objective(sb, tb, update, args.epochs*args.steps_per_epoch, args)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, update {update}")
            if step == len(loaders[0]) or (args.max_train_batches and step == args.max_train_batches):
                gradients = gradient_diagnostics(model, details["classification"], details["weighted_transfer"])
                gradients["VIS"], gradients["IR"] = gradients.pop("RGB"), gradients.pop("NIR")
                gradients.pop("AIS")
                event(output, "gradients.jsonl", {"epoch": epoch, "update": update+1, "gradients": gradients})
            norms = feature_norms(model.last_features, len(sb["label"]), names=("VIS", "IR"))
            loss.backward()
            unclipped_norm = (float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm,
                                                               error_if_nonfinite=True))
                              if args.gradient_clip_norm else None)
            optimizer.step()
            source_correct += int((details["source_logits"].argmax(1) == sb["label"]).sum())
            source_seen += len(sb["label"])
            target_seen += len(tb["rgb"])
            update += 1
            steps += 1
            values = {key: float(details[key].detach()) for key in totals}
            for key in totals:
                totals[key] += values[key]
            event(output, "updates.jsonl", {"epoch": epoch, "update": update, **values,
                  "transfer_weight": details["transfer_weight"], "grl_coefficient": details["grl_coefficient"],
                  "adaptation_graph_in_loss": details["adaptation_graph_in_loss"],
                  "domain_accuracy": details["domain_accuracy"], "feature_norms": norms,
                  "entropy_grl_coefficient": details["entropy_grl_coefficient"],
                  "auxiliary_source_ce": float(details["auxiliary_source_ce"].detach()),
                  "auxiliary_source_weight": details["auxiliary_source_weight"],
                  "gradient_norm_before_clipping": unclipped_norm,
                  "input_augmentation": input_augmentation,
                  "source_ce_unweighted": float(details["source_ce_unweighted"]),
                  "domain_feature_moments": details["domain_feature_moments"],
                  "method_feature_norms": details["method_feature_norms"],
                  "logit_norms": details["logit_norms"],
                  "normalized_feature_norms": feature_norms(model.last_normalized_features, len(sb["label"]), names=("VIS", "IR")),
                  "modality_weights": model.modality_weights().detach().cpu().tolist(),
                  "learning_rates": [g["lr"] for g in optimizer.param_groups]})
            if scheduler is not None and schedule_per_update:
                step_scaled_scheduler(scheduler, optimizer, validation_control.lr_scale)
            if step == 1 or step % 5 == 0 or step == len(loaders[0]):
                print(f"Epoch [{epoch}/{args.epochs}] batch [{step}/{len(loaders[0])}] "
                      f"source_CE={values['source_ce']:.4f} DA={values['transfer']:.4f} "
                      f"weight={details['transfer_weight']:.4f} target_labels_used=False", flush=True)
            if args.max_train_batches and step >= args.max_train_batches:
                break
        synchronize(device)
        times.append(time.perf_counter()-started)
        if device.type == "cuda":
            peak_reserved = max(peak_reserved or 0, torch.cuda.max_memory_reserved(device)/1024**3)
        if not args.max_train_batches and (len(seen_source) != len(source) or len(seen_target) != len(target)):
            raise RuntimeError("Incomplete training coverage")
        event(output, "training.jsonl", {"epoch": epoch, **{k: v/steps for k, v in totals.items()},
              "updates": steps, "source_sample_exposures": source_seen, "target_sample_exposures": target_seen,
              "source_unique_samples": len(seen_source), "target_unique_samples": len(seen_target),
              "source_train_accuracy": source_correct/source_seen, "target_training_labels_used": False,
              "training_time_s": times[-1]})
        normalization_record = {"epoch": epoch, **bn_policy}
        if args.domain_running_stats:
            from weather.domain_running_stats import statistics_summary
            normalization_record["domain_running_statistics"] = statistics_summary(model)
        event(output, "normalization_policy.jsonl", normalization_record)
        for ds in (source_eval, target_eval):
            evaluate(model, ds, device, epoch, output, args.eval_batch_size, event_prefix="WEATHER_EVALUATION")
        metrics, records, val_details = evaluate(model, validation, device, epoch, output, args.eval_batch_size,
                                               event_prefix="WEATHER_EVALUATION", return_details=True)
        if args.domain_running_stats:
            comparison = output/"normalization_comparison/pooled"
            comparison.mkdir(parents=True, exist_ok=True)
            pooled, _ = evaluate(model, validation, device, epoch, comparison, args.eval_batch_size,
                                 event_prefix="WEATHER_POOLED_DIAGNOSTIC", normalization_domain_override="pooled")
            event(output, "normalization_comparison.jsonl", {"epoch": epoch, "identical_model_weights": True,
                  "selected_protocol": "target_training_running_statistics", "target_statistics": metrics, "pooled_statistics": pooled})
            model.set_bn_domain("target")
        history.append({"epoch": epoch, **metrics})
        if better_metrics(metrics, best_metrics):
            best_metrics, best_records, best_epoch, stale = dict(metrics), records, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        control = validation_control.observe(val_details["cross_entropy"], epoch, optimizer)
        if scheduler is not None and not schedule_per_update:
            step_scaled_scheduler(scheduler, optimizer, validation_control.lr_scale)
        stopping_stale = validation_control.bad_epochs if args.early_stopping_monitor == "cross_entropy" else stale
        stop = bool(args.early_stopping_patience and epoch >= args.early_stopping_min_epochs
                    and stopping_stale >= args.early_stopping_patience)
        control.update(epoch=epoch, selected_ACC_epoch=best_epoch, ACC_F1_bad_epochs=stale,
                       early_stop=stop, next_learning_rates=[g["lr"] for g in optimizer.param_groups])
        event(output, "validation_control.jsonl", control)
        print("WEATHER_VALIDATION_CONTROL: "+json.dumps(control), flush=True)
        if stop:
            stop_reason = "validation_"+args.early_stopping_monitor+"_patience"
            break
    model.load_state_dict(best_state, strict=True)
    sensitivity = None
    if args.modality_diagnostics:
        sensitivity = validation_modality_diagnostics(model, validation, device, best_epoch, output,
                                                       batch_size=args.eval_batch_size)
        if sensitivity["full"] != best_metrics:
            raise RuntimeError("Restored validation metrics differ from selected metrics")
        model.load_state_dict(best_state, strict=True)
    write_json(output/"selected_target_val_predictions.json", best_records)
    complexity = profile(model, validation, device, times, peak_reserved)
    for key in ("total_params", "trainable_params", "inference_GFLOPs", "total_train_time_s", "train_epoch_mean_s", "latency_mean_ms", "throughput_samples_s"):
        if not math.isfinite(complexity[key]) or complexity[key] <= 0:
            raise RuntimeError(f"Invalid complexity metric: {key}")
    if device.type == "cuda" and not peak_reserved:
        raise RuntimeError("Missing GPU training memory measurement")
    checked = validate_diagnostics(output, (source_eval, target_eval, validation), len(history), best_epoch,
                                   num_classes=classes, modalities=("VIS", "IR"))
    completeness = {"status": "PASS", "epochs": len(history), "selected_epoch": best_epoch,
                    "protocol": "UDA", "target_training_labels_used": False,
                    "target_test_available": False, "selection_partition": "target_val",
                    "prediction_rows_checked": checked, "partial_training_smoke": bool(args.max_train_batches)}
    write_json(output/"completeness.json", completeness)
    result = {"method": args.method, "target": args.target, "seed": args.seed, "protocol": "UDA",
              "core": "vds2raw.adaptation.AdaptationModel", "modalities": ["VIS", "IR"],
              "optimizer": type(optimizer).__name__,
              "weather_profile": args.weather_profile, "domain_forward": args.domain_forward,
              "image_backbone": args.image_backbone, "freeze_image_encoders": getattr(model, "freeze_image_encoders", args.freeze_image_encoders),
              "encoder_training_scope": getattr(model, "encoder_training_scope", "all"),
              "unit_method_features": args.unit_method_features,
              "classifier_layout": args.classifier_layout,
              "selected_epoch": best_epoch, "completed_epochs": len(history), "optimizer_updates": update,
              "stop_reason": stop_reason, "planned_optimizer_updates": args.epochs*args.steps_per_epoch,
              "freeze_bn_after_epochs": args.freeze_bn_after_epochs, "modality_diagnostics": sensitivity,
              "freeze_bn_scope": args.freeze_bn_scope, "feature_normalization": args.feature_normalization,
              "domain_running_stats": args.domain_running_stats,
              "auxiliary_source_weight": args.auxiliary_source_weight,
              "screening_only": screening, "planned_epochs": args.epochs,
              "input_augmentation": {"FDA_beta": args.fda_beta, "FDA_probability": args.fda_probability,
                                     "source_zoom_min_area": args.source_zoom_min_area,
                                     "source_zoom_probability": args.source_zoom_probability},
              "evaluation_protocol": "best_target_val_within_budget", "target_test_available": False,
              "target_training_labels_used": False, "target_val": best_metrics, "complexity": complexity,
              "modality_weights_at_selected_epoch": model.modality_weights().detach().cpu().tolist(),
              "smoke": bool(args.max_train_batches)}
    write_json(output/"result.json", result)
    print("FINAL_WEATHER_METRICS_JSON: "+json.dumps(result), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--data-dir", required=True, help="prepared exact-pair lists")
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--weather-profile", choices=("baseline", "timing_v2", "bn_stable", "balanced_v4", "domain_stats_v5"), default="baseline")
    p.add_argument("--domain-running-stats", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--freeze-bn-after-epochs", type=int, default=None)
    p.add_argument("--freeze-bn-scope", choices=("all", "encoders"), default=None)
    p.add_argument("--feature-normalization", choices=("none", "l2", "unit"), default=None)
    p.add_argument("--auxiliary-source-weight", type=float, default=None)
    p.add_argument("--source-class-balance", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gradient-clip-norm", type=float, default=None)
    p.add_argument("--modality-diagnostics", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--domain-forward", choices=("reference", "joint"), default=None)
    p.add_argument("--log-feature-moments", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--save-dir", default=str(ROOT/"runs/weather_vds_core"))
    p.add_argument("--weights", default=str(ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"))
    p.add_argument("--image-backbone", choices=("resnet18", "convnext_small"), default="resnet18")
    p.add_argument("--freeze-image-encoders", action="store_true", help="Freeze encoder weights and stochastic layers; train method bottleneck/heads/gates")
    p.add_argument("--encoder-training-scope", choices=("all", "frozen", "ir_last_stage", "last_stages"), default=None)
    p.add_argument("--image-lr-override", type=float, default=None, help="Absolute LR for trainable encoder parameters; keeps the method head LR separate")
    p.add_argument("--initial-vis-weight", type=float, default=None, help="Optional initialization of the still-trainable global VIS/IR gate")
    p.add_argument("--gate-lr-multiplier", type=float, default=1., help="Relative to the method new-layer LR")
    p.add_argument("--gate-weight-decay", type=float, default=None)
    p.add_argument("--unit-method-features", action="store_true", help="Normalize the method bottleneck output before classification/adaptation")
    p.add_argument("--classifier-layout", choices=("reference", "linear"), default="reference",
                   help="Opt-in direct linear classifier; reference keeps each method's existing bottleneck/head")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--stop-after-epochs", type=int, default=0,
                   help="Screen a prefix of the full epoch/LR/GRL budget without changing its clocks")
    p.add_argument("--cache-evaluation", action="store_true", help="Cache identical deterministic CPU eval inputs to avoid repeated JPEG decoding")
    p.add_argument("--fda-beta", type=float, default=0., help="Opt-in FDA experiment; 0 preserves JMDA augmentation exactly")
    p.add_argument("--fda-probability", type=float, default=.5)
    p.add_argument("--source-zoom-min-area", type=float, default=1.,
                   help="Separate opt-in center-zoom experiment; 1 is the exact original augmentation")
    p.add_argument("--source-zoom-probability", type=float, default=.5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float)
    p.add_argument("--optimizer", choices=("reference", "adam", "sgd"), default="reference")
    p.add_argument("--trade-off", type=float)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--lr-scheduler", choices=("reference", "constant", "cosine"), default=None)
    p.add_argument("--lr-gamma", type=float, default=None)
    p.add_argument("--lr-decay", type=float, default=.75)
    p.add_argument("--adaptation-warmup-epochs", type=int, default=None)
    p.add_argument("--adaptation-ramp-epochs", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--early-stopping-min-epochs", type=int, default=None)
    p.add_argument("--early-stopping-monitor", choices=("accuracy_f1", "cross_entropy"), default=None)
    p.add_argument("--val-plateau-patience", type=int, default=None)
    p.add_argument("--validation-control-start-epoch", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.set_defaults(profile="baseline", ais_lr_multiplier=1.)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if ((args.epochs is not None and args.epochs < 1) or args.batch_size < 2 or args.eval_batch_size < 1
            or args.cpu_threads < 1 or args.num_workers < 0 or args.max_train_batches < 0 or args.stop_after_epochs < 0):
        raise ValueError("Invalid training budget or loader settings")
    if (args.label_smoothing is not None and not 0 <= args.label_smoothing < 1) or (args.lr is not None and args.lr <= 0):
        raise ValueError("Invalid smoothing or learning rate")
    if args.freeze_bn_after_epochs is not None and args.freeze_bn_after_epochs < -1:
        raise ValueError("BN freeze epoch must be -1 (off) or nonnegative")
    if (args.trade_off is not None and args.trade_off < 0) or min(args.adaptation_warmup_epochs or 0, args.adaptation_ramp_epochs or 0,
            args.early_stopping_patience or 0, args.early_stopping_min_epochs or 0, args.weight_decay) < 0:
        raise ValueError("Negative adaptation or regularization argument")
    if min(args.auxiliary_source_weight or 0., args.gradient_clip_norm or 0., args.val_plateau_patience or 0,
           args.validation_control_start_epoch or 0) < 0:
        raise ValueError("Negative auxiliary/gradient/validation control argument")
    if not 0 <= args.fda_beta <= .25 or not 0 <= args.fda_probability <= 1:
        raise ValueError("Invalid FDA beta/probability")
    if not .1 <= args.source_zoom_min_area <= 1 or not 0 <= args.source_zoom_probability <= 1:
        raise ValueError("Invalid source center-zoom area/probability")
    run(args)
