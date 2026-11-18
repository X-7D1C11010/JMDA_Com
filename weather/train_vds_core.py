"""Weather VIS+IR training with five method cores shared with VDS2Raw.

Default UDA keeps target instance labels out of training. Explicit label-guided
modes separately declare supervised sampling/alignment. Optional locked
holdouts are evaluated once after validation-only model selection.
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


def configure_allocator_budget(device, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("CUDA allocator fraction must lie in (0,1]")
    if device.type=="cuda" and fraction<1:
        index=device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(fraction,index)


def paired_epoch_samplers(source, target, source_batch_size, target_batch_size, seed, method):
    target_batch_size = target_batch_size if target_batch_size is not None else source_batch_size
    if min(source_batch_size, target_batch_size) < 2:
        raise ValueError("Both training domain batch sizes must be at least two")
    if method == "mdd" and source_batch_size != target_batch_size:
        raise ValueError("The released MDD objective requires equal source/target batch sizes")
    steps = max(math.ceil(len(source)/source_batch_size), math.ceil(len(target)/target_batch_size))
    samplers = [PairedEpochSampler(ds, size, steps, seed+offset)
                for ds, size, offset in ((source, source_batch_size, 0), (target, target_batch_size, 10000))]
    return steps, samplers


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
        if hasattr(self,"source_object_views"):
            if self.partition!="source_train":raise RuntimeError("Extra object views are source-only")
            result["source_object_rgb"],result["source_object_is_roi"]=self.source_object_views.sample(index)
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
             "source_train_class_ids_compact": sorted({label for _,label in source.samples}),
             "target_train_class_ids_compact": sorted({label_map[label] for _,label in read_image_list(target_path)}),
             "class_support_check": "exact source_train == target_train; shared mapping; fail before training on mismatch",
             "target_label_metadata_usage": "known class support and cohort audit only; no target labels in training batches",
             "source_samples_excluded_outside_target_support": source.skipped,
             "counts": {ds.partition: len(ds) for ds in datasets},
             "pairing": "exact VIS/IR filename stem", "cross_partition_path_overlap": 0,
             "target_training_labels_used": False,
             "category_support_policy": "known target category set; shared compact mapping; no target instance labels in training"}
    return datasets, audit


def run(args):
    if args.method not in METHODS:
        raise ValueError("Unknown weather adaptation method")
    if not math.isfinite(args.mcc_temperature) or args.mcc_temperature<=0:
        raise ValueError("MCC temperature must be finite and positive")
    if args.checkpoint_vis_blocks and args.image_backbone!="convnext_small":
        raise ValueError("VIS block checkpointing is defined for ConvNeXt")
    if (args.calibration_components<2 or args.calibration_shrinkage<=0 or args.cluster_teacher_weight<0 or
            args.cluster_teacher_temperature<=0 or not 0<=args.cluster_teacher_confidence<=1):
        raise ValueError("Invalid feature calibration/cluster-teacher settings")
    if not math.isfinite(args.calibration_adapter_scale) or args.calibration_adapter_scale<=0:
        raise ValueError("Calibration adapter scale must be finite and positive")
    if not 0<=args.cluster_association_vis_weight<=1:raise ValueError("Invalid cross-modal association weight")
    if (not all(math.isfinite(x) and x>=0 for x in (args.source_crossmodal_weight,args.target_crossmodal_weight))
            or not math.isfinite(args.crossmodal_temperature) or args.crossmodal_temperature<=0):
        raise ValueError("Invalid paired crossmodal loss settings")
    if args.crossmodal_projection_dim<0 or args.crossmodal_projection_dim==1:raise ValueError("Invalid contrastive projection width")
    if (not math.isfinite(args.prediction_teacher_weight) or not 0<=args.prediction_teacher_weight<1
            or not all(math.isfinite(x) and x>0 for x in
                (args.prediction_teacher_temperature,args.prediction_student_temperature))):
        raise ValueError("Invalid frozen prediction teacher settings")
    if (not math.isfinite(args.prediction_distillation_weight) or args.prediction_distillation_weight<0
            or not math.isfinite(args.prediction_distillation_sharpening) or args.prediction_distillation_sharpening<=0
            or args.prediction_distillation_start_epoch<1):
        raise ValueError("Invalid prediction distillation settings")
    if not math.isfinite(args.prediction_distillation_encoder_lr_scale) or args.prediction_distillation_encoder_lr_scale<=0:
        raise ValueError("Invalid post-cache encoder LR scale")
    if args.prediction_distillation_weight and not args.prediction_teacher_weight:
        raise ValueError("Prediction distillation requires a frozen probability teacher")
    if args.prediction_teacher_weight and (args.feature_calibration or args.source_crossmodal_weight
            or args.target_crossmodal_weight or args.source_prototype_weight or args.domain_confusion_weight
            or args.target_feature_style_probability or args.source_object_ce_weight or args.source_object_consistency_weight):
        raise ValueError("Screen the frozen prediction blend independently of other enhancements")
    if args.target_crossmodal_objective=="vicreg" and args.target_crossmodal_weight and not args.crossmodal_projection_dim:
        raise ValueError("VICReg crossmodal screening requires explicit training-only projections")
    if not math.isfinite(args.source_ir_inversion_probability) or not 0<=args.source_ir_inversion_probability<=1:
        raise ValueError("Invalid source IR inversion probability")
    if not math.isfinite(args.target_representation_anchor_weight) or args.target_representation_anchor_weight<0:
        raise ValueError("Invalid target representation anchor weight")
    if args.target_representation_anchor_weight and (args.feature_calibration or args.prediction_teacher_weight or args.target_feature_style_probability):
        raise ValueError("Screen target representation anchoring independently of other teacher/style variants")
    if not all(math.isfinite(x) and 0<=x<=1 for x in (args.source_ir_statistics_probability,args.source_ir_statistics_strength)):
        raise ValueError("Invalid source IR-statistics augmentation settings")
    if not math.isfinite(args.source_prototype_weight) or args.source_prototype_weight<0:raise ValueError("Invalid source prototype weight")
    if not math.isfinite(args.domain_confusion_weight) or args.domain_confusion_weight<0:
        raise ValueError("Invalid encoder domain-confusion weight")
    if args.domain_confusion_weight and args.method not in {"dann","cdan"}:
        raise ValueError("Encoder domain confusion is only defined for DANN/CDAN")
    if not 0<=args.target_feature_style_probability<=1 or not 0<=args.target_feature_style_source_max<=1:
        raise ValueError("Invalid target feature-style settings")
    if not all(math.isfinite(x) and x>=0 for x in (args.source_object_ce_weight,args.source_object_consistency_weight)):
        raise ValueError("Invalid source object loss weights")
    object_enabled=bool(args.source_object_ce_weight or args.source_object_consistency_weight)
    if object_enabled and (not args.source_object_manifest or args.feature_calibration or args.target_feature_style_probability
                           or args.image_backbone!="convnext_small" or args.domain_running_stats):
        raise ValueError("Source object views require a manifest and an independent ConvNeXt/no-calibration screen")
    if args.cluster_teacher_weight and not args.feature_calibration:
        raise ValueError("Cluster teacher loss requires the explicitly enabled enhancement")
    if args.weather_profile in {"timing_v2", "bn_stable"} and args.method not in {"dann", "deepcoral"}:
        raise ValueError("Historical timing/BN comparison is only defined for DANN and Deep CORAL")
    args.seed = seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    resolve_weather_settings(args)
    if not 1<=args.selection_min_epoch<=args.epochs:
        raise ValueError('Selection minimum epoch must lie inside the planned budget')
    if args.stop_after_epochs and args.stop_after_epochs<args.selection_min_epoch:
        raise ValueError('Observed prefix ends before any model is eligible for selection')
    if object_enabled and args.domain_running_stats:
        raise ValueError("Source object screening does not combine separate domain BN")
    if args.target_feature_style_probability and (args.domain_forward!="joint" or args.domain_running_stats or args.feature_calibration):
        raise ValueError("Feature-style screening requires joint domain forward and no separate calibration/domain BN")
    split_manifest=Path(args.data_dir)/'holdout_protocol_manifest.json'
    split_info=None
    if split_manifest.is_file():
        split_info=json.loads(split_manifest.read_text(encoding='utf-8'))
        for name,expected in split_info['output_list_sha256'].items():
            if hashlib.sha256((Path(args.data_dir)/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Locked retraining split list changed: '+name)
    from weather.class_matching import protocol_metadata,make_matched_samplers
    guided=getattr(args,"target_class_matching","none")=="target_truth"
    fewshot=getattr(args,'target_labels_per_class',0)
    alignment_weight=getattr(args,'matched_alignment_weight',0.)
    if (not math.isfinite(alignment_weight) or alignment_weight<0 or args.matched_alignment_ramp_epochs<0
            or not 0<=args.matched_alignment_margin<2):
        raise ValueError('Invalid label-guided alignment settings')
    if fewshot<0:raise ValueError('Target labels per class must be nonnegative')
    source_ema=getattr(args,'source_anchor_ema_momentum',None)
    if source_ema is not None and (not guided or not fewshot or alignment_weight<=0 or not 0<=source_ema<1):
        raise ValueError('Source EMA anchors require positive few-shot alignment and a finite momentum in [0,1)')
    supervision=protocol_metadata(guided,alignment_weight,fewshot)
    datasets, audit = prepare_datasets(args.data_dir, args.target, args.augment)
    source, target, source_eval, target_eval, validation = datasets
    from weather.generalization import validate_holdout,selection_improved,labelled_training_diagnostic
    heldout=None
    if args.holdout_list:
        heldout=WeatherView(paired_dataset(str(args.holdout_list),jmda_val_transform(),audit['compact_label_map']),'target_test')
        audit['heldout_evaluation']=validate_holdout(heldout,(source,target,validation),audit['compact_label_map'])
        audit['counts']['target_test']=len(heldout)
        if split_info is not None:
            audit['heldout_evaluation'].update(grouping=split_info['grouping'],
                historically_unseen_test=split_info['historically_unseen_test'],
                split_seed=split_info['split_seed'])
    matching_plan=None
    if fewshot:
        from weather.fewshot import TargetLabelBudget
        matching_plan=TargetLabelBudget.from_dataset(target,audit['compact_label_map'],fewshot,args.seed)
        audit.update(supervision)
        audit.update(target_label_metadata_usage='fixed per-class labelled subset only; all other target truth excluded from pairing and losses',
                     target_label_budget=fewshot,category_support_policy='shared class support; independent native domain batches')
    elif guided:
        matching_plan,matched_samplers=make_matched_samplers(source,target,args.batch_size,args.target_batch_size,
                                                           audit["compact_label_map"],args.seed)
        audit.update(supervision)
        audit["target_label_metadata_usage"]=("true target_train classes used for supervised sampling"
            +(" and cross-domain class feature alignment" if alignment_weight else "")+"; no target classification CE")
        audit['category_support_policy']='shared class support; per-instance target_train truth guides sampling'+(
            ' and cross-domain class alignment' if alignment_weight else '')
    training_cache={"enabled":args.cache_training_images}
    if args.cache_training_images:
        from weather.training_image_cache import cache_training_resizes
        training_cache.update({"source_train":cache_training_resizes(source.dataset),
                               "target_train":cache_training_resizes(target.dataset)})
    object_views=None
    if object_enabled:
        from weather.source_object_views import SourceObjectViews
        source.source_object_views=SourceObjectViews(source,args.source_object_manifest,audit["compact_label_map"],args.seed)
        object_views=source.source_object_views.metadata
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
    if guided and not fewshot:
        args.steps_per_epoch, samplers = matching_plan.steps, matched_samplers
    else:
        args.steps_per_epoch, samplers = paired_epoch_samplers(
            source, target, args.batch_size, args.target_batch_size, args.seed, args.method)
    if fewshot:
        matching_plan.steps=args.steps_per_epoch
        matching_plan.source_samples=len(source)
        matching_plan.batch_sizes=(args.batch_size,args.target_batch_size or args.batch_size)
    if guided:write_json(output/"label_guided_sampling.json",matching_plan.metadata(alignment_weight))
    loader_args = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    if args.num_workers:
        loader_args.update(persistent_workers=True, prefetch_factor=2, timeout=300)
    loaders = [DataLoader(ds, batch_sampler=sampler,
                          generator=torch.Generator().manual_seed(args.seed+20000+i), **loader_args)
               for i, (ds, sampler) in enumerate(zip((source, target), samplers))]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_allocator_budget(device,args.cuda_memory_fraction)
    classes = len(audit["original_class_ids"])
    model = AdaptationModel(args.method, args.weights, not args.no_pretrained,
                            num_classes=classes, use_ais=False, ir_channels=3,
                            feature_normalization=args.feature_normalization,
                            auxiliary_source_weight=args.auxiliary_source_weight,
                            image_backbone=args.image_backbone,
                            freeze_image_encoders=args.freeze_image_encoders,
                            unit_method_features=args.unit_method_features,
                            encoder_training_scope=args.encoder_training_scope,
                            classifier_layout=args.classifier_layout,
                            ir_backbone=args.ir_backbone, ir_weights_path=args.ir_weights,
                            ir_feature_mode=args.ir_feature_mode,
                            calibration_feature_dim=args.calibration_components if args.feature_calibration else None).to(device)
    model.encoder_precision = args.encoder_precision
    model.vis_gradient_checkpointing=args.checkpoint_vis_blocks
    if args.method=="mcc":
        model.mcc.temperature=args.mcc_temperature
    if args.ir_training_scope=="all_used_layers":
        if not hasattr(model.nir_encoder,"unfreeze_all_used_layers"):
            raise ValueError("Full IR layer policy requires the declared UNIP encoder")
        model.nir_encoder.unfreeze_all_used_layers()
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
    enhancement={"enabled":False}
    if args.prediction_teacher_weight:
        from weather.prediction_teacher import prepare_prediction_teacher
        prediction_metadata=prepare_prediction_teacher(model,source_eval,target_eval,device,
            args.prediction_teacher_weights,args.prediction_teacher_weight,
            args.prediction_teacher_temperature,args.prediction_student_temperature,args.seed)
        enhancement={"enabled":True,"variant_family":"core_plus_frozen_cluster_probability_blend",
            "prediction_teacher":prediction_metadata}
        write_json(output/"prediction_teacher_audit.json",prediction_metadata)
        if args.prediction_distillation_weight:
            from weather.prediction_distillation import TargetProbabilityCache
            model.prediction_teacher_inference=False
            model.target_probability_cache=TargetProbabilityCache(target_eval.paths,classes).to(device)
            prediction_metadata.update(extra_inference_image_encoders=0,training_only=True,
                student_objective_changed=True,inference="student only; ensemble prepares training pseudo-probabilities")
            enhancement.update(variant_family="core_plus_cached_ensemble_target_distillation",
                prediction_distillation_weight=args.prediction_distillation_weight,
                prediction_distillation_start_epoch=args.prediction_distillation_start_epoch,
                prediction_distillation_encoder_lr_scale=args.prediction_distillation_encoder_lr_scale,
                prediction_distillation_sharpening=args.prediction_distillation_sharpening)
            write_json(output/"prediction_teacher_audit.json",prediction_metadata)
    model.source_object_enabled=object_enabled
    if args.source_crossmodal_weight or args.target_crossmodal_weight or args.source_prototype_weight:
        if model.image_feature_dim!=model.ir_feature_dim and not args.crossmodal_projection_dim:
            raise ValueError("Unequal encoder widths require a declared train-only contrastive projection")
        if (not args.crossmodal_projection_dim and
                not any(p.requires_grad for encoder in (model.rgb_encoder,model.nir_encoder) for p in encoder.parameters())):
            raise ValueError("Frozen crossmodal encoders require a trainable contrastive projection")
        if args.feature_calibration:raise ValueError("Screen contrastive and cluster enhancements separately")
        model.crossmodal_alignment_enabled=True
        model.crossmodal_stop_gradient=args.crossmodal_stop_gradient
        projection={"width":0,"trainable_parameters":0,"type":"raw equal-width features"}
        if args.crossmodal_projection_dim:
            from vds2raw.crossmodal_loss import install_crossmodal_projectors
            projection=install_crossmodal_projectors(model,args.crossmodal_projection_dim,args.seed)
        if args.source_prototype_weight:
            from vds2raw.crossmodal_loss import SourceCategoryMemory
            model.source_category_memory=SourceCategoryMemory(classes,args.crossmodal_projection_dim or model.image_feature_dim).to(device)
        enhancement={"enabled":True,"variant_family":"core_plus_paired_crossmodal_alignment",
            "source_supervised_crossmodal_weight":args.source_crossmodal_weight,
            "source_prototype_weight":args.source_prototype_weight,
            "target_instance_crossmodal_weight":args.target_crossmodal_weight,"temperature":args.crossmodal_temperature,
            "target_crossmodal_objective":args.target_crossmodal_objective,
            "target_crossmodal_coefficients":({"invariance":1.,"variance":1.,"covariance":.04}
                if args.target_crossmodal_objective=="vicreg" else None),
            "target_positives":"exact VIS/IR pair index; no target class labels",
            "source_positives":"source class labels across VIS and IR",
            "paired_alignment_changes_image_preprocessing":False,"projection":projection,
            "stop_contrastive_gradient_at_encoder":args.crossmodal_stop_gradient,
            "extra_trainable_parameters":projection["trainable_parameters"]}
        if args.target_crossmodal_weight and args.target_crossmodal_objective=="vicreg":
            enhancement["variant_family"] += "_negative_free_target_pairs"
    if args.feature_calibration:
        from weather.prepare_calibrated_initialization import prepare_calibrated_initialization
        enhancement=prepare_calibrated_initialization(model,source_eval,target_eval,device,args)
        write_json(output/"calibration_audit.json",enhancement)
    if args.source_feature_mixup_weight:
        family=(enhancement["variant_family"]+"_and_source_feature_mixup" if enhancement["enabled"] else "core_plus_source_feature_mixup")
        enhancement.update(enabled=True,variant_family=family,source_feature_mixup={
            "fraction":args.source_feature_mixup_weight,"alpha":args.source_feature_mixup_alpha,
            "source_main_CE_formula":"(1-fraction)*original_CE + fraction*mixed_feature_CE",
            "source_auxiliary_losses_unchanged":True,"target_labels_used":False,
            "extra_encoder_forwards":0,"extra_inference_parameters":0})
    if args.domain_confusion_weight:
        family=enhancement["variant_family"]+"_and_domain_confusion" if enhancement["enabled"] else "core_plus_encoder_domain_confusion"
        enhancement.update(enabled=True,variant_family=family,domain_confusion_weight=args.domain_confusion_weight,
            domain_confusion={"objective":"uniform-domain BCE; encoder-only additional loss",
                "discriminator_parameters_detached":True,"class_probabilities_and_entropy_weights_detached":True,
                "original_discriminator_BCE_retained":True,"target_instance_labels_used":False})
    if args.target_feature_style_probability:
        from weather.target_feature_style import install_target_feature_style
        style=install_target_feature_style(model,args.target_feature_style_probability,args.target_feature_style_source_max,args.seed)
        family=enhancement["variant_family"]+"_and_target_feature_style" if enhancement["enabled"] else "core_plus_target_feature_style"
        enhancement.update(enabled=True,variant_family=family,target_feature_style=style)
    if object_enabled:
        family=enhancement["variant_family"]+"_and_source_object_views" if enhancement["enabled"] else "core_plus_source_object_views"
        enhancement.update(enabled=True,variant_family=family,source_object_views=object_views,
            source_object_ce_weight=args.source_object_ce_weight,
            source_object_consistency_weight=args.source_object_consistency_weight)
    if args.source_ir_statistics_probability:
        family=enhancement["variant_family"]+"_and_source_IR_statistics" if enhancement["enabled"] else "core_plus_source_IR_statistics"
        enhancement.update(enabled=True,variant_family=family,source_IR_statistics={
            "probability":args.source_ir_statistics_probability,"strength":args.source_ir_statistics_strength,
            "scope":"source IR only; target_train moments detached; evaluation unchanged",
            "core_loss_changed":False,"extra_inference_parameters":0})
    if args.target_representation_anchor_weight:
        from weather.target_representation_anchor import prepare_target_representation_cache
        model.target_representation_cache,anchor_metadata=prepare_target_representation_cache(model,target_eval,device,output,args.eval_batch_size)
        family=enhancement["variant_family"]+"_and_initial_target_anchor" if enhancement["enabled"] else "core_plus_initial_target_anchor"
        enhancement.update(enabled=True,variant_family=family,target_representation_anchor=anchor_metadata,
            target_representation_anchor_weight=args.target_representation_anchor_weight,
            target_representation_anchor_active_from_update=1)
    optimizer, scheduler, schedule_per_update = make_optimizer(model, args)
    fda_generator = torch.Generator(device=device).manual_seed(args.seed+33000) if args.fda_beta else None
    zoom_generator = (torch.Generator(device=device).manual_seed(args.seed+34000)
                      if args.source_zoom_min_area < 1 and args.source_zoom_probability else None)
    weather_generator = (torch.Generator(device=device).manual_seed(args.seed+35000)
                         if args.source_weather_probability else None)
    aspect_generator = (torch.Generator(device=device).manual_seed(args.seed+36000)
                        if args.source_aspect_min != 1 or args.source_aspect_max != 1 else None)
    polarity_generator=(torch.Generator(device=device).manual_seed(args.seed+38000) if args.source_ir_inversion_probability else None)
    ir_statistics_generator=(torch.Generator(device=device).manual_seed(args.seed+39000) if args.source_ir_statistics_probability else None)
    validation_control = ValidationLossControl(args)
    inputs = list(Path(args.data_dir).glob("*.txt"))+list(Path(args.data_dir).glob("*.tsv"))
    if object_enabled:inputs.append(Path(args.source_object_manifest))
    if not args.no_pretrained:
        inputs.append(Path(args.weights))
    if args.ir_backbone:
        inputs.append(Path(args.ir_weights))
    if args.prediction_teacher_weight:inputs.append(Path(args.prediction_teacher_weights))
    if args.holdout_list:
        inputs.append(Path(args.holdout_list))
        holdout_manifest=Path(args.data_dir)/'holdout_protocol_manifest.json'
        if holdout_manifest.is_file():inputs.append(holdout_manifest)
    evaluation_protocol=('minimum_target_val_CE_within_budget' if args.selection_metric=='cross_entropy'
                         else 'best_target_val_within_budget')
    config = {"method": args.method, **supervision, "source": "sunny", "target": args.target,
              'configuration_tag':args.configuration_tag,
              "modalities": ["VIS", "IR"], "core": "vds2raw.adaptation.AdaptationModel",
              "enhancement":enhancement,
              "weather_profile": args.weather_profile, "domain_forward": args.domain_forward,
              "target_test_available": heldout is not None,
              "selection_partition": "target_val", "evaluation_protocol": evaluation_protocol,
              'selection_metric':args.selection_metric,
              'selection_min_epoch':args.selection_min_epoch,
              "architecture": (f"ImageNet {args.image_backbone} VIS; official {args.ir_backbone} IR; learned global weights; feature concatenation"
                               if args.ir_backbone else f"two independent ImageNet {args.image_backbone} encoders; learned global weights; feature concatenation"),
              "image_backbone": args.image_backbone, "freeze_image_encoders": getattr(model, "freeze_image_encoders", args.freeze_image_encoders),
              "ir_backbone": args.ir_backbone or args.image_backbone,
              "encoder_precision": args.encoder_precision,
              "checkpoint_vis_blocks":args.checkpoint_vis_blocks,
              "ir_training_scope": args.ir_training_scope,
              "infrared_pretraining": getattr(model.nir_encoder, "provenance", None),
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
              "training_resize_cache":training_cache,
              "augmentation": "JMDA paired Resize256x256/RandomCrop224/HFlip; eval Resize224x224; VIS ImageNet norm, IR .5/.5",
              "extra_training_augmentation": {"FDA_beta": args.fda_beta, "FDA_probability": args.fda_probability,
                                               "source_zoom_min_area": args.source_zoom_min_area,
                                               "source_zoom_probability": args.source_zoom_probability,
                                               "source_weather_probability": args.source_weather_probability,
                                               "source_ir_inversion_probability":args.source_ir_inversion_probability,
                                               "source_ir_statistics_probability":args.source_ir_statistics_probability,
                                               "source_ir_statistics_strength":args.source_ir_statistics_strength,
                                               "source_aspect_min": args.source_aspect_min, "source_aspect_max": args.source_aspect_max,
                                               "weather_augmentation_scope": "source_train only; validation unchanged"},
              "planned_optimizer_updates": args.epochs*args.steps_per_epoch,
              "per_domain_batch_sizes": {"source":args.batch_size,"target":args.target_batch_size or args.batch_size},
              "sampler": (matching_plan.metadata(alignment_weight) if guided else "complete larger-domain coverage; cycle shorter domain; fixed full per-domain batches"),
              "optimizer": type(optimizer).__name__, "dataset_audit": audit,
              "optimizer_groups": [{"lr": g["lr"], "role": g["role"], "weight_decay": g["weight_decay"],
                                    "parameters": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups],
              "arguments": vars(args), "model": repr(model),
              "provenance": build_run_provenance(args, args.data_dir, inputs),
              "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (Path(__file__), ROOT/"vds2raw/adaptation.py", ROOT/"vds2raw/train_adaptation.py",
                               ROOT/"tools/uda_multimodal.py", ROOT/"tools/uda_data.py")}}
    if alignment_weight:
        family=(enhancement['variant_family']+'_and_true_class_cross_domain_alignment'
                if enhancement['enabled'] else 'core_plus_true_class_cross_domain_alignment')
        enhancement.update(enabled=True,variant_family=family,matched_class_alignment={
            'weight':alignment_weight,'margin':args.matched_alignment_margin,
            'ramp_epochs':args.matched_alignment_ramp_epochs,'source_centroids_detached':True,
            'target_true_classes_used':True,'target_classifier_CE_used':False,
            'not_an_original_UDA_method':True,
            'source_anchor_mode':'source_ema' if source_ema is not None else 'current_batch',
            'source_anchor_ema_momentum':source_ema,'prototype_updates_from':'source_train_only',
            **({'target_label_scope':'fixed labelled subset only','labels_per_class':fewshot} if fewshot else {})})
    write_json(output/"run_config.json", config)
    print("WEATHER_PROTOCOL: "+json.dumps({**audit, "arguments": vars(args)},default=str), flush=True)
    if args.feature_calibration:
        initial_dir=output/"initialization_diagnostic"
        initial_dir.mkdir()
        initial_metrics,_=evaluate(model,validation,device,0,initial_dir,args.eval_batch_size,
                                   event_prefix="INITIALIZATION_DIAGNOSTIC_NOT_TRAINED_RESULT")
        write_json(output/"initialization_diagnostic.json",{"target_val":initial_metrics,
            "student_optimizer_updates":0,"formal_method_result":False,"used_for_best_epoch_selection":False})
    best_state = best_metrics = best_records = None
    peak_metrics=None;peak_epoch=0;best_validation_ce=math.inf
    best_epoch = update = stale = 0
    screening = bool(args.stop_after_epochs and args.stop_after_epochs < args.epochs)
    epoch_limit = min(args.epochs, args.stop_after_epochs) if screening else args.epochs
    stop_reason = "screening_epoch_limit" if screening else "max_epochs"
    times, history, peak_reserved = [], [], None
    for epoch in range(1, epoch_limit+1):
        if args.prediction_distillation_weight and epoch==args.prediction_distillation_start_epoch:
            from weather.prediction_distillation import prepare_cached_target_probabilities
            enhancement["prediction_distillation"]=prepare_cached_target_probabilities(model,target_eval,device,
                output,epoch,args.prediction_distillation_sharpening,args.seed)
            from weather.encoder_schedule import scale_image_lr_for_cached_distillation
            enhancement["prediction_distillation"]["encoder_LR_phase"]=scale_image_lr_for_cached_distillation(
                optimizer,scheduler,args.prediction_distillation_encoder_lr_scale)
            write_json(output/"prediction_distillation_audit.json",enhancement["prediction_distillation"])
        for sampler in samplers:
            sampler.epoch = epoch-1
        model.train()
        from weather.encoder_schedule import apply_encoder_warmup
        encoder_policy = apply_encoder_warmup(model, epoch, args.encoder_warmup_freeze_epochs)
        from weather.encoder_schedule import lower_image_lr_after_source_warmup
        encoder_policy.update(lower_image_lr_after_source_warmup(optimizer, scheduler, epoch,
            args.adaptation_warmup_epochs, args.image_adaptation_lr_scale))
        encoder_policy["phase"] = ("source_supervised" if epoch <= args.adaptation_warmup_epochs else "domain_adaptation")
        bn_policy = apply_batchnorm_policy(model, epoch, args.freeze_bn_after_epochs, args.freeze_bn_scope)
        event(output, "encoder_trainability.jsonl", {"epoch": epoch, **encoder_policy})
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        totals = {key: 0.0 for key in ("classification", "source_ce", "transfer", "weighted_transfer")}
        if args.source_feature_mixup_weight:
            totals.update(source_feature_mixup_loss=0.,source_feature_mixup_correction=0.)
        if alignment_weight:totals.update(matched_alignment_loss=0.,weighted_matched_alignment=0.)
        if args.prediction_distillation_weight:
            totals.update(prediction_distillation_loss=0.,weighted_prediction_distillation=0.)
        if args.source_crossmodal_weight or args.target_crossmodal_weight or args.source_prototype_weight:
            totals.update({key:0. for key in ("source_crossmodal_loss","target_crossmodal_loss","weighted_source_crossmodal","weighted_target_crossmodal")})
        if args.target_crossmodal_weight and args.target_crossmodal_objective=="vicreg":
            totals.update({"target_vicreg_"+key:0. for key in ("invariance","variance","covariance","VIS_mean_std","IR_mean_std")})
        if args.source_prototype_weight:totals.update(source_prototype_loss=0.,weighted_source_prototype=0.)
        if args.domain_confusion_weight:totals.update(domain_confusion_loss=0.,weighted_domain_confusion=0.)
        if object_enabled:totals.update(source_object_ce=0.,source_object_consistency=0.,weighted_source_object=0.)
        seen_source, seen_target = set(), set()
        source_correct = source_seen = target_seen = steps = 0
        target_matched_correct=0
        for step, (sb, tb) in enumerate(zip(*loaders), 1):
            seen_source.update(sb["path"])
            seen_target.update(tb["path"])
            if "label" in tb:
                raise RuntimeError("Target batch must not contain labels")
            matched_truth=(matching_plan.assert_batch(sb["path"],tb["path"],sb["label"].tolist()) if guided and not fewshot else None)
            if fewshot:
                tb['alignment_labels']=torch.tensor(matching_plan.labels_for(tb['path']),dtype=torch.long)
            sb, tb = move(sb, device), move(tb, device)
            from weather.fourier_style import augment_source_style
            sb, input_augmentation = augment_source_style(sb, tb, args.fda_beta, args.fda_probability, fda_generator)
            from weather.source_zoom import augment_source_zoom
            sb, zoom_record = augment_source_zoom(sb, args.source_zoom_min_area, args.source_zoom_probability, zoom_generator)
            input_augmentation.update(zoom_record)
            from weather.weather_augmentation import augment_source_weather
            sb, weather_record = augment_source_weather(sb, args.target, args.source_weather_probability, weather_generator)
            input_augmentation.update(weather_record)
            if args.source_ir_statistics_probability:
                from weather.infrared_statistics import augment_source_ir_statistics
                sb,ir_statistics_record=augment_source_ir_statistics(sb,tb,args.source_ir_statistics_probability,
                    args.source_ir_statistics_strength,ir_statistics_generator)
                input_augmentation.update(ir_statistics_record)
            if args.source_ir_inversion_probability:
                from weather.infrared_polarity import augment_source_ir_polarity
                sb,polarity_record=augment_source_ir_polarity(sb,args.source_ir_inversion_probability,polarity_generator)
                input_augmentation.update(polarity_record)
            from weather.source_aspect import augment_source_aspect
            sb, aspect_record = augment_source_aspect(sb, args.source_aspect_min, args.source_aspect_max,
                                                    args.source_aspect_probability, aspect_generator)
            input_augmentation.update(aspect_record)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.objective(sb, tb, update, args.epochs*args.steps_per_epoch, args)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, update {update}")
            if step == len(loaders[0]) or (args.max_train_batches and step == args.max_train_batches):
                gradients = gradient_diagnostics(model, details["classification"], details["weighted_transfer"])
                gradients["VIS"], gradients["IR"] = gradients.pop("RGB"), gradients.pop("NIR")
                gradients.pop("AIS")
                separate={}
                if args.feature_calibration:
                    separate={"core":gradient_diagnostics(model,details["classification"],details["weighted_core_transfer"]),
                              "cluster_teacher":gradient_diagnostics(model,details["classification"],details["weighted_cluster_teacher"])}
                if args.source_crossmodal_weight or args.target_crossmodal_weight or args.source_prototype_weight:
                    separate={"core":gradient_diagnostics(model,details["source_ce"],details["weighted_core_transfer"]),
                        "source_crossmodal":gradient_diagnostics(model,details["source_ce"],details["weighted_source_crossmodal"]),
                        "target_crossmodal":gradient_diagnostics(model,details["source_ce"],details["weighted_target_crossmodal"])}
                    if args.source_prototype_weight:
                        separate["source_prototypes"]=gradient_diagnostics(model,details["source_ce"],details["weighted_source_prototype"])
                if args.domain_confusion_weight:
                    if "core" not in separate:
                        separate["core"]=gradient_diagnostics(model,details["source_ce"],details["weighted_core_transfer"])
                    separate["domain_confusion"]=gradient_diagnostics(model,details["source_ce"],details["weighted_domain_confusion"])
                if object_enabled:
                    if "core" not in separate:separate["core"]=gradient_diagnostics(model,details["source_ce"],details["weighted_core_transfer"])
                    separate["source_object_views"]=gradient_diagnostics(model,details["source_ce"],details["weighted_source_object"])
                if args.prediction_distillation_weight:
                    if "core" not in separate:separate["core"]=gradient_diagnostics(model,details["source_ce"],details["weighted_core_transfer"])
                    separate["prediction_distillation"]=gradient_diagnostics(model,details["source_ce"],details["weighted_prediction_distillation"])
                if args.target_representation_anchor_weight:
                    if "core" not in separate:separate["core"]=gradient_diagnostics(model,details["source_ce"],details["weighted_core_transfer"])
                    separate["target_representation_anchor"]=gradient_diagnostics(model,details["source_ce"],details["weighted_target_representation_anchor"])
                if args.source_feature_mixup_weight:
                    if "core" not in separate:separate["core"]=gradient_diagnostics(model,details["source_ce"],details["weighted_core_transfer"])
                    separate["source_feature_mixup_correction"]=gradient_diagnostics(model,details["source_ce"],details["source_feature_mixup_correction"])
                if alignment_weight:
                    if 'core' not in separate:
                        separate['core']=gradient_diagnostics(model,details['source_ce'],details['weighted_core_transfer'])
                    separate['matched_class_alignment']=gradient_diagnostics(model,details['source_ce'],details['weighted_matched_alignment'])
                event(output, "gradients.jsonl", {"epoch": epoch, "update": update+1, "gradients": gradients,"enhancement_components":separate})
            norms = feature_norms(model.last_features, len(sb["label"]), names=("VIS", "IR"))
            target_diagnostics = None
            if args.log_feature_moments:
                from weather.target_batch_diagnostics import target_batch_diagnostics
                target_diagnostics = target_batch_diagnostics(details["target_logits"],tb["path"],
                    model.mcc.temperature if args.method=="mcc" else None)
            loss.backward()
            unclipped_norm = (float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm,
                                                               error_if_nonfinite=True))
                              if args.gradient_clip_norm else None)
            sampled = None
            if args.log_parameter_updates and (step == len(loaders[0]) or (args.max_train_batches and step == args.max_train_batches)):
                from weather.encoder_schedule import capture_parameter_sample
                sampled = capture_parameter_sample(model)
            optimizer.step()
            if sampled is not None:
                from weather.encoder_schedule import parameter_update_record
                event(output, "parameter_updates.jsonl", {"epoch": epoch, "update": update+1,
                      "sampled_parameters": parameter_update_record(sampled, optimizer)})
            source_correct += int((details["source_logits"].argmax(1) == sb["label"]).sum())
            if guided and not fewshot:
                target_matched_correct+=int((details["target_logits"].detach().argmax(1).cpu()
                                             ==torch.tensor(matched_truth)).sum())
            source_seen += len(sb["label"])
            target_seen += len(tb["rgb"])
            update += 1
            steps += 1
            values = {key: float(details[key].detach()) for key in totals}
            for key in totals:
                totals[key] += values[key]
            event(output, "updates.jsonl", {"epoch": epoch, "update": update, **values,
                  "transfer_weight": details["transfer_weight"], "grl_coefficient": details["grl_coefficient"],
                  **({**supervision,
                      "same_true_class_at_every_position":True,"source_true_classes":sb["label"].detach().cpu().tolist(),
                      "target_true_classes_for_sampling":matched_truth,"target_classification_CE_used":False} if guided and not fewshot else {}),
                  **({**supervision,'restricted_target_labels_only':True,
                      'native_batches_true_class_matched':False,
                      'source_true_classes':sb['label'].detach().cpu().tolist(),
                      'target_known_alignment_labels':tb['alignment_labels'].detach().cpu().tolist(),
                      'target_batch_paths':tb['path'],
                      'target_classification_CE_used':False} if fewshot else {}),
                  "adaptation_graph_in_loss": details["adaptation_graph_in_loss"],
                  "core_adaptation_graph_in_loss":details["core_adaptation_graph_in_loss"],
                  "domain_accuracy": details["domain_accuracy"], "feature_norms": norms,
                  "entropy_grl_coefficient": details["entropy_grl_coefficient"],
                  "auxiliary_source_ce": float(details["auxiliary_source_ce"].detach()),
                  "auxiliary_source_weight": details["auxiliary_source_weight"],
                  "weighted_core_transfer":float(details["weighted_core_transfer"].detach()),
                  "cluster_teacher_loss":float(details["cluster_teacher_loss"].detach()),
                  "weighted_cluster_teacher":float(details["weighted_cluster_teacher"].detach()),
                  "cluster_teacher_selected":details["cluster_teacher_selected"],
                  "prediction_distillation_loss":float(details["prediction_distillation_loss"].detach()),
                  "weighted_prediction_distillation":float(details["weighted_prediction_distillation"].detach()),
                  "prediction_distillation_ready":details["prediction_distillation_ready"],
                  **({"source_feature_mixup_details":details["source_feature_mixup_details"]} if args.source_feature_mixup_weight else {}),
                  **({"matched_alignment_loss":float(details['matched_alignment_loss'].detach()),
                      "weighted_matched_alignment":float(details['weighted_matched_alignment'].detach()),
                      "matched_alignment_details":details['matched_alignment_details']} if alignment_weight else {}),
                  "target_representation_anchor_loss":float(details["target_representation_anchor_loss"].detach()),
                  "weighted_target_representation_anchor":float(details["weighted_target_representation_anchor"].detach()),
                  "target_representation_cosines":details["target_representation_cosines"],
                  "source_crossmodal_loss":float(details["source_crossmodal_loss"].detach()),
                  "target_crossmodal_loss":float(details["target_crossmodal_loss"].detach()),
                  "target_crossmodal_objective":args.target_crossmodal_objective,
                  **({"target_vicreg_"+k:float(v.detach()) for k,v in details["target_crossmodal_components"].items()}
                     if details["target_crossmodal_components"] is not None else {}),
                  "weighted_source_crossmodal":float(details["weighted_source_crossmodal"].detach()),
                  "weighted_target_crossmodal":float(details["weighted_target_crossmodal"].detach()),
                  "source_pair_cosine":float(details["source_pair_cosine"]) if details["source_pair_cosine"] is not None else None,
                  "target_pair_cosine":float(details["target_pair_cosine"]) if details["target_pair_cosine"] is not None else None,
                  "source_prototype_loss":float(details["source_prototype_loss"].detach()),
                  "weighted_source_prototype":float(details["weighted_source_prototype"].detach()),
                  "source_prototype_classes_seen":details["source_prototype_classes_seen"],
                  "domain_confusion_loss":float(details["domain_confusion_loss"].detach()),
                  "weighted_domain_confusion":float(details["weighted_domain_confusion"].detach()),
                  "source_object_ce":float(details["source_object_ce"].detach()),
                  "source_object_consistency":float(details["source_object_consistency"].detach()),
                  "weighted_source_object":float(details["weighted_source_object"].detach()),
                  "source_object_correct":details["source_object_correct"],
                  "source_object_roi_samples":int(sb["source_object_is_roi"].sum()) if object_enabled else 0,
                  "target_feature_style":model.target_feature_style.diagnostics() if args.target_feature_style_probability else None,
                  "gradient_norm_before_clipping": unclipped_norm,
                  "input_augmentation": input_augmentation,
                  "source_batch_unique_samples": len(set(sb["path"])),
                  "target_batch_diagnostics": target_diagnostics,
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
                      f"weight={details['transfer_weight']:.4f} protocol={supervision['protocol']} "
                      f"target_labels_used={guided}", flush=True)
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
              "source_train_accuracy": source_correct/source_seen, **supervision,
              **({"matched_target_train_accuracy":target_matched_correct/target_seen,
                  "label_guided_batch_checks":steps,"all_batch_positions_class_matched":True} if guided and not fewshot else {}),
              **({'restricted_target_labels_only':True,'all_batch_positions_class_matched':False,
                  'labelled_target_samples':len(matching_plan.target_lookup)} if fewshot else {}),
              "training_time_s": times[-1]})
        normalization_record = {"epoch": epoch, **bn_policy}
        if args.domain_running_stats:
            from weather.domain_running_stats import statistics_summary
            normalization_record["domain_running_statistics"] = statistics_summary(model)
        event(output, "normalization_policy.jsonl", normalization_record)
        for ds in (source_eval, target_eval):
            _,diagnostic_records=evaluate(model, ds, device, epoch, output, args.eval_batch_size, event_prefix="WEATHER_EVALUATION")
            if guided and ds.partition=='target_train':
                if fewshot:
                    diagnostic_records=[r for r in diagnostic_records if r['sample_id'] in matching_plan.target_lookup]
                fit=labelled_training_diagnostic(diagnostic_records,matching_plan.target_lookup,classes)
                event(output,'target_train_fit.jsonl',{'epoch':epoch,**fit})
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
        if better_metrics(metrics,peak_metrics):
            peak_metrics,peak_epoch,stale=dict(metrics),epoch,0
        else:
            stale+=1
        if epoch>=args.selection_min_epoch and selection_improved(args.selection_metric,metrics,val_details['cross_entropy'],best_metrics,best_validation_ce):
            best_metrics, best_records, best_epoch = dict(metrics), records, epoch
            best_validation_ce=val_details['cross_entropy']
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        control = validation_control.observe(val_details["cross_entropy"], epoch, optimizer)
        if scheduler is not None and not schedule_per_update:
            step_scaled_scheduler(scheduler, optimizer, validation_control.lr_scale)
        stopping_stale = validation_control.bad_epochs if args.early_stopping_monitor == "cross_entropy" else stale
        stop = bool(args.early_stopping_patience and epoch >= args.early_stopping_min_epochs
                    and stopping_stale >= args.early_stopping_patience)
        control.update(epoch=epoch, selected_ACC_epoch=peak_epoch,selected_model_epoch=best_epoch,
                       selection_metric=args.selection_metric,ACC_F1_bad_epochs=stale,
                       early_stop=stop, next_learning_rates=[g["lr"] for g in optimizer.param_groups])
        event(output, "validation_control.jsonl", control)
        print("WEATHER_VALIDATION_CONTROL: "+json.dumps(control), flush=True)
        if stop:
            stop_reason = "validation_"+args.early_stopping_monitor+"_patience"
            break
    if args.source_branch_probe:
        from weather.branch_probe import source_branch_probe
        final_probe = source_branch_probe(model, source_eval, validation, device, args.eval_batch_size, args.seed)
        final_probe.update(epoch=len(history), stage="last_epoch_before_best_restore")
        write_json(output/"source_branch_probe_final.json", final_probe)
    model.load_state_dict(best_state, strict=True)
    sensitivity = None
    if args.modality_diagnostics:
        sensitivity = validation_modality_diagnostics(model, validation, device, best_epoch, output,
                                                       batch_size=args.eval_batch_size)
        if sensitivity["full"] != best_metrics:
            raise RuntimeError("Restored validation metrics differ from selected metrics")
        model.load_state_dict(best_state, strict=True)
    write_json(output/"selected_target_val_predictions.json", best_records)
    selected_features=None
    if args.cache_best_features:
        from weather.feature_cache import cache_selected_features
        selected_features=cache_selected_features(model,(source_eval,target_eval,validation),device,output,best_epoch,args.eval_batch_size)
    if args.source_branch_probe:
        from weather.branch_probe import source_branch_probe
        probe = source_branch_probe(model, source_eval, validation, device, args.eval_batch_size, args.seed)
        probe["selected_epoch"] = best_epoch
        write_json(output/"source_branch_probe.json", probe)
    complexity = profile(model, validation, device, times, peak_reserved)
    if args.prediction_teacher_weight:
        complexity["teacher_preparation_time_s"]=enhancement["prediction_teacher"]["preparation_time_s"]
        complexity["pseudo_probability_preparation_time_s"]=enhancement.get("prediction_distillation",{}).get("preparation_time_s",0.)
        complexity["optimizer_and_teacher_preparation_time_s"]=(complexity["total_train_time_s"]
            +complexity["teacher_preparation_time_s"]+complexity["pseudo_probability_preparation_time_s"])
    if args.target_representation_anchor_weight:
        complexity["target_representation_preparation_time_s"]=enhancement["target_representation_anchor"]["preparation_time_s"]
        complexity["optimizer_and_representation_preparation_time_s"]=(complexity["total_train_time_s"]
            +complexity["target_representation_preparation_time_s"])
    for key in ("total_params", "trainable_params", "inference_GFLOPs", "total_train_time_s", "train_epoch_mean_s", "latency_mean_ms", "throughput_samples_s"):
        if not math.isfinite(complexity[key]) or complexity[key] <= 0:
            raise RuntimeError(f"Invalid complexity metric: {key}")
    if device.type == "cuda" and not peak_reserved:
        raise RuntimeError("Missing GPU training memory measurement")
    test_metrics=None
    if heldout is not None:
        test_metrics,test_records=evaluate(model,heldout,device,best_epoch,output,args.eval_batch_size,
                                            event_prefix='FINAL_HELDOUT_TEST_ONCE')
        write_json(output/'heldout_test_predictions.json',test_records)
    diagnostic_datasets=(source_eval,target_eval,validation)+((heldout,) if heldout is not None else ())
    checked = validate_diagnostics(output, diagnostic_datasets, len(history), best_epoch,
                                   num_classes=classes, modalities=("VIS", "IR"))
    completeness = {"status": "PASS", "epochs": len(history), "selected_epoch": best_epoch,
                    **supervision,
                    "target_test_available": heldout is not None, "selection_partition": "target_val",
                    'test_evaluations':int(heldout is not None),'selection_metric':args.selection_metric,
                    "prediction_rows_checked": checked, "partial_training_smoke": bool(args.max_train_batches)}
    write_json(output/"completeness.json", completeness)
    result = {"method": args.method, "target": args.target, "seed": args.seed, **supervision,
              'configuration_tag':args.configuration_tag,
              "core": "vds2raw.adaptation.AdaptationModel", "modalities": ["VIS", "IR"],
              **({"label_guided_sampling":matching_plan.metadata(alignment_weight)} if guided else {}),
              "optimizer": type(optimizer).__name__,
              "selected_feature_cache":selected_features,
              "enhancement":enhancement,
              "mcc_temperature": args.mcc_temperature if args.method=="mcc" else None,
              "weather_profile": args.weather_profile, "domain_forward": args.domain_forward,
              "image_backbone": args.image_backbone, "freeze_image_encoders": getattr(model, "freeze_image_encoders", args.freeze_image_encoders),
              "ir_backbone": args.ir_backbone or args.image_backbone,
              "encoder_precision": args.encoder_precision,
              "ir_training_scope": args.ir_training_scope,
              "infrared_pretraining": getattr(model.nir_encoder, "provenance", None),
              "image_adaptation_lr_scale": args.image_adaptation_lr_scale,
              "encoder_training_scope": getattr(model, "encoder_training_scope", "all"),
              "unit_method_features": args.unit_method_features,
              "classifier_layout": args.classifier_layout,
              "selected_epoch": best_epoch, "completed_epochs": len(history), "optimizer_updates": update,
              'selection_metric':args.selection_metric,'selected_validation_CE':best_validation_ce,
              'selection_min_epoch':args.selection_min_epoch,
              'peak_target_val':peak_metrics,'peak_target_val_epoch':peak_epoch,
              'target_test':test_metrics,'test_evaluations':int(heldout is not None),
              'heldout_evaluation':audit.get('heldout_evaluation'),
              "per_domain_batch_sizes": {"source":args.batch_size,"target":args.target_batch_size or args.batch_size},
              "stop_reason": stop_reason, "planned_optimizer_updates": args.epochs*args.steps_per_epoch,
              "freeze_bn_after_epochs": args.freeze_bn_after_epochs, "modality_diagnostics": sensitivity,
              "freeze_bn_scope": args.freeze_bn_scope, "feature_normalization": args.feature_normalization,
              "domain_running_stats": args.domain_running_stats,
              "auxiliary_source_weight": args.auxiliary_source_weight,
              "screening_only": screening, "planned_epochs": args.epochs,
              "input_augmentation": {"FDA_beta": args.fda_beta, "FDA_probability": args.fda_probability,
                                     "source_zoom_min_area": args.source_zoom_min_area,
                                     "source_zoom_probability": args.source_zoom_probability,
                                     "source_weather_probability": args.source_weather_probability,
                                     "source_ir_inversion_probability":args.source_ir_inversion_probability,
                                     "source_ir_statistics_probability":args.source_ir_statistics_probability,
                                     "source_ir_statistics_strength":args.source_ir_statistics_strength},
              "source_aspect": {"minimum": args.source_aspect_min, "maximum": args.source_aspect_max,
                                "probability": args.source_aspect_probability},
              "evaluation_protocol": evaluation_protocol, "target_test_available": heldout is not None,
              "target_val": best_metrics, "complexity": complexity,
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
    p.add_argument("--feature-calibration",action="store_true",help="Explicit enhanced variant: training-only statistics plus source-mapped target cluster initialization")
    p.add_argument("--calibration-components",type=int,default=32)
    p.add_argument("--calibration-shrinkage",type=float,default=.1)
    p.add_argument("--calibration-adapter-scale",type=float,default=.1)
    p.add_argument("--cluster-teacher-weight",type=float,default=0.)
    p.add_argument("--cluster-teacher-confidence",type=float,default=.4)
    p.add_argument("--cluster-teacher-temperature",type=float,default=.1)
    p.add_argument("--source-crossmodal-weight",type=float,default=0.)
    p.add_argument("--source-prototype-weight",type=float,default=0.)
    p.add_argument("--domain-confusion-weight",type=float,default=0.,
                   help="Opt-in additional encoder-only uniform-domain loss for saturated DANN/CDAN discriminators")
    p.add_argument("--target-feature-style-probability",type=float,default=0.,
                   help="Explicit enhancement: source feature statistics mixed with unlabelled target references during joint training")
    p.add_argument("--target-feature-style-source-max",type=float,default=.5)
    p.add_argument("--source-object-manifest",default=None)
    p.add_argument("--source-object-ce-weight",type=float,default=0.)
    p.add_argument("--source-object-consistency-weight",type=float,default=0.)
    p.add_argument("--cache-training-images",action="store_true")
    p.add_argument("--target-crossmodal-weight",type=float,default=0.)
    p.add_argument("--prediction-teacher-weight",type=float,default=0.)
    p.add_argument("--prediction-distillation-weight",type=float,default=0.)
    p.add_argument("--target-representation-anchor-weight",type=float,default=0.,
                   help="Opt-in training cosine loss to initial target_train encoder features; no target labels or added inference encoder")
    p.add_argument("--prediction-distillation-encoder-lr-scale",type=float,default=1.)
    p.add_argument("--prediction-distillation-start-epoch",type=int,default=2)
    p.add_argument("--prediction-distillation-sharpening",type=float,default=.5)
    p.add_argument("--prediction-teacher-temperature",type=float,default=1.)
    p.add_argument("--prediction-student-temperature",type=float,default=1.)
    p.add_argument("--prediction-teacher-weights",default=None)
    p.add_argument("--target-crossmodal-objective",choices=("contrastive","vicreg"),default="contrastive",
                   help="Opt-in negative-free target-pair regularizer; original method cores remain separate")
    p.add_argument("--crossmodal-temperature",type=float,default=.2)
    p.add_argument("--crossmodal-projection-dim",type=int,default=0,help="Opt-in training-only nonlinear projection; required for unequal modality widths")
    p.add_argument("--crossmodal-stop-gradient",choices=("none","vis","ir"),default="none")
    p.add_argument("--source-ir-inversion-probability",type=float,default=0.)
    p.add_argument("--cache-best-features",action="store_true")
    p.add_argument("--cluster-association",choices=("source_votes","source_prototypes","source_neighbors","source_trimmed"),default="source_votes")
    p.add_argument("--cluster-neighbors",type=int,default=3)
    p.add_argument("--cluster-trim-fraction",type=float,default=.25)
    p.add_argument("--cluster-crossmodal-association",action="store_true")
    p.add_argument("--cluster-association-vis-weight",type=float,default=.75)
    p.add_argument("--cluster-initialization",choices=("kmeans++","source_prototypes"),default="kmeans++")
    p.add_argument("--auxiliary-source-weight", type=float, default=None)
    p.add_argument("--source-feature-mixup-weight",type=float,default=0.,help="Opt-in convex source main CE mixture; target DA loss unchanged")
    p.add_argument("--source-feature-mixup-alpha",type=float,default=.4)
    p.add_argument("--target-class-matching",choices=("none","target_truth"),default="none",
                   help="Explicit label-guided protocol, not strict UDA; sampling and optional class feature alignment")
    p.add_argument('--matched-alignment-weight',type=float,default=0.,help='Explicit supervised class feature alignment; default off')
    p.add_argument('--matched-alignment-margin',type=float,default=.2)
    p.add_argument('--matched-alignment-ramp-epochs',type=int,default=3)
    p.add_argument('--source-anchor-ema-momentum',type=float,default=None,
                   help='Opt-in source-only cross-batch EMA anchors; target labels remain restricted to the fixed few-shot budget')
    p.add_argument('--target-labels-per-class',type=int,default=0,
                   help='Positive fixed label budget; only selected targets enter extra alignment, all targets enter native DA')
    p.add_argument('--selection-metric',choices=('accuracy_f1','cross_entropy'),default='accuracy_f1')
    p.add_argument('--configuration-tag',default='reference',help='Recorded experiment label; does not transform optimizer settings')
    p.add_argument('--selection-min-epoch',type=int,default=1)
    p.add_argument('--holdout-list',type=Path,default=None,help='Optional fixed target_test metadata; evaluate once after validation selection')
    p.add_argument("--source-class-balance", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gradient-clip-norm", type=float, default=None)
    p.add_argument("--modality-diagnostics", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--source-branch-probe", action="store_true",
                   help="Separate diagnostic: fit branch linear probes using source labels only")
    p.add_argument("--encoder-warmup-freeze-epochs", type=int, default=0,
                   help="Optional: keep encoder weights fixed before joint image/domain adaptation")
    p.add_argument("--log-parameter-updates", action="store_true",
                   help="Record actual optimizer changes in representative encoder/head/gate parameters")
    p.add_argument("--domain-forward", choices=("reference", "joint"), default=None)
    p.add_argument("--log-feature-moments", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--save-dir", default=str(ROOT/"runs/weather_vds_core"))
    p.add_argument("--weights", default=str(ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"))
    p.add_argument("--image-backbone", choices=("resnet18", "convnext_small"), default="resnet18")
    p.add_argument("--checkpoint-vis-blocks",action="store_true",
                   help="Optional activation checkpointing inside the original ConvNeXt; keeps its module structure and outputs")
    p.add_argument("--ir-backbone", choices=("unip_tiny", "unip_small"), default=None,
                   help="Opt-in official infrared pretraining; existing defaults stay unchanged")
    p.add_argument("--ir-weights", default=None)
    p.add_argument("--ir-feature-mode", choices=("patch_mean", "cls", "qk_mean", "last4", "cls_attention"), default="patch_mean")
    p.add_argument("--encoder-precision", choices=("fp32", "bf16"), default="fp32",
                   help="Optional CUDA BF16 encoder forward; method heads and all losses remain FP32")
    p.add_argument("--ir-training-scope", choices=("inherit","all_used_layers"), default="inherit")
    p.add_argument("--freeze-image-encoders", action="store_true", help="Freeze encoder weights and stochastic layers; train method bottleneck/heads/gates")
    p.add_argument("--encoder-training-scope", choices=("all", "frozen", "ir_last_stage", "last_stages"), default=None)
    p.add_argument("--image-lr-override", type=float, default=None, help="Absolute LR for trainable encoder parameters; keeps the method head LR separate")
    p.add_argument("--ir-lr-override", type=float, default=None, help="Optional separate IR encoder LR; does not alter method-head LR")
    p.add_argument("--adversarial-lr-override", type=float, default=None,
                   help="Optional separate discriminator/MDD adversarial-head LR; the method loss and classifier LR stay unchanged")
    p.add_argument("--image-adaptation-lr-scale", type=float, default=1.,
                   help="Optional encoder LR reduction after source-supervision warmup; keeps head LR unchanged")
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
    p.add_argument("--source-weather-probability", type=float, default=0.,
                   help="Explicit source-only gray/exposure/fog/blur/IR-contrast variant; evaluation stays unchanged")
    p.add_argument("--source-ir-statistics-probability", type=float, default=0.,
                   help="Opt-in IR-only source intensity transfer from unlabelled target_train; validation unchanged")
    p.add_argument("--source-ir-statistics-strength", type=float, default=1.)
    p.add_argument("--source-aspect-min", type=float, default=1.)
    p.add_argument("--source-aspect-max", type=float, default=1.)
    p.add_argument("--source-aspect-probability", type=float, default=.75)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--target-batch-size", type=int, default=None,
                   help="Optional target-only batch size, e.g. for MCC class-confusion estimation; MDD requires equal batches")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float)
    p.add_argument("--optimizer", choices=("reference", "adam", "adamw", "sgd"), default="reference")
    p.add_argument("--trade-off", type=float)
    p.add_argument("--mcc-temperature",type=float,default=2.5,
                   help="MCC softmax temperature; default preserves the existing released loss setting")
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
    p.add_argument("--cuda-memory-fraction", type=float, default=1.,help="Optional per-worker allocator cap for local resource-controlled screening")
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
    if (not math.isfinite(args.source_feature_mixup_weight) or not 0<=args.source_feature_mixup_weight<1
            or not math.isfinite(args.source_feature_mixup_alpha) or args.source_feature_mixup_alpha<=0):
        raise ValueError("Invalid source feature mixup fraction/alpha")
    if not 0 <= args.fda_beta <= .25 or not 0 <= args.fda_probability <= 1:
        raise ValueError("Invalid FDA beta/probability")
    if not .1 <= args.source_zoom_min_area <= 1 or not 0 <= args.source_zoom_probability <= 1:
        raise ValueError("Invalid source center-zoom area/probability")
    if args.encoder_warmup_freeze_epochs < 0:
        raise ValueError("Encoder warm-up freeze must be nonnegative")
    run(args)
