"""Bounded seed42 weather screens with explicit reference-config overrides."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot


def recorded_arguments_to_cli(arguments):
    """Serialize recognized parser arguments, preserving a real reference run."""
    from weather.train_vds_core import parser
    command=[]
    for action in parser()._actions:
        if action.dest not in arguments or action.dest=="help" or not action.option_strings:continue
        value=arguments[action.dest]
        if value is None:continue
        option=action.option_strings[0]
        if isinstance(action,argparse.BooleanOptionalAction):
            command.append(next(opt for opt in action.option_strings if opt.startswith("--no-")==bool(not value)))
        elif isinstance(action,argparse._StoreTrueAction):
            if value:command.append(option)
        elif isinstance(action,argparse._StoreFalseAction):
            if not value:command.append(option)
        else:
            command.append(option)
            command.extend(str(x) for x in value) if isinstance(value,(list,tuple)) else command.append(str(value))
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--targets",nargs="+",choices=("rainy","foggy","dark","backlight"),default=["foggy","dark","rainy"])
    p.add_argument("--methods",nargs="+",choices=("dann","deepcoral","cdan","mdd","mcc"),default=["mcc"])
    p.add_argument("--source-weights",nargs="+",type=float,default=[.1,.3])
    p.add_argument("--target-weight",type=float,default=.05)
    p.add_argument("--target-objective",choices=("contrastive","vicreg"),default="contrastive")
    p.add_argument("--prediction-teacher-weight",type=float,default=0.)
    p.add_argument("--prediction-distillation-weight",type=float,default=0.)
    p.add_argument("--target-representation-anchor-weight",type=float,default=None)
    p.add_argument("--prediction-distillation-encoder-lr-scale",type=float,default=1.)
    p.add_argument("--prediction-distillation-start-epoch",type=int,default=2)
    p.add_argument("--prediction-distillation-sharpening",type=float,default=.5)
    p.add_argument("--prediction-teacher-temperature",type=float,default=1.)
    p.add_argument("--prediction-student-temperature",type=float,default=1.)
    p.add_argument("--source-prototype-weight",type=float,default=0.)
    p.add_argument("--domain-confusion-weight",type=float,default=0.)
    p.add_argument("--target-feature-style-probability",type=float,default=0.)
    p.add_argument("--target-feature-style-source-max",type=float,default=.5)
    p.add_argument("--trade-off",type=float,default=None)
    p.add_argument("--learning-rate",type=float,default=None,help="Explicit base LR override; preserve separate encoder/adversary overrides")
    p.add_argument("--adaptation-warmup-epochs",type=int,default=None)
    p.add_argument("--adaptation-ramp-epochs",type=int,default=None)
    p.add_argument("--image-adaptation-lr-scale",type=float,default=None)
    p.add_argument("--reference-config",type=Path,default=None)
    p.add_argument("--classifier-layout",choices=("reference","linear"),default=None)
    p.add_argument("--auxiliary-source-weight",type=float,default=None)
    p.add_argument("--source-feature-mixup-weight",type=float,default=None)
    p.add_argument("--source-feature-mixup-alpha",type=float,default=None)
    p.add_argument("--ir-backbone",choices=("unip_tiny","unip_small"),default=None)
    p.add_argument("--ir-weights",type=Path,default=None)
    p.add_argument("--ir-feature-mode",choices=("patch_mean","cls","qk_mean","last4","cls_attention"),default=None)
    p.add_argument("--ir-training-scope",choices=("inherit","all_used_layers"),default=None)
    p.add_argument("--source-object-manifest",type=Path,default=None)
    p.add_argument("--source-object-ce-weight",type=float,default=0.)
    p.add_argument("--source-object-consistency-weight",type=float,default=0.)
    p.add_argument("--projection-dim",type=int,default=0)
    p.add_argument("--stop-gradient",choices=("none","vis","ir"),default=None,
        help="Preserve reference setting unless explicitly overridden; no reference defaults to none")
    p.add_argument("--source-ir-inversion-probability",type=float,default=0.)
    p.add_argument("--source-ir-statistics-probability",type=float,default=0.)
    p.add_argument("--source-ir-statistics-strength",type=float,default=1.)
    p.add_argument("--initial-vis-weight",type=float,default=None)
    p.add_argument("--fda-beta",type=float,default=None)
    p.add_argument("--source-weather-probability",type=float,default=None)
    p.add_argument("--cache-best-features",action="store_true")
    p.add_argument("--scope",choices=("all","last_stages","ir_last_stage"),default=None,
        help="Explicit override; otherwise preserve a reference run's encoder scope")
    p.add_argument("--checkpoint-vis-blocks",action=argparse.BooleanOptionalAction,default=None)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--stop-after-epochs",type=int,default=0,help="Observe an explicit prefix while preserving full LR/GRL clocks")
    p.add_argument("--minimum-epochs",type=int,default=30)
    p.add_argument("--parallel",type=int,choices=(1,2),default=2)
    p.add_argument("--plan-only",action="store_true",help="Write exact commands and reference differences without launching training")
    args=p.parse_args()
    if any(len(x)!=len(set(x)) for x in (args.targets,args.methods,args.source_weights)):p.error("Duplicate trials")
    values=args.source_weights+[args.target_weight,args.source_prototype_weight,args.domain_confusion_weight,
        args.source_object_ce_weight,args.source_object_consistency_weight]+([] if args.trade_off is None else [args.trade_off])
    if (not all(math.isfinite(x) and x>=0 for x in values) or args.epochs<1 or args.minimum_epochs>args.epochs
            or args.projection_dim<0 or args.projection_dim==1 or args.stop_after_epochs<0
            or args.stop_after_epochs>args.epochs):p.error("Invalid trial settings")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate<=0):
        p.error("Learning rate must be finite and positive")
    if any(value is not None and value<0 for value in
           (args.adaptation_warmup_epochs,args.adaptation_ramp_epochs)):
        p.error("Adaptation warmup and ramp must be nonnegative")
    if args.image_adaptation_lr_scale is not None and (not math.isfinite(args.image_adaptation_lr_scale)
            or not 0<args.image_adaptation_lr_scale<=1):
        p.error("Post-warmup image LR factor must lie in (0,1]")
    if args.auxiliary_source_weight is not None and (not math.isfinite(args.auxiliary_source_weight) or args.auxiliary_source_weight<0):
        p.error("Auxiliary source weight must be finite and nonnegative")
    if args.source_feature_mixup_weight is not None and (not math.isfinite(args.source_feature_mixup_weight) or not 0<=args.source_feature_mixup_weight<1):
        p.error("Source mixup fraction must lie in [0,1)")
    if args.source_feature_mixup_alpha is not None and (not math.isfinite(args.source_feature_mixup_alpha) or args.source_feature_mixup_alpha<=0):
        p.error("Source mixup alpha must be positive")
    if (not math.isfinite(args.prediction_teacher_weight) or not 0<=args.prediction_teacher_weight<1
            or not all(math.isfinite(x) and x>0 for x in
                (args.prediction_teacher_temperature,args.prediction_student_temperature))):
        p.error("Invalid prediction blend settings")
    if (not math.isfinite(args.prediction_distillation_weight) or args.prediction_distillation_weight<0
            or not math.isfinite(args.prediction_distillation_sharpening) or args.prediction_distillation_sharpening<=0
            or args.prediction_distillation_start_epoch<1
            or args.prediction_distillation_weight and not args.prediction_teacher_weight):
        p.error("Invalid cached teacher distillation settings")
    if not math.isfinite(args.prediction_distillation_encoder_lr_scale) or args.prediction_distillation_encoder_lr_scale<=0:
        p.error("Invalid post-cache encoder learning-rate scale")
    reference=json.loads(args.reference_config.read_text(encoding="utf-8")) if args.reference_config else None
    effective_ir=args.ir_backbone or (reference["arguments"].get("ir_backbone") if reference else None)
    effective_ir_weights=args.ir_weights or (reference["arguments"].get("ir_weights") if reference else None)
    if (args.ir_backbone and not effective_ir_weights) or (args.ir_weights and not effective_ir):
        p.error("Explicit infrared encoder/weights must form a complete configuration")
    if args.ir_training_scope=="all_used_layers" and not effective_ir:
        p.error("Full infrared layer training requires a UNIP encoder")
    if args.stop_gradient is None:
        args.stop_gradient=reference["arguments"].get("crossmodal_stop_gradient","none") if reference else "none"
    if args.target_representation_anchor_weight is None:
        args.target_representation_anchor_weight=reference["arguments"].get("target_representation_anchor_weight",0.) if reference else 0.
    if not math.isfinite(args.target_representation_anchor_weight) or args.target_representation_anchor_weight<0:
        p.error("Invalid target representation anchor weight")
    if reference and any(method!=reference["method"] for method in args.methods):p.error("Reference method must match the selected method")
    if args.domain_confusion_weight and any(method not in {"dann","cdan"} for method in args.methods):
        p.error("Encoder domain confusion is only defined for DANN/CDAN")
    if args.target_objective=="vicreg" and args.target_weight and not args.projection_dim:
        p.error("VICReg screening requires explicit training-only projections")
    if not 0<=args.target_feature_style_probability<=1 or not 0<=args.target_feature_style_source_max<=1:
        p.error("Invalid target feature-style settings")
    if not all(math.isfinite(x) and 0<=x<=1 for x in (args.source_ir_statistics_probability,args.source_ir_statistics_strength)):
        p.error("Invalid source IR statistics settings")
    if args.target_feature_style_probability and (not reference or reference["image_backbone"]!="convnext_small"):
        p.error("Target feature style requires a reference weather ConvNeXt configuration")
    object_enabled=bool(args.source_object_ce_weight or args.source_object_consistency_weight)
    if object_enabled and (not reference or reference["image_backbone"]!="convnext_small" or not args.source_object_manifest):
        p.error("Source object views require a ConvNeXt reference configuration and reviewed manifest")
    snapshot=prepare_snapshot(ROOT)
    data=ROOT/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/prepared_exact_pairs"
    weights=ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
    args.output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for target in args.targets:
        for method in args.methods:
            for source_weight in args.source_weights:
                name=f"{method}_{target}_source{source_weight:g}_target{args.target_weight:g}"
                cmd=[sys.executable,"-B","-u",str(snapshot/"weather/train_vds_core.py"),"--method",method,"--target",target,
                    "--data-dir",str(data),"--save-dir",str(args.output.resolve()/name),"--weights",str(weights),
                    "--image-backbone","resnet18","--encoder-training-scope",args.scope or "last_stages","--classifier-layout",args.classifier_layout or "linear",
                    "--feature-normalization","unit","--source-class-balance","--label-smoothing",".05","--auxiliary-source-weight",".2",
                    "--source-crossmodal-weight",str(source_weight),"--target-crossmodal-weight",str(args.target_weight),
                    "--target-crossmodal-objective",args.target_objective,
                    "--crossmodal-temperature",".2","--initial-vis-weight",{"rainy":".9","foggy":".5","dark":".25","backlight":".5"}[target],
                    "--optimizer","adamw","--lr",{"dann":".0002","deepcoral":".00005","cdan":".0001","mdd":".0015","mcc":".0015"}[method],
                    "--lr-scheduler","cosine","--weight-decay",".0001","--gradient-clip-norm","5",
                    "--gate-lr-multiplier",".1","--gate-weight-decay","0","--mcc-temperature","1",
                    "--trade-off",str(args.trade_off) if args.trade_off is not None else {"dann":".1","deepcoral":".1","cdan":".1","mdd":".2","mcc":"1"}[method],
                    "--adaptation-warmup-epochs","3","--adaptation-ramp-epochs","10","--freeze-bn-after-epochs","3","--freeze-bn-scope","encoders",
                    "--epochs",str(args.epochs),"--early-stopping-patience","15","--early-stopping-min-epochs",str(args.minimum_epochs),
                    "--batch-size","8","--target-batch-size","8" if method=="mdd" else "16","--seed","42","--num-workers","2","--cpu-threads","2",
                    "--cache-evaluation","--cache-training-images","--log-feature-moments","--modality-diagnostics","--log-parameter-updates",
                    "--cuda-memory-fraction",".5"]
                if method in {"dann","cdan","mdd"}:cmd += ["--adversarial-lr-override",".00005"]
                if args.projection_dim:cmd += ["--crossmodal-projection-dim",str(args.projection_dim)]
                cmd += ["--crossmodal-stop-gradient",args.stop_gradient,"--source-ir-inversion-probability",str(args.source_ir_inversion_probability)]
                cmd += ["--target-representation-anchor-weight",str(args.target_representation_anchor_weight)]
                cmd += ["--source-ir-statistics-probability",str(args.source_ir_statistics_probability),
                        "--source-ir-statistics-strength",str(args.source_ir_statistics_strength)]
                cmd += ["--source-prototype-weight",str(args.source_prototype_weight)]
                cmd += ["--domain-confusion-weight",str(args.domain_confusion_weight)]
                cmd += ["--prediction-teacher-weight",str(args.prediction_teacher_weight),
                    "--prediction-distillation-weight",str(args.prediction_distillation_weight),
                    "--prediction-distillation-encoder-lr-scale",str(args.prediction_distillation_encoder_lr_scale),
                    "--prediction-distillation-start-epoch",str(args.prediction_distillation_start_epoch),
                    "--prediction-distillation-sharpening",str(args.prediction_distillation_sharpening),
                    "--prediction-teacher-temperature",str(args.prediction_teacher_temperature),
                    "--prediction-student-temperature",str(args.prediction_student_temperature)]
                if args.prediction_teacher_weight:cmd += ["--prediction-teacher-weights",str(weights)]
                if args.cache_best_features:cmd += ["--cache-best-features"]
                if args.checkpoint_vis_blocks:cmd += ["--checkpoint-vis-blocks"]
                if args.initial_vis_weight is not None:cmd[cmd.index("--initial-vis-weight")+1]=str(args.initial_vis_weight)
                if args.learning_rate is not None:cmd[cmd.index("--lr")+1]=str(args.learning_rate)
                for key in ("adaptation_warmup_epochs","adaptation_ramp_epochs","image_adaptation_lr_scale"):
                    value=getattr(args,key)
                    if value is not None:
                        option="--"+key.replace("_","-")
                        if option in cmd:cmd[cmd.index(option)+1]=str(value)
                        else:cmd += [option,str(value)]
                if args.auxiliary_source_weight is not None:cmd[cmd.index("--auxiliary-source-weight")+1]=str(args.auxiliary_source_weight)
                if args.stop_after_epochs:cmd += ["--stop-after-epochs",str(args.stop_after_epochs)]
                for key in ("source_feature_mixup_weight","source_feature_mixup_alpha"):
                    if getattr(args,key) is not None:cmd += ["--"+key.replace("_","-"),str(getattr(args,key))]
                if args.fda_beta is not None:cmd += ["--fda-beta",str(args.fda_beta)]
                if args.source_weather_probability is not None:cmd += ["--source-weather-probability",str(args.source_weather_probability)]
                for key in ("ir_backbone","ir_weights","ir_feature_mode","ir_training_scope"):
                    value=getattr(args,key)
                    if value is not None:
                        value=str(value.resolve()) if key=="ir_weights" else str(value)
                        cmd += ["--"+key.replace("_","-"),value]
                if reference:
                    values=dict(reference["arguments"])
                    values.update(method=method,target=target,data_dir=str(data),save_dir=str(args.output.resolve()/name),seed=42,
                        epochs=args.epochs,early_stopping_min_epochs=args.minimum_epochs,early_stopping_patience=15,
                        stop_after_epochs=args.stop_after_epochs,max_train_batches=0,source_crossmodal_weight=source_weight,target_crossmodal_weight=args.target_weight,
                        crossmodal_temperature=.2,crossmodal_projection_dim=args.projection_dim,cache_training_images=True,
                        target_crossmodal_objective=args.target_objective,
                        prediction_teacher_weight=args.prediction_teacher_weight,
                        prediction_distillation_weight=args.prediction_distillation_weight,
                        prediction_distillation_encoder_lr_scale=args.prediction_distillation_encoder_lr_scale,
                        prediction_distillation_start_epoch=args.prediction_distillation_start_epoch,
                        prediction_distillation_sharpening=args.prediction_distillation_sharpening,
                        prediction_teacher_temperature=args.prediction_teacher_temperature,
                        prediction_student_temperature=args.prediction_student_temperature,
                        prediction_teacher_weights=str(weights) if args.prediction_teacher_weight else None,
                        crossmodal_stop_gradient=args.stop_gradient,source_ir_inversion_probability=args.source_ir_inversion_probability,
                        target_representation_anchor_weight=args.target_representation_anchor_weight,
                        source_ir_statistics_probability=args.source_ir_statistics_probability,
                        source_ir_statistics_strength=args.source_ir_statistics_strength,
                        source_prototype_weight=args.source_prototype_weight,
                        domain_confusion_weight=args.domain_confusion_weight,
                        target_feature_style_probability=args.target_feature_style_probability,
                        target_feature_style_source_max=args.target_feature_style_source_max,
                        source_object_manifest=str(args.source_object_manifest.resolve()) if args.source_object_manifest else None,
                        source_object_ce_weight=args.source_object_ce_weight,
                        source_object_consistency_weight=args.source_object_consistency_weight,
                        cache_best_features=args.cache_best_features,
                        cache_evaluation=True,num_workers=2,cpu_threads=2,cuda_memory_fraction=.4)
                    if args.trade_off is not None:values["trade_off"]=args.trade_off
                    if args.learning_rate is not None:values["lr"]=args.learning_rate
                    for key in ("adaptation_warmup_epochs","adaptation_ramp_epochs","image_adaptation_lr_scale"):
                        if getattr(args,key) is not None:values[key]=getattr(args,key)
                    if args.initial_vis_weight is not None:values["initial_vis_weight"]=args.initial_vis_weight
                    if args.fda_beta is not None:values["fda_beta"]=args.fda_beta
                    if args.source_weather_probability is not None:values["source_weather_probability"]=args.source_weather_probability
                    if args.scope is not None:
                        values["encoder_training_scope"]=args.scope
                        values["freeze_image_encoders"]=False
                    if args.classifier_layout is not None:values["classifier_layout"]=args.classifier_layout
                    if args.auxiliary_source_weight is not None:values["auxiliary_source_weight"]=args.auxiliary_source_weight
                    for key in ("source_feature_mixup_weight","source_feature_mixup_alpha"):
                        if getattr(args,key) is not None:values[key]=getattr(args,key)
                    for key in ("ir_backbone","ir_weights","ir_feature_mode","ir_training_scope"):
                        value=getattr(args,key)
                        if value is not None:values[key]=str(value.resolve()) if key=="ir_weights" else value
                    if args.checkpoint_vis_blocks is not None:values["checkpoint_vis_blocks"]=args.checkpoint_vis_blocks
                    if args.target_feature_style_probability:values["domain_forward"]="joint"
                    cmd=[sys.executable,"-B","-u",str(snapshot/"weather/train_vds_core.py"),*recorded_arguments_to_cli(values)]
                changes={}
                if reference:
                    from weather.train_vds_core import parser as backend_parser
                    defaults={action.dest:action.default for action in backend_parser()._actions}
                    original=reference["arguments"]
                    for key in original.keys()|values.keys():
                        old=original.get(key,defaults.get(key))
                        candidate=values.get(key,defaults.get(key))
                        if old!=candidate:
                            changes[key]={"reference":old,"candidate":candidate}
                            if key not in original:changes[key]["reference_default_used"]=True
                jobs.append({"name":name,"command":cmd,"reference_argument_changes":changes})
    family=("core_plus_encoder_domain_confusion" if args.domain_confusion_weight and not any(args.source_weights)
            and not args.target_weight and not args.source_prototype_weight else "core_plus_paired_crossmodal_alignment"
            + ("_and_domain_confusion" if args.domain_confusion_weight else ""))
    if args.target_feature_style_probability:
        family=(family+"_and_target_feature_style" if any(args.source_weights) or args.target_weight
                or args.source_prototype_weight or args.domain_confusion_weight else "core_plus_target_feature_style")
    if not (any(args.source_weights) or args.target_weight or args.source_prototype_weight
            or args.domain_confusion_weight or args.target_feature_style_probability or object_enabled):
        family="original_method_with_reference_config_overrides"
    if object_enabled:
        family=(family+"_and_source_object_views" if any(args.source_weights) or args.target_weight or args.source_prototype_weight
                or args.domain_confusion_weight or args.target_feature_style_probability else "core_plus_source_object_views")
    if args.target_weight and args.target_objective=="vicreg":family += "_negative_free_target_pairs"
    if args.prediction_teacher_weight:family="core_plus_frozen_cluster_probability_blend"
    if args.prediction_distillation_weight:family="core_plus_cached_ensemble_target_distillation"
    if args.source_ir_statistics_probability:family += "_and_source_IR_statistics"
    if args.target_representation_anchor_weight:family += "_and_initial_target_anchor"
    effective_mixup=(args.source_feature_mixup_weight if args.source_feature_mixup_weight is not None
                    else reference["arguments"].get("source_feature_mixup_weight",0.) if reference else 0.)
    if effective_mixup:
        family=("core_plus_source_feature_mixup" if family=="original_method_with_reference_config_overrides"
                else family+"_and_source_feature_mixup")
    (args.output/"plan.json").write_text(json.dumps({"created_UTC":datetime.now(timezone.utc).isoformat(),"launcher_pid":os.getpid(),
        "target_training_labels_used":False,"variant_family":family,
        "reference_config":str(args.reference_config.resolve()) if args.reference_config else None,"jobs":jobs},indent=2),encoding="utf-8")
    if args.plan_only:
        print("PLAN_ONLY="+str(args.output/"plan.json"),flush=True)
        return

    def execute_in_slot(job,preflight=False):
        import psutil
        from weather.local_capacity import commit_free_bytes
        while True:
            free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0])
            commit=commit_free_bytes()
            if free>=1800 and psutil.virtual_memory().available>=6*1024**3 and (commit is None or commit>=6*1024**3):break
            print(f"WAITING_HEADROOM {job['name']}; GPU free={free}MiB; commit free={commit}",flush=True)
            time.sleep(10)
        cmd=job["command"].copy();name=job["name"]
        if preflight:
            name="actual_CUDA_preflight"
            cmd[cmd.index("--save-dir")+1]=str(ROOT/"runs/weather_local_preflight"/(args.output.name+"_crossmodal"))
            cmd += ["--max-train-batches","1","--stop-after-epochs","1",
                    "--adaptation-warmup-epochs","0","--adaptation-ramp-epochs","1"]
            if args.prediction_distillation_weight:cmd += ["--prediction-distillation-start-epoch","1"]
        print("START="+name,flush=True)
        with (args.output/(name+".log")).open("x",encoding="utf-8") as stream:
            env=dict(os.environ,JMDA_DETERMINISTIC="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",LOKY_MAX_CPU_COUNT="2",PYTHONUNBUFFERED="1")
            code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        print(f"FINISHED={name}; exit={code}",flush=True)
        return {"name":name,"exit":code}

    def execute(job,preflight=False):
        from weather.local_capacity import local_training_lease
        with local_training_lease(ROOT/"runs/weather_local_screen/.local_training.lock"):
            return execute_in_slot(job,preflight)

    status=execute(jobs[0],True)
    if status["exit"]:
        (args.output/"preflight_failure.json").write_text(json.dumps(status),encoding="utf-8")
        raise SystemExit(status["exit"])
    finished=[]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for future in as_completed([pool.submit(execute,job) for job in jobs]):
            finished.append(future.result())
            (args.output/"queue_status.json").write_text(json.dumps({"requested":len(jobs),"finished":finished,
                "status":"RUNNING" if len(finished)<len(jobs) else "FAIL" if any(x["exit"] for x in finished) else "PASS"},indent=2),encoding="utf-8")
    if any(x["exit"] for x in finished):raise SystemExit(1)


if __name__=="__main__":main()
