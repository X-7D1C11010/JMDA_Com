"""Fit the authorized initialization from source/train and unlabelled target/train."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from weather.statistical_calibration import TrainFeatureCalibration,ClusterInitialization,ResidualFeatureAdapter,unit


def prepare_calibrated_initialization(model,source_eval,target_eval,device,args):
    if (source_eval.partition!="source_train" or not source_eval.labelled or
            target_eval.partition!="target_train" or target_eval.labelled):
        raise ValueError("Only labelled source_train and label-free target_train may fit calibration")
    if (model.encoder_training_scope!="frozen" or args.feature_normalization!="unit" or
            args.classifier_layout!="linear" or args.encoder_precision!="fp32" or args.domain_running_stats):
        raise ValueError("Calibration screening requires frozen FP32 image encoders, unit fusion, linear heads, and its own training statistics")
    model.eval()
    fixed=model.modality_weights().detach().clone()
    def collect(view):
        arrays=[];labels=[];ids=[]
        with torch.no_grad():
            for batch in DataLoader(view,batch_size=args.eval_batch_size,num_workers=0,shuffle=False,
                                    generator=torch.Generator().manual_seed(args.seed+37000)):
                if not view.labelled and "label" in batch:raise RuntimeError("Target training labels entered calibration")
                rgb=model.rgb_encoder(batch["rgb"].to(device));ir=model.nir_encoder(batch["nir"].to(device))
                values=F.normalize(torch.cat((fixed[0]*F.normalize(rgb,dim=1),fixed[1]*F.normalize(ir,dim=1)),1),dim=1)
                arrays.append(values.cpu().numpy());ids.extend(batch["path"])
                if view.labelled:labels.extend(batch["label"].tolist())
        return np.concatenate(arrays),np.asarray(labels,dtype=np.int64),ids
    source,labels,source_ids=collect(source_eval)
    target,absent_labels,target_ids=collect(target_eval)
    if len(absent_labels):raise RuntimeError("Unlabelled target statistics received labels")
    calibrator=TrainFeatureCalibration.fit(source,target,args.calibration_components,args.calibration_shrinkage)
    if calibrator.output_dim!=args.calibration_components:
        raise ValueError("Calibration dimension exceeds available training samples/features")
    with torch.no_grad():
        s=calibrator(torch.tensor(source),"source").numpy()
        t=calibrator(torch.tensor(target),"target").numpy()
    associated_source=associated_target=None
    association_audits=[]
    if args.cluster_crossmodal_association:
        half=model.image_feature_dim
        parts=[]
        for part in (slice(None,half),slice(half,None)):
            a,b=unit(source[:,part]),unit(target[:,part])
            component_count=min(32,a.shape[1],len(a)+len(b)-1)
            c=TrainFeatureCalibration.fit(a,b,component_count,args.calibration_shrinkage)
            with torch.no_grad():parts.append((c(torch.tensor(a,dtype=torch.float32),"source").numpy(),
                                              c(torch.tensor(b,dtype=torch.float32),"target").numpy()))
            association_audits.append(c.metadata)
        w=args.cluster_association_vis_weight
        associated_source=unit(np.concatenate((w*parts[0][0],(1-w)*parts[1][0]),1))
        associated_target=unit(np.concatenate((w*parts[0][1],(1-w)*parts[1][1]),1))
    teacher=ClusterInitialization.fit(s,t,labels,model.num_classes,args.cluster_teacher_temperature,args.seed,
        args.cluster_association,args.cluster_initialization,args.cluster_neighbors,args.cluster_trim_fraction,
        association_source=associated_source,association_target=associated_target)
    model.feature_calibrator=calibrator.to(device)
    model.cluster_teacher=teacher.to(device)
    model.feature_adapter=ResidualFeatureAdapter(calibrator.output_dim,args.calibration_adapter_scale).to(device)
    model.register_buffer("teacher_modality_weights",fixed)
    model.cluster_teacher.initialize_head(model.head)
    if model.method=="mdd":model.cluster_teacher.initialize_head(model.adv_head)
    model.feature_domain="target"
    return {"enabled":True,"variant_family":"core_plus_train_statistics_cluster_initialization",
            "source_cohort_sha256":hashlib.sha256("\n".join(source_ids).encode()).hexdigest(),
            "target_cohort_sha256":hashlib.sha256("\n".join(target_ids).encode()).hexdigest(),
            "calibration":calibrator.metadata,"cluster_initialization":teacher.metadata,
            "crossmodal_association":{"enabled":args.cluster_crossmodal_association,
                "VIS_weight":args.cluster_association_vis_weight,"training_calibrations":association_audits},
            "teacher_modality_weights":fixed.cpu().tolist(),"cluster_teacher_weight":args.cluster_teacher_weight,
            "cluster_teacher_confidence":args.cluster_teacher_confidence,"teacher_logit_temperature":teacher.temperature,
            "student_feature_adapter":{"type":"zero-initialized residual linear map","scale":args.calibration_adapter_scale,"trainable":True},
            "source_initialization_ACC":float((teacher(torch.tensor(s).to(device)).argmax(1).cpu().numpy()==labels).mean()),
            "target_training_labels_used":False,"validation_used_for_fitting":False,
            "student_model_checkpoints_saved":False}
