"""Three-weather, fixed-budget few-shot variants; historical presets stay intact."""
from pathlib import Path

from weather.label_guided_presets import build_preset
from weather.rainy_presets import arguments_for as rainy_arguments

METHODS=('dann','deepcoral','cdan','mdd','mcc')
TARGETS=('dark','foggy','rainy')
PAIRS=tuple(f'{method}_{target}' for target in TARGETS for method in METHODS)
ALLOCATION={
    'server':tuple([f'{m}_dark' for m in METHODS]+['mdd_foggy','mcc_foggy']),
    'local':tuple([f'{m}_rainy' for m in METHODS]+['dann_foggy','deepcoral_foggy','cdan_foggy']),
}


def build_fewshot_preset(root,data_dir,output,method,target,seed):
    if f'{method}_{target}' not in PAIRS:raise ValueError('Unknown few-shot method/weather')
    if target=='rainy':
        values=rainy_arguments(method,root,data_dir,output,seed)
        # Without distillation there is no epoch-2 cached-teacher LR boost.
        # Preserve the old effective adaptation-phase encoder LR explicitly;
        # this variant uses it from epoch 1 and records that deliberate change.
        boost=values.get('prediction_distillation_encoder_lr_scale',1.)
        if boost!=1:
            for key in ('image_lr_override','ir_lr_override'):
                if values.get(key) is not None:values[key]*=boost
    else:
        values=build_preset(root,data_dir,output,method,target,seed)
    values.update(epochs=40,stop_after_epochs=0,max_train_batches=0,device='cuda',
        early_stopping_patience=0,target_class_matching='target_truth',target_labels_per_class=3,
        matched_alignment_weight=.1,matched_alignment_margin=.2,matched_alignment_ramp_epochs=3,
        configuration_tag='fewshot3_alignment0p1_40epoch_v1',log_parameter_updates=True,
        prediction_distillation_weight=0.,prediction_teacher_weight=0.,prediction_teacher_weights=None,
        prediction_distillation_encoder_lr_scale=1.,cluster_teacher_weight=0.,feature_calibration=False,
        target_representation_anchor_weight=0.,target_crossmodal_weight=0.,domain_confusion_weight=0.,
        target_feature_style_probability=0.)
    values['target_batch_size']=values['batch_size']
    if values['adaptation_warmup_epochs']+values['adaptation_ramp_epochs']>=40:
        raise ValueError('Fixed budget leaves no epochs after the native adaptation ramp')
    return values
