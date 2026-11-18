"""Declared, unvalidated transfer of the successful foggy CDAN design."""
from pathlib import Path

from weather.fewshot_presets import ALLOCATION,build_fewshot_preset

RECIPE='server_adaptive_v2'
PAIRS=ALLOCATION['server']


def build_server_adaptive_preset(root,data_dir,output,method,target,seed):
    if f'{method}_{target}' not in PAIRS:
        raise ValueError('The adaptive server recipe is restricted to dark five methods and foggy MDD/MCC')
    root=Path(root)
    values=build_fewshot_preset(root,data_dir,output,method,target,seed)
    # Each method keeps its own objective, GRL, classifier and trade-off.
    # "reference" restores DANN/MDD/MCC bottlenecks and the CDAN linear
    # projection. Deep CORAL deliberately retains its Identity bottleneck.
    values.update(optimizer='adamw',lr=.0003 if method in {'deepcoral','cdan'} else .001,
        image_backbone='convnext_small',ir_backbone='unip_tiny',ir_feature_mode='patch_mean',
        weights=str(root/'pretrained_weights/torch/hub/checkpoints/convnext_small-0c510722.pth'),
        ir_weights=str(root/'pretrained_weights/unip/unip_vit_tiny_by_ibot_large.pth'),
        encoder_precision='bf16',encoder_training_scope='last_stages',ir_training_scope='all_used_layers',
        freeze_image_encoders=False,image_lr_override=5e-5,ir_lr_override=1e-4,
        image_adaptation_lr_scale=.1,adversarial_lr_override=None,
        classifier_layout='reference',feature_normalization='unit',unit_method_features=False,
        weight_decay=1e-4,label_smoothing=.05,gradient_clip_norm=5.,
        gate_lr_multiplier=.1,gate_weight_decay=0.,auxiliary_source_weight=.5,
        source_class_balance=True,matched_alignment_weight=.2,source_anchor_ema_momentum=.9,
        adaptation_warmup_epochs=4,adaptation_ramp_epochs=10 if method=='deepcoral' else 6,
        configuration_tag=RECIPE+'_fewshot3_alignment0p2_source_ema0p9_reference',
        cuda_memory_fraction=.5,epochs=40,early_stopping_patience=0)
    # Delay validation-based LR cuts until the native adaptation ramp finishes.
    values.update(validation_control_start_epoch=values['adaptation_warmup_epochs']+values['adaptation_ramp_epochs']+1,
                  val_plateau_patience=8)
    return values
