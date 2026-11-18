"""Check the fixed-seed phase-LR trial cannot silently alter other settings."""
from pathlib import Path
import unittest
import torch

from tools.run_label_guided_matrix import build_jobs
from tools.run_cdan_foggy_phase_lr_test import ROOT,build_test_job,build_discriminator_job
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from weather.train_vds_core import parser
from weather.encoder_schedule import lower_image_lr_after_source_warmup


class CdanPhaseLRChecks(unittest.TestCase):
    def test_discriminator_variant_keeps_encoder_and_all_other_training_settings(self):
        original=build_jobs(Path('baseline'),Path('lists'),['cdan_foggy'],[46],ROOT,True,
            overrides={'matched_alignment_weight':.2},evaluation_mode='validation_ce',
            unified_fewshot=True,fewshot_alignment_variant=True)[0]
        job=build_discriminator_job(original,Path('new'))
        self.assertEqual(set(job['changed_argument_keys']),
            {'adversarial_lr_override','configuration_tag','save_dir'})
        args=parser().parse_args(recorded_arguments_to_cli(job['values']))
        self.assertEqual(args.adversarial_lr_override,3e-4)
        self.assertEqual(args.image_adaptation_lr_scale,.1)
        self.assertEqual(args.val_plateau_patience,4)
        self.assertEqual(args.epochs,40)
        self.assertEqual(args.target_labels_per_class,3)
        self.assertEqual(args.matched_alignment_weight,.2)

    def test_discriminator_optimizer_group_does_not_change_classifier_lr(self):
        from vds2raw.train_adaptation import make_optimizer
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rgb_encoder=torch.nn.Linear(2,2)
                self.nir_encoder=torch.nn.Linear(2,2)
                self.head=torch.nn.Linear(4,2)
                self.discriminator=torch.nn.Linear(4,1)
                self.modality_logits=torch.nn.Parameter(torch.zeros(2))
                self.method='cdan'
        model=ToyModel()
        args=parser().parse_args(['--method','cdan','--data-dir','lists','--target','foggy','--lr','.0003','--optimizer','adamw',
            '--image-lr-override','5e-5','--ir-lr-override','1e-4','--adversarial-lr-override','3e-4',
            '--gate-lr-multiplier','.1','--gate-weight-decay','0','--lr-scheduler','reference'])
        optimizer,_,_=make_optimizer(model,args)
        groups={id(p):g for g in optimizer.param_groups for p in g['params']}
        self.assertEqual(groups[id(model.discriminator.weight)]['lr'],3e-4)
        self.assertEqual(groups[id(model.discriminator.weight)]['role'],'adversarial')
        self.assertAlmostEqual(groups[id(model.head.weight)]['lr'],3e-3)
        self.assertEqual(groups[id(model.rgb_encoder.weight)]['lr'],5e-5)
        self.assertEqual(groups[id(model.nir_encoder.weight)]['lr'],1e-4)

    def test_only_one_training_parameter_changes_and_all_budgets_stay_fixed(self):
        original=build_jobs(Path('baseline'),Path('lists'),['cdan_foggy'],[46],ROOT,True,
            overrides={'matched_alignment_weight':.2},evaluation_mode='validation_ce',
            unified_fewshot=True,fewshot_alignment_variant=True)[0]
        new=build_test_job(original,Path('new'),.3)
        self.assertEqual(set(new['changed_argument_keys']),
            {'image_adaptation_lr_scale','configuration_tag','save_dir'})
        args=parser().parse_args(recorded_arguments_to_cli(new['values']))
        self.assertEqual(args.seed,46)
        self.assertEqual(args.image_adaptation_lr_scale,.3)
        self.assertEqual(args.matched_alignment_weight,.2)
        self.assertEqual(args.target_labels_per_class,3)
        self.assertEqual(args.epochs,40)
        self.assertEqual(args.val_plateau_patience,4)
        self.assertIsNone(args.adversarial_lr_override)
        self.assertEqual(args.early_stopping_patience,0)

    def test_phase_factor_applies_once_to_encoders_and_not_heads(self):
        parameters=[torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
        optimizer=torch.optim.SGD([
            {'params':[parameters[0]],'role':'pretrained_RGB','lr':5e-5},
            {'params':[parameters[1]],'role':'pretrained_NIR','lr':1e-4},
            {'params':[parameters[2]],'role':'new_layers','lr':3e-3}])
        for epoch in range(1,7):lower_image_lr_after_source_warmup(optimizer,None,epoch,4,.3)
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[1.5e-5,3e-5,3e-3])


if __name__=='__main__':unittest.main()
