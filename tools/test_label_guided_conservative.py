"""Conservative profiles affect actual optimizer groups without altering DA losses."""
from pathlib import Path
import tempfile
import unittest

import torch

from weather.label_guided_presets import REFERENCE_PARAMETERS, build_preset
from weather.train_vds_core import parser, resolve_weather_settings
from vds2raw.train_adaptation import make_optimizer
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.run_label_guided_matrix import build_jobs, write_pair_summary
from tools.test_weather_vds_core import fast_five_method_model


class ConservativeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_actual_group_rates_halved_and_replay_not_scaled_twice(self):
        root=Path(__file__).resolve().parents[1]
        for pair in REFERENCE_PARAMETERS:
            with self.subTest(pair=pair):
                method,target=pair.rsplit('_',1)
                before=build_preset(root,root/'lists',root/'output',method,target)
                after=build_preset(root,root/'lists',root/'output',method,target,parameter_profile='conservative')
                self.assertEqual(after['configuration_tag'],'conservative_v1')
                for key in ('trade_off','matched_alignment_weight','adaptation_warmup_epochs',
                            'adaptation_ramp_epochs','epochs','batch_size','image_backbone','ir_backbone'):
                    self.assertEqual(before.get(key),after.get(key),key)
                self.assertGreaterEqual(after['weight_decay'],2*before['weight_decay'])
                self.assertGreaterEqual(after['label_smoothing'],.1)
                self.assertEqual(after['gradient_clip_norm'],5.)
                model=fast_five_method_model(method,num_classes=3,feature_normalization='none',auxiliary_source_weight=0.)
                groups=[]
                for values in (before,after):
                    args=parser().parse_args(recorded_arguments_to_cli(values))
                    resolve_weather_settings(args)
                    self.assertEqual(args.lr,values['lr'])
                    self.assertEqual(args.image_lr_override,values['image_lr_override'])
                    optimizer,*_=make_optimizer(model,args)
                    groups.append({g['role']:g['lr'] for g in optimizer.param_groups})
                self.assertEqual(groups[0].keys(),groups[1].keys())
                for role,rate in groups[0].items():
                    self.assertAlmostEqual(groups[1][role],rate*.5,places=12,msg=role)

    def test_tuning_never_requests_holdout_and_keeps_ce_controls(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            jobs=build_jobs(root/'out',root/'lists',list(REFERENCE_PARAMETERS),[42],root/'snapshot',True,
                            evaluation_mode='validation_ce',parameter_profile='conservative')
            for job in jobs:
                values=job['values']
                self.assertIsNone(values['holdout_list'])
                self.assertNotIn('--holdout-list',job['command'])
                self.assertEqual(values['selection_metric'],'cross_entropy')
                self.assertEqual(values['early_stopping_monitor'],'cross_entropy')
                self.assertEqual(values['early_stopping_patience'],6)
                self.assertEqual(values['val_plateau_patience'],3)
                self.assertEqual(values['selection_min_epoch'],values['adaptation_warmup_epochs']+1)
            rows=[{'method':'dann','target':'dark','configuration_tag':tag} for tag in ('reference','conservative_v1')]
            with self.assertRaises(ValueError):write_pair_summary(root,rows,[42])

    def test_lr_only_is_a_regularization_rollback_with_identical_training_control(self):
        root=Path(__file__).resolve().parents[1]
        for pair in REFERENCE_PARAMETERS:
            method,target=pair.rsplit('_',1)
            ref=build_preset(root,root/'lists',root/'out',method,target)
            conservative=build_preset(root,root/'lists',root/'out',method,target,parameter_profile='conservative')
            trial=build_preset(root,root/'lists',root/'out',method,target,parameter_profile='lr_only')
            self.assertEqual(trial['configuration_tag'],'lr_only_v1')
            for key in ('lr','image_lr_override','ir_lr_override','adversarial_lr_override'):
                self.assertEqual(trial.get(key),conservative.get(key))
            for key in ('weight_decay','label_smoothing','gradient_clip_norm','matched_alignment_weight',
                        'trade_off','epochs','batch_size','encoder_training_scope'):
                self.assertEqual(trial.get(key),ref.get(key))
            jobs=[build_jobs(root/'out',root/'lists',[pair],[42],root/'snapshot',True,
                             evaluation_mode='validation_ce',parameter_profile=p)[0]
                  for p in ('conservative','lr_only')]
            for key in ('selection_metric','selection_min_epoch','early_stopping_monitor',
                        'early_stopping_patience','early_stopping_min_epochs','val_plateau_patience'):
                self.assertEqual(jobs[0]['values'][key],jobs[1]['values'][key])

    def test_reference_restores_previous_rates_and_regularization_for_all_pairs(self):
        root=Path(__file__).resolve().parents[1]
        jobs=build_jobs(root/'out',root/'lists',list(REFERENCE_PARAMETERS),[42],root/'snapshot',True,
                        evaluation_mode='validation_ce',parameter_profile='reference')
        for job in jobs:
            values=job['values'];original=REFERENCE_PARAMETERS[job['pair']]['arguments']
            for key in ('lr','image_lr_override','ir_lr_override','adversarial_lr_override',
                        'weight_decay','label_smoothing','gradient_clip_norm','optimizer',
                        'lr_scheduler','lr_gamma','lr_decay','image_backbone','ir_backbone',
                        'trade_off','batch_size','epochs'):
                self.assertEqual(values.get(key),original.get(key),(job['pair'],key))
            self.assertEqual(values['configuration_tag'],'reference')
            self.assertEqual(values['early_stopping_patience'],8)
            self.assertEqual(values['val_plateau_patience'],4)
            self.assertEqual(values['selection_metric'],'cross_entropy')
            self.assertIsNone(values['holdout_list'])


if __name__=='__main__':unittest.main()
