"""Full allocation, parser, teacher removal and coherent peak-result reporting."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.run_weather_fewshot_suite import build_plan,required_weights
from tools.run_label_guided_matrix import write_peak_summary,main,build_jobs
from weather.fewshot_presets import ALLOCATION,PAIRS
from weather.rainy_presets import arguments_for
from weather.train_vds_core import parser,resolve_weather_settings
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli


class FewShotSuiteChecks(unittest.TestCase):
    def test_coefficient_variants_preserve_every_other_training_setting(self):
        for seed in (42,43,46):
            original=build_jobs(Path('out'),Path('lists'),['cdan_foggy'],[seed],Path('snapshot'),True,
                                evaluation_mode='validation_ce',unified_fewshot=True)[0]['values']
            for coefficient in (.2,.3):
                variant=build_jobs(Path('out'),Path('lists'),['cdan_foggy'],[seed],Path('snapshot'),True,
                    overrides={'matched_alignment_weight':coefficient},evaluation_mode='validation_ce',
                    unified_fewshot=True,fewshot_alignment_variant=True)[0]['values']
                changed={k for k in set(original)|set(variant) if original.get(k)!=variant.get(k)}
                self.assertEqual(changed,{'matched_alignment_weight','configuration_tag'})
                args=parser().parse_args(recorded_arguments_to_cli(variant))
                self.assertEqual(args.epochs,40)
                self.assertEqual(args.target_labels_per_class,3)
                self.assertEqual(args.early_stopping_patience,0)
                self.assertEqual(args.matched_alignment_weight,coefficient)

    def test_coefficient_variant_cannot_change_full_machine_queue_or_disable_guidance(self):
        base=['runner','--data-dir','missing','--unified-fewshot','--fewshot-alignment-variant',
              '--pairs','cdan_foggy','--target-labels-per-class','3']
        for extra in (['--matched-alignment-weight','0'],['--matched-alignment-weight','nan'],
                      ['--matched-alignment-weight','.2','--machine-split','local']):
            with patch('sys.argv',base+extra),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as e:main()
                self.assertEqual(e.exception.code,2)

    def test_exact_split_disjoint_complete_and_default_seeds(self):
        local=build_plan('local',Path('out'),Path('lists'),Path('snapshot'))
        server=build_plan('server',Path('out'),Path('lists'),Path('snapshot'))
        self.assertEqual(local['total'],40);self.assertEqual(server['total'],35)
        self.assertEqual(local['counts_by_weather'],{'foggy':15,'rainy':25})
        self.assertEqual(server['counts_by_weather'],{'dark':25,'foggy':10})
        self.assertFalse(set(local['pairs'])&set(server['pairs']))
        self.assertEqual(set(local['pairs'])|set(server['pairs']),set(PAIRS))
        jobs=local['jobs']+server['jobs']
        self.assertEqual(len({j['name'] for j in jobs}),75)
        self.assertEqual(len({j['result_file'] for j in jobs}),75)
        self.assertEqual({j['values']['seed'] for j in jobs},set(range(42,47)))

    def test_all_fifteen_variants_parse_with_native_losses_and_fixed_protocol(self):
        for machine in ALLOCATION:
            plan=build_plan(machine,Path('out'),Path('lists'),Path('snapshot'),(42,))
            for job in plan['jobs']:
                with self.subTest(job=job['name']):
                    args=parser().parse_args(recorded_arguments_to_cli(job['values']))
                    resolve_weather_settings(args)
                    self.assertEqual(args.target_labels_per_class,3)
                    self.assertEqual(args.matched_alignment_weight,.1)
                    self.assertEqual(args.target_class_matching,'target_truth')
                    self.assertEqual(args.epochs,40);self.assertEqual(args.early_stopping_patience,0)
                    self.assertEqual(args.stop_after_epochs,0);self.assertIsNone(args.holdout_list)
                    self.assertEqual(args.selection_metric,'cross_entropy')
                    self.assertEqual(args.prediction_teacher_weight,0)
                    self.assertEqual(args.prediction_distillation_weight,0)
                    self.assertEqual(args.cluster_teacher_weight,0)
                    self.assertEqual(args.target_crossmodal_weight,0)
                    self.assertEqual(args.target_representation_anchor_weight,0)
                    self.assertGreater(args.trade_off,0)
                    self.assertLess(args.adaptation_warmup_epochs+args.adaptation_ramp_epochs,40)
                    self.assertTrue(args.log_parameter_updates)

    def test_removed_rain_teacher_does_not_silently_remove_encoder_rate_boost(self):
        plan=build_plan('local',Path('out'),Path('lists'),Path('snapshot'),(42,))
        for method in ('dann','mdd','mcc'):
            old=arguments_for(method,Path.cwd(),Path('lists'),Path('out'),42)
            new=next(j['values'] for j in plan['jobs'] if j['values']['target']=='rainy' and j['values']['method']==method)
            self.assertAlmostEqual(new['image_lr_override'],3e-4)
            self.assertAlmostEqual(old['image_lr_override'],1e-5)
            self.assertEqual(old['prediction_distillation_weight'],4.)
            self.assertEqual(new['prediction_distillation_encoder_lr_scale'],1.)

    def test_machine_specific_dependencies_and_invalid_seed_lists(self):
        server=build_plan('server',Path('out'),Path('lists'),Path('snapshot'),(42,))
        local=build_plan('local',Path('out'),Path('lists'),Path('snapshot'),(42,))
        self.assertNotIn('unip_small',required_weights(server))
        self.assertIn('unip_small',required_weights(local))
        for seeds in ((),(42,42),(-1,)):
            with self.assertRaises(ValueError):build_plan('local',Path('out'),Path('lists'),Path('snapshot'),seeds)

    def test_peak_summary_does_not_mix_independent_metric_maxima(self):
        with tempfile.TemporaryDirectory() as name:
            row={'method':'dann','target':'foggy','seed':42,'protocol':'FEWSHOT_TARGET_CLASS_ALIGNMENT',
                 'configuration_tag':'test','screening_only':False,'peak_validation_epoch':7,
                 'peak_validation_ACC_percent':75.,'peak_validation_Precision_macro_percent':71.,
                 'peak_validation_Recall_macro_percent':72.,'peak_validation_F1_macro_percent':73.}
            write_peak_summary(Path(name),[row],[42])
            with (Path(name)/'peak_acc_runs.csv').open(encoding='utf-8-sig') as f:record=next(csv.DictReader(f))
            self.assertEqual(record['selected_epoch'],'7')
            self.assertEqual(record['Precision_macro_percent'],'71.0')
            with (Path(name)/'peak_acc_summary.csv').open(encoding='utf-8-sig') as f:summary=next(csv.DictReader(f))
            self.assertEqual(summary['F1_macro_percent_mean'],'73.0')
            self.assertEqual(summary['ACC_percent_std'],'')

    def test_wrong_machine_list_rejected_before_dataset_loading(self):
        argv=['runner','--data-dir','missing','--unified-fewshot','--machine-split','server',
              '--pairs','dann_rainy','--matched-alignment-weight','0.1','--target-labels-per-class','3']
        with patch('sys.argv',argv),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:main()
            self.assertEqual(e.exception.code,2)


if __name__=='__main__':unittest.main()
