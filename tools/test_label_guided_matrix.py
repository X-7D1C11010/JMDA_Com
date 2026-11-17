"""Portable preset and matrix-plan contracts without training or downloads."""
from pathlib import Path
import tempfile
import unittest

from weather.label_guided_presets import REFERENCE_PARAMETERS,build_preset
from weather.train_vds_core import parser
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.run_label_guided_matrix import build_jobs,write_pair_summary,preflight_command,alignment_overrides


class GuidedMatrixChecks(unittest.TestCase):
    def test_weight_variant_changes_only_alignment_and_metadata(self):
        self.assertEqual(alignment_overrides(None),{})
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):alignment_overrides(value)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for pair in ('dann_foggy','deepcoral_foggy'):
                original=build_jobs(root/'output',root/'lists',[pair],[42],root/'snapshot',True,
                    evaluation_mode='validation_ce')[0]
                variant=build_jobs(root/'output',root/'lists',[pair],[42],root/'snapshot',True,
                    overrides=alignment_overrides(1.),evaluation_mode='validation_ce')[0]
                changed={k for k in variant['values'] if original['values'].get(k)!=variant['values'][k]}
                self.assertEqual(changed,{'matched_alignment_weight','configuration_tag'})
                args=parser().parse_args(recorded_arguments_to_cli(variant['values']))
                self.assertEqual(args.matched_alignment_weight,1.)
                self.assertEqual(args.target_class_matching,'target_truth')
                self.assertEqual(args.selection_metric,'cross_entropy')
                self.assertEqual(args.configuration_tag,'reference_alignment1_v1')
                self.assertEqual(args.epochs,40)
                smoke=preflight_command(variant['command'],root/'smoke')
                smoke_args=parser().parse_args(smoke[4:])
                self.assertEqual(smoke_args.matched_alignment_weight,1.)
                self.assertEqual(smoke_args.stop_after_epochs,1)

    def test_zero_weight_keeps_matching_and_native_method_schedule(self):
        from weather.class_matching import PROTOCOL,protocol_metadata
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for pair in ('dann_foggy','deepcoral_foggy'):
                old=build_jobs(root/'output',root/'lists',[pair],[42],root/'snapshot',True,
                    evaluation_mode='validation_ce')[0]
                new=build_jobs(root/'output',root/'lists',[pair],[42],root/'snapshot',True,
                    overrides=alignment_overrides(0.),evaluation_mode='validation_ce')[0]
                changed={k for k in new['values'] if old['values'].get(k)!=new['values'][k]}
                self.assertEqual(changed,{'matched_alignment_weight','configuration_tag'})
                args=parser().parse_args(recorded_arguments_to_cli(new['values']))
                self.assertEqual(args.matched_alignment_weight,0.)
                self.assertEqual(args.target_class_matching,'target_truth')
                self.assertEqual(args.adaptation_warmup_epochs,8)
                meta=protocol_metadata(True,args.matched_alignment_weight)
                self.assertEqual(meta['protocol'],PROTOCOL)
                self.assertFalse(meta['target_true_labels_used_in_alignment_loss'])
                self.assertFalse(meta['strict_unsupervised'])
                smoke=parser().parse_args(preflight_command(new['command'],root/'smoke')[4:])
                self.assertEqual(smoke.adaptation_warmup_epochs,0)
                self.assertEqual(smoke.stop_after_epochs,2)
                self.assertEqual(smoke.max_train_batches,2)
            rows=[{'protocol':PROTOCOL,'method':'dann','target':'foggy','seed':42,'screening_only':False,
                   **{k:80. for k in ('ACC_percent','Precision_macro_percent','Recall_macro_percent','F1_macro_percent')}}]
            self.assertEqual(write_pair_summary(root,rows,[42])[0]['protocol'],PROTOCOL)
            with self.assertRaisesRegex(ValueError,'mix alignment'):
                write_pair_summary(root,rows+[dict(rows[0],seed=43,protocol='LABEL_GUIDED_CLASS_ALIGNMENT')],[42,43])

    def test_all_dark_weight_one_variants_keep_reference_training_parameters(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            pairs=[method+'_dark' for method in ('dann','deepcoral','cdan','mdd','mcc')]
            before=build_jobs(root/'output',root/'lists',pairs,[42],root/'snapshot',True,
                              evaluation_mode='validation_ce')
            after=build_jobs(root/'output',root/'lists',pairs,[42],root/'snapshot',True,
                overrides=alignment_overrides(1.),evaluation_mode='validation_ce')
            self.assertEqual(len(after),5)
            for old,new in zip(before,after):
                changed={k for k in new['values'] if old['values'].get(k)!=new['values'][k]}
                self.assertTrue({'matched_alignment_weight','configuration_tag'}<=changed)
                self.assertTrue(changed<={'matched_alignment_weight','configuration_tag','log_parameter_updates'})
                parsed=parser().parse_args(recorded_arguments_to_cli(new['values']))
                self.assertEqual(parsed.target,'dark')
                self.assertEqual(parsed.target_class_matching,'target_truth')
                self.assertEqual(parsed.matched_alignment_weight,1.)
                self.assertEqual(parsed.selection_metric,'cross_entropy')

    def test_all_ten_presets_parse_without_reading_original_run_directories(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for pair in REFERENCE_PARAMETERS:
                with self.subTest(pair=pair):
                    method,target=pair.rsplit('_',1)
                    values=build_preset(root,root/'lists',root/'output',method,target,43)
                    args=parser().parse_args(recorded_arguments_to_cli(values))
                    self.assertEqual(args.seed,43)
                    self.assertEqual(args.target_class_matching,'target_truth')
                    self.assertEqual(args.matched_alignment_weight,5.)
                    self.assertEqual(args.batch_size,args.target_batch_size)
                    self.assertTrue(Path(args.weights).is_relative_to(root))
                    if args.ir_weights:self.assertTrue(Path(args.ir_weights).is_relative_to(root))
                    original=REFERENCE_PARAMETERS[pair]['arguments']
                    self.assertEqual(args.epochs,original['epochs'])
                    self.assertEqual(args.lr,original['lr'])
                    self.assertEqual(args.image_backbone,original['image_backbone'])
                    self.assertEqual(args.ir_backbone,original.get('ir_backbone'))

    def test_matrix_seeds_and_results_are_separate(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            jobs=build_jobs(root/'output',root/'lists',['dann_dark','mcc_foggy'],[42,43],root/'snapshot')
            self.assertEqual(len(jobs),4)
            self.assertEqual(len({j['result_file'] for j in jobs}),4)
            self.assertTrue(all(j['required_free_MiB']>=2048 for j in jobs))
            self.assertEqual(jobs[0]['values']['adaptation_warmup_epochs'],1)
            full=build_jobs(root/'full',root/'lists',['dann_dark'],[42],root/'snapshot',True)
            self.assertEqual(full[0]['values']['stop_after_epochs'],0)
            self.assertEqual(full[0]['values']['epochs'],jobs[0]['values']['epochs'])
            early=build_jobs(root/'early',root/'lists',['mcc_foggy'],[42],root/'snapshot',True,
                             {'early_stopping_patience':10,'early_stopping_min_epochs':20})
            self.assertEqual(early[0]['values']['early_stopping_patience'],10)
            with self.assertRaises(ValueError):
                build_jobs(root/'bad',root/'lists',['mcc_foggy'],[42],root/'snapshot',True,
                           {'early_stopping_min_epochs':21})
            heldout=build_jobs(root/'holdout',root/'lists',['mcc_foggy'],[42],root/'snapshot',True,
                               evaluation_mode='holdout_ce')[0]
            self.assertEqual(heldout['values']['selection_metric'],'cross_entropy')
            self.assertEqual(heldout['values']['early_stopping_monitor'],'cross_entropy')
            smoke=preflight_command(heldout['command'],root/'smoke')
            self.assertNotIn('--holdout-list',smoke)
            self.assertIn('--holdout-list',heldout['command'])

    def test_summary_uses_same_epoch_metrics_and_marks_missing_seeds(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            rows=[{'method':'dann','target':'dark','seed':seed,'screening_only':False,
                   **{k:value for k in ('ACC_percent','Precision_macro_percent','Recall_macro_percent','F1_macro_percent')}}
                  for seed,value in [(42,80.),(43,90.)]]
            summary=write_pair_summary(root,rows,[42,43,44])
            self.assertEqual(summary[0]['ACC_percent_mean'],85.)
            self.assertEqual(summary[0]['ACC_percent_std'],5.)
            self.assertFalse(summary[0]['all_requested_seeds_completed'])
            with self.assertRaises(ValueError):write_pair_summary(root,rows+rows[:1],[42,43])


if __name__=='__main__':unittest.main()
