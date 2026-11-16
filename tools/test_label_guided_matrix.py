"""Portable preset and matrix-plan contracts without training or downloads."""
from pathlib import Path
import tempfile
import unittest

from weather.label_guided_presets import REFERENCE_PARAMETERS,build_preset
from weather.train_vds_core import parser
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.run_label_guided_matrix import build_jobs,write_pair_summary,preflight_command


class GuidedMatrixChecks(unittest.TestCase):
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
