"""Prevent unintended training/protocol changes in the bounded CORAL tests."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from tools.run_deepcoral_dark_tuning import build_trials,validate_cohort,validate_locked_inputs,TRIALS
from tools.run_label_guided_matrix import build_jobs
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.test_weather_vds_core import fixture,fast_five_method_model
from weather import train_vds_core as weather
from weather.compact_diagnostics import export_run_diagnostics
from weather.server_adaptive_presets import RECIPE


def baseline_values():
    return build_jobs(Path('out'),Path('lists'),['deepcoral_dark'],[46],Path('.'),True,
        evaluation_mode='validation_ce',unified_fewshot=True,fewshot_recipe=RECIPE)[0]['values']


class DarkCoralTuningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_four_separate_trials_change_only_declared_setting_and_metadata(self):
        baseline=baseline_values();before=copy.deepcopy(baseline)
        jobs=build_trials(baseline,Path('output'),Path('snapshot'))
        self.assertEqual(len(jobs),4);self.assertEqual(baseline,before)
        for job in jobs:
            trial=job['trial'];values=job['values']
            changed={k for k in set(baseline)|set(values) if baseline.get(k)!=values.get(k)}
            self.assertEqual(changed,set(TRIALS[trial])|{'save_dir','configuration_tag'})
            args=weather.parser().parse_args(recorded_arguments_to_cli(values));weather.resolve_weather_settings(args)
            self.assertEqual(args.method,'deepcoral');self.assertEqual(args.target,'dark');self.assertEqual(args.seed,46)
            self.assertEqual(args.epochs,40);self.assertEqual(args.early_stopping_patience,0)
            self.assertEqual(args.target_labels_per_class,3);self.assertEqual(args.matched_alignment_weight,.2)
            self.assertEqual(args.source_anchor_ema_momentum,.9);self.assertIsNone(args.holdout_list)
            self.assertEqual(args.prediction_distillation_weight,0.)
            self.assertEqual(args.classifier_layout,'reference')
        self.assertEqual(len({job['result_file'] for job in jobs}),4)

    def test_rejects_wrong_baseline_unknown_or_repeated_trials_and_changed_label_pool(self):
        for key,value in (('method','mdd'),('target_labels_per_class',0),('seed',42),('matched_alignment_weight',.3)):
            baseline=baseline_values();baseline[key]=value
            with self.assertRaises(ValueError):build_trials(baseline,Path('out'),Path('.'))
        for trials in ([],['missing'],['coral0p1','coral0p1']):
            with self.assertRaises(ValueError):build_trials(baseline_values(),Path('out'),Path('.'),trials)
        pool={'labels_per_class':3,'labelled_target_samples':24,'unlabelled_target_samples':349,
              'labelled_target_class_counts':{'0':3},'label_pool_portable_sha256':'fixed',
              'unselected_truth_used_for_pairing_or_alignment':False}
        before={'label_budget':pool}
        result={'protocol':'FEWSHOT_TARGET_CLASS_ALIGNMENT','test_evaluations':0,
                'target_true_labels_used_in_classification_loss':False,'label_guided_sampling':copy.deepcopy(pool)}
        validate_cohort(before,result)
        result['label_guided_sampling']['label_pool_portable_sha256']='changed'
        with self.assertRaises(ValueError):validate_cohort(before,result)

    def test_actual_source_target_validation_and_pair_hashes_are_locked(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);hashes={}
            for filename in ('sunny_train.txt','dark_train.txt','dark_val.txt','dark_train_pairs.tsv'):
                path=root/filename;path.write_text('same cohort\n',encoding='utf-8')
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            baseline={'code_provenance':{'explicit_input_sha256':hashes}}
            validate_locked_inputs(baseline,root)
            (root/'dark_train_pairs.tsv').write_text('other cohort\n',encoding='utf-8')
            with self.assertRaises(ValueError):validate_locked_inputs(baseline,root)

    def test_actual_coral_training_and_namespaced_diagnostics_keep_all_four_trials(self):
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            for job in build_trials(baseline_values(),root/'trials',root):
                # Synthetic CPU plumbing check: native CORAL and enhancement
                # losses stay real; small image encoders/fixture keep it short.
                args=weather.parser().parse_args(['--method','deepcoral','--target','backlight',
                    '--data-dir',str(lists),'--save-dir',str(root/job['trial']),'--epochs','2',
                    '--batch-size','2','--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained',
                    '--optimizer','adamw','--lr','.001','--target-class-matching','target_truth',
                    '--target-labels-per-class','1','--matched-alignment-weight','.2','--source-anchor-ema-momentum','.9',
                    '--log-parameter-updates','--auxiliary-source-weight','.5','--adaptation-warmup-epochs','0',
                    '--adaptation-ramp-epochs','0','--crossmodal-projection-dim','8'])
                args.source_crossmodal_weight=.1;args.crossmodal_stop_gradient='vis'
                for key,value in TRIALS[job['trial']].items():setattr(args,key,value)
                with patch.object(weather,'AdaptationModel',fast_five_method_model),contextlib.redirect_stdout(io.StringIO()):
                    result=weather.run(args)
                folder=root/job['trial']/'deepcoral_backlight_seed42'
                exported=export_run_diagnostics(root/'diagnostics',folder/'result.json',job['trial'])
                self.assertEqual(exported['peak_target_val']['metrics'],result['peak_target_val'])
                self.assertTrue(exported['run_id'].startswith(job['trial']+':'))
                # Repeated export updates only this trial, preserving its peers.
                export_run_diagnostics(root/'diagnostics',folder/'result.json',job['trial'])
            rows=[json.loads(s) for s in (root/'diagnostics/run_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
            epochs=[json.loads(s) for s in (root/'diagnostics/epoch_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows),4);self.assertEqual(len(epochs),8)
            self.assertEqual(len({r['run_id'] for r in rows}),4)


if __name__=='__main__':unittest.main()
