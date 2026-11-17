"""Scope, mixed-protocol separation and selected-prediction audit contracts."""
import csv
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tools import run_weather_all as runner
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.test_weather_encoder_screen import TinyConvNeXt
from tools.test_weather_vds_core import fixture,fast_five_method_model
from weather.class_matching import ALIGNMENT_PROTOCOL
from weather.train_vds_core import parser,resolve_weather_settings
from vds2raw.adaptation import AdaptationModel


class AllWeatherChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_exact_requested_scope_and_protocols(self):
        plan=runner.build_plan(Path('out'),Path('snapshot'))
        self.assertEqual(plan['total'],110)
        self.assertEqual(plan['counts_by_weather'],{'backlight':10,'rainy':50,'dark':25,'foggy':25})
        self.assertEqual(len({j['name'] for j in plan['jobs']}),110)
        self.assertEqual(len({j['result_file'] for j in plan['jobs']}),110)
        for target in ('backlight','rainy','dark','foggy'):
            jobs=[j for j in plan['jobs'] if j['target']==target]
            self.assertEqual({j['method'] for j in jobs},{'cdan','mcc'} if target=='backlight' else set(runner.METHODS))
            self.assertEqual({j['seed'] for j in jobs},set(range(42,47)))
            for job in jobs:
                guided=target in ('dark','foggy')
                self.assertEqual(job['target_training_labels_used'],guided)
                self.assertEqual(job['protocol'],ALIGNMENT_PROTOCOL if guided else 'UDA')
                self.assertEqual(job['values']['target_class_matching'],'target_truth' if guided else 'none')
                self.assertEqual(job['values']['matched_alignment_weight'],5. if guided else 0.)
                self.assertTrue(job['values']['log_parameter_updates'])
        self.assertFalse(plan['score_caps_applied'])
        self.assertFalse(any(j['comparison_role']=='source_only_control' for j in plan['jobs']))

    def test_single_rain_weight_has_seventeen_method_weather_pairs(self):
        plan=runner.build_plan(Path('out'),Path('snapshot'),seeds=(42,),rainy_weights=(1.,))
        self.assertEqual(plan['total'],17)
        for job in plan['jobs']:
            args=parser().parse_args(recorded_arguments_to_cli(job['values']));resolve_weather_settings(args)
            self.assertEqual(args.seed,42)
            if job['target']=='backlight':self.assertEqual(args.image_lr_override,1e-4)
            if job['target'] in ('dark','foggy'):
                self.assertEqual(args.selection_metric,'cross_entropy');self.assertIsNone(args.holdout_list)

    def test_machine_split_is_disjoint_and_covers_full_plan(self):
        server=runner.build_plan(Path('out'),Path('snapshot'),targets=('rainy','dark'))
        local=runner.build_plan(Path('out'),Path('snapshot'),targets=('backlight','foggy'))
        full=runner.build_plan(Path('out'),Path('snapshot'))
        self.assertEqual(server['counts_by_weather'],{'rainy':50,'dark':25})
        self.assertEqual(local['counts_by_weather'],{'backlight':10,'foggy':25})
        self.assertEqual(server['total'],75);self.assertEqual(local['total'],35)
        names=lambda plan:{j['name'] for j in plan['jobs']}
        self.assertFalse(names(server)&names(local))
        self.assertEqual(names(server)|names(local),names(full))
        for plan in (server,local):
            for job in plan['jobs']:
                self.assertEqual(job['target_training_labels_used'],job['target'] in ('dark','foggy'))
                counterpart=next(j for j in full['jobs'] if j['name']==job['name'])
                self.assertEqual(job['values'],counterpart['values'])

    def test_invalid_weather_targets_rejected(self):
        for targets in ((),('rainy','rainy'),('unknown',)):
            with self.subTest(targets=targets),self.assertRaises(ValueError):
                runner.build_plan(Path('out'),Path('snapshot'),targets=targets)

    def test_machine_allocation_rejected_before_dispatch(self):
        for argv in (['--execute','--targets','backlight'],['--local','--execute','--targets','rainy'],
                     ['--check-environment','--targets','foggy']):
            with self.subTest(argv=argv),patch('sys.argv',['runner',*argv]),patch.object(runner,'check_all') as check,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:runner.main()
                self.assertEqual(error.exception.code,2);check.assert_not_called()

    def test_small_unip_weight_only_required_for_local_foggy(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(runner,'check_environment') as environment,patch('tools.download_official_uda_weights.require_weight') as require:
            runner.check_all(directory,('rainy','dark'))
            environment.assert_called_once_with('both');require.assert_not_called()
            runner.check_all(directory,('backlight','foggy'))
            require.assert_called_once_with('unip_small',runner.ROOT/'pretrained_weights')

    def test_cdan_newly_enabled_stages_receive_core_gradients(self):
        job=next(j for j in runner.build_plan(Path('out'),Path('snapshot'),seeds=(42,))['jobs'] if j['target']=='backlight' and j['method']=='cdan')
        args=parser().parse_args(recorded_arguments_to_cli(job['values']));resolve_weather_settings(args);args.steps_per_epoch=2
        with patch('vds2raw.adaptation.convnext_small',side_effect=lambda **_:TinyConvNeXt()):
            model=AdaptationModel('cdan',pretrained=False,num_classes=9,use_ais=False,ir_channels=3,
                image_backbone='convnext_small',encoder_training_scope='last_stages',feature_normalization='unit')
        model.unit_method_features=args.unit_method_features;model.train()
        source={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'label':torch.arange(4)}
        target={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8)}
        _,details=model.objective(source,target,20,120,args)
        for encoder in (model.rgb_encoder,model.nir_encoder):
            gradients=torch.autograd.grad(details['weighted_core_transfer'],[p for p in encoder.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients if g is not None),0.)

    def test_guided_probe_rejects_target_classifier_ce_or_broken_matching(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'result.json';result={'protocol':ALIGNMENT_PROTOCOL,'target_training_labels_used':True,
                'target_true_labels_used_in_alignment_loss':True,'target_true_labels_used_in_classification_loss':False}
            p.write_text(json.dumps(result));updates=Path(name)/'updates.jsonl'
            updates.write_text(json.dumps({'same_true_class_at_every_position':False,'weighted_matched_alignment':1.})+'\n')
            job={'protocol':ALIGNMENT_PROTOCOL}
            with self.assertRaisesRegex(ValueError,'class matching'):runner.validate_probe(job,p)
            updates.write_text(json.dumps({'same_true_class_at_every_position':True,'weighted_matched_alignment':1.})+'\n')
            self.assertEqual(runner.validate_probe(job,p)['status'],'PASS')
            result['target_true_labels_used_in_classification_loss']=True;p.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'Wrong label-guided'):runner.validate_probe(job,p)

    def test_statistics_never_pool_protocols_or_rain_variants(self):
        with tempfile.TemporaryDirectory() as name:
            jobs=runner.build_plan(Path(name),Path('snapshot'),seeds=(42,43))['jobs'];rows=[]
            chosen=[j for j in jobs if j['method']=='cdan' and j['target'] in ('backlight','rainy','dark')]
            for job in chosen:
                result={'selected_epoch':2,'completed_epochs':20,'selection_metric':job['values'].get('selection_metric','accuracy_f1'),
                        'target_val':dict(accuracy=.8+(job['seed']-42)*.1,precision_macro=.8,recall_macro=.8,f1_macro=.8)}
                rows.append(runner.make_row(job,result))
            runner.summarize(Path(name),rows,[],jobs)
            with (Path(name)/'summary.csv').open(encoding='utf-8-sig') as f:means=list(csv.DictReader(f))
            self.assertEqual(len(means),4)
            self.assertTrue(all(int(r['num_seeds'])==2 for r in means))
            self.assertTrue(all(abs(float(r['ACC_mean'])-85.)<1e-8 and abs(float(r['ACC_std'])-5.)<1e-8 for r in means))
            self.assertEqual({r['protocol'] for r in means},{'UDA',ALIGNMENT_PROTOCOL})
            self.assertTrue(all(r['all_requested_seeds_completed']=='True' for r in means))

    def test_prediction_fingerprint_separates_same_classes_from_different_probabilities(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);job={'result_file':str(root/'result.json'),'method':'mcc','target':'backlight',
                                'seed':42,'variant':'test','protocol':'UDA'}
            result={'selected_epoch':2,'target_val':{key:1. for key in runner.METRICS.values()},'modality_weights_at_selected_epoch':[.5,.5]}
            for file,payload in (('run_config.json',{'arguments':{}}),('selected_target_val_predictions.json',
                [{'sample_id':'a','y_true':0,'y_pred':0,'probabilities':[.7,.3]}])):
                (root/file).write_text(json.dumps(payload))
            (root/'gradients.jsonl').write_text(json.dumps({'epoch':2,'gradients':{}})+'\n')
            (root/'parameter_updates.jsonl').write_text(json.dumps({'epoch':2,'sampled_parameters':{}})+'\n')
            first=runner.make_diagnostic(job,result)
            (root/'selected_target_val_predictions.json').write_text(json.dumps(
                [{'sample_id':'a','y_true':0,'y_pred':0,'probabilities':[.6,.4]}]))
            second=runner.make_diagnostic(job,result)
            self.assertEqual(first['predicted_class_sha256'],second['predicted_class_sha256'])
            self.assertNotEqual(first['probability_sha256'],second['probability_sha256'])
            self.assertEqual(second['selected_predictions'][0]['y_true'],0)
            second['variant']='second'
            compared=runner.write_prediction_comparisons(root,[first,second])
            self.assertEqual(len(compared),1)
            self.assertTrue(compared[0]['same_class_predictions'])
            self.assertFalse(compared[0]['same_probabilities'])
            self.assertTrue(compared[0]['all_four_metrics_equal'])
            second['selected_predictions'][0]['y_true']=1
            with self.assertRaisesRegex(ValueError,'class mapping'):runner.write_prediction_comparisons(root,[first,second])

    def test_both_protocols_complete_real_cpu_training_validation_and_export(self):
        from weather import train_vds_core as weather
        for guided in (False,True):
            with self.subTest(guided=guided),tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
                root=Path(name);lists=fixture(root)
                args=parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                    '--save-dir',str(root/'output'),'--device','cpu','--no-pretrained','--epochs','2',
                    '--batch-size','4','--target-batch-size','4','--num-workers','0','--cpu-threads','2',
                    '--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','0',
                    '--log-parameter-updates','--feature-normalization','unit',
                    '--target-class-matching','target_truth' if guided else 'none',
                    '--matched-alignment-weight','5' if guided else '0'])
                resolve_weather_settings(args);values=dict(vars(args))
                with patch.object(weather,'AdaptationModel',fast_five_method_model),contextlib.redirect_stdout(io.StringIO()):
                    result=weather.run(args)
                job={'name':'fixture','method':'mcc','target':'backlight','seed':42,'variant':'test',
                    'protocol':ALIGNMENT_PROTOCOL if guided else 'UDA','target_training_labels_used':guided,
                    'comparison_role':'label_guided_adaptation' if guided else 'adaptation',
                    'minimum_epochs':2,'values':values,'result_file':str(root/'output/mcc_backlight_seed42/result.json')}
                validated=runner.validate_formal(job)
                self.assertEqual(validated['protocol'],job['protocol'])
                self.assertEqual(runner.validate_probe(job,job['result_file'])['status'],'PASS')
                diagnostic=runner.make_diagnostic(job,result)
                self.assertEqual(len(diagnostic['selected_predictions']),4)
                self.assertEqual(len(diagnostic['predicted_class_sha256']),64)


if __name__=='__main__':unittest.main()
