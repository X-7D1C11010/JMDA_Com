"""Parameter isolation and live native-DA encoder gradients for new trials."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import torch

from tools.run_weather_stability import build_jobs,preflight_values,runtime_audit,completed_result
from tools.test_weather_encoder_screen import TinyConvNeXt
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from weather.rainy_presets import arguments_for
from weather.train_vds_core import parser,resolve_weather_settings
from vds2raw.adaptation import AdaptationModel
from vds2raw.train_adaptation import make_optimizer


class StabilityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_backlight_changes_only_new_encoder_training_and_logging(self):
        root=Path(__file__).resolve().parents[1]
        jobs=build_jobs(root,root/'out',scope='backlight',source_controls=True)
        self.assertEqual(len(jobs),6)
        for job in jobs:
            values=job['values'];args=parser().parse_args(recorded_arguments_to_cli(values));resolve_weather_settings(args)
            self.assertEqual(args.encoder_training_scope,'last_stages')
            self.assertEqual(args.image_lr_override,1e-5)
            self.assertEqual(args.target_class_matching,'none')
            self.assertEqual(args.matched_alignment_weight,0.)
            self.assertTrue(args.log_parameter_updates)
            self.assertEqual(args.epochs,60)
            if job['comparison_role']=='source_only_control':self.assertEqual(args.trade_off,0.)
        self.assertEqual(len({j['result_file'] for j in jobs}),6)

    def test_rainy_only_changes_distillation_strength_and_logging(self):
        root=Path(__file__).resolve().parents[1]
        jobs=build_jobs(root,root/'out',scope='rainy')
        self.assertEqual(len(jobs),20)
        skip={'save_dir','configuration_tag','prediction_distillation_weight','log_parameter_updates',
              'target_class_matching','matched_alignment_weight'}
        for job in jobs:
            values=job['values'];ref=arguments_for(job['method'],root,root/'out/lists/rainy',root/'out',job['seed'])
            for key in ref.keys()-skip:self.assertEqual(values[key],ref[key],(job['name'],key))
            self.assertIn(values['prediction_distillation_weight'],(1.,.5))
        self.assertEqual(len({j['result_file'] for j in jobs}),20)

    def test_full_server_plan_has_eighty_trials_and_isolates_the_encoder_lr_change(self):
        root=Path(__file__).resolve().parents[1]
        full=build_jobs(root,root/'full',backlight_seeds=tuple(range(42,47)),rainy_seeds=tuple(range(42,47)),
                        source_controls=True,backlight_image_lr=1e-4)
        self.assertEqual(len(full),80)
        self.assertEqual(len({j['result_file'] for j in full}),80)
        self.assertEqual(sum(j['target']=='backlight' for j in full),30)
        self.assertEqual(sum(j['target']=='rainy' for j in full),50)
        for target in ('backlight','rainy'):
            self.assertEqual({j['seed'] for j in full if j['target']==target},set(range(42,47)))
        old=build_jobs(root,root/'old',scope='backlight',source_controls=True)
        new=build_jobs(root,root/'new',scope='backlight',source_controls=True,backlight_image_lr=1e-4)
        for before,after in zip(old,new):
            self.assertNotEqual(before['variant'],after['variant'])
            self.assertEqual(after['values']['image_lr_override'],1e-4)
            for key in before['values'].keys()-{'save_dir','configuration_tag','data_dir','image_lr_override'}:
                self.assertEqual(before['values'][key],after['values'][key],key)

    def test_invalid_encoder_lr_is_rejected(self):
        for rate in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):build_jobs(Path('.'),Path('out'),backlight_image_lr=rate)

    def test_environment_check_failure_never_dispatches_or_creates_a_run(self):
        from tools import run_weather_stability as runner
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/'not_created'
            with patch.object(sys,'argv',['check','--check-environment','--data-root',name,'--output',str(output)]), \
                    patch.object(runner,'check_environment',side_effect=ImportError('missing sklearn')), \
                    patch.object(runner.subprocess,'run') as dispatch:
                with self.assertRaisesRegex(ImportError,'missing sklearn'):runner.main()
            dispatch.assert_not_called();self.assertFalse(output.exists())

    def test_last_stage_gets_native_da_gradients_for_each_backlight_method(self):
        root=Path(__file__).resolve().parents[1]
        for job in build_jobs(root,root/'out',scope='backlight'):
            args=parser().parse_args(recorded_arguments_to_cli(job['values']));resolve_weather_settings(args)
            args.steps_per_epoch=2
            with patch('vds2raw.adaptation.convnext_small',side_effect=lambda **_:TinyConvNeXt()):
                model=AdaptationModel(args.method,pretrained=False,use_ais=False,ir_channels=3,
                    num_classes=4,image_backbone='convnext_small',encoder_training_scope=args.encoder_training_scope,
                    feature_normalization=args.feature_normalization,classifier_layout=args.classifier_layout)
            model.train()
            source={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'label':torch.arange(4)}
            target={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8)}
            loss,details=model.objective(source,target,20,120,args)
            for encoder in (model.rgb_encoder,model.nir_encoder):
                params=[p for p in encoder.parameters() if p.requires_grad]
                native=torch.autograd.grad(details['weighted_core_transfer'],params,retain_graph=True,allow_unused=True)
                self.assertGreater(sum(float(g.abs().sum()) for g in native if g is not None),0.)
                self.assertTrue(all(not p.requires_grad for p in encoder.features[0].parameters()))
            optimizer,*_=make_optimizer(model,args)
            before=[p.detach().clone() for p in model.rgb_encoder.features[-1].parameters()]
            loss.backward();optimizer.step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.rgb_encoder.features[-1].parameters())))

    def test_preflight_exercises_core_and_distillation_without_heldout(self):
        original={'holdout_list':'forbidden','epochs':60,'encoder_warmup_freeze_epochs':5}
        probe=preflight_values(original,Path('preflight'))
        self.assertIsNone(probe['holdout_list'])
        self.assertEqual((probe['max_train_batches'],probe['stop_after_epochs']),(2,2))
        self.assertEqual(probe['adaptation_warmup_epochs'],0)
        self.assertEqual(probe['encoder_warmup_freeze_epochs'],0)
        self.assertEqual(original['encoder_warmup_freeze_epochs'],5)

    def test_runtime_audit_rejects_zero_native_encoder_gradient(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)
            (p/'result.json').write_text(json.dumps({'protocol':'UDA','target_training_labels_used':False}))
            (p/'run_config.json').write_text(json.dumps({'arguments':{'target_class_matching':'none'}}))
            (p/'gradients.jsonl').write_text(json.dumps({'gradients':{k:{'CE_norm':1.,'weighted_DA_norm':0.} for k in ('VIS','IR')}})+'\n')
            (p/'parameter_updates.jsonl').write_text(json.dumps({'sampled_parameters':{k+'_last_trainable_matrix':{'update_L2':.1} for k in ('VIS','IR')}})+'\n')
            with self.assertRaisesRegex(ValueError,'Inactive VIS'):runtime_audit(p/'result.json')
            self.assertEqual(runtime_audit(p/'result.json',False)['status'],'PASS')

    def test_resume_rejects_partial_formal_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);job={'result_file':str(root/'trial/result.json')}
            self.assertIsNone(completed_result(job))
            (root/'trial').mkdir();(root/'trial/progress.log').write_text('partial')
            with self.assertRaisesRegex(ValueError,'Partial formal output'):completed_result(job)

    def test_resume_checks_saved_parameters_before_skipping_training(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as name:
            job=build_jobs(root,Path(name),scope='backlight')[0];p=Path(job['result_file']);p.parent.mkdir(parents=True)
            result={**{k:job[k] for k in ('method','target','seed')},'completed_epochs':60,'smoke':False,'screening_only':False}
            p.write_text(json.dumps(result));(p.parent/'completeness.json').write_text(json.dumps({'status':'PASS'}))
            args=parser().parse_args(recorded_arguments_to_cli(job['values']));resolve_weather_settings(args)
            config=p.parent/'run_config.json';config.write_text(json.dumps({'arguments':vars(args)},default=str))
            with patch('tools.run_weather_stability.runtime_audit'):
                self.assertEqual(completed_result(job)['seed'],42)
                args.lr*=2;config.write_text(json.dumps({'arguments':vars(args)},default=str))
                with self.assertRaisesRegex(ValueError,'Saved configuration differs: lr'):completed_result(job)

    def test_resuming_all_complete_jobs_dispatches_no_worker_or_data_preparation(self):
        from tools import run_weather_stability as runner
        root=Path(__file__).resolve().parents[1];parent=root/'runs/weather_stability';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as name:
            out=Path(name);jobs=build_jobs(root,out,scope='backlight')
            snapshot=out/'snapshot';snapshot.mkdir();(snapshot/'runtime_snapshot_manifest.json').write_text(json.dumps({'files':{}}))
            (out/'plan.json').write_text(json.dumps({'jobs':jobs,'snapshot':str(snapshot),'data_root':str(out/'data')}))
            def done(job):return {'selected_epoch':10,'completed_epochs':60,'target_val':dict(accuracy=.8,precision_macro=.8,recall_macro=.8,f1_macro=.8)}
            with patch.object(sys,'argv',['queue','--resume',str(out),'--execute']), \
                    patch.object(runner,'completed_result',side_effect=done), \
                    patch('tools.prepare_mcc_weather_v2_snapshot._verified',return_value=True), \
                    patch.object(runner,'check_environment'), \
                    patch.object(runner.subprocess,'run') as dispatch,contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            dispatch.assert_not_called()
            status=json.loads((out/'queue_status.json').read_text())
            self.assertEqual((status['status'],status['completed']),('PASS',3))


if __name__=='__main__':unittest.main()
