"""Server candidate scope, native architectures and recorded diagnostic integrity."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from tools.run_weather_fewshot_suite import build_plan,required_weights
from tools.run_label_guided_matrix import build_jobs,main
from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
from tools.test_weather_vds_core import fixture,fast_five_method_model
from weather import train_vds_core as weather
from weather.compact_diagnostics import export_run_diagnostics
from weather.fewshot_presets import ALLOCATION
from weather.server_adaptive_presets import RECIPE
from weather.summarize_class_matching import summarize
from vds2raw.adaptation import AdaptationModel


class ServerAdaptiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_all_seven_presets_parse_and_keep_native_loss_and_label_protocol(self):
        root=Path.cwd()
        old=build_plan('server',root/'out',root/'lists',root,(46,))
        new=build_plan('server',root/'out',root/'lists',root,(46,),RECIPE)
        self.assertEqual(new['pairs'],list(ALLOCATION['server']))
        self.assertEqual(new['total'],7)
        self.assertEqual(required_weights(new),['convnext_small','unip_tiny'])
        for previous,job in zip(old['jobs'],new['jobs']):
            a=weather.parser().parse_args(recorded_arguments_to_cli(job['values']))
            weather.resolve_weather_settings(a)
            with self.subTest(job=job['name']):
                self.assertEqual(a.trade_off,previous['values']['trade_off'])
                self.assertEqual(a.initial_vis_weight,previous['values']['initial_vis_weight'])
                for key in ('source_weather_probability','source_aspect_min','source_aspect_max','source_zoom_min_area'):
                    self.assertEqual(job['values'].get(key),previous['values'].get(key))
                self.assertEqual(a.batch_size,previous['values']['batch_size'])
                self.assertEqual(a.target_batch_size,a.batch_size)
                self.assertEqual(a.target_labels_per_class,3)
                self.assertEqual(a.target_class_matching,'target_truth')
                self.assertEqual(a.source_anchor_ema_momentum,.9)
                self.assertEqual(a.matched_alignment_weight,.2)
                self.assertEqual(a.classifier_layout,'reference')
                self.assertEqual(a.epochs,40);self.assertEqual(a.early_stopping_patience,0)
                self.assertIsNone(a.holdout_list)
                self.assertEqual(a.prediction_distillation_weight,0)
                self.assertEqual(a.prediction_teacher_weight,0)
                self.assertEqual(a.domain_confusion_weight,0)
                self.assertEqual(a.validation_control_start_epoch,a.adaptation_warmup_epochs+a.adaptation_ramp_epochs+1)
                self.assertEqual(a.image_lr_override,5e-5)
                self.assertEqual(a.ir_lr_override,1e-4)
                self.assertEqual(a.image_adaptation_lr_scale,.1)
                self.assertEqual(job['required_free_MiB'],6144)
                self.assertIn('matched_alignment_weight',job['changes_from_previous_server_preset'])
        again=build_plan('server',root/'out',root/'lists',root,(46,))
        self.assertEqual(old,again)
        self.assertEqual(build_plan('server',root/'out',root/'lists',root,tuple(range(42,47)),RECIPE)['total'],35)

    def test_native_classifier_shapes_are_method_specific(self):
        for method in ('dann','deepcoral','cdan','mdd','mcc'):
            model=AdaptationModel(method,pretrained=False,num_classes=8,use_ais=False,
                                  ir_channels=3,classifier_layout='reference')
            if method=='deepcoral':self.assertIsInstance(model.bottleneck,nn.Identity)
            elif method=='cdan':self.assertIsInstance(model.bottleneck,nn.Linear)
            else:self.assertIsInstance(model.bottleneck,nn.Sequential)
            if method=='mdd':self.assertIsInstance(model.adv_head,nn.Sequential)
            self.assertEqual(model.mdd.margin,4)
            self.assertEqual(model.mcc.temperature,2.5)
            del model

    def test_new_recipe_cannot_leak_into_local_or_unrelated_protocol(self):
        for machine in ('local',):
            with self.assertRaises(ValueError):build_plan(machine,Path('out'),Path('lists'),Path('snap'),(46,),RECIPE)
        with self.assertRaises(ValueError):
            build_jobs(Path('out'),Path('lists'),['dann_rainy'],[46],Path('snap'),unified_fewshot=True,fewshot_recipe=RECIPE)
        argv=['runner','--data-dir','missing','--fewshot-recipe',RECIPE,'--unified-fewshot',
            '--machine-split','server','--pairs',*ALLOCATION['server'],'--target-labels-per-class','3']
        with patch('sys.argv',argv+['--matched-alignment-weight','.1']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:main()
            self.assertEqual(error.exception.code,2)

    def test_cpu_all_five_native_losses_ema_and_export_keep_actual_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            for method in ('dann','deepcoral','cdan','mdd','mcc'):
                with self.subTest(method=method):
                    args=weather.parser().parse_args(['--method',method,'--target','backlight','--data-dir',str(lists),
                        '--save-dir',str(root/method),'--epochs','2','--batch-size','2','--num-workers','0',
                        '--cpu-threads','2','--device','cpu','--no-pretrained','--classifier-layout','reference',
                        '--optimizer','adamw','--lr','.001','--log-parameter-updates','--auxiliary-source-weight','.5',
                        '--target-class-matching','target_truth','--target-labels-per-class','1',
                        '--matched-alignment-weight','.2','--source-anchor-ema-momentum','.9',
                        '--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','0','--gradient-clip-norm','5'])
                    with patch.object(weather,'AdaptationModel',fast_five_method_model),contextlib.redirect_stdout(io.StringIO()):
                        result=weather.run(args)
                    folder=root/method/f'{method}_backlight_seed42'
                    summarize([folder/'result.json'],root/'summary'/method)
                    exported=export_run_diagnostics(root/'exports',folder/'result.json')
                    self.assertEqual(exported['peak_target_val']['metrics'],result['peak_target_val'])
                    self.assertEqual(exported['CE_selected_target_val']['metrics'],result['target_val'])
                    self.assertEqual(exported['test_evaluations'],0)
                    self.assertEqual(exported['final_source_prototype_diagnostic']['target_samples_in_prototype_updates'],0)
                    # Re-export replaces this run only; neither duplicate nor loss of other runs.
                    export_run_diagnostics(root/'exports',folder/'result.json')
                    gradients=[json.loads(s) for s in (folder/'gradients.jsonl').read_text().splitlines()]
                    self.assertIn('core',gradients[-1]['enhancement_components'])
                    self.assertIn('matched_class_alignment',gradients[-1]['enhancement_components'])
                    corrupted=copy.deepcopy(result);corrupted['peak_target_val']['accuracy']+=.01
                    (folder/'result.json').write_text(json.dumps(corrupted),encoding='utf-8')
                    with self.assertRaises(ValueError):export_run_diagnostics(root/'bad',folder/'result.json')
                    (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
            runs=[json.loads(s) for s in (root/'exports/run_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
            epochs=[json.loads(s) for s in (root/'exports/epoch_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(runs),5);self.assertEqual(len(epochs),10)
            self.assertEqual(len({r['run_id'] for r in runs}),5)


if __name__=='__main__':unittest.main()
