import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from weather import train_vds_core as weather
from tools import run_weather_crossmodal_screen as controller


class ScreenReferenceChecks(unittest.TestCase):
    def test_actual_controller_preserves_reference_stop_gradient_and_records_explicit_override(self):
        args=weather.parser().parse_args(['--method','deepcoral','--target','dark','--data-dir','original',
            '--save-dir','original','--source-crossmodal-weight','.1','--target-crossmodal-weight','.1',
            '--crossmodal-projection-dim','128','--crossmodal-stop-gradient','vis'])
        weather.resolve_weather_settings(args)
        # A real historical configuration predates the new anchor option.
        delattr(args,'target_representation_anchor_weight')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);reference=root/'reference.json'
            reference.write_text(json.dumps({'method':'deepcoral','image_backbone':'resnet18','arguments':vars(args)}))
            for explicit,expected,label in (([], 'vis','preserve'),(['--stop-gradient','ir'],'ir','override'),
                    (['--target-representation-anchor-weight','1'],'vis','new_option'),
                    (['--auxiliary-source-weight','.5'],'vis','auxiliary_source'),
                    (['--adaptation-warmup-epochs','12','--adaptation-ramp-epochs','8',
                      '--image-adaptation-lr-scale','.2'],'vis','phased_adaptation'),
                    (['--ir-backbone','unip_tiny','--ir-weights',str(root/'thermal weights.pth'),
                      '--ir-training-scope','all_used_layers'],'vis','thermal_ir')):
                output=root/label
                argv=['screen','--reference-config',str(reference),'--output',str(output),'--methods','deepcoral',
                    '--targets','dark','--source-weights','.1','--target-weight','.1','--projection-dim','128',
                    '--epochs','20','--minimum-epochs','20','--plan-only',*explicit]
                with patch.object(sys,'argv',argv),patch.object(controller,'prepare_snapshot',return_value=root/'snapshot'),contextlib.redirect_stdout(io.StringIO()):
                    controller.main()
                plan=json.loads((output/'plan.json').read_text())
                job=plan['jobs'][0];command=job['command']
                self.assertEqual(command[command.index('--crossmodal-stop-gradient')+1],expected)
                changes=job['reference_argument_changes']
                if '--stop-gradient' in explicit:
                    self.assertEqual(changes['crossmodal_stop_gradient'],{'reference':'vis','candidate':'ir'})
                else:self.assertNotIn('crossmodal_stop_gradient',changes)
                if '--target-representation-anchor-weight' in explicit:
                    self.assertEqual(changes['target_representation_anchor_weight'],
                        {'reference':0.,'candidate':1.,'reference_default_used':True})
                if '--ir-backbone' in explicit:
                    self.assertEqual(command[command.index('--ir-backbone')+1],'unip_tiny')
                    self.assertEqual(command[command.index('--ir-weights')+1],str((root/'thermal weights.pth').resolve()))
                    self.assertEqual(changes['ir_training_scope'],{'reference':'inherit','candidate':'all_used_layers'})
                if '--auxiliary-source-weight' in explicit:
                    self.assertEqual(changes['auxiliary_source_weight'],{'reference':0.,'candidate':.5})
                for key,value in (('adaptation_warmup_epochs',12),('adaptation_ramp_epochs',8),
                                  ('image_adaptation_lr_scale',.2)):
                    if '--'+key.replace('_','-') in explicit:
                        self.assertEqual(changes[key]['candidate'],value)
                    else:self.assertNotIn(key,changes)
                self.assertFalse((output/'queue_status.json').exists())

    def test_existing_IR_configuration_is_preserved_when_no_override_is_given(self):
        args=weather.parser().parse_args(['--method','mcc','--target','foggy','--data-dir','original','--save-dir','original',
            '--ir-backbone','unip_tiny','--ir-weights','original thermal weights.pth',
            '--ir-feature-mode','cls','--ir-training-scope','all_used_layers'])
        weather.resolve_weather_settings(args)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);ref=root/'reference.json'
            ref.write_text(json.dumps({'method':'mcc','image_backbone':'resnet18','arguments':vars(args)}))
            argv=['screen','--reference-config',str(ref),'--output',str(root/'out'),'--methods','mcc','--targets','foggy',
                  '--source-weights','0','--target-weight','0','--epochs','20','--minimum-epochs','20','--plan-only']
            with patch.object(sys,'argv',argv),patch.object(controller,'prepare_snapshot',return_value=root/'snapshot'),contextlib.redirect_stdout(io.StringIO()):controller.main()
            job=json.loads((root/'out/plan.json').read_text())['jobs'][0]
            for key in ('ir_backbone','ir_weights','ir_feature_mode','ir_training_scope'):
                self.assertNotIn(key,job['reference_argument_changes'])
                option='--'+key.replace('_','-');self.assertEqual(job['command'][job['command'].index(option)+1],getattr(args,key))


if __name__=='__main__':unittest.main()
