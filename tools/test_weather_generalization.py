"""Locked splits, CE selection and heldout isolation through real training."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from weather import train_vds_core as weather
from weather.generalization import selection_improved
from weather.summarize_class_matching import summarize
from tools.prepare_weather_generalization_split import prepare
from tools.test_weather_vds_core import fixture,fast_five_method_model


class GeneralizationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_CE_selection_can_prefer_lower_accuracy_without_erasing_peak(self):
        before={'accuracy':.9,'f1_macro':.9};later={'accuracy':.95,'f1_macro':.95}
        self.assertTrue(selection_improved('accuracy_f1',later,1.,before,.5))
        self.assertFalse(selection_improved('cross_entropy',later,1.,before,.5))
        self.assertTrue(selection_improved('cross_entropy',before,.4,later,1.))

    def test_locked_split_preserves_classes_and_exact_pair_groups(self):
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);original=fixture(root)
            a=prepare(original,root/'a',['backlight'],.25,123)
            b=prepare(original,root/'b',['backlight'],.25,123)
            self.assertEqual(a['output_list_sha256'],b['output_list_sha256'])
            self.assertEqual(a['targets']['backlight']['target_train'],4)
            self.assertEqual(a['targets']['backlight']['target_test'],4)
            self.assertEqual((root/'a/backlight_val.txt').read_bytes(),(original/'backlight_val.txt').read_bytes())
            self.assertFalse(a['historically_unseen_test'])
            # Copy an identical original pair into validation under another path:
            # path-disjoint splits must still reject identical content.
            from tools.uda_data import read_image_list
            from tools.data_paths import exact_ir_partner
            s=read_image_list(original/'backlight_train.txt')[0][0]
            v=read_image_list(original/'backlight_val.txt')[0][0]
            Path(v).write_bytes(Path(s).read_bytes())
            Path(exact_ir_partner(v)).write_bytes(Path(exact_ir_partner(s)).read_bytes())
            with self.assertRaises(ValueError):prepare(original,root/'bad',['backlight'],.25,123)

    def test_all_five_entries_keep_test_out_of_training_and_evaluate_it_once(self):
        for method in ['dann','deepcoral','cdan','mdd','mcc']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
                root=Path(name);original=fixture(root);lists=root/'split'
                prepare(original,lists,['backlight'],.25,123)
                args=weather.parser().parse_args(['--method',method,'--target','backlight','--data-dir',str(lists),
                    '--save-dir',str(root/'output'),'--epochs','2','--batch-size','2','--num-workers','0','--cpu-threads','2',
                    '--device','cpu','--no-pretrained','--target-class-matching','target_truth','--matched-alignment-weight','1',
                    '--selection-metric','cross_entropy','--selection-min-epoch','2','--early-stopping-monitor','cross_entropy',
                    '--holdout-list',str(lists/'backlight_test.txt'),'--no-modality-diagnostics'])
                from tools.uda_data import read_image_list
                heldout={path for path,label in read_image_list(lists/'backlight_test.txt')}
                order=[];original_evaluate=weather.evaluate
                def evaluate(model,dataset,*rest,**kwargs):
                    order.append(dataset.partition)
                    if dataset.partition=='target_test':
                        self.assertEqual(order.count('target_val'),2)
                    return original_evaluate(model,dataset,*rest,**kwargs)
                def factory(*a,**kw):
                    model=fast_five_method_model(*a,**kw);original_objective=model.objective
                    def objective(source,target,*rest):
                        self.assertFalse(heldout&(set(source['path'])|set(target['path'])))
                        self.assertNotIn('target_test',order)
                        return original_objective(source,target,*rest)
                    model.objective=objective;return model
                with patch.object(weather,'AdaptationModel',factory),patch.object(weather,'evaluate',evaluate),contextlib.redirect_stdout(io.StringIO()):
                    r=weather.run(args)
                self.assertEqual(order.count('target_test'),1);self.assertEqual(order[-1],'target_test')
                self.assertEqual(r['test_evaluations'],1);self.assertEqual(r['target_test']['val_samples'],4)
                out=root/f'output/{method}_backlight_seed42'
                evaluation=[json.loads(l) for l in (out/'evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
                vals=[e for e in evaluation if e['partition']=='target_val']
                selected=next(e for e in vals if e['epoch']==r['selected_epoch'])
                self.assertEqual(selected['cross_entropy'],min(e['cross_entropy'] for e in vals if e['epoch']>=2))
                self.assertEqual(r['peak_target_val']['accuracy'],max(e['metrics']['accuracy'] for e in vals))
                self.assertEqual(len((out/'target_train_fit.jsonl').read_text(encoding='utf-8').splitlines()),2)
                row=summarize([out/'result.json'],root/'summary')[0]
                self.assertEqual(row['heldout_ACC_percent'],100*r['target_test']['accuracy'])
                self.assertEqual(row['ACC_percent'],row['heldout_ACC_percent'])
                self.assertEqual(row['reporting_partition'],'target_test')

    def test_locked_lists_reject_semantic_noop_edits_before_model_creation(self):
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);original=fixture(root);lists=root/'split';prepare(original,lists,['backlight'],.25,123)
            with (lists/'backlight_train.txt').open('a',encoding='utf-8') as stream:stream.write('\n')
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--holdout-list',str(lists/'backlight_test.txt'),'--device','cpu','--no-pretrained'])
            with patch.object(weather,'AdaptationModel',side_effect=AssertionError('Model must not be created')):
                with self.assertRaisesRegex(ValueError,'Locked retraining split list changed'):weather.run(args)

    def test_actual_loop_stops_on_validation_CE_plateau(self):
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'out'),'--epochs','5','--batch-size','2','--num-workers','0','--cpu-threads','2',
                '--device','cpu','--no-pretrained','--selection-metric','cross_entropy',
                '--early-stopping-monitor','cross_entropy','--early-stopping-patience','1','--early-stopping-min-epochs','2',
                '--validation-control-start-epoch','1','--no-modality-diagnostics'])
            original_CE=torch.nn.functional.cross_entropy
            def flat_validation_CE(logits,labels,*a,**kw):
                # Only evaluation requests per-sample CE; training means keep
                # the actual loss/gradient. This synthetic plateau is a test.
                if kw.get('reduction')=='none':return logits.new_ones(len(labels))
                return original_CE(logits,labels,*a,**kw)
            with patch.object(weather,'AdaptationModel',fast_five_method_model), \
                    patch('torch.nn.functional.cross_entropy',flat_validation_CE),contextlib.redirect_stdout(io.StringIO()):
                result=weather.run(args)
            self.assertEqual(result['completed_epochs'],2)
            self.assertEqual(result['stop_reason'],'validation_cross_entropy_patience')


if __name__=='__main__':unittest.main()
