"""Explicit supervision, exact class pairing and full-coverage sampler checks."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from weather.class_matching import MatchedClassPlan,MatchedClassBatchSampler,PROTOCOL
from weather import train_vds_core as weather
from tools.test_weather_vds_core import fixture,fast_five_method_model
from weather.summarize_class_matching import summarize as summarize_guided
from weather.summarize_vds_core import summarize as summarize_UDA
from tools.run_weather_class_matching import build_values


class LabelGuidedMatchingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_matching_covers_every_sample_without_cartesian_pair_explosion(self):
        a=[0,0,0,0,0,1,1,2];b=[0,1,1,1,1,2,2,2,2]
        plan=MatchedClassPlan(a,b,[f's{i}' for i in range(len(a))],[f't{i}' for i in range(len(b))],4,42)
        source,target=[MatchedClassBatchSampler(plan,i) for i in [0,1]]
        s,t=list(source),list(target)
        self.assertEqual(set(sum(s,[])),set(range(len(a))))
        self.assertEqual(set(sum(t,[])),set(range(len(b))))
        self.assertEqual(plan.pairs_per_epoch,5+4+4)
        for si,ti in zip(s,t):
            self.assertEqual([a[i] for i in si],[b[i] for i in ti])
            plan.assert_batch([f's{i}' for i in si],[f't{i}' for i in ti],[a[i] for i in si])
        self.assertEqual(s,list(source));self.assertEqual(t,list(target))
        source.epoch=target.epoch=1
        self.assertNotEqual(s,list(source))
        self.assertEqual(plan.metadata()['target_true_labels_used_in_classification_loss'],False)
        with self.assertRaises(RuntimeError):plan.assert_batch(['s0'],['t1'],[0])
        with self.assertRaises(ValueError):MatchedClassPlan([0],[1],['s'],['t'],2,42)

    def test_all_five_actual_entries_mark_label_guidance_and_keep_target_labels_out_of_objective(self):
        for method in ['dann','deepcoral','cdan','mdd','mcc']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
                root=Path(name);lists=fixture(root)
                args=weather.parser().parse_args(['--method',method,'--target','backlight','--data-dir',str(lists),
                    '--save-dir',str(root/'output'),'--epochs','1','--batch-size','2','--num-workers','0',
                    '--cpu-threads','2','--device','cpu','--no-pretrained','--target-class-matching','target_truth'])
                observed=[]
                def factory(*a,**kw):
                    model=fast_five_method_model(*a,**kw);original=model.objective
                    def objective(source,target,*rest):
                        self.assertNotIn('label',target);observed.append(1)
                        return original(source,target,*rest)
                    model.objective=objective
                    return model
                with patch.object(weather,'AdaptationModel',factory),contextlib.redirect_stdout(io.StringIO()):r=weather.run(args)
                self.assertEqual(r['protocol'],PROTOCOL);self.assertTrue(r['target_training_labels_used'])
                self.assertFalse(r['strict_unsupervised']);self.assertFalse(r['target_true_labels_used_in_classification_loss'])
                output=root/f'output/{method}_backlight_seed42'
                cfg=json.loads((output/'run_config.json').read_text());self.assertEqual(cfg['sampler']['protocol'],PROTOCOL)
                epoch=json.loads((output/'training.jsonl').read_text().splitlines()[0])
                self.assertEqual(epoch['source_unique_samples'],8);self.assertEqual(epoch['target_unique_samples'],8)
                self.assertTrue(epoch['all_batch_positions_class_matched'])
                for line in (output/'updates.jsonl').read_text().splitlines():
                    row=json.loads(line)
                    self.assertEqual(row['source_true_classes'],row['target_true_classes_for_sampling'])
                    self.assertTrue(row['target_training_labels_used']);self.assertFalse(row['target_classification_CE_used'])
                self.assertGreater(len(observed),0)
                rows=summarize_guided([output/'result.json'],root/'guided_summary')
                self.assertEqual(rows[0]['protocol'],PROTOCOL)
                with self.assertRaises(ValueError):summarize_UDA(output.parent)

    def test_strict_UDA_default_never_constructs_the_truth_guided_sampler(self):
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','1','--max-train-batches','1','--batch-size','2',
                '--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained'])
            with patch.object(weather,'AdaptationModel',fast_five_method_model), \
                    patch('weather.class_matching.make_matched_samplers',side_effect=AssertionError('UDA must not read target classes for sampling')),contextlib.redirect_stdout(io.StringIO()):r=weather.run(args)
            self.assertEqual(r['protocol'],'UDA');self.assertFalse(r['target_training_labels_used'])
            with self.assertRaises(ValueError):
                summarize_guided([root/'output/mcc_backlight_seed42/result.json'],root/'bad_summary')

    def test_launcher_preserves_reference_except_explicit_protocol_overrides(self):
        original={'method':'mcc','target':'foggy','seed':42,'epochs':20,'batch_size':8,'target_batch_size':32,
                  'lr':.001,'mcc_temperature':1.,'initial_vis_weight':.5,'source_crossmodal_weight':.1}
        values=build_values({'arguments':original},Path('output'),8)
        self.assertEqual(values['batch_size'],values['target_batch_size'])
        self.assertEqual(values['target_class_matching'],'target_truth')
        for key in original:
            if key not in ('batch_size','target_batch_size'):self.assertEqual(values[key],original[key])
        smoke=build_values({'arguments':original},Path('output'),8,smoke=True)
        self.assertEqual(smoke['max_train_batches'],1);self.assertEqual(smoke['stop_after_epochs'],1)
        self.assertNotIn('target_class_matching',original)


if __name__=='__main__':unittest.main()
