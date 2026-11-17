"""Fixed label budgets, zero unknown-target gradients, and actual method loops."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from weather.fewshot import TargetLabelBudget
from weather.matched_alignment import fewshot_class_alignment,matched_class_alignment
from weather.class_matching import FEWSHOT_PROTOCOL
from weather import train_vds_core as weather
from weather.summarize_class_matching import summarize
from tools.test_weather_vds_core import fixture,fast_five_method_model
from tools.run_label_guided_matrix import alignment_overrides


class FewShotChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_fixed_three_per_class_is_shared_and_unknowns_are_hidden(self):
        paths=[f'{c}_{i}' for c in range(9) for i in range(5)]
        labels=[c for c in range(9) for _ in range(5)]
        a=TargetLabelBudget(paths,labels,3,42)
        b=TargetLabelBudget(list(reversed(paths)),list(reversed(labels)),3,42)
        self.assertEqual(a.target_lookup,b.target_lookup)
        self.assertEqual(len(a.target_lookup),27)
        self.assertEqual(sum(y==-1 for y in a.labels_for(paths)),18)
        a.steps=3;a.batch_sizes=(16,16)
        m=a.metadata(.1)
        self.assertEqual(m['protocol'],FEWSHOT_PROTOCOL)
        self.assertEqual(set(m['labelled_target_class_counts'].values()),{3})
        self.assertFalse(m['unselected_truth_used_for_pairing_or_alignment'])
        portable=TargetLabelBudget(['other/root/'+p for p in paths],labels,3,42)
        portable.steps=3;portable.batch_sizes=(16,16)
        self.assertEqual(m['label_pool_portable_sha256'],portable.metadata(.1)['label_pool_portable_sha256'])
        self.assertEqual(m['unlabelled_target_samples'],18)
        self.assertEqual(alignment_overrides(.1,'reference',3)['target_labels_per_class'],3)
        with self.assertRaises(ValueError):alignment_overrides(0.,'reference',3)
        with self.assertRaises(ValueError):TargetLabelBudget(paths,labels,6,42)

    def test_extra_loss_has_zero_gradient_on_unknown_targets(self):
        torch.manual_seed(1)
        source=torch.randn(6,5,requires_grad=True)
        target=torch.randn(5,5,requires_grad=True)
        ys=torch.tensor([0,0,1,1,2,2]);yt=torch.tensor([0,-1,2,-1,7])
        loss,details=fewshot_class_alignment(source,target,ys,yt)
        loss.backward()
        self.assertIsNone(source.grad)
        self.assertGreater(float(target.grad[[0,2]].norm()),0)
        self.assertEqual(float(target.grad[[1,3,4]].norm()),0.)
        changed=target.detach().clone();changed[[1,3,4]]*=100
        again,_=fewshot_class_alignment(source.detach(),changed,ys,yt)
        torch.testing.assert_close(loss.detach(),again)
        self.assertEqual(details['aligned_labelled_targets'],2)
        self.assertEqual(details['unlabelled_targets_in_loss'],0)
        zero,_=fewshot_class_alignment(source,target,ys,torch.full((5,),-1))
        self.assertEqual(float(zero),0.)

    def test_fully_known_matched_case_reproduces_original_class_alignment(self):
        torch.manual_seed(2)
        s=torch.randn(6,5);t=torch.randn(6,5);y=torch.tensor([0,0,1,1,2,2])
        old,_=matched_class_alignment(s,t,y)
        new,_=fewshot_class_alignment(s,t,y,y)
        torch.testing.assert_close(old,new)

    def test_all_five_actual_methods_keep_all_targets_but_only_budget_labels(self):
        for method in ('dann','deepcoral','cdan','mdd','mcc'):
            with self.subTest(method=method),tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
                root=Path(name);lists=fixture(root)
                args=weather.parser().parse_args(['--method',method,'--target','backlight','--data-dir',str(lists),
                    '--save-dir',str(root/'output'),'--epochs','3','--batch-size','2','--num-workers','0',
                    '--cpu-threads','2','--device','cpu','--no-pretrained','--target-class-matching','target_truth',
                    '--target-labels-per-class','1','--matched-alignment-weight','.1',
                    '--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','0'])
                with patch.object(weather,'AdaptationModel',fast_five_method_model),contextlib.redirect_stdout(io.StringIO()):
                    result=weather.run(args)
                self.assertEqual(result['protocol'],FEWSHOT_PROTOCOL)
                self.assertEqual(result['label_guided_sampling']['labelled_target_samples'],4)
                self.assertEqual(result['label_guided_sampling']['unlabelled_target_samples'],4)
                p=root/f'output/{method}_backlight_seed42'
                rows=[json.loads(s) for s in (p/'training.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['target_unique_samples']==8 and r['source_unique_samples']==8 for r in rows))
                updates=[json.loads(s) for s in (p/'updates.jsonl').read_text().splitlines()]
                self.assertTrue(any(e['weighted_matched_alignment']>0 for e in updates))
                self.assertTrue(any(e['weighted_core_transfer']!=0 for e in updates))
                self.assertEqual(summarize([p/'result.json'],root/'summary')[0]['protocol'],FEWSHOT_PROTOCOL)
                for e in updates:
                    if -1 in e['target_known_alignment_labels']:
                        e['target_known_alignment_labels'][e['target_known_alignment_labels'].index(-1)]=0
                        break
                (p/'updates.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in updates))
                with self.assertRaisesRegex(ValueError,'Unselected target truth'):
                    summarize([p/'result.json'],root/'bad_summary')


if __name__=='__main__':unittest.main()
