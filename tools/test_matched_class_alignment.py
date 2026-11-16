"""Supervision guards, class-centroid math and all five actual objectives."""
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

from weather.matched_alignment import matched_class_alignment
from weather.class_matching import ALIGNMENT_PROTOCOL
from weather import train_vds_core as weather
from weather.summarize_class_matching import summarize
from tools.test_weather_vds_core import fixture,fast_five_method_model


class MatchedAlignmentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_centroid_math_gradients_permutation_and_single_class(self):
        s=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],requires_grad=True)
        t=s.detach().clone().requires_grad_();y=torch.tensor([0,0,1,1])
        loss,details=matched_class_alignment(s,t,y)
        self.assertAlmostEqual(float(loss),0.,places=6)
        self.assertEqual(details['negative_margin_violation_fraction'],0.)
        wrong=torch.flip(t.detach(),[0]).requires_grad_()
        bad,d=matched_class_alignment(s,wrong,y)
        self.assertGreater(float(bad),1.)
        bad.backward();self.assertIsNone(s.grad);self.assertGreater(float(wrong.grad.norm()),0.)
        permutation=torch.tensor([2,0,3,1])
        reordered,_=matched_class_alignment(s.detach()[permutation],wrong.detach()[permutation],y[permutation])
        self.assertTrue(torch.allclose(bad.detach(),reordered))
        one,_=matched_class_alignment(s[:2],t[:2],y[:2]);self.assertTrue(torch.isfinite(one))

    def test_all_five_entries_expose_live_alignment_and_preserve_native_losses(self):
        for method in ['dann','deepcoral','cdan','mdd','mcc']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
                root=Path(name);lists=fixture(root)
                args=weather.parser().parse_args(['--method',method,'--target','backlight','--data-dir',str(lists),
                    '--save-dir',str(root/'output'),'--epochs','1','--batch-size','2','--num-workers','0',
                    '--cpu-threads','2','--device','cpu','--no-pretrained','--target-class-matching','target_truth',
                    '--matched-alignment-weight','5','--matched-alignment-ramp-epochs','0'])
                observed=[]
                def factory(*a,**kw):
                    model=fast_five_method_model(*a,**kw);original=model.objective
                    def objective(source,target,*rest):
                        self.assertNotIn('label',target)
                        baseline=copy.deepcopy(model)
                        before_rng=torch.get_rng_state();total,details=original(source,target,*rest)
                        after_rng=torch.get_rng_state()
                        torch.set_rng_state(before_rng);args.matched_alignment_weight=0.
                        base,plain=type(model).objective(baseline,source,target,*rest);args.matched_alignment_weight=5.
                        torch.set_rng_state(after_rng)
                        self.assertTrue(torch.allclose(total-base,details['weighted_matched_alignment'],atol=1e-6))
                        self.assertTrue(torch.equal(details['transfer'],plain['transfer']))
                        self.assertTrue(torch.equal(details['source_ce'],plain['source_ce']))
                        self.assertTrue(details['weighted_matched_alignment'].requires_grad)
                        observed.append(float(details['weighted_matched_alignment']))
                        return total,details
                    model.objective=objective;return model
                with patch.object(weather,'AdaptationModel',factory),contextlib.redirect_stdout(io.StringIO()):r=weather.run(args)
                self.assertEqual(r['protocol'],ALIGNMENT_PROTOCOL)
                self.assertTrue(r['target_true_labels_used_in_alignment_loss'])
                self.assertFalse(r['target_true_labels_used_in_classification_loss'])
                output=root/f'output/{method}_backlight_seed42'
                updates=[json.loads(l) for l in (output/'updates.jsonl').read_text().splitlines()]
                self.assertTrue(all(x['matched_alignment_details']['source_centroids_detached'] for x in updates))
                self.assertGreater(sum(observed),0.)
                self.assertEqual(summarize([output/'result.json'],root/'summary')[0]['protocol'],ALIGNMENT_PROTOCOL)

    def test_strict_UDA_rejects_true_class_alignment_before_loading_data(self):
        args=weather.parser().parse_args(['--method','mcc','--target','dark','--matched-alignment-weight','1','--data-dir','missing'])
        # The supervision guard also protects direct objective calls.
        from weather.class_matching import protocol_metadata
        with self.assertRaises(ValueError):protocol_metadata(False,1.)
        with patch.object(weather,'prepare_datasets',side_effect=AssertionError('Must reject before loading')):
            with self.assertRaises(ValueError):weather.run(args)


if __name__=='__main__':unittest.main()
