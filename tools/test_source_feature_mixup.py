"""Source-only mixup math, RNG isolation and actual training-record checks."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools import test_crossmodal_loss as helpers
from tools.test_weather_vds_core import fixture
from vds2raw.source_feature_mixup import source_feature_mixup
from weather import train_vds_core as weather


class SourceFeatureMixupChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_weighted_soft_label_formula_and_no_target_or_rng_side_effects(self):
        torch.manual_seed(9);head=nn.Linear(5,3)
        source=torch.randn(6,5,requires_grad=True);target=torch.randn(4,5,requires_grad=True)
        labels=torch.tensor([0,1,2,0,1,2]);weights=torch.tensor([.5,1.,2.])
        state=torch.get_rng_state();loss,details=source_feature_mixup(head,source,labels,.4,42,7,weights,.05)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        rng=np.random.default_rng(np.random.SeedSequence([42,7,17019]));coefficient=float(rng.beta(.4,.4));perm=torch.tensor(rng.permutation(6))
        logits=head(coefficient*source+(1-coefficient)*source[perm])
        expected=coefficient*F.cross_entropy(logits,labels,weight=weights,label_smoothing=.05)+(1-coefficient)*F.cross_entropy(logits,labels[perm],weight=weights,label_smoothing=.05)
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        grads=torch.autograd.grad(loss,[source,target],allow_unused=True)
        self.assertGreater(float(grads[0].norm()),0);self.assertIsNone(grads[1])
        self.assertEqual(details['target_samples'],0)
        dropout=nn.Sequential(nn.Linear(5,5),nn.Dropout(.5),nn.Linear(5,3)).train()
        state=torch.get_rng_state();source_feature_mixup(dropout,source,labels,.4,42,7)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_zero_default_matches_prior_snapshot_and_positive_case_keeps_all_five_native_losses(self):
        path=Path(__file__).resolve().parents[1]/'experiment_updates/mcc_weather_v2_b38d7eab12c57437/vds2raw/adaptation.py'
        spec=importlib.util.spec_from_file_location('before_source_mixup',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        helper=helpers.CrossmodalChecks()
        for method in ['dann','deepcoral','cdan','mdd','mcc']:
            with self.subTest(method=method):
                torch.manual_seed(42);model=helper.make_model(method);args=helper.args(method)
                args.source_crossmodal_weight=args.target_crossmodal_weight=0;args.trade_off=.2
                source={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'label':torch.tensor([0,1,2,0])}
                target={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8)}
                parameters=[p for p in model.parameters() if p.requires_grad]
                state=torch.get_rng_state();baseline,old_details=old.AdaptationModel.objective(model,source,target,2,8,args)
                old_grad=torch.autograd.grad(baseline,parameters,allow_unused=True);after=torch.get_rng_state()
                torch.set_rng_state(state);current,details=model.objective(source,target,2,8,args)
                new_grad=torch.autograd.grad(current,parameters,allow_unused=True)
                self.assertTrue(torch.equal(current,baseline));self.assertTrue(torch.equal(after,torch.get_rng_state()))
                for a,b in zip(old_grad,new_grad):
                    self.assertEqual(a is None,b is None)
                    if a is not None:self.assertTrue(torch.equal(a,b))
                args.source_feature_mixup_weight=.5
                torch.set_rng_state(state);mixed,details=model.objective(source,target,2,8,args)
                self.assertTrue(torch.equal(details['weighted_core_transfer'],old_details['weighted_core_transfer']))
                self.assertAlmostEqual(float(mixed),float(baseline+.5*(details['source_feature_mixup_loss']-details['source_ce'])),places=5)
                self.assertTrue(torch.equal(after,torch.get_rng_state()))
                torch.set_rng_state(state)
                poisoned=model.objective(source,dict(target,label=torch.full((4,),999)),2,8,args)[0]
                self.assertTrue(torch.equal(mixed,poisoned))

    def test_actual_entry_records_source_correction_gradients_and_enhancement(self):
        helper=helpers.CrossmodalChecks()
        def model(method,*unused,**kwargs):return helper.make_model(method,kwargs['num_classes'])
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','1','--max-train-batches','1','--batch-size','2',
                '--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained',
                '--source-feature-mixup-weight','.5'])
            with patch.object(weather,'AdaptationModel',model),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            output=root/'output/mcc_backlight_seed42'
            r=json.loads((output/'result.json').read_text());u=json.loads((output/'updates.jsonl').read_text().splitlines()[0])
            epoch=json.loads((output/'training.jsonl').read_text().splitlines()[0]);grad=json.loads((output/'gradients.jsonl').read_text().splitlines()[0])
            self.assertEqual(r['enhancement']['source_feature_mixup']['fraction'],.5)
            self.assertFalse(r['target_training_labels_used'])
            self.assertIn('source_feature_mixup_loss',u);self.assertIn('source_feature_mixup_correction',epoch)
            self.assertEqual(u['source_feature_mixup_details']['target_samples'],0)
            self.assertIn('source_feature_mixup_correction',grad['enhancement_components'])


if __name__=='__main__':unittest.main()
