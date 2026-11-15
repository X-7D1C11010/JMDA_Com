from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import Dataset
from weather.target_representation_anchor import TargetRepresentationCache,prepare_target_representation_cache
from weather import train_vds_core as weather
from tools import test_crossmodal_loss as crossmodal_helpers
from vds2raw.adaptation import METHODS


class View(Dataset):
    partition='target_train';labelled=False
    def __init__(self):
        self.paths=['target_a','target_b','target_c']
        g=torch.Generator().manual_seed(987)
        self.rgb=torch.rand(3,3,8,8,generator=g);self.nir=torch.rand(3,3,8,8,generator=g)
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):return {'rgb':self.rgb[index],'nir':self.nir[index],'path':self.paths[index]}


class TargetRepresentationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def cache(self):
        g=torch.Generator().manual_seed(988)
        return TargetRepresentationCache(['a','b','c'],[torch.randn(3,512,generator=g),torch.randn(3,512,generator=g)])

    def args(self,method,weight):
        args=weather.parser().parse_args(['--method',method,'--target','dark','--data-dir','unused',
            '--trade-off','.1','--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','1',
            '--target-representation-anchor-weight',str(weight)])
        weather.resolve_weather_settings(args);args.steps_per_epoch=1
        return args

    def batches(self):
        source={'rgb':torch.eye(3).reshape(3,3,1,1).expand(3,3,8,8).clone(),
                'nir':torch.randn(3,3,8,8),'label':torch.arange(3)}
        target={'rgb':torch.full((3,3,8,8),.2),'nir':torch.full((3,3,8,8),.1),'path':['a','b','c']}
        return source,target

    def test_cache_uses_path_identity_and_has_no_trainable_parameters(self):
        cache=self.cache();features=[cache.reference_0[[2,0,2]],cache.reference_1[[2,0,2]]]
        loss,cos=cache(features,['c','a','c'])
        self.assertLess(float(loss),1e-6);self.assertEqual(sum(p.numel() for p in cache.parameters()),0)
        self.assertTrue(all(v>.99999 for v in cos.values()))
        with self.assertRaises(ValueError):cache(features,['c','unknown','c'])
        with self.assertRaises(ValueError):TargetRepresentationCache(['a','a'],[torch.ones(2,3),torch.ones(2,3)])

    def test_all_five_core_losses_preserved_and_anchor_reaches_both_encoders_only(self):
        helper=crossmodal_helpers.CrossmodalChecks()
        for method in METHODS:
            with self.subTest(method=method),torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(989)
                model=helper.make_model(method);model.crossmodal_stop_gradient='vis'
                baseline=deepcopy(model);model.target_representation_cache=self.cache()
                source,target=self.batches();state=torch.get_rng_state()
                _,base=baseline.objective(source,target,2,6,self.args(method,0))
                torch.set_rng_state(state)
                loss,details=model.objective(source,target,2,6,self.args(method,1))
                torch.testing.assert_close(details['weighted_core_transfer'],base['weighted_core_transfer'],rtol=0,atol=0)
                torch.testing.assert_close(details['source_ce'],base['source_ce'],rtol=0,atol=0)
                params=[model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight,model.head.weight,model.modality_logits]
                gradients=torch.autograd.grad(details['weighted_target_representation_anchor'],params,retain_graph=True,allow_unused=True)
                self.assertTrue(all(g is not None and float(g.norm())>0 for g in gradients[:2]))
                self.assertIsNone(gradients[2]);self.assertIsNone(gradients[3])
                self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_disabled_loss_is_identical_even_when_a_cache_is_attached(self):
        model=crossmodal_helpers.CrossmodalChecks().make_model('mcc');source,target=self.batches()
        _,base=model.objective(source,target,2,6,self.args('mcc',0))
        model.target_representation_cache=self.cache()
        _,new=model.objective(source,target,2,6,self.args('mcc',0))
        for k in ['source_ce','weighted_transfer','weighted_core_transfer']:torch.testing.assert_close(base[k],new[k],rtol=0,atol=0)

    def test_initialization_reads_target_train_only_and_preserves_rng(self):
        model=crossmodal_helpers.CrossmodalChecks().make_model('mcc');view=View();state=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as name:
            cache,audit=prepare_target_representation_cache(model,view,torch.device('cpu'),Path(name),2)
            self.assertTrue(model.training);self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertFalse(audit['target_training_labels_used']);self.assertFalse(audit['validation_used'])
            with np.load(Path(name)/'target_initial_representations.npz',allow_pickle=False) as data:
                self.assertEqual(set(data.files),{'paths','VIS','IR'});self.assertEqual(data['paths'].tolist(),view.paths)
            view.partition='target_val';view.labelled=True
            with self.assertRaises(ValueError):prepare_target_representation_cache(model,view,torch.device('cpu'),Path(name),2)

    def test_attached_buffers_do_not_change_inference_or_state_shape(self):
        model=crossmodal_helpers.CrossmodalChecks().make_model('mcc').eval();batch={'rgb':torch.rand(3,3,8,8),'nir':torch.rand(3,3,8,8)}
        with torch.no_grad():before=model(batch)
        parameters=sum(p.numel() for p in model.parameters());model.target_representation_cache=self.cache()
        with torch.no_grad():after=model(batch)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertEqual(parameters,sum(p.numel() for p in model.parameters()))
        state={k:v.clone() for k,v in model.state_dict().items()};model.load_state_dict(state)

    def test_real_entry_records_initialization_loss_gradients_and_student_only_profile(self):
        from tools.test_weather_vds_core import fixture,fast_five_method_model
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','2','--max-train-batches','1','--batch-size','2',
                '--device','cpu','--no-pretrained','--num-workers','0','--cpu-threads','2',
                '--target-representation-anchor-weight','1','--feature-normalization','unit'])
            with patch.object(weather,'AdaptationModel',fast_five_method_model),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            output=root/'output/mcc_backlight_seed42';result=json.loads((output/'result.json').read_text())
            self.assertEqual(result['enhancement']['variant_family'],'core_plus_initial_target_anchor')
            updates=[json.loads(l) for l in (output/'updates.jsonl').read_text().splitlines()]
            self.assertGreater(updates[0]['weighted_target_representation_anchor'],0)
            self.assertEqual(set(updates[0]['target_representation_cosines']),{'VIS','IR'})
            gradients=[json.loads(l) for l in (output/'gradients.jsonl').read_text().splitlines()]
            self.assertIn('target_representation_anchor',gradients[0]['enhancement_components'])
            self.assertGreater(result['complexity']['target_representation_preparation_time_s'],0)
            self.assertFalse(result['enhancement']['target_representation_anchor']['inference_changed'])


if __name__=='__main__':unittest.main()
