import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from weather.prediction_distillation import TargetProbabilityCache
from weather import train_vds_core as weather
from tools import test_crossmodal_loss as helpers


class PredictionDistillationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_cache_checks_identity_and_restores_initial_state(self):
        cache=TargetProbabilityCache(['one','two'],3)
        initial={k:v.clone() for k,v in cache.state_dict().items()}
        logits=torch.randn(2,3,requires_grad=True)
        with self.assertRaises(RuntimeError):cache(logits,['one','two'])
        cache.assign(torch.tensor([[.8,.1,.1],[.1,.8,.1]]));loss=cache(logits,['two','one'])
        loss.backward();self.assertGreater(float(logits.grad.norm()),0.)
        with self.assertRaisesRegex(ValueError,'Unknown'):cache(logits,['not found','one'])
        cache.load_state_dict(initial,strict=True);self.assertFalse(bool(cache.ready))

    def test_post_cache_encoder_scale_preserves_head_momentum_and_scheduler(self):
        from weather.encoder_schedule import scale_image_lr_for_cached_distillation
        a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.ones(2))
        optimizer=torch.optim.SGD([{'params':[a],'lr':1e-5,'role':'pretrained_RGB_NIR'},
            {'params':[b],'lr':.1,'role':'new_layers'}],momentum=.9)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:1/(1+i*.001))
        a.grad=torch.ones_like(a);b.grad=torch.ones_like(b);optimizer.step();scheduler.step()
        first=optimizer.param_groups[0]['lr'];head=optimizer.param_groups[1]['lr']
        momentum=optimizer.state[a]['momentum_buffer'].clone()
        report=scale_image_lr_for_cached_distillation(optimizer,scheduler,30.)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],30*first)
        self.assertEqual(optimizer.param_groups[1]['lr'],head)
        torch.testing.assert_close(momentum,optimizer.state[a]['momentum_buffer'])
        optimizer.step();scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.0003/(1+.002))
        self.assertFalse(report['source_bootstrap_and_pseudo_probabilities_changed'])

    def test_distillation_updates_native_encoders_heads_for_all_cores_without_target_GT(self):
        helper=helpers.CrossmodalChecks()
        for method in ['dann','deepcoral','cdan','mdd','mcc']:
            model=helper.make_model(method);args=helper.args(method)
            args.source_crossmodal_weight=args.target_crossmodal_weight=0
            args.prediction_distillation_weight=4.
            model.target_probability_cache=TargetProbabilityCache(['a','b','c','d'],3)
            model.target_probability_cache.assign(torch.tensor([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.8,.1,.1]]))
            source={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'label':torch.tensor([0,1,2,0])}
            target={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'path':['a','b','c','d']}
            rng=torch.get_rng_state();loss,details=model.objective(source,target,2,8,args)
            grads=torch.autograd.grad(details['weighted_prediction_distillation'],
                [model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight,model.head.weight],retain_graph=True)
            self.assertTrue(all(float(x.norm())>0 and torch.isfinite(x).all() for x in grads))
            torch.set_rng_state(rng);poisoned=model.objective(source,dict(target,label=torch.full((4,),999)),2,8,args)[0]
            torch.testing.assert_close(loss,poisoned,rtol=0,atol=0)

    def test_actual_entry_caches_unlabelled_target_and_inference_omits_teacher(self):
        from tools.test_weather_vds_core import fixture
        helper=helpers.CrossmodalChecks()
        def small_model(method,*unused,**kwargs):return helper.make_model(method,kwargs['num_classes'])
        weights=Path(__file__).resolve().parents[1]/'pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth'
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(temporary);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','2','--max-train-batches','1','--batch-size','2',
                '--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained','--feature-normalization','unit',
                '--prediction-teacher-weight','.25','--prediction-teacher-weights',str(weights),
                '--prediction-distillation-weight','4','--prediction-distillation-start-epoch','2'])
            args.prediction_distillation_encoder_lr_scale=30.
            with patch.object(weather,'AdaptationModel',small_model),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            output=root/'output/mcc_backlight_seed42'
            result=json.loads((output/'result.json').read_text());audit=json.loads((output/'prediction_distillation_audit.json').read_text())
            self.assertEqual(result['enhancement']['variant_family'],'core_plus_cached_ensemble_target_distillation')
            self.assertFalse(audit['target_training_labels_used']);self.assertFalse(audit['validation_used'])
            self.assertFalse(audit['inference_includes_teacher'])
            self.assertGreater(audit['preparation_time_s'],0)
            self.assertGreater(result['complexity']['optimizer_and_teacher_preparation_time_s'],result['complexity']['total_train_time_s'])
            self.assertEqual(audit['current_source_supervised_student_epoch'],1)
            self.assertEqual(audit['encoder_LR_phase']['encoder_LR_scale_after_cache'],30.)
            updates=[json.loads(x) for x in (output/'updates.jsonl').read_text().splitlines()]
            self.assertFalse(updates[0]['prediction_distillation_ready']);self.assertTrue(updates[1]['prediction_distillation_ready'])
            self.assertGreater(updates[1]['weighted_prediction_distillation'],0)
            evaluations=[json.loads(x) for x in (output/'evaluation.jsonl').read_text().splitlines()]
            self.assertTrue(all('unblended_student_metrics' not in row for row in evaluations))
            # Frozen teacher convolutions are absent from the inference graph.
            self.assertLess(result['complexity']['inference_GFLOPs'],.01)


if __name__=='__main__':unittest.main()
