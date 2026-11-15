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
from torch import nn
from weather.prediction_teacher import FrozenClusterPrediction,prepare_prediction_teacher
from weather import train_vds_core as weather
from tools import test_crossmodal_loss as helpers


class PredictionTeacherChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def tiny_teacher(self):
        encoders=[nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,8),nn.BatchNorm1d(8)) for _ in range(2)]
        teacher=FrozenClusterPrediction(encoders,.25,4.,.25)
        generator=torch.Generator().manual_seed(13)
        source=[torch.randn(18,8,generator=generator) for _ in range(2)]
        target=[torch.randn(15,8,generator=generator)+.4 for _ in range(2)]
        teacher.fit(source,target,np.arange(18)%3,3)
        return teacher

    def test_blends_softmax_probabilities_not_logits(self):
        teacher=self.tiny_teacher();a=torch.tensor([[.1,2.,1.]]);b=torch.tensor([[1.,.2,3.]])
        expected=.75*(a/.25).softmax(1)+.25*(b/4).softmax(1)
        actual=teacher.combine(a,b).exp()
        torch.testing.assert_close(actual,expected)
        self.assertGreater(float((actual-(.75*a+.25*b).softmax(1)).abs().sum()),.1)

    def test_fitting_is_rng_independent_and_teacher_is_frozen(self):
        teacher=self.tiny_teacher()
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        teacher.train(True);self.assertFalse(teacher.training)
        self.assertTrue(all(not m.training for m in teacher.modules()))
        batch={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8)}
        student=torch.randn(4,3,requires_grad=True);rng=torch.get_rng_state();before=[x.clone() for x in teacher.buffers()]
        a=teacher(student,batch);b=teacher(student,dict(batch,label=torch.full((4,),999),path=['unused']*4))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for old,new in zip(before,teacher.buffers()):torch.testing.assert_close(old,new,rtol=0,atol=0)
        a.sum().backward();self.assertGreater(float(student.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_core_objective_is_identical_with_inference_teacher_attached(self):
        helper=helpers.CrossmodalChecks()
        for method in ['dann','deepcoral','cdan','mdd','mcc']:
            model=helper.make_model(method);args=helper.args(method)
            args.source_crossmodal_weight=args.target_crossmodal_weight=0;args.trade_off=.1
            source={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'label':torch.tensor([0,1,2,0])}
            target={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8)}
            rng=torch.get_rng_state();one=model.objective(source,target,2,8,args)[0];after=torch.get_rng_state()
            model.prediction_teacher=self.tiny_teacher();torch.set_rng_state(rng)
            two=model.objective(source,target,2,8,args)[0]
            torch.testing.assert_close(one,two,rtol=0,atol=0)
            self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_actual_entry_fits_train_only_logs_native_metrics_and_counts_teacher(self):
        from tools.test_weather_vds_core import fixture
        helper=helpers.CrossmodalChecks()
        def small_model(method,*unused,**kwargs):return helper.make_model(method,kwargs['num_classes'])
        weights=Path(__file__).resolve().parents[1]/'pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth'
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(temporary);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','1','--max-train-batches','1','--batch-size','2',
                '--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained','--feature-normalization','unit',
                '--prediction-teacher-weight','.25','--prediction-teacher-temperature','4',
                '--prediction-student-temperature','.25','--prediction-teacher-weights',str(weights)])
            with patch.object(weather,'AdaptationModel',small_model),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            output=root/'output/mcc_backlight_seed42'
            result=json.loads((output/'result.json').read_text())
            audit=json.loads((output/'prediction_teacher_audit.json').read_text())
            self.assertFalse(audit['target_training_labels_used']);self.assertFalse(audit['validation_used_for_fitting'])
            self.assertEqual(audit['fit_partitions'],['source_train','target_train'])
            self.assertEqual(result['enhancement']['variant_family'],'core_plus_frozen_cluster_probability_blend')
            self.assertGreater(result['complexity']['total_params'],22_000_000)
            self.assertLess(result['complexity']['trainable_params'],2_000_000)
            events=[json.loads(x) for x in (output/'evaluation.jsonl').read_text().splitlines()]
            self.assertTrue(all('unblended_student_metrics' in x for x in events))
            pred=json.loads((output/'predictions.jsonl').read_text().splitlines()[0])
            self.assertIn('unblended_student_probabilities',pred)
            args.target_crossmodal_weight=.1
            with self.assertRaisesRegex(ValueError,'independently'):weather.run(args)

    def test_invalid_partition_and_teacher_weights_rejected_before_fit(self):
        class View:partition='target_val';labelled=True
        with self.assertRaisesRegex(ValueError,'source_train'):
            prepare_prediction_teacher(None,View(),View(),'cpu',None,.25)


if __name__=='__main__':unittest.main()
