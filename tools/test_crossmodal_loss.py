from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from vds2raw.crossmodal_loss import paired_crossmodal_loss,install_crossmodal_projectors
from vds2raw.adaptation import AdaptationModel,METHODS
from weather import train_vds_core as weather


class CrossmodalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_target_uses_pair_identity_and_source_uses_class_positives(self):
        features=torch.eye(4)
        aligned,_=paired_crossmodal_loss(features,features,.1)
        mismatched,_=paired_crossmodal_loss(features,features.roll(1,0),.1)
        self.assertLess(float(aligned),.001)
        self.assertGreater(float(mismatched),5.)
        labels=torch.tensor([0,0,1,1])
        a,_=paired_crossmodal_loss(features,features,.2,labels)
        b,_=paired_crossmodal_loss(features,features[[1,0,3,2]],.2,labels)
        torch.testing.assert_close(a,b)

    def test_unequal_modality_projection_preserves_rng_and_is_absent_from_inference(self):
        torch.manual_seed(42)
        model=self.make_model("mcc")
        model.nir_encoder[-1]=nn.Linear(3,128);model.ir_feature_dim=128
        model.head=nn.Linear(640,3)
        rng=torch.get_rng_state();record=install_crossmodal_projectors(model,64)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(record["input_widths"],[512,128])
        source={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8),"label":torch.tensor([0,0,1,2])}
        target={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8)}
        _,details=model.objective(source,target,2,8,self.args("mcc"))
        gradients=torch.autograd.grad(details["weighted_target_crossmodal"],
            [model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight,model.crossmodal_projectors[0][0].weight,model.crossmodal_projectors[1][0].weight])
        self.assertTrue(all(float(g.norm())>0 for g in gradients))
        calls=[]
        hooks=[p.register_forward_hook(lambda *unused:calls.append(1)) for p in model.crossmodal_projectors]
        model.eval()
        with torch.no_grad():model.encode(target)
        for h in hooks:h.remove()
        self.assertEqual(calls,[])

    def test_stop_gradient_protects_reference_encoder_while_both_classifiers_remain_trainable(self):
        model=self.make_model("mcc");model.crossmodal_stop_gradient="vis"
        install_crossmodal_projectors(model,64)
        source={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8),"label":torch.tensor([0,1,2,0])}
        target={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8)}
        loss,details=model.objective(source,target,2,8,self.args("mcc"))
        gradients=torch.autograd.grad(details["weighted_target_crossmodal"],
            [model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight],allow_unused=True,retain_graph=True)
        self.assertIsNone(gradients[0]);self.assertGreater(float(gradients[1].norm()),0.)
        loss.backward()
        self.assertGreater(float(model.rgb_encoder[-1].weight.grad.norm()),0.)

    def test_reference_argument_roundtrip_retains_modalities_paths_and_boolean_settings(self):
        from tools.run_weather_crossmodal_screen import recorded_arguments_to_cli
        arguments=vars(weather.parser().parse_args(["--method","mcc","--target","foggy","--data-dir","data with spaces","--save-dir","output",
            "--image-backbone","convnext_small","--ir-backbone","unip_tiny","--ir-training-scope","all_used_layers",
            "--encoder-training-scope","last_stages","--encoder-precision","bf16","--no-domain-running-stats",
            "--cache-evaluation","--source-class-balance","--source-weather-probability",".75","--source-aspect-min",".5","--source-aspect-max","2",
            "--batch-size","8","--target-batch-size","32","--feature-normalization","unit"]))
        restored=weather.parser().parse_args(recorded_arguments_to_cli(arguments))
        for action in weather.parser()._actions:
            key=action.dest
            if key in arguments and arguments[key] is not None:
                self.assertEqual(getattr(restored,key),arguments[key],key)

    def test_resnet_last_stage_policy_unfreezes_only_requested_modalities(self):
        for scope in ("last_stages","ir_last_stage"):
            model=AdaptationModel("mcc",pretrained=False,num_classes=3,use_ais=False,ir_channels=3,
                encoder_training_scope=scope)
            model.train()
            self.assertFalse(model.rgb_encoder.conv1.weight.requires_grad)
            self.assertFalse(model.nir_encoder.conv1.weight.requires_grad)
            self.assertTrue(model.nir_encoder.layer4[0].conv1.weight.requires_grad)
            self.assertEqual(model.rgb_encoder.layer4[0].conv1.weight.requires_grad,scope=="last_stages")
            self.assertFalse(model.nir_encoder.layer4[0].bn1.training)

    def make_model(self,method,classes=3):
        model=AdaptationModel(method,pretrained=False,num_classes=classes,use_ais=False,ir_channels=3,
            classifier_layout="linear",feature_normalization="unit")
        for key in ("rgb_encoder","nir_encoder"):
            setattr(model,key,nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,512)))
        model.crossmodal_alignment_enabled=True
        return model

    def args(self,method):
        args=weather.parser().parse_args(["--method",method,"--target","backlight","--data-dir","unused","--save-dir","unused",
            "--source-crossmodal-weight",".1","--target-crossmodal-weight",".05","--trade-off","0"])
        weather.resolve_weather_settings(args);args.steps_per_epoch=2
        return args

    def test_all_cores_send_both_losses_to_both_encoders_without_target_labels(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(42)
                model=self.make_model(method)
                args=self.args(method)
                source={"rgb":torch.randn(4,3,12,12),"nir":torch.randn(4,3,12,12),"label":torch.tensor([0,0,1,2])}
                target={"rgb":torch.randn(4,3,12,12),"nir":torch.randn(4,3,12,12)}
                rng=torch.get_rng_state()
                loss,details=model.objective(source,target,2,8,args)
                weights=[model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight]
                for key in ("weighted_source_crossmodal","weighted_target_crossmodal"):
                    gradients=torch.autograd.grad(details[key],weights,retain_graph=True)
                    self.assertTrue(all(float(gradient.norm())>0 for gradient in gradients))
                self.assertTrue(details["adaptation_graph_in_loss"])
                self.assertFalse(details["core_adaptation_graph_in_loss"])
                torch.set_rng_state(rng)
                poisoned=model.objective(source,dict(target,label=torch.full((4,),999),path=["arbitrary"]*4),2,8,args)[0]
                torch.testing.assert_close(loss,poisoned,rtol=0,atol=0)
                loss.backward()
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in weights))

    def test_zero_weights_leave_original_math_and_rng_unchanged(self):
        for method in METHODS:
            torch.manual_seed(42)
            model=self.make_model(method);args=self.args(method)
            args.source_crossmodal_weight=args.target_crossmodal_weight=0
            args.trade_off=.2
            source={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8),"label":torch.tensor([0,1,2,0])}
            target={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8)}
            rng=torch.get_rng_state();a=model.objective(source,target,2,8,args)[0];after=torch.get_rng_state()
            torch.set_rng_state(rng);model.crossmodal_alignment_enabled=False;b=model.objective(source,target,2,8,args)[0]
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_source_alignment_is_active_during_warmup_without_target_or_adversarial_gradients(self):
        model=self.make_model("dann");args=self.args("dann");args.adaptation_warmup_epochs=2
        source={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8),"label":torch.tensor([0,1,2,0])}
        target={"rgb":torch.randn(4,3,8,8,requires_grad=True),"nir":torch.randn(4,3,8,8,requires_grad=True)}
        loss,details=model.objective(source,target,0,8,args)
        self.assertGreater(float(details["weighted_source_crossmodal"].detach()),0)
        self.assertEqual(float(details["weighted_target_crossmodal"]),0)
        loss.backward()
        self.assertTrue(target["rgb"].grad is None or torch.count_nonzero(target["rgb"].grad)==0)
        self.assertTrue(all(parameter.grad is None for parameter in model.discriminator.parameters()))

    def test_real_entry_logs_distinct_enhancement_and_loss_components(self):
        from tools.test_weather_vds_core import fixture
        def small_model(method,*unused,**kwargs):return self.make_model(method,kwargs["num_classes"])
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
            root=Path(temporary);lists=fixture(root)
            args=weather.parser().parse_args(["--method","mcc","--target","backlight","--data-dir",str(lists),
                "--save-dir",str(root/"output"),"--epochs","1","--max-train-batches","1","--batch-size","2",
                "--num-workers","0","--cpu-threads","2","--device","cpu","--no-pretrained","--feature-normalization","unit",
                "--source-crossmodal-weight",".1","--target-crossmodal-weight",".05"])
            with patch.object(weather,"AdaptationModel",small_model),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            output=root/"output/mcc_backlight_seed42"
            result=json.loads((output/"result.json").read_text())
            self.assertEqual(result["enhancement"]["variant_family"],"core_plus_paired_crossmodal_alignment")
            record=json.loads((output/"updates.jsonl").read_text().splitlines()[0])
            self.assertGreater(record["weighted_target_crossmodal"],0)
            self.assertGreater(record["weighted_source_crossmodal"],0)
            self.assertFalse(result["target_training_labels_used"])
            gradients=json.loads((output/"gradients.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(gradients["enhancement_components"]),{"core","source_crossmodal","target_crossmodal"})


if __name__=="__main__":unittest.main()
