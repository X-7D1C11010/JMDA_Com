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
from torch.utils.checkpoint import checkpoint
from vds2raw.adaptation import AdaptationModel,METHODS
from weather.target_feature_style import TargetFeatureStyle,install_target_feature_style
from weather import train_vds_core as weather


class TinyImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,4,1),nn.Sequential(nn.ReLU(),nn.Conv2d(4,4,1)),
                                    nn.AvgPool2d(2),nn.Sequential(nn.ReLU(),nn.Conv2d(4,4,1)))
        self.output=nn.Linear(4,512)
    def forward(self,x):return self.output(self.features(x).mean((-2,-1)))


class TinyIR(nn.Module):
    def __init__(self):
        super().__init__();self.trunk=nn.Module();self.trunk.patch_embed=nn.Module()
        self.trunk.patch_embed.proj=nn.Conv2d(3,4,1)
        self.trunk.blocks=nn.ModuleList([nn.LayerNorm(4),nn.LayerNorm(4)])
        self.output=nn.Linear(4,512)
    def forward(self,x):
        z=self.trunk.patch_embed.proj(x).flatten(2).transpose(1,2)
        z=torch.cat((torch.zeros(len(z),1,4,device=z.device),z),1)
        for block in self.trunk.blocks:z=block(z)
        return self.output(z[:,1:].mean(1))


class TargetFeatureStyleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,method,classes=3,unip=False):
        model=AdaptationModel(method,pretrained=False,num_classes=classes,use_ais=False,ir_channels=3,
                              classifier_layout="linear",feature_normalization="unit")
        model.rgb_encoder=TinyImage();model.nir_encoder=TinyIR() if unip else TinyImage()
        model.image_backbone="convnext_small"
        return model

    def args(self,method):
        args=weather.parser().parse_args(["--method",method,"--target","rainy","--data-dir","unused","--save-dir","unused",
            "--domain-forward","joint","--target-feature-style-probability","1","--trade-off",".1",
            "--adaptation-warmup-epochs","0","--adaptation-ramp-epochs","1"])
        weather.resolve_weather_settings(args);args.steps_per_epoch=2
        return args

    def batches(self):
        return ({"rgb":torch.randn(3,3,8,8),"nir":torch.randn(3,3,8,8),"label":torch.tensor([0,1,2])},
                {"rgb":torch.randn(3,3,8,8)+2,"nir":torch.randn(3,3,8,8)-1})

    def test_source_moments_move_target_unchanged_and_no_target_gradient(self):
        source=torch.randn(2,3,8,8,requires_grad=True)
        target=(torch.randn(3,3,8,8)+4).requires_grad_()
        mixer=TargetFeatureStyle(1,0);mixer.begin_step(0,2,3)
        output=mixer.spatial(torch.cat((source,target)),"test")
        torch.testing.assert_close(output[2:],target,rtol=0,atol=0)
        indices=mixer.context["target_indices"]
        torch.testing.assert_close(output[:2].mean((-2,-1)),target[indices].mean((-2,-1)),rtol=1e-5,atol=1e-6)
        output[:2].square().mean().backward()
        self.assertGreater(float(source.grad.norm()),0)
        self.assertEqual(torch.count_nonzero(target.grad),0)
        trace=mixer.diagnostics()["layers"]["test"]
        self.assertLess(trace["moment_mean_gap_after"],trace["moment_mean_gap_before"])

    def test_pair_modalities_share_references_and_do_not_consume_global_rng(self):
        mixer=TargetFeatureStyle(.7,.3)
        inputs=torch.randn(7,3,8,8);rng=torch.get_rng_state()
        mixer.begin_step(5,3,4);a=mixer.spatial(inputs,"VIS");b=mixer.spatial(inputs,"IR")
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        mixer.begin_step(5,3,4);again=mixer.spatial(inputs,"VIS")
        torch.testing.assert_close(a,again,rtol=0,atol=0)

    def test_zero_and_eval_are_exact_identity_and_cls_is_retained(self):
        inputs=torch.randn(6,10,4)
        mixer=TargetFeatureStyle(1,.3);mixer.begin_step(1,3,3)
        output=mixer.tokens(inputs,"IR")
        torch.testing.assert_close(output[:,:1],inputs[:,:1],rtol=0,atol=0)
        torch.testing.assert_close(output[3:],inputs[3:],rtol=0,atol=0)
        mixer.eval();self.assertIs(mixer.tokens(inputs,"IR"),inputs)
        zero=TargetFeatureStyle(0);self.assertIs(zero.tokens(inputs,"IR"),inputs)

    def test_hooks_cover_convnext_and_ir_tokens_and_are_absent_in_eval(self):
        model=self.model("mcc",unip=True);x=torch.randn(6,3,8,8)
        model.eval()
        before=model.encode({"rgb":x,"nir":x}).detach()
        metadata=install_target_feature_style(model,1)
        self.assertEqual(len(metadata["layers"]),4)
        model.eval();after=model.encode({"rgb":x,"nir":x}).detach()
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertEqual(model.target_feature_style.records,{})
        model.train();model.target_feature_style.begin_step(0,3,3)
        model.encode({"rgb":x,"nir":x})
        self.assertEqual(len(model.target_feature_style.records),4)

    def test_checkpoint_recomputations_use_identical_style_and_gradients(self):
        model=self.model("mcc");install_target_feature_style(model,1)
        x=torch.randn(5,3,8,8,requires_grad=True);p=model.rgb_encoder.features[0].weight
        model.target_feature_style.begin_step(11,2,3)
        a=model.rgb_encoder(x);ga=torch.autograd.grad(a.square().mean(),p)[0]
        model.target_feature_style.begin_step(11,2,3)
        b=checkpoint(model.rgb_encoder,x,use_reentrant=False,preserve_rng_state=True)
        gb=torch.autograd.grad(b.square().mean(),p)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(ga,gb,rtol=1e-6,atol=1e-8)

    def test_all_five_cores_have_finite_gradients_without_using_target_labels(self):
        for method in METHODS:
            torch.manual_seed(42);model=self.model(method);install_target_feature_style(model,1)
            source,target=self.batches();args=self.args(method);rng=torch.get_rng_state()
            loss,_=model.objective(source,target,2,20,args)
            torch.set_rng_state(rng)
            bad=model.objective(source,dict(target,label=torch.full((3,),999),path=["ignored"]*3),2,20,args)[0]
            torch.testing.assert_close(loss,bad,rtol=0,atol=0)
            loss.backward()
            for encoder in (model.rgb_encoder,model.nir_encoder):
                gradient=encoder.features[0].weight.grad
                self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0)

    def test_warmup_source_classification_does_not_differentiate_target_statistics(self):
        model=self.model("dann");install_target_feature_style(model,1)
        args=self.args("dann");args.adaptation_warmup_epochs=3
        source,target=self.batches();target["rgb"].requires_grad_();target["nir"].requires_grad_()
        model.objective(source,target,0,20,args)[0].backward()
        self.assertEqual(torch.count_nonzero(target["rgb"].grad),0)
        self.assertEqual(torch.count_nonzero(target["nir"].grad),0)

    def test_real_entry_logs_source_only_style_and_original_method_loss(self):
        from tools.test_weather_vds_core import fixture
        def small(method,*unused,**kw):return self.model(method,kw["num_classes"])
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
            root=Path(tmp);lists=fixture(root)
            args=weather.parser().parse_args(["--method","mcc","--target","backlight","--data-dir",str(lists),
                "--save-dir",str(root/"out"),"--epochs","1","--max-train-batches","1","--batch-size","2",
                "--num-workers","0","--cpu-threads","2","--device","cpu","--no-pretrained","--image-backbone","convnext_small",
                "--domain-forward","joint","--target-feature-style-probability","1"])
            with patch.object(weather,"AdaptationModel",small),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            directory=root/"out/mcc_backlight_seed42"
            result=json.loads((directory/"result.json").read_text())
            update=json.loads((directory/"updates.jsonl").read_text().splitlines()[0])
            self.assertEqual(result["enhancement"]["variant_family"],"core_plus_target_feature_style")
            self.assertTrue(update["target_feature_style"]["active"])
            self.assertEqual(len(update["target_feature_style"]["layers"]),4)
            self.assertFalse(result["target_training_labels_used"])


if __name__=="__main__":unittest.main()
