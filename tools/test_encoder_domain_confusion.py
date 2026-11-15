"""Gradient ownership and real-entry checks for the optional new objective."""
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
from torch.nn import functional as F
from vds2raw.adaptation import AdaptationModel, reverse
from vds2raw.domain_confusion import encoder_domain_confusion
from weather import train_vds_core as weather


class DomainConfusionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,method,classes=3):
        model=AdaptationModel(method,pretrained=False,num_classes=classes,use_ais=False,ir_channels=3,
                              classifier_layout="linear",feature_normalization="unit")
        for key in ("rgb_encoder","nir_encoder"):
            setattr(model,key,nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,512)))
        return model

    def args(self,method):
        args=weather.parser().parse_args(["--method",method,"--target","rainy","--data-dir","unused","--save-dir","unused",
            "--domain-confusion-weight",".05","--trade-off",".1","--adaptation-warmup-epochs","0","--adaptation-ramp-epochs","1"])
        weather.resolve_weather_settings(args);args.steps_per_epoch=2
        return args

    def batches(self):
        return ({"rgb":torch.randn(3,3,8,8),"nir":torch.randn(3,3,8,8),"label":torch.tensor([0,1,2])},
                {"rgb":torch.randn(2,3,8,8),"nir":torch.randn(2,3,8,8)})

    def test_saturation_keeps_feature_gradient_and_never_updates_discriminator_or_class_head(self):
        for method in ("dann","cdan"):
            x=torch.tensor([[50.,0.],[50.,0.],[-50.,0.]],requires_grad=True)
            class_logits=torch.zeros(3,3,requires_grad=True)
            discriminator=nn.Linear(2 if method=="dann" else 6,1,bias=False)
            with torch.no_grad():
                discriminator.weight.zero_();discriminator.weight[:,::2]=1
            original=F.binary_cross_entropy_with_logits(reverse(x[:,0],1),torch.tensor([1.,1.,0.]))
            original_gradient=torch.autograd.grad(original,x,retain_graph=True)[0]
            self.assertLess(float(original_gradient.norm()),1e-18)
            loss=encoder_domain_confusion(x,class_logits,discriminator,2,method)
            gradients=torch.autograd.grad(loss,(x,class_logits,*discriminator.parameters()),allow_unused=True)
            self.assertGreater(float(gradients[0].norm()),.1)
            self.assertTrue(all(g is None for g in gradients[1:]))

    def test_both_encoder_gradients_and_discriminator_core_updates_survive_poisoned_target_metadata(self):
        for method in ("dann","cdan"):
            torch.manual_seed(42);model=self.model(method);args=self.args(method)
            source,target=self.batches();rng=torch.get_rng_state()
            loss,details=model.objective(source,target,2,20,args)
            names=(model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight)
            gradients=torch.autograd.grad(details["weighted_domain_confusion"],(*names,*model.discriminator.parameters()),retain_graph=True,allow_unused=True)
            self.assertTrue(all(g is not None and float(g.norm())>0 for g in gradients[:2]))
            self.assertTrue(all(g is None for g in gradients[2:]))
            torch.set_rng_state(rng)
            poisoned=model.objective(source,dict(target,label=torch.tensor([999,999]),path=["unused"]*2),2,20,args)[0]
            torch.testing.assert_close(loss,poisoned,rtol=0,atol=0)
            loss.backward()
            self.assertGreater(sum(float(p.grad.norm()) for p in model.discriminator.parameters() if p.grad is not None),0)

    def test_disabled_flag_is_bitwise_legacy_math_gradients_and_rng(self):
        for method in ("dann","cdan"):
            torch.manual_seed(42);model=self.model(method);args=self.args(method);args.domain_confusion_weight=0
            source,target=self.batches();rng=torch.get_rng_state()
            first=model.objective(source,target,2,20,args)[0];after=torch.get_rng_state()
            gradient=torch.autograd.grad(first,model.rgb_encoder[-1].weight)
            del args.domain_confusion_weight;torch.set_rng_state(rng)
            legacy=model.objective(source,target,2,20,args)[0]
            legacy_gradient=torch.autograd.grad(legacy,model.rgb_encoder[-1].weight)
            torch.testing.assert_close(first,legacy,rtol=0,atol=0)
            torch.testing.assert_close(gradient[0],legacy_gradient[0],rtol=0,atol=0)
            self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_warmup_does_not_compute_extra_discriminator_forward_or_target_gradient(self):
        model=self.model("dann");args=self.args("dann");args.adaptation_warmup_epochs=3
        source,target=self.batches();target["rgb"].requires_grad_()
        loss,details=model.objective(source,target,0,20,args)
        self.assertEqual(float(details["weighted_domain_confusion"]),0)
        loss.backward()
        self.assertTrue(target["rgb"].grad is None or torch.count_nonzero(target["rgb"].grad)==0)
        self.assertTrue(all(p.grad is None for p in model.discriminator.parameters()))

    def test_real_entry_separately_records_enabled_loss_and_gradient(self):
        from tools.test_weather_vds_core import fixture
        def small(method,*unused,**kw):return self.model(method,kw["num_classes"])
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
            root=Path(tmp);lists=fixture(root)
            args=weather.parser().parse_args(["--method","cdan","--target","backlight","--data-dir",str(lists),
                "--save-dir",str(root/"out"),"--epochs","1","--max-train-batches","1","--batch-size","2",
                "--num-workers","0","--cpu-threads","2","--device","cpu","--no-pretrained","--classifier-layout","linear",
                "--domain-confusion-weight",".05","--trade-off",".1","--adaptation-ramp-epochs","1"])
            with patch.object(weather,"AdaptationModel",small),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            directory=root/"out/cdan_backlight_seed42"
            result=json.loads((directory/"result.json").read_text())
            self.assertEqual(result["enhancement"]["variant_family"],"core_plus_encoder_domain_confusion")
            records=[json.loads(line) for line in (directory/"updates.jsonl").read_text().splitlines()]
            self.assertGreater(records[0]["weighted_domain_confusion"],0)
            diagnostics=json.loads((directory/"gradients.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(diagnostics["enhancement_components"]),{"core","domain_confusion"})
            self.assertFalse(result["target_training_labels_used"])

    def test_summary_does_not_merge_different_confusion_weights(self):
        from tools import test_weather_summary_protocol as fixtures
        from weather.summarize_vds_core import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,method,weight in (("dann_seed42","dann",.01),("cdan_seed42","cdan",.1)):
                fixtures.record(root,name,method=method)
                path=root/name/"result.json";value=json.loads(path.read_text())
                value["enhancement"]={"enabled":True,"variant_family":"core_plus_encoder_domain_confusion","domain_confusion_weight":weight}
                path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"encoder or fusion-normalization"):
                summarize(root)


if __name__=="__main__":unittest.main()
