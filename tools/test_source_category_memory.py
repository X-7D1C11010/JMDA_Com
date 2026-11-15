import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from vds2raw.crossmodal_loss import SourceCategoryMemory,install_crossmodal_projectors
from vds2raw.adaptation import METHODS
from tools import test_crossmodal_loss as checks
from tools.test_weather_vds_core import fixture
from weather import train_vds_core as weather


class SourceMemoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_prototypes_retain_class_information_across_batches_and_backpropagate(self):
        memory=SourceCategoryMemory(3,5)
        v=torch.randn(4,5,requires_grad=True);i=torch.randn(4,5,requires_grad=True)
        loss,seen=memory(v,i,torch.tensor([0,0,1,1]))
        self.assertEqual(seen,2);self.assertEqual(memory.observations.tolist(),[2,2,0])
        loss.backward();self.assertGreater(float(v.grad.norm()),0);self.assertGreater(float(i.grad.norm()),0)
        memory(torch.randn(2,5),torch.randn(2,5),torch.tensor([2,2]))
        self.assertEqual(memory.observations.tolist(),[2,2,2])
        self.assertTrue(torch.isfinite(loss))

    def test_each_core_uses_only_source_labels_for_memory(self):
        for method in METHODS:
            torch.manual_seed(42);model=checks.CrossmodalChecks().make_model(method)
            install_crossmodal_projectors(model,64);model.source_category_memory=SourceCategoryMemory(3,64)
            args=checks.CrossmodalChecks().args(method);args.source_prototype_weight=.2;args.target_crossmodal_weight=0
            source={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8),"label":torch.tensor([0,0,1,2])}
            target={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8)}
            state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state()
            loss,detail=model.objective(source,target,2,8,args)
            grads=torch.autograd.grad(detail["weighted_source_prototype"],[model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight],retain_graph=True)
            self.assertTrue(all(float(g.norm())>0 for g in grads))
            self.assertEqual(model.source_category_memory.observations.tolist(),[2,1,1])
            model.load_state_dict(state);torch.set_rng_state(rng)
            poisoned=model.objective(source,dict(target,label=torch.full((4,),999)),2,8,args)[0]
            torch.testing.assert_close(loss,poisoned,rtol=0,atol=0)
            before=model.source_category_memory.observations.clone()
            model.eval()
            with torch.no_grad():model.encode(target)
            torch.testing.assert_close(before,model.source_category_memory.observations,rtol=0,atol=0)

    def test_real_entry_prototype_only_loss_and_feature_export_do_not_fit_eval_labels(self):
        created=[]
        def small_model(method,*unused,**kwargs):
            model=checks.CrossmodalChecks().make_model(method,kwargs["num_classes"]);created.append(model);return model
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);lists=fixture(root)
            args=weather.parser().parse_args(["--method","mcc","--target","backlight","--data-dir",str(lists),"--save-dir",str(root/"out"),
                "--epochs","1","--max-train-batches","1","--batch-size","2","--num-workers","0","--cpu-threads","2","--device","cpu","--no-pretrained",
                "--source-prototype-weight",".2","--crossmodal-projection-dim","64","--cache-best-features"])
            with patch.object(weather,"AdaptationModel",small_model),contextlib.redirect_stdout(io.StringIO()):result=weather.run(args)
            self.assertEqual(int(created[0].source_category_memory.observations.sum()),2)
            self.assertEqual(result["enhancement"]["source_prototype_weight"],.2)
            self.assertEqual(result["selected_feature_cache"]["epoch"],1)
            self.assertFalse(result["target_training_labels_used"])


if __name__=="__main__":unittest.main()
