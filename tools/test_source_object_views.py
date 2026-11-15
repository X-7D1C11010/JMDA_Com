import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from tools import test_target_feature_style as toy
from tools.test_weather_vds_core import fixture
from vds2raw.adaptation import METHODS
from weather import train_vds_core as weather
from weather.source_object_views import SourceObjectViews,source_object_loss


def manifest(root,source,audit):
    path=Path(source.paths[0]);inverse={int(v):int(k) for k,v in audit["compact_label_map"].items()}
    label=source.dataset.samples[0][1]
    with Image.open(path) as image:size=list(image.size)
    record={"image_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"original_source_class_id":inverse[label],
            "size":size,"box":[1,1,size[0]-1,size[1]-1]}
    target=root/"source_objects.json"
    target.write_text(json.dumps({"kind":"reviewed_source_only_object_views","source_domain":"sunny",
        "target_images_used":False,"records":[record]}),encoding="utf-8")
    return target


class SourceObjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,method,classes=3,layout="linear"):
        model=toy.TargetFeatureStyleChecks().model(method,classes)
        if layout=="reference":
            model.bottleneck=torch.nn.Sequential(torch.nn.Linear(1024,256),torch.nn.BatchNorm1d(256),torch.nn.ReLU())
            model.head=torch.nn.Linear(256,classes)
        model.source_object_enabled=True
        return model

    def args(self,method):
        args=weather.parser().parse_args(["--method",method,"--target","rainy","--data-dir","unused","--save-dir","unused",
            "--domain-forward","joint","--source-object-ce-weight",".25","--source-object-consistency-weight",".1",
            "--adaptation-warmup-epochs","0","--adaptation-ramp-epochs","1","--trade-off",".1"])
        weather.resolve_weather_settings(args);args.steps_per_epoch=2
        return args

    def test_extra_view_preserves_main_pair_labels_and_all_rng(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
            root=Path(tmp);lists=fixture(root);views,audit=weather.prepare_datasets(lists,"backlight",True)
            source=views[0];path=manifest(root,source,audit)
            objects=SourceObjectViews(source,path,audit["compact_label_map"],42)
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            before=source[0];after_main=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
            source.source_object_views=objects;after=source[0]
            for key in ("rgb","nir","label"):torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            self.assertEqual(random.getstate(),after_main[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],after_main[1][1]))
            self.assertEqual(np.random.get_state()[2:],after_main[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(),after_main[2]))
            self.assertTrue(after["source_object_is_roi"])
            self.assertNotIn("source_object_rgb",views[1][0]);self.assertNotIn("source_object_rgb",views[-1][0])
            self.assertEqual(objects.metadata["source_samples"],len(source))
            self.assertEqual(objects.metadata["reviewed_roi_samples"],1)
            with self.assertRaises(ValueError):SourceObjectViews(views[1],path,audit["compact_label_map"],42)

    def test_invalid_box_or_source_class_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
            root=Path(tmp);lists=fixture(root);views,audit=weather.prepare_datasets(lists,"backlight",True)
            path=manifest(root,views[0],audit);original=json.loads(path.read_text())
            wrong=copy.deepcopy(original);wrong["records"][0]["box"]=[-1,0,5,5];path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,"box"):SourceObjectViews(views[0],path,audit["compact_label_map"],42)
            wrong=copy.deepcopy(original);wrong["records"][0]["original_source_class_id"]=max(map(int,audit["compact_label_map"]))
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,"class mapping"):SourceObjectViews(views[0],path,audit["compact_label_map"],42)

    def test_all_cores_get_source_only_object_gradients_and_ignore_target_labels(self):
        for method in METHODS:
            torch.manual_seed(42);model=self.model(method);args=self.args(method)
            source,target=toy.TargetFeatureStyleChecks().batches()
            source["source_object_rgb"]=torch.randn_like(source["rgb"])
            target["rgb"].requires_grad_();rng=torch.get_rng_state()
            loss,details=model.objective(source,target,2,20,args)
            params=(model.rgb_encoder.features[0].weight,model.nir_encoder.features[0].weight,target["rgb"])
            gradients=torch.autograd.grad(details["weighted_source_object"],params,retain_graph=True,allow_unused=True)
            self.assertTrue(all(g is not None and float(g.norm())>0 for g in gradients[:2]))
            self.assertTrue(gradients[2] is None or torch.count_nonzero(gradients[2])==0)
            self.assertEqual(len(model.last_features[0]),6)
            torch.set_rng_state(rng)
            poisoned=model.objective(source,dict(target,label=torch.full((3,),999),source_object_rgb=object()),2,20,args)[0]
            torch.testing.assert_close(loss,poisoned,rtol=0,atol=0)

    def test_auxiliary_head_does_not_refit_bn_and_accepts_one_source_example(self):
        model=self.model("mcc",layout="reference");source,_=toy.TargetFeatureStyleChecks().batches()
        source["source_object_rgb"]=torch.randn_like(source["rgb"])
        model.encode(source);branches=model.last_object_features
        bn=model.bottleneck[1];before=[v.clone() for v in (bn.running_mean,bn.running_var,bn.num_batches_tracked)]
        subset={k:v[:1] for k,v in source.items()}
        ce,consistency,_=source_object_loss(model,subset,tuple(v[:1] for v in branches))
        self.assertTrue(ce.requires_grad and consistency.requires_grad)
        for x,y in zip(before,(bn.running_mean,bn.running_var,bn.num_batches_tracked)):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertTrue(bn.training)

    def test_real_entry_keeps_full_cohort_and_records_source_only_extra_loss(self):
        def small(method,*unused,**kw):return self.model(method,kw["num_classes"])
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
            root=Path(tmp);lists=fixture(root);views,audit=weather.prepare_datasets(lists,"backlight",True)
            path=manifest(root,views[0],audit)
            args=weather.parser().parse_args(["--method","mcc","--target","backlight","--data-dir",str(lists),
                "--save-dir",str(root/"out"),"--epochs","1","--max-train-batches","1","--batch-size","2",
                "--num-workers","0","--cpu-threads","2","--device","cpu","--no-pretrained","--image-backbone","convnext_small",
                "--source-object-manifest",str(path),"--source-object-ce-weight",".25","--source-object-consistency-weight",".1"])
            with patch.object(weather,"AdaptationModel",small),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            directory=root/"out/mcc_backlight_seed42";result=json.loads((directory/"result.json").read_text())
            record=json.loads((directory/"updates.jsonl").read_text().splitlines()[0])
            self.assertEqual(result["enhancement"]["variant_family"],"core_plus_source_object_views")
            self.assertEqual(result["enhancement"]["source_object_views"]["source_samples"],len(views[0]))
            self.assertGreater(record["weighted_source_object"],0)
            self.assertFalse(result["target_training_labels_used"])


if __name__=="__main__":unittest.main()
