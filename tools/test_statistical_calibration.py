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
from weather.statistical_calibration import TrainFeatureCalibration,ClusterInitialization,ResidualFeatureAdapter,fused_features
from vds2raw.adaptation import AdaptationModel,METHODS
from weather import train_vds_core as weather


class CalibrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        rng=np.random.default_rng(42)
        source=rng.normal(size=(30,8));target=rng.normal(size=(24,8))*2+1
        labels=np.arange(30)%3
        cal=TrainFeatureCalibration.fit(source,target,4)
        s=cal(torch.tensor(source,dtype=torch.float32),"source")
        t=cal(torch.tensor(target,dtype=torch.float32),"target")
        teacher=ClusterInitialization.fit(s.numpy(),t.numpy(),labels,3)
        return cal,teacher,source,target

    def test_fusion_accepts_declared_unequal_modal_widths(self):
        raw=np.array([[3.,4.,2.,0.,0.],[0.,2.,0.,3.,4.]])
        actual=fused_features(raw,.75,visible_dimension=2)
        scale=np.sqrt(.75**2+.25**2)
        expected=np.array([[.75*.6,.75*.8,.25,0.,0.],[0.,.75,0.,.25*.6,.25*.8]])/scale
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        with self.assertRaises(ValueError):fused_features(raw,.75)
        with self.assertRaises(ValueError):fused_features(raw,.75,visible_dimension=5)

    def test_frozen_statistics_and_mixed_domain_split_do_not_depend_on_batch_members(self):
        cal,_,source,target=self.fixture()
        a,b=torch.tensor(source,dtype=torch.float32),torch.tensor(target,dtype=torch.float32)
        before={k:v.clone() for k,v in cal.state_dict().items()}
        mixed=cal(torch.cat((a,b)),source_count=len(a))
        torch.testing.assert_close(mixed[:len(a)],cal(a,"source"))
        torch.testing.assert_close(mixed[len(a):],cal(b,"target"))
        torch.testing.assert_close(cal(b[:1],"target"),cal(b,"target")[:1],rtol=1e-5,atol=1e-6)
        cal.train();cal(b,"target")
        for k,v in before.items():torch.testing.assert_close(v,cal.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(list(cal.parameters()),[])
        self.assertEqual(cal.metadata["fit_samples"],{"source_train":30,"target_train":24})

    def test_linear_head_initialization_reproduces_source_associated_cluster_predictions(self):
        _,teacher,_,_=self.fixture()
        head=nn.Linear(4,3)
        teacher.initialize_head(head)
        x=torch.randn(8,4)
        torch.testing.assert_close(head(x),teacher(x),rtol=1e-5,atol=1e-5)
        self.assertEqual(set(teacher.metadata["cluster_to_source_class"]),{0,1,2})
        adapter=ResidualFeatureAdapter(4)
        torch.testing.assert_close(adapter(x),x,rtol=0,atol=0)
        adapter(x).square().mean().backward()
        self.assertGreater(float(adapter.delta.weight.grad.norm()),0.)

    def test_gradient_logger_handles_a_batch_without_any_eligible_teacher_sample(self):
        from vds2raw.train_adaptation import gradient_diagnostics
        model=nn.Module();model.head=nn.Linear(2,3)
        classification=model.head(torch.randn(4,2)).square().mean()
        result=gradient_diagnostics(model,classification,torch.zeros(()))
        self.assertEqual(result["head"]["weighted_DA_norm"],0.)

    def test_auxiliary_modalities_associate_clusters_from_source_labels_only(self):
        cal,_,source,target=self.fixture()
        s=cal(torch.tensor(source,dtype=torch.float32),"source").numpy()
        t=cal(torch.tensor(target,dtype=torch.float32),"target").numpy()
        for method in ("source_votes","source_prototypes","source_neighbors","source_trimmed"):
            teacher=ClusterInitialization.fit(s,t,np.arange(30)%3,3,matching=method,
                association_source=s[:,:2],association_target=t[:,:2])
            self.assertTrue(teacher.metadata["auxiliary_paired_modalities_for_source_association"])
            self.assertEqual(set(teacher.metadata["cluster_to_source_class"]),{0,1,2})
            self.assertTrue(torch.isfinite(teacher(torch.tensor(t))).all())

    def test_all_named_cores_use_trainable_adapter_and_ignore_target_label_metadata(self):
        for method in METHODS:
            torch.manual_seed(42)
            model=AdaptationModel(method,pretrained=False,num_classes=3,use_ais=False,ir_channels=3,
                feature_normalization="unit",freeze_image_encoders=True,classifier_layout="linear",calibration_feature_dim=4)
            for key in ("rgb_encoder","nir_encoder"):
                setattr(model,key,nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,512)).requires_grad_(False))
            rng=np.random.default_rng(42)
            source=rng.normal(size=(30,1024));target=rng.normal(size=(24,1024))+1
            model.feature_calibrator=TrainFeatureCalibration.fit(source,target,4)
            with torch.no_grad():
                a=model.feature_calibrator(torch.tensor(source,dtype=torch.float32),"source")
                b=model.feature_calibrator(torch.tensor(target,dtype=torch.float32),"target")
            model.cluster_teacher=ClusterInitialization.fit(a.numpy(),b.numpy(),np.arange(30)%3,3)
            model.register_buffer("teacher_modality_weights",torch.tensor([.5,.5]))
            model.feature_adapter=ResidualFeatureAdapter(4)
            model.cluster_teacher.initialize_head(model.head)
            with torch.no_grad():model.head.bias[0]+=.3
            args=weather.parser().parse_args(["--method",method,"--target","rainy","--data-dir",".",
                "--cluster-teacher-weight","1","--cluster-teacher-confidence","0"])
            weather.resolve_weather_settings(args);args.steps_per_epoch=10
            s={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8),"label":torch.arange(4)%3}
            t={"rgb":torch.randn(4,3,8,8),"nir":torch.randn(4,3,8,8)}
            model.eval()
            torch.manual_seed(0);loss,detail=model.objective(s,t,100,1000,args)
            torch.manual_seed(0);poisoned,_=model.objective(s,{**t,"label":object(),"path":["999"]*4},100,1000,args)
            torch.testing.assert_close(loss,poisoned,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(loss))
            core_grad=torch.autograd.grad(detail["weighted_core_transfer"],model.feature_adapter.delta.weight,retain_graph=True)[0]
            self.assertGreater(float(core_grad.norm()),0.)
            self.assertTrue(torch.isfinite(core_grad).all())
            loss.backward()
            self.assertIsNotNone(model.feature_adapter.delta.weight.grad)

    def test_real_entry_fits_training_partitions_and_records_separate_losses(self):
        from tools.test_weather_vds_core import fixture
        def small_model(method,*unused,**kwargs):
            m=AdaptationModel(method,pretrained=False,num_classes=kwargs["num_classes"],use_ais=False,ir_channels=3,
                feature_normalization="unit",freeze_image_encoders=True,classifier_layout="linear",calibration_feature_dim=4)
            for key in ("rgb_encoder","nir_encoder"):
                setattr(m,key,nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,512)).requires_grad_(False))
            return m
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
            root=Path(temporary);lists=fixture(root)
            args=weather.parser().parse_args(["--method","mcc","--target","backlight","--data-dir",str(lists),
                "--save-dir",str(root/"output"),"--epochs","1","--max-train-batches","1","--num-workers","0",
                "--device","cpu","--no-pretrained","--feature-calibration","--calibration-components","4",
                "--feature-normalization","unit","--classifier-layout","linear","--freeze-image-encoders",
                "--cluster-teacher-weight","1","--batch-size","4","--log-feature-moments"])
            with patch.object(weather,"AdaptationModel",small_model),contextlib.redirect_stdout(io.StringIO()):
                result=weather.run(args)
            audit=json.loads((root/"output/mcc_backlight_seed42/calibration_audit.json").read_text())
            update=json.loads((root/"output/mcc_backlight_seed42/updates.jsonl").read_text().splitlines()[0])
            self.assertTrue(result["enhancement"]["enabled"])
            self.assertEqual(audit["calibration"]["fit_samples"],{"source_train":8,"target_train":8})
            self.assertFalse(audit["validation_used_for_fitting"])
            self.assertIn("weighted_cluster_teacher",update)
            self.assertTrue(result["smoke"])


if __name__=="__main__":unittest.main()
