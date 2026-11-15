"""Test target-batch scaling without dropping samples or changing source updates."""
from types import SimpleNamespace
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
from weather.train_vds_core import paired_epoch_samplers
from vds2raw.adaptation import AdaptationModel
from weather.target_batch_diagnostics import target_batch_diagnostics


class TargetBatchChecks(unittest.TestCase):
    def test_larger_unlabelled_batch_preserves_source_sampler_and_update_count(self):
        for source_count,target_count in ((493,465),(401,96),(515,440)):
            steps,original=paired_epoch_samplers(range(source_count),range(target_count),8,None,42,"mcc")
            larger_steps,larger=paired_epoch_samplers(range(source_count),range(target_count),8,32,42,"mcc")
            self.assertEqual(steps,larger_steps)
            self.assertEqual(list(original[0]),list(larger[0]))
            targets=list(larger[1])
            self.assertTrue(all(len(batch)==32 for batch in targets))
            self.assertEqual(set(sum(targets,[])),set(range(target_count)))
        with self.assertRaisesRegex(ValueError,"equal"):
            paired_epoch_samplers(range(100),range(94),8,32,42,"mdd")

    def test_mcc_target_only_loss_and_encoder_gradients_with_unequal_batches(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=AdaptationModel("mcc",pretrained=False,num_classes=3,use_ais=False,ir_channels=3,
                              classifier_layout="linear")
        for name in ("rgb_encoder","nir_encoder"):
            setattr(model,name,nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,512)))
        model.eval()
        source={"rgb":torch.randn(8,3,8,8),"nir":torch.randn(8,3,8,8),"label":torch.arange(8)%3}
        target={"rgb":torch.randn(32,3,8,8),"nir":torch.randn(32,3,8,8)}
        args=SimpleNamespace(label_smoothing=0.,steps_per_epoch=10,adaptation_warmup_epochs=0,
                             adaptation_ramp_epochs=1,trade_off=.5)
        loss,detail=model.objective(source,target,20,100,args)
        self.assertEqual(detail["source_logits"].shape,(8,3))
        self.assertEqual(detail["target_logits"].shape,(32,3))
        torch.testing.assert_close(detail["transfer"],model.mcc(detail["target_logits"]))
        gradients=torch.autograd.grad(detail["weighted_transfer"],tuple(model.nir_encoder.parameters()),retain_graph=True)
        self.assertGreater(float(sum(g.square().sum() for g in gradients)),0.)
        rng=torch.get_rng_state().clone()
        observed=target_batch_diagnostics(detail["target_logits"],[str(i%29) for i in range(32)],model.mcc.temperature)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(observed["unique_samples"],29)
        self.assertEqual(sum(observed["predicted_class_counts"]),32)
        self.assertAlmostEqual(observed["MCC"]["loss_reconstructed"],float(detail["transfer"].detach()),places=6)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_real_training_entry_logs_confusion_without_changing_updates(self):
        from weather import train_vds_core as weather
        from tools.test_weather_vds_core import fixture,fast_five_method_model
        observed=[]
        for diagnostics in (False,True):
            with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{"JMDA_DATA_ROOT":""}):
                root=Path(temporary);lists=fixture(root)
                argv=["--method","mcc","--target","backlight","--data-dir",str(lists),
                    "--save-dir",str(root/"output"),"--epochs","1","--batch-size","4",
                    "--target-batch-size","8","--max-train-batches","1","--num-workers","0",
                    "--mcc-temperature",".5",
                    "--device","cpu","--no-pretrained"]
                if diagnostics:argv.append("--log-feature-moments")
                args=weather.parser().parse_args(argv)
                with patch.object(weather,"AdaptationModel",fast_five_method_model),contextlib.redirect_stdout(io.StringIO()):
                    result=weather.run(args)
                folder=root/"output/mcc_backlight_seed42"
                update=json.loads((folder/"updates.jsonl").read_text().splitlines()[0])
                predictions=[json.loads(line)["logits"] for line in (folder/"predictions.jsonl").read_text().splitlines()]
                observed.append((update["source_ce"],update["transfer"],predictions))
                self.assertEqual(result["per_domain_batch_sizes"],{"source":4,"target":8})
                self.assertTrue(result["smoke"])
                if diagnostics:
                    batch=update["target_batch_diagnostics"]
                    self.assertEqual((batch["samples"],batch["unique_samples"]),(8,8))
                    self.assertFalse(batch["real_labels_used"])
                    self.assertEqual(batch["MCC"]["temperature"],.5)
                    self.assertAlmostEqual(batch["MCC"]["loss_reconstructed"],update["transfer"],places=6)
        self.assertEqual(observed[0],observed[1])


if __name__=="__main__":unittest.main()
