import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from weather.infrared_statistics import augment_source_ir_statistics


class RejectLabels(dict):
    def __getitem__(self, key):
        if key in {"label", "labels", "y_true", "class"}:
            raise AssertionError("Target truth was accessed")
        return super().__getitem__(key)


class InfraredStatisticsChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def batches(self):
        ramp = torch.linspace(-1,1,64).reshape(1,1,8,8).repeat(3,3,1,1)
        source = {"nir":ramp*.8,"rgb":torch.randn(3,3,8,8),"label":torch.tensor([0,1,2])}
        target = RejectLabels(nir=ramp[:2]*.2+.1, label="forbidden")
        return source,target

    def test_off_is_exact_identity_and_does_not_consume_rng(self):
        source,target=self.batches();state=torch.get_rng_state().clone()
        result,_=augment_source_ir_statistics(source,target)
        self.assertIs(result,source);self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_full_transfer_matches_target_moments_preserves_VIS_labels_and_geometry(self):
        source,target=self.batches();original=source['nir'].clone();ref=target['nir'].clone()
        output,record=augment_source_ir_statistics(source,target,1,1,torch.Generator().manual_seed(42))
        torch.testing.assert_close(output['nir'],target['nir'][:1].expand_as(original),atol=1e-6,rtol=1e-5)
        self.assertIs(output['rgb'],source['rgb']);self.assertIs(output['label'],source['label'])
        self.assertTrue(torch.equal(original,source['nir']));self.assertTrue(torch.equal(ref,target['nir']))
        self.assertEqual(record['source_IR_statistics_augmented'],3)
        self.assertFalse(record['IR_statistics_target_labels_used'])
        self.assertEqual(record['source_IR_clipped_fraction'],0)

    def test_private_generator_reproducibility_and_constant_images_are_finite(self):
        source,target=self.batches();source['nir'].zero_();state=torch.get_rng_state().clone()
        a,ra=augment_source_ir_statistics(source,target,.5,.8,torch.Generator().manual_seed(42))
        b,rb=augment_source_ir_statistics(source,target,.5,.8,torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a['nir'],b['nir']));self.assertEqual(ra,rb)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.isfinite(a['nir']).all())

    def test_all_five_core_losses_and_encoder_gradients_remain_active(self):
        from tools.test_crossmodal_loss import CrossmodalChecks
        from weather import train_vds_core as weather
        from vds2raw.adaptation import METHODS
        helper=CrossmodalChecks()
        for method in METHODS:
            with self.subTest(method=method),torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(782)
                model=helper.make_model(method,3)
                source,target=self.batches()
                # Deliberately different domain covariance. Identical inputs
                # correctly give zero CORAL and must not be used to assert >0.
                source['rgb']=torch.eye(3).reshape(3,3,1,1).expand(3,3,8,8).clone()
                target=RejectLabels(nir=source['nir'].clone()*.25+.1,rgb=torch.full_like(source['rgb'],.2))
                source,_=augment_source_ir_statistics(source,target,1,1,torch.Generator().manual_seed(42))
                args=weather.parser().parse_args(['--method',method,'--target','foggy','--data-dir','unused',
                    '--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','1','--trade-off','.1'])
                weather.resolve_weather_settings(args);args.steps_per_epoch=1
                loss,details=model.objective(source,target,0,3,args);loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(float(details['weighted_core_transfer'].abs()),0)
                for encoder in (model.rgb_encoder,model.nir_encoder):
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None),0)

    def test_actual_entry_records_augmentation_and_no_target_truth(self):
        from tools.test_weather_vds_core import fixture,fast_five_method_model
        from weather import train_vds_core as weather
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(temporary);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','1','--max-train-batches','1','--batch-size','2',
                '--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained',
                '--auxiliary-source-weight','.2','--feature-normalization','unit',
                '--source-ir-statistics-probability','1','--source-ir-statistics-strength','.8'])
            with patch.object(weather,'AdaptationModel',fast_five_method_model),contextlib.redirect_stdout(io.StringIO()):
                weather.run(args)
            output=root/'output/mcc_backlight_seed42'
            result=json.loads((output/'result.json').read_text())
            self.assertEqual(result['enhancement']['variant_family'],'core_plus_source_IR_statistics')
            self.assertEqual(result['input_augmentation']['source_ir_statistics_strength'],.8)
            updates=[json.loads(line) for line in (output/'updates.jsonl').read_text().splitlines()]
            self.assertIn('source_IR_statistics_augmented',str(updates))
            for line in (output/'predictions.jsonl').read_text().splitlines():
                row=json.loads(line)
                self.assertEqual(set(row['auxiliary_probabilities']),{'VIS','IR'})
                for values in row['auxiliary_probabilities'].values():self.assertAlmostEqual(sum(values),1.,places=5)
                if row['partition']=='target_train':self.assertNotIn('y_true',row)


if __name__=='__main__':unittest.main()
