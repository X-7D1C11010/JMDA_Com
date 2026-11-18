"""Source-only prototype history, unknown-target masking and live CDAN checks."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from weather.matched_alignment import SourceClassPrototypeEMA,fewshot_ema_class_alignment
from weather import train_vds_core as weather
from weather.summarize_class_matching import summarize
from tools.test_weather_vds_core import fixture,fast_five_method_model
from tools.run_label_guided_matrix import build_jobs
from tools.run_cdan_foggy_phase_lr_test import remaining_seed_gate,build_source_ema_job


class SourceEMAChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_source_only_history_ema_math_and_cached_class_gradient(self):
        bank=SourceClassPrototypeEMA(3,2,.5)
        bank.update_source(torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([0,1]))
        bank.update_source(torch.tensor([[1.,0.]]),torch.tensor([1]))
        torch.testing.assert_close(bank.centroids[1],torch.tensor([2**-.5,2**-.5]))
        self.assertEqual(bank.sample_counts.tolist(),[1,2,0])
        self.assertEqual(sum(p.numel() for p in bank.parameters()),0)
        source=torch.tensor([[1.,0.]],requires_grad=True)
        target=torch.tensor([[.5,.5],[2.,-3.]],requires_grad=True)
        loss,details=fewshot_ema_class_alignment(source,target,torch.tensor([1]),torch.tensor([0,-1]),bank)
        loss.backward()
        self.assertIsNone(source.grad)
        self.assertGreater(float(target.grad[0].norm()),0.)
        self.assertEqual(float(target.grad[1].norm()),0.)
        self.assertEqual(details['aligned_from_cache_not_current_batch'],1)
        self.assertEqual(details['aligned_target_positions'],[0])
        self.assertEqual(details['target_samples_in_prototype_updates'],0)
        self.assertEqual(bank.sample_counts.tolist(),[1,3,0])
        restored=SourceClassPrototypeEMA(3,2,.5)
        restored.load_state_dict(bank.state_dict())
        torch.testing.assert_close(restored.centroids,bank.centroids)

    def test_threshold_uses_real_score_and_starts_only_the_four_other_seeds(self):
        self.assertEqual(remaining_seed_gate(64.999,65.,46),[])
        self.assertEqual(remaining_seed_gate(65.,65.,46),[42,43,44,45])
        self.assertEqual(remaining_seed_gate(80.,65.,42),[43,44,45,46])
        with self.assertRaises(ValueError):remaining_seed_gate(float('nan'),65.,46)

    def test_actual_cdan_native_losses_unchanged_and_history_audit_rejects_contamination(self):
        with tempfile.TemporaryDirectory() as name,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(name);lists=fixture(root)
            args=weather.parser().parse_args(['--method','cdan','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','3','--batch-size','2','--num-workers','0',
                '--cpu-threads','2','--device','cpu','--no-pretrained','--target-class-matching','target_truth',
                '--target-labels-per-class','1','--matched-alignment-weight','.2','--source-anchor-ema-momentum','.9',
                '--adaptation-warmup-epochs','0','--adaptation-ramp-epochs','0'])
            checked=[]
            def factory(*a,**kw):
                model=fast_five_method_model(*a,**kw);original=model.objective
                def objective(source,target,*rest):
                    if checked:return original(source,target,*rest)
                    baseline=copy.deepcopy(model);before=torch.get_rng_state()
                    total,details=original(source,target,*rest);after=torch.get_rng_state()
                    plain_args=copy.copy(rest[-1]);plain_args.matched_alignment_weight=0.
                    torch.set_rng_state(before)
                    plain,plain_details=type(model).objective(baseline,source,target,*rest[:-1],plain_args)
                    torch.set_rng_state(after)
                    torch.testing.assert_close(total-plain,details['weighted_matched_alignment'])
                    self.assertTrue(torch.equal(details['transfer'],plain_details['transfer']))
                    self.assertTrue(torch.equal(details['source_ce'],plain_details['source_ce']))
                    checked.append(True)
                    return total,details
                model.objective=objective;return model
            with patch.object(weather,'AdaptationModel',factory),contextlib.redirect_stdout(io.StringIO()):
                result=weather.run(args)
            self.assertTrue(checked)
            self.assertEqual(result['enhancement']['matched_class_alignment']['source_anchor_mode'],'source_ema')
            p=root/'output/cdan_backlight_seed42'
            updates=[json.loads(s) for s in (p/'updates.jsonl').read_text().splitlines()]
            self.assertTrue(any(u['matched_alignment_details']['aligned_from_cache_not_current_batch']>0 for u in updates))
            summarize([p/'result.json'],root/'summary')
            updates[-1]['matched_alignment_details']['source_prototype_sample_counts'][0]+=1
            (p/'updates.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in updates))
            with self.assertRaisesRegex(ValueError,'EMA source prototype history'):
                summarize([p/'result.json'],root/'bad_summary')

    def test_invalid_protocol_is_rejected_before_data_is_read(self):
        args=weather.parser().parse_args(['--method','cdan','--target','foggy','--data-dir','missing',
                                         '--source-anchor-ema-momentum','.9'])
        with patch.object(weather,'prepare_datasets',side_effect=AssertionError('Must not load data')):
            with self.assertRaises(ValueError):weather.run(args)

    def test_preset_only_changes_source_anchor_policy_and_metadata(self):
        root=Path('root')
        baseline=build_jobs(root,root/'lists',['cdan_foggy'],[46],root,True,
            overrides={'matched_alignment_weight':.2},evaluation_mode='validation_ce',
            unified_fewshot=True,fewshot_alignment_variant=True)[0]['values']
        new=build_jobs(root,root/'lists',['cdan_foggy'],[46],root,True,
            overrides={'matched_alignment_weight':.2,'source_anchor_ema_momentum':.9},
            evaluation_mode='validation_ce',unified_fewshot=True,fewshot_alignment_variant=True)[0]['values']
        self.assertEqual({k for k in set(baseline)|set(new) if baseline.get(k)!=new.get(k)},
                         {'source_anchor_ema_momentum','configuration_tag'})
        self.assertEqual(new['image_adaptation_lr_scale'],.1)
        self.assertIsNone(new.get('adversarial_lr_override'))
        self.assertEqual(new['target_labels_per_class'],3)
        original={'values':baseline}
        job=build_source_ema_job(original,root/'new',.9)
        self.assertEqual(set(job['changed_argument_keys']),
            {'source_anchor_ema_momentum','configuration_tag','save_dir'})
        enhanced=build_source_ema_job(original,root/'extra',.9,.01)
        self.assertEqual(set(enhanced['changed_argument_keys']),
            {'source_anchor_ema_momentum','domain_confusion_weight','configuration_tag','save_dir'})
        self.assertEqual(enhanced['values']['domain_confusion_weight'],.01)
        reference=build_source_ema_job(original,root/'reference',.9,0.,'reference')
        self.assertEqual(set(reference['changed_argument_keys']),
            {'source_anchor_ema_momentum','classifier_layout','configuration_tag','save_dir'})
        self.assertEqual(reference['values']['classifier_layout'],'reference')


if __name__=='__main__':unittest.main()
