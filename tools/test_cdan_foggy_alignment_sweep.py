"""The sweep must retain the real historical training and supervision budget."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.run_cdan_foggy_alignment_sweep import ROOT,build_variant,save_comparison
from tools.run_label_guided_matrix import build_jobs


class CdanFoggyCoefficientChecks(unittest.TestCase):
    def test_seed42_reuse_summary_marks_partial_and_aggregates_five_unique_seeds(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            baselines=[{'seed':s,'alignment_weight':.1,'peak_epoch':1,
                **{m:50. for m in ('ACC','Precision','Recall','F1')}} for s in range(42,47)]
            reused={'seed':42,'alignment_weight':.2,'peak_epoch':30,'ACC':60.,'Precision':61.,'Recall':62.,'F1':63.}
            save_comparison(out,baselines+[reused])
            partial=next(r for r in json.loads((out/'summary.json').read_text()) if r['alignment_weight']==.2)
            self.assertEqual(partial['completed_seeds'],1)
            self.assertFalse(partial['all_requested_seeds_completed'])
            self.assertIsNone(partial['ACC_std'])
            completed=[dict(reused,seed=s,ACC=60.+s-42) for s in range(43,47)]
            save_comparison(out,baselines+[reused]+completed)
            summary=next(r for r in json.loads((out/'summary.json').read_text()) if r['alignment_weight']==.2)
            self.assertEqual(summary['completed_seeds'],5)
            self.assertTrue(summary['all_requested_seeds_completed'])
            self.assertEqual(summary['ACC_mean'],62.)
            self.assertAlmostEqual(summary['ACC_std'],2**.5)
            self.assertEqual(summary['F1_mean'],63.)
            with self.assertRaisesRegex(ValueError,'Repeated'):
                save_comparison(out,baselines+[reused,reused])

    def test_variants_change_only_coefficient_tag_and_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);baseline=root/'baseline';baseline.mkdir()
            jobs=build_jobs(baseline,root/'lists',['cdan_foggy'],[42],ROOT,True,
                            evaluation_mode='validation_ce',unified_fewshot=True)
            (baseline/'plan.json').write_text(json.dumps({'unified_fewshot':True,'jobs':jobs}))
            for weight in (.2,.3):
                job=build_variant(baseline,root/f'weight{weight}',42,weight)
                self.assertEqual(set(job['changed_argument_keys']),
                    {'matched_alignment_weight','configuration_tag','save_dir'})
                self.assertEqual(job['values']['data_dir'],jobs[0]['values']['data_dir'])
                self.assertEqual(job['baseline_result_file'],jobs[0]['result_file'])
            # Accidental learning-rate drift must fail before a GPU job starts.
            jobs[0]['values']['image_lr_override']*=2
            (baseline/'plan.json').write_text(json.dumps({'unified_fewshot':True,'jobs':jobs}))
            with self.assertRaisesRegex(ValueError,'Unexpected changes'):
                build_variant(baseline,root/'bad',42,.2)


if __name__=='__main__':unittest.main()
