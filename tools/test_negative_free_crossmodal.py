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
from vds2raw.adaptation import METHODS
from vds2raw.crossmodal_loss import paired_variance_covariance_loss,install_crossmodal_projectors
from weather import train_vds_core as weather
from tools import test_crossmodal_loss as helpers


class NegativeFreeCrossmodalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_components_match_independent_numpy_formula(self):
        a=np.arange(20,dtype=np.float32).reshape(5,4)/10
        b=a[::-1].copy()+.3
        loss,cosine,details=paired_variance_covariance_loss(torch.tensor(a),torch.tensor(b))
        inv=np.mean((a-b)**2)
        var=sum(np.maximum(1-np.sqrt(x.var(0,ddof=1)+1e-4),0).mean() for x in (a,b))/2
        cov=0.
        for x in (a,b):
            c=np.cov(x,rowvar=False);cov+=(np.square(c).sum()-np.square(np.diag(c)).sum())/x.shape[1]
        self.assertAlmostEqual(float(details['invariance']),inv,places=6)
        self.assertAlmostEqual(float(details['variance']),var,places=6)
        self.assertAlmostEqual(float(details['covariance']),cov,places=5)
        self.assertAlmostEqual(float(loss),inv+var+cov/25,places=6)
        self.assertTrue(torch.isfinite(cosine))

    def test_pairing_changes_invariance_but_not_marginal_statistics(self):
        a=torch.tensor([[1.,0.,2.],[0.,2.,1.],[3.,0.,1.],[1.,3.,0.]])
        same,_,one=paired_variance_covariance_loss(a,a)
        wrong,_,two=paired_variance_covariance_loss(a,a.roll(1,0))
        self.assertGreater(float(wrong),float(same))
        for key in ['variance','covariance','VIS_mean_std','IR_mean_std']:
            torch.testing.assert_close(one[key],two[key])
        # Identical rows have a positive anti-collapse penalty, not zero loss.
        self.assertGreater(float(paired_variance_covariance_loss(torch.ones(4,3),torch.ones(4,3))[0]),.9)
        for x,y in [(torch.ones(1,3),torch.ones(1,3)),(torch.ones(2,3),torch.ones(2,4))]:
            with self.assertRaises(ValueError):paired_variance_covariance_loss(x,y)

    def model_args(self,method):
        helper=helpers.CrossmodalChecks()
        model=helper.make_model(method);args=helper.args(method)
        args.target_crossmodal_objective='vicreg';args.crossmodal_projection_dim=8
        install_crossmodal_projectors(model,8)
        return model,args

    def test_all_core_losses_keep_target_GT_out_and_receive_both_encoder_gradients(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(42);model,args=self.model_args(method)
                source={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'label':torch.tensor([0,0,1,2])}
                target={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8)}
                rng=torch.get_rng_state();loss,details=model.objective(source,target,2,8,args)
                self.assertIsNotNone(details['target_crossmodal_components'])
                variables=[model.rgb_encoder[-1].weight,model.nir_encoder[-1].weight,
                           model.crossmodal_projectors[0][0].weight,model.crossmodal_projectors[1][0].weight]
                gradients=torch.autograd.grad(details['weighted_target_crossmodal'],variables,retain_graph=True)
                self.assertTrue(all(float(g.norm())>0 and torch.isfinite(g).all() for g in gradients))
                torch.set_rng_state(rng)
                poisoned=model.objective(source,dict(target,label=torch.full((4,),999),path=['dummy']*4),2,8,args)[0]
                torch.testing.assert_close(loss,poisoned,rtol=0,atol=0)

    def test_warmup_and_zero_weight_preserve_original_behavior(self):
        model,args=self.model_args('cdan');args.adaptation_warmup_epochs=2
        source={'rgb':torch.randn(4,3,8,8),'nir':torch.randn(4,3,8,8),'label':torch.tensor([0,0,1,2])}
        target={'rgb':torch.randn(4,3,8,8,requires_grad=True),'nir':torch.randn(4,3,8,8,requires_grad=True)}
        loss,details=model.objective(source,target,0,8,args)
        self.assertEqual(float(details['weighted_target_crossmodal']),0.)
        loss.backward()
        for key in ['rgb','nir']:
            self.assertTrue(target[key].grad is None or int(torch.count_nonzero(target[key].grad))==0)
        args.target_crossmodal_weight=0;args.adaptation_warmup_epochs=0
        rng=torch.get_rng_state()
        from vds2raw.crossmodal_loss import paired_crossmodal_loss
        with patch('vds2raw.crossmodal_loss.paired_crossmodal_loss',wraps=paired_crossmodal_loss) as paired:
            first,disabled=model.objective(source,target,2,8,args)
        # Only source alignment is evaluated. An inactive target objective
        # must not be replaced by a different unweighted loss in the log.
        self.assertEqual(paired.call_count,1)
        self.assertEqual(float(disabled['target_crossmodal_loss']),0.)
        self.assertEqual(float(disabled['weighted_target_crossmodal']),0.)
        self.assertIsNone(disabled['target_pair_cosine'])
        self.assertIsNone(disabled['target_crossmodal_components'])
        self.assertGreater(float(disabled['weighted_source_crossmodal']),0.)
        torch.testing.assert_close(first,disabled['classification'],rtol=0,atol=0)
        after=torch.get_rng_state()
        torch.set_rng_state(rng);args.target_crossmodal_objective='contrastive'
        second=model.objective(source,target,2,8,args)[0]
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_real_entry_logs_components_gradients_and_training_only_projection(self):
        from tools.test_weather_vds_core import fixture
        helper=helpers.CrossmodalChecks()
        def small_model(method,*unused,**kwargs):return helper.make_model(method,kwargs['num_classes'])
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,{'JMDA_DATA_ROOT':''}):
            root=Path(temporary);lists=fixture(root)
            args=weather.parser().parse_args(['--method','mcc','--target','backlight','--data-dir',str(lists),
                '--save-dir',str(root/'output'),'--epochs','1','--max-train-batches','1','--batch-size','2',
                '--num-workers','0','--cpu-threads','2','--device','cpu','--no-pretrained','--feature-normalization','unit',
                '--target-crossmodal-weight','.1','--target-crossmodal-objective','vicreg','--crossmodal-projection-dim','8'])
            with patch.object(weather,'AdaptationModel',small_model),contextlib.redirect_stdout(io.StringIO()):weather.run(args)
            output=root/'output/mcc_backlight_seed42'
            result=json.loads((output/'result.json').read_text())
            update=json.loads((output/'updates.jsonl').read_text().splitlines()[0])
            epoch=json.loads((output/'training.jsonl').read_text().splitlines()[0])
            self.assertEqual(result['enhancement']['target_crossmodal_objective'],'vicreg')
            self.assertFalse(result['target_training_labels_used'])
            self.assertEqual(update['target_crossmodal_objective'],'vicreg')
            for key in ['invariance','variance','covariance','VIS_mean_std','IR_mean_std']:
                self.assertIn('target_vicreg_'+key,update);self.assertIn('target_vicreg_'+key,epoch)
            grad=json.loads((output/'gradients.jsonl').read_text().splitlines()[0])
            self.assertIn('target_crossmodal',grad['enhancement_components'])
            args.crossmodal_projection_dim=0
            with self.assertRaisesRegex(ValueError,'projections'):weather.run(args)


if __name__=='__main__':unittest.main()
