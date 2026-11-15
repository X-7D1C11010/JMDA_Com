import unittest
import torch
from weather.ir_input_normalizer import InfraredInputNormalizer


class IRNormalizationChecks(unittest.TestCase):
    def test_identity_has_no_rng_or_tensor_changes(self):
        x=torch.randn(2,3,16,16);state=torch.get_rng_state().clone()
        self.assertIs(InfraredInputNormalizer()(x),x)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_per_sample_computation_is_batch_independent_and_never_fits_dataset_buffers(self):
        x=torch.linspace(-.1,.1,256).view(1,1,16,16).repeat(1,3,1,1)
        for mode in ('zscore','percentile'):
            op=InfraredInputNormalizer(mode)
            before={k:v.clone() for k,v in op.state_dict().items()}
            y=op(x);z=op(torch.cat((x,torch.rand_like(x)*2-1),0))[:1]
            torch.testing.assert_close(y,z)
            self.assertEqual(list(before),list(op.state_dict()))
            self.assertEqual(len(list(op.parameters())),0)
            self.assertGreater(float(y.std()),float(x.std()))
            self.assertTrue(torch.isfinite(op(torch.zeros_like(x))).all())
            self.assertTrue(torch.all(y.flatten()[1:256]-y.flatten()[:255]>0))


if __name__=='__main__':unittest.main()
