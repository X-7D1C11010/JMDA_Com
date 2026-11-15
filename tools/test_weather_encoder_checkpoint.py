from copy import deepcopy
import unittest
import torch
from torchvision.models.convnext import ConvNeXt,CNBlockConfig
from weather.encoder_checkpoint import checkpointed_convnext


class ConvNeXtCheckpointChecks(unittest.TestCase):
    def test_forward_gradients_and_stochastic_depth_rng_match_without_wrapping_encoder(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        original=ConvNeXt([CNBlockConfig(8,16,2),CNBlockConfig(16,None,2)],
                          stochastic_depth_prob=.2,num_classes=3).train()
        recomputed=deepcopy(original)
        image=torch.randn(2,3,32,32)
        state=torch.get_rng_state().clone()
        expected=original(image)
        expected_grads=torch.autograd.grad(expected.square().sum(),tuple(original.parameters()))
        final_rng=torch.get_rng_state().clone()
        torch.set_rng_state(state)
        actual=checkpointed_convnext(recomputed,image)
        actual_grads=torch.autograd.grad(actual.square().sum(),tuple(recomputed.parameters()),retain_graph=True)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        for a,b in zip(expected_grads,actual_grads):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(final_rng,torch.get_rng_state()))
        self.assertIsInstance(recomputed,ConvNeXt)
        self.assertEqual(list(original.state_dict()),list(recomputed.state_dict()))
        self.assertGreater(float(actual_grads[0].norm()),0.)

    def test_evaluation_without_grad_keeps_the_original_forward(self):
        model=ConvNeXt([CNBlockConfig(8,None,2)],num_classes=3).eval()
        x=torch.randn(1,3,32,32)
        with torch.no_grad():
            torch.testing.assert_close(checkpointed_convnext(model,x),model(x),rtol=0,atol=0)


if __name__=="__main__":unittest.main()
