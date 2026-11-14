import unittest

import torch
from torch import nn

from weather.spatial_pooling import FixedGeM


class SpatialPoolingTests(unittest.TestCase):
    def test_p_one_preserves_existing_average_output_and_gradient_exactly(self):
        features = torch.randn(2, 3, 5, 7, dtype=torch.float64, requires_grad=True)
        expected = nn.AdaptiveAvgPool2d(1)(features)
        actual = FixedGeM(1.)(features)
        self.assertTrue(torch.equal(actual, expected))
        left = torch.autograd.grad(actual.sum(), features, retain_graph=True)[0]
        right = torch.autograd.grad(expected.sum(), features)[0]
        self.assertTrue(torch.equal(left, right))

    def test_gem_highlights_peaks_with_finite_gradient_and_unchanged_width(self):
        features = torch.tensor([0., 1., 2., 8.], dtype=torch.float64).reshape(1, 1, 2, 2).requires_grad_()
        pooled = FixedGeM(3.)(features)
        self.assertEqual(pooled.shape, (1, 1, 1, 1))
        self.assertGreater(pooled.item(), features.mean().item())
        self.assertLess(pooled.item(), features.max().item())
        pooled.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.flatten()[-1].item(), features.grad.flatten()[1].item())
        self.assertEqual(list(FixedGeM(3.).parameters()), [])


if __name__ == "__main__":
    unittest.main()
