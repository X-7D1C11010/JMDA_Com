import unittest

import torch
from torch import nn

from weather.ibn_probe import IBN, resnet50_ibn_a_probe


class IBNProbeTests(unittest.TestCase):
    def test_instance_half_removes_per_image_brightness_while_bn_half_preserves_it(self):
        module = IBN(8).eval()
        images = torch.randn(2, 8, 12, 12)
        normal, shifted = module(images), module(images*3.+8.)
        torch.testing.assert_close(normal[:, :4], shifted[:, :4], rtol=1e-4, atol=1e-4)
        self.assertGreater(float((normal[:, 4:]-shifted[:, 4:]).abs().mean()), 3.)

    def test_official_a_layout_only_changes_first_three_stages(self):
        model = resnet50_ibn_a_probe()
        self.assertEqual(sum(isinstance(m, IBN) for m in model.modules()), 13)
        self.assertIsInstance(model.layer4[0].bn1, nn.BatchNorm2d)
        self.assertEqual(model.fc.in_features, 2048)
        self.assertEqual(model.avgpool.kernel_size, 7)


if __name__ == "__main__":
    unittest.main()
