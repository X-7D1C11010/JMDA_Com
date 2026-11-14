from copy import deepcopy
import unittest
import torch
from torch import nn
from weather.pretrained_ir import calibrate_half_normalized_stem


class PretrainedIRTests(unittest.TestCase):
    def test_original_imagenet_activation_is_preserved_under_half_normalized_input(self):
        torch.manual_seed(42)
        original = nn.Conv2d(3, 8, 4, stride=4)
        original_state = {k: v.clone() for k, v in original.state_dict().items()}
        calibrated = calibrate_half_normalized_stem(deepcopy(original))
        grey = torch.rand(3, 1, 28, 28).repeat(1, 3, 1, 1)
        mean = grey.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = grey.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        torch.testing.assert_close(calibrated((grey-.5)/.5), original((grey-mean)/std), atol=8e-7, rtol=1e-5)
        self.assertTrue(all(torch.equal(v, original.state_dict()[k]) for k, v in original_state.items()))

    def test_rejects_padded_stems_where_bias_correction_is_not_exact(self):
        with self.assertRaisesRegex(ValueError, "unpadded"):
            calibrate_half_normalized_stem(nn.Conv2d(3, 8, 3, padding=1))


if __name__ == "__main__":
    unittest.main()
