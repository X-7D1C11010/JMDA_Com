"""Domain running-statistics isolation, eval safety and reference gradients."""
from copy import deepcopy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from weather.domain_running_stats import DomainRunningStats, install_domain_statistics, set_domain
from vds2raw.adaptation import AdaptationModel


class DomainStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_separate_buffers_share_affine_and_evaluation_never_fits(self):
        original = nn.BatchNorm1d(2, momentum=1.)
        layer = DomainRunningStats(original).train()
        source = torch.tensor([[1., 4.], [3., 6.]])
        target = source+100
        layer(source)
        source_mean = original.running_mean.clone()
        set_domain(layer, "target")
        layer(target)
        torch.testing.assert_close(original.running_mean, source_mean)
        torch.testing.assert_close(layer.target_mean, source_mean+100)
        self.assertIs(layer.bn.weight, original.weight)
        self.assertEqual((int(original.num_batches_tracked), int(layer.target_batches)), (1, 1))
        before = {k: v.clone() for k, v in layer.state_dict().items()}
        layer.eval()(torch.randn(7, 2)*500)
        for key, value in layer.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_separate_forward_methods_preserve_training_objectives_and_encoder_gradients(self):
        source = {"rgb": torch.randn(4, 3, 64, 64), "nir": torch.randn(4, 3, 64, 64), "label": torch.arange(4)}
        target = {"rgb": torch.randn(4, 3, 64, 64), "nir": torch.randn(4, 3, 64, 64)}
        args = SimpleNamespace(domain_forward="reference", label_smoothing=0., steps_per_epoch=2,
                               adaptation_warmup_epochs=0, adaptation_ramp_epochs=1, trade_off=.2)
        for method in ("dann", "deepcoral", "cdan"):
            torch.manual_seed(42)
            reference = AdaptationModel(method, pretrained=False, num_classes=4, use_ais=False, ir_channels=3).train()
            variant = deepcopy(reference)
            self.assertGreater(install_domain_statistics(variant), 0)
            variant.domain_running_stats = True
            state = torch.get_rng_state()
            expected, detail = reference.objective(source, target, 10, 200, args)
            torch.set_rng_state(state)
            actual, changed = variant.objective(source, target, 10, 200, args)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            torch.testing.assert_close(detail["source_logits"], changed["source_logits"], rtol=0, atol=0)
            torch.testing.assert_close(detail["target_logits"], changed["target_logits"], rtol=0, atol=0)
            expected.backward()
            actual.backward()
            torch.testing.assert_close(reference.rgb_encoder.conv1.weight.grad, variant.rgb_encoder.conv1.weight.grad, rtol=0, atol=0)
            self.assertEqual(int(variant.rgb_encoder.bn1.bn.num_batches_tracked), 1)
            self.assertEqual(int(variant.rgb_encoder.bn1.target_batches), 1)
            torch.testing.assert_close(variant.rgb_encoder.bn1.pooled_mean, reference.rgb_encoder.bn1.running_mean, rtol=0, atol=0)
            torch.testing.assert_close(variant.rgb_encoder.bn1.pooled_var, reference.rgb_encoder.bn1.running_var, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
