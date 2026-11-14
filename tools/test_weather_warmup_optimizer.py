"""Inactive adaptation heads must not decay during source-CE warm-up."""

import unittest
from unittest.mock import patch

import torch

from tools.test_weather_encoder_screen import TinyConvNeXt
from vds2raw.adaptation import AdaptationModel
from vds2raw.train_adaptation import make_optimizer
from weather.train_vds_core import parser, resolve_weather_settings


class WarmupOptimizerTests(unittest.TestCase):
    def test_source_warmup_preserves_adversary_then_adaptation_updates_it(self):
        torch.set_num_threads(2)
        for method in ("dann", "cdan", "mdd"):
            with self.subTest(method=method), patch("vds2raw.adaptation.convnext_small", side_effect=lambda **_: TinyConvNeXt()):
                torch.manual_seed(42)
                model = AdaptationModel(method, pretrained=False, use_ais=False, ir_channels=3,
                                        image_backbone="convnext_small", freeze_image_encoders=True, num_classes=4)
                args = parser().parse_args(["--method", method, "--data-dir", ".", "--target", "backlight",
                    "--optimizer", "adam", "--lr", ".0003", "--adaptation-warmup-epochs", "3",
                    "--adaptation-ramp-epochs", "1", "--weight-decay", ".0005"])
                resolve_weather_settings(args)
                args.steps_per_epoch = 2
                optimizer, _, _ = make_optimizer(model, args)
                source = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8), "label": torch.arange(4)}
                target = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8)}
                adversary = model.adv_head if method == "mdd" else model.discriminator
                before = [p.detach().clone() for p in adversary.parameters()]
                before_head = [p.detach().clone() for p in model.head.parameters()]
                optimizer.zero_grad(set_to_none=True)
                loss, details = model.objective(source, target, 0, 40, args)
                self.assertEqual(details["transfer_weight"], 0.)
                loss.backward()
                optimizer.step()
                changed = sum(not torch.equal(old, new) for old, new in zip(before, adversary.parameters()))
                self.assertEqual(changed, 0, "Source warm-up changed an inactive adaptation head")
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before_head, model.head.parameters())))
                optimizer.zero_grad(set_to_none=True)
                loss, details = model.objective(source, target, 8, 40, args)
                self.assertGreater(details["transfer_weight"], 0.)
                loss.backward()
                optimizer.step()
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, adversary.parameters())))


if __name__ == "__main__":
    unittest.main()
