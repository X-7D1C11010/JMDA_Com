"""Regression coverage for opt-in encoder screening; not a new default."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from vds2raw.adaptation import AdaptationModel, METHODS, optimizer_groups
from weather.train_vds_core import parser, resolve_weather_settings
from vds2raw.train_adaptation import make_optimizer


class TinyConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Sequential(nn.Conv2d(3, 768, 1), nn.Dropout(.9)),
                                      nn.Sequential(nn.Conv2d(768, 768, 1), nn.Dropout(.9)))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Flatten(1), nn.Dropout(.9), nn.Linear(768, 1000))

    def forward(self, images):
        return self.classifier(self.avgpool(self.features(images)).flatten(1))


class EncoderScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_encoder_keeps_stochastic_layers_off_but_trains_all_five_method_heads(self):
        for method in METHODS:
            with self.subTest(method=method), patch("vds2raw.adaptation.convnext_small", side_effect=lambda **_: TinyConvNeXt()):
                model = AdaptationModel(method, pretrained=False, use_ais=False, ir_channels=3,
                                        image_backbone="convnext_small", freeze_image_encoders=True,
                                        num_classes=4, feature_normalization="l2", unit_method_features=True)
                model.train()
                self.assertFalse(model.rgb_encoder.training)
                self.assertFalse(model.nir_encoder.classifier[2].training)
                self.assertTrue(model.head.training)
                source = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8), "label": torch.arange(4)}
                target = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8), "label": object()}
                torch.testing.assert_close(model.encode(source).norm(dim=1), torch.ones(4))
                args = parser().parse_args(["--method", method, "--target", "backlight", "--data-dir", "."])
                resolve_weather_settings(args)
                args.steps_per_epoch, args.adaptation_ramp_epochs = 2, 1
                loss, details = model.objective(source, target, 100, 200, args)
                self.assertEqual(details["target_logits"].shape, (4, 4))
                self.assertEqual(tuple(x.shape[1] for x in model.last_features), (768, 768))
                adaptation_parameters = [p for p in model.parameters() if p.requires_grad]
                adaptation_gradients = torch.autograd.grad(details["weighted_transfer"], adaptation_parameters,
                                                           retain_graph=True, allow_unused=True)
                self.assertTrue(any(g is not None and bool(g.norm() > 0) for g in adaptation_gradients))
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.rgb_encoder.parameters()))
                self.assertTrue(any(p.grad is not None and bool(p.grad.norm() > 0) for p in model.head.parameters()))
                groups = optimizer_groups(model, .001)
                registered = [id(p) for group in groups for p in group["params"]]
                self.assertEqual(len(registered), len(set(registered)))
                self.assertEqual(set(registered), {id(p) for p in model.parameters() if p.requires_grad})

    def test_explicit_legacy_encoder_options_preserve_seeded_default_model_state(self):
        torch.manual_seed(42)
        original = AdaptationModel("deepcoral", pretrained=False, use_ais=False, ir_channels=3)
        torch.manual_seed(42)
        explicit = AdaptationModel("deepcoral", pretrained=False, use_ais=False, ir_channels=3,
                                   image_backbone="resnet18", freeze_image_encoders=False,
                                   classifier_layout="reference")
        self.assertEqual(original.state_dict().keys(), explicit.state_dict().keys())
        self.assertTrue(all(torch.equal(value, explicit.state_dict()[name]) for name, value in original.state_dict().items()))

    def test_experimental_weather_encoder_cannot_silently_change_vds_ais_architecture(self):
        with self.assertRaisesRegex(ValueError, "weather VIS/IR only"):
            AdaptationModel("mcc", pretrained=False, image_backbone="convnext_small")

    def test_unit_fusion_and_separate_gate_scale_preserve_trainable_parameters(self):
        with patch("vds2raw.adaptation.convnext_small", side_effect=lambda **_: TinyConvNeXt()):
            model = AdaptationModel("deepcoral", pretrained=False, use_ais=False, ir_channels=3,
                                    image_backbone="convnext_small", freeze_image_encoders=True,
                                    feature_normalization="unit", num_classes=4)
        batch = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8)}
        model.encode(batch)
        torch.testing.assert_close(model.last_fused.norm(dim=1), torch.ones(4))
        for branch in model.last_normalized_features:
            torch.testing.assert_close(branch.norm(dim=1), torch.ones(4))
        groups = optimizer_groups(model, .01, gate_lr_multiplier=.01, gate_weight_decay=0.)
        gate = next(group for group in groups if group["role"] == "modality_gate")
        self.assertEqual((gate["lr"], gate["weight_decay"]), (.001, 0.))
        self.assertEqual(gate["params"], [model.modality_logits])
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        model.eval()
        model.encode(batch)
        full = model.last_fused.clone()
        model.eval_disabled_modalities = (1,)
        model.encode(batch)
        torch.testing.assert_close(model.last_fused[:, :768], full[:, :768])
        self.assertEqual(float(model.last_fused[:, 768:].abs().max()), 0.)

    def test_optimizer_override_keeps_reference_defaults_and_method_lr_ratios(self):
        for method, expected in (("dann", torch.optim.Adam), ("mcc", torch.optim.SGD)):
            with patch("vds2raw.adaptation.convnext_small", side_effect=lambda **_: TinyConvNeXt()):
                model = AdaptationModel(method, pretrained=False, use_ais=False, ir_channels=3,
                                        image_backbone="convnext_small", freeze_image_encoders=True)
            args = parser().parse_args(["--method", method, "--target", "backlight", "--data-dir", "."])
            resolve_weather_settings(args)
            optimizer, _, _ = make_optimizer(model, args)
            self.assertIsInstance(optimizer, expected)
            original_lrs = [group["lr"] for group in optimizer.param_groups]
            args.optimizer = "adam"
            optimizer, _, _ = make_optimizer(model, args)
            self.assertIsInstance(optimizer, torch.optim.Adam)
            self.assertEqual([group["lr"] for group in optimizer.param_groups], original_lrs)

    def test_ir_last_stage_has_gradients_while_vis_and_early_ir_stay_fixed(self):
        with patch("vds2raw.adaptation.convnext_small", side_effect=lambda **_: TinyConvNeXt()):
            model = AdaptationModel("deepcoral", pretrained=False, use_ais=False, ir_channels=3,
                                    image_backbone="convnext_small", encoder_training_scope="ir_last_stage",
                                    feature_normalization="unit", num_classes=4)
        model.train()
        self.assertFalse(model.freeze_image_encoders)
        self.assertFalse(model.rgb_encoder.training)
        self.assertFalse(model.nir_encoder.features[-1].training)
        self.assertTrue(all(not p.requires_grad for p in model.rgb_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.nir_encoder.features[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.nir_encoder.features[-1].parameters()))
        args = parser().parse_args(["--method", "deepcoral", "--target", "dark", "--data-dir", ".", "--image-lr-override", ".0001"])
        resolve_weather_settings(args)
        args.steps_per_epoch, args.adaptation_ramp_epochs = 2, 1
        source = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8), "label": torch.arange(4)}
        target = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8)}
        loss, _ = model.objective(source, target, 10, 40, args)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.rgb_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.norm() > 0) for p in model.nir_encoder.features[-1].parameters()))
        optimizer, _, _ = make_optimizer(model, args)
        self.assertEqual(optimizer.param_groups[0]["lr"], .0001)

    def test_last_stages_updates_both_modalities_without_unfreezing_early_stages(self):
        with patch("vds2raw.adaptation.convnext_small", side_effect=lambda **_: TinyConvNeXt()):
            model = AdaptationModel("deepcoral", pretrained=False, use_ais=False, ir_channels=3,
                                    image_backbone="convnext_small", encoder_training_scope="last_stages",
                                    feature_normalization="unit", num_classes=4)
        model.train()
        batch = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8)}
        torch.nn.functional.cross_entropy(model(batch), torch.arange(4)).backward()
        for encoder in (model.rgb_encoder, model.nir_encoder):
            self.assertFalse(encoder.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.features[0].parameters()))
            self.assertTrue(any(p.grad is not None and bool(p.grad.norm() > 0) for p in encoder.features[-1].parameters()))

    def test_direct_linear_layout_preserves_all_five_adaptation_gradients(self):
        for method in METHODS:
            with self.subTest(method=method), patch("vds2raw.adaptation.convnext_small", side_effect=lambda **_: TinyConvNeXt()):
                model = AdaptationModel(method, pretrained=False, use_ais=False, ir_channels=3,
                                        image_backbone="convnext_small", encoder_training_scope="frozen",
                                        feature_normalization="unit", classifier_layout="linear", num_classes=4)
                self.assertIsInstance(model.bottleneck, nn.Identity)
                self.assertIsInstance(model.head, nn.Linear)
                if method == "mdd":
                    self.assertIsInstance(model.adv_head, nn.Linear)
                source = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8), "label": torch.arange(4)}
                target = {"rgb": torch.randn(4, 3, 8, 8), "nir": torch.randn(4, 3, 8, 8), "label": object()}
                args = parser().parse_args(["--method", method, "--target", "backlight", "--data-dir", "."])
                resolve_weather_settings(args)
                args.steps_per_epoch, args.adaptation_ramp_epochs = 2, 1
                loss, details = model.objective(source, target, 100, 200, args)
                params = [p for p in model.parameters() if p.requires_grad]
                gradients = torch.autograd.grad(details["weighted_transfer"], params, retain_graph=True, allow_unused=True)
                self.assertTrue(any(g is not None and bool(g.norm() > 0) for g in gradients))
                loss.backward()
                self.assertTrue(bool(model.head.weight.grad.norm() > 0))
                self.assertTrue(bool(model.modality_logits.grad.norm() > 0))


if __name__ == "__main__":
    unittest.main()
