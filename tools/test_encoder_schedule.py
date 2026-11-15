import unittest
import torch
from torch import nn
from weather.encoder_schedule import apply_encoder_warmup, capture_parameter_sample, parameter_update_record


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.nir_encoder = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(16, 2)
        self.modality_logits = nn.Parameter(torch.ones(2))

    def forward(self, x):
        a, b = self.rgb_encoder(x), self.nir_encoder(x)
        return self.head(torch.cat((a*self.modality_logits[0], b*self.modality_logits[1]), 1))


class EncoderScheduleTests(unittest.TestCase):
    def test_warmup_skips_weight_decay_and_momentum_but_classifier_learns_then_encoder_updates(self):
        torch.manual_seed(42); model = Tiny()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.1)
        x, labels = torch.randn(4, 3, 8, 8), torch.tensor([0, 0, 1, 1])
        policy = apply_encoder_warmup(model, 1, 1)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), labels).backward()
        before = capture_parameter_sample(model); optimizer.step(); record = parameter_update_record(before, optimizer)
        self.assertTrue(policy["warmup_frozen"])
        for name in ("VIS_first_conv", "IR_first_conv"):
            self.assertEqual(record[name]["update_L2"], 0.)
            self.assertFalse(record[name]["gradient_present"])
            self.assertFalse(record[name]["momentum_buffer_present"])
        self.assertGreater(record["classifier_last_weight"]["update_L2"], 0.)
        self.assertFalse(apply_encoder_warmup(model, 2, 1)["warmup_frozen"])
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), labels).backward()
        before = capture_parameter_sample(model); optimizer.step(); record = parameter_update_record(before, optimizer)
        self.assertGreater(record["IR_first_conv"]["update_L2"], 0.)
        self.assertTrue(record["IR_first_conv"]["momentum_buffer_present"])

    def test_disabled_policy_preserves_existing_partial_freeze_and_rng(self):
        model = Tiny(); model.rgb_encoder.requires_grad_(False)
        flags = [p.requires_grad for p in model.parameters()]; state = torch.get_rng_state()
        apply_encoder_warmup(model, 1, 0)
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        apply_encoder_warmup(model, 1, 1); apply_encoder_warmup(model, 2, 1)
        self.assertTrue(all(not p.requires_grad for p in model.rgb_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.nir_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
