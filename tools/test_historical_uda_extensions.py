"""Regression checks for the opt-in historical UDA extensions."""

import argparse
from contextlib import redirect_stdout
import io
from itertools import islice
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from historical_uda_extensions import (  # noqa: E402
    ContinuousShuffleSampler, EMATargetTeacher, pretrain_ir_encoder,
)


class HistoricalUDAExtensionsTests(unittest.TestCase):
    def test_source_sampler_carries_remainder_across_full_batches(self):
        sampler = ContinuousShuffleSampler(5, 42)
        loader = DataLoader(torch.arange(5), batch_size=3, sampler=sampler)
        samples = torch.cat(list(islice(loader, 4))).tolist()
        self.assertEqual(sorted(samples[:5]), list(range(5)))
        self.assertEqual(sorted(samples[5:10]), list(range(5)))
        self.assertEqual(samples, list(islice(iter(sampler), 12)))

    def test_ema_uses_second_view_and_masks_low_confidence(self):
        student = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            student.weight.copy_(torch.tensor([[3., -3.], [-3., 3.]]))
        with redirect_stdout(io.StringIO()):
            teacher = EMATargetTeacher(student, lambda model, x: model(x))
        examples = torch.tensor([[1., 0.], [0.5, 0.5]])
        logits = student(examples).detach().requires_grad_()
        loss = teacher.loss(logits, examples)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertEqual(teacher.accepted.item(), 1)
        self.assertGreater(float(logits.grad[0].abs().sum()), 0)
        torch.testing.assert_close(logits.grad[1], torch.zeros(2))
        self.assertIsNone(teacher.model.weight.grad)
        original = teacher.model.weight.clone()
        with torch.no_grad():
            student.weight.add_(1)
        teacher.update(student)
        torch.testing.assert_close(teacher.model.weight, original + 0.01)

    def test_ir_initialization_changes_encoder_and_restores_torch_rng(self):
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(3, 2))
        encoder.out_features = 2
        pairs = torch.arange(24, dtype=torch.float32).reshape(4, 6, 1, 1) / 24
        dataset = TensorDataset(pairs, torch.tensor([0, 1, 0, 1]))
        args = argparse.Namespace(batch_size=2, seed=42, num_workers=0,
                                  prefetch_factor=4, worker_timeout_seconds=300,
                                  pin_memory=False, persistent_workers=False)
        original = encoder[1].weight.detach().clone()
        rng = torch.random.get_rng_state()
        with redirect_stdout(io.StringIO()):
            pretrain_ir_encoder(encoder, dataset, 2, args, torch.device("cpu"))
        self.assertFalse(torch.equal(original, encoder[1].weight))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
