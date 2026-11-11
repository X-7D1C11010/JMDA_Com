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
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from historical_uda_extensions import (  # noqa: E402
    ContinuousShuffleSampler, EMATargetTeacher, TargetBNProbe, pretrain_ir_encoder,
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

    def test_target_bn_uses_all_training_images_and_restores_model_and_rng(self):
        class LabelsMustBeIgnored:
            def __int__(self):
                raise AssertionError("Target training labels were consumed")

        class TargetImages:
            def __init__(self):
                self.seen = []

            def __len__(self):
                return 5

            def __getitem__(self, index):
                self.seen.append(index)
                return torch.full((6, 4, 4), float(index + 2)), LabelsMustBeIgnored()

        model = nn.Sequential(
            nn.Conv2d(6, 2, 1), nn.BatchNorm2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Linear(2, 2),
        ).train()
        dataset = TargetImages()
        probe = TargetBNProbe(
            model, dataset,
            transforms.Compose([transforms.Normalize([0.5] * 3, [0.5] * 3)]),
            lambda network, images: network(images), torch.device("cpu"), 2,
        )
        state = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.random.get_rng_state()
        metrics = {"accuracy": 0.5, "precision_macro": 0.5,
                   "recall_macro": 0.5, "f1_macro": 0.5}

        def evaluate():
            self.assertFalse(model.training)
            self.assertGreater(model[1].running_mean.abs().sum().item(), 0)
            return metrics

        with redirect_stdout(io.StringIO()):
            result = probe.evaluate(metrics, evaluate, 1)
        self.assertEqual(result, metrics)
        self.assertEqual(dataset.seen, list(range(5)))
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(probe.summary()["bn_fit_pairs"], 5)

        def fail_validation():
            raise RuntimeError("synthetic validation failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic validation failure"):
            probe.evaluate(metrics, fail_validation, 2)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
