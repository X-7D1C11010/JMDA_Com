import csv
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

try:
    from tools.domain_pairing import (
        ControlledCrossDomainPairDataset,
        CrossDomainPairDataset,
    )
    from tools.metrics_io import save_metric_history
except ImportError:  # Direct execution from the tools directory.
    from domain_pairing import ControlledCrossDomainPairDataset, CrossDomainPairDataset
    from metrics_io import save_metric_history


class _ToyDataset:
    def __init__(self, labels):
        self.samples = [(f'image-{index}', label) for index, label in enumerate(labels)]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class OfficialMethodSupportTests(unittest.TestCase):
    def test_pair_table_keeps_all_positives_and_requested_negatives(self):
        source = _ToyDataset([0, 0, 1])
        target = _ToyDataset([0, 1])
        pairs = CrossDomainPairDataset(source, target, negative_ratio=1, seed=7)
        self.assertEqual(pairs.positive_pairs, 3)
        self.assertEqual(pairs.negative_pairs, 3)
        self.assertEqual(len(pairs), 6)

    def test_ccsa_pair_table_matches_released_reference_procedure(self):
        source = _ToyDataset([0, 0, 1, 2])
        target = _ToyDataset([0, 1, 2])
        seed = 19
        pairs = CrossDomainPairDataset(source, target, negative_ratio=3, seed=seed)

        positives, negatives = [], []
        for source_index, (_, source_label) in enumerate(source.samples):
            for target_index, (_, target_label) in enumerate(target.samples):
                item = (
                    source_index,
                    target_index,
                    int(source_label == target_label),
                )
                (positives if item[2] else negatives).append(item)
        rng = random.Random(seed)
        rng.shuffle(negatives)
        expected = positives + negatives[: 3 * len(positives)]
        rng.shuffle(expected)

        self.assertEqual(pairs.positive_pairs, 4)
        # With three classes only eight unique negatives exist. The released
        # slice keeps all eight; it does not repeat negatives to force 3:1.
        self.assertEqual(pairs.negative_pairs, 8)
        self.assertEqual(pairs.pairs, expected)

    def test_ccsa_loss_matches_released_formula(self):
        from vds2raw.models import ccsa_contrastive_loss

        pair_labels = torch.tensor([1.0, 0.0])
        distances = torch.tensor([[0.5], [0.25]])
        released = torch.mean(
            pair_labels.view(-1, 1) * distances.square()
            + (1.0 - pair_labels.view(-1, 1))
            * torch.clamp(1.0 - distances, min=0.0).square()
        )
        actual = ccsa_contrastive_loss(pair_labels, distances, margin=1.0)
        torch.testing.assert_close(actual, released)

    def test_ccsa_microbatch_update_matches_official_effective_batch(self):
        from vds2raw.models import ccsa_contrastive_loss, euclidean_distance
        from vds2raw.train_ccsa import train_direction

        class TinyCCSA(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(2, 2, bias=False)
                self.classifier = nn.Linear(2, 2, bias=False)

            def forward(self, first, second=None):
                first_features = self.embedding(first)
                first_logits = self.classifier(first_features)
                if second is None:
                    return first_logits
                second_features = self.embedding(second)
                second_logits = self.classifier(second_features)
                return (
                    first_logits,
                    second_logits,
                    first_features,
                    second_features,
                )

        torch.manual_seed(7)
        actual_model = TinyCCSA()
        reference_model = TinyCCSA()
        reference_model.load_state_dict(actual_model.state_dict())
        first = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.2]])
        second = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.1], [1.0, 0.0]])
        first_labels = torch.tensor([0, 1, 0, 1])
        second_labels = torch.tensor([0, 1, 1, 0])
        pair_labels = torch.tensor([1, 1, 0, 0])
        criterion = nn.CrossEntropyLoss()
        args = SimpleNamespace(micro_batch_size=2, alpha=0.25, margin=1.0)
        actual_optimizer = torch.optim.SGD(actual_model.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)

        actual_ce, actual_csa = train_direction(
            actual_model,
            actual_optimizer,
            criterion,
            first,
            first_labels,
            second,
            pair_labels,
            args,
        )
        actual_target_ce, actual_target_csa = train_direction(
            actual_model,
            actual_optimizer,
            criterion,
            second,
            second_labels,
            first,
            pair_labels,
            args,
        )

        logits, _, first_features, second_features = reference_model(first, second)
        reference_ce = criterion(logits, first_labels)
        reference_csa = ccsa_contrastive_loss(
            pair_labels,
            euclidean_distance(first_features, second_features),
            margin=1.0,
        )
        reference_loss = 0.75 * reference_ce + 0.25 * reference_csa
        reference_optimizer.zero_grad()
        reference_loss.backward()
        reference_optimizer.step()

        target_logits, _, target_features, source_features = reference_model(
            second, first
        )
        reference_target_ce = criterion(target_logits, second_labels)
        reference_target_csa = ccsa_contrastive_loss(
            pair_labels,
            euclidean_distance(target_features, source_features),
            margin=1.0,
        )
        reference_target_loss = (
            0.75 * reference_target_ce + 0.25 * reference_target_csa
        )
        reference_optimizer.zero_grad()
        reference_target_loss.backward()
        reference_optimizer.step()

        self.assertAlmostEqual(actual_ce, reference_ce.item(), places=6)
        self.assertAlmostEqual(actual_csa, reference_csa.item(), places=6)
        self.assertAlmostEqual(
            actual_target_ce, reference_target_ce.item(), places=6
        )
        self.assertAlmostEqual(
            actual_target_csa, reference_target_csa.item(), places=6
        )
        for actual, reference in zip(
            actual_model.parameters(), reference_model.parameters()
        ):
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)

    def test_controlled_pairs_are_fixed_balanced_and_epoch_resampled(self):
        source = _ToyDataset([0, 0, 1, 1, 2, 2])
        target = _ToyDataset([0, 0, 1, 1, 2, 2])
        pairs = ControlledCrossDomainPairDataset(
            source,
            target,
            pairs_per_epoch=24,
            negative_ratio=3,
            seed=7,
        )
        epoch_zero = list(pairs.pairs)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(pairs.positive_pairs, 6)
        self.assertEqual(pairs.negative_pairs, 18)
        target_use = {index: 0 for index in range(len(target))}
        for _, target_index, _ in pairs.pairs:
            target_use[target_index] += 1
        self.assertEqual(set(target_use.values()), {4})
        pairs.set_epoch(1)
        self.assertNotEqual(epoch_zero, pairs.pairs)

    def test_metric_files_contain_requested_fields(self):
        metrics = {
            'accuracy': 0.75,
            'precision_macro': 0.70,
            'recall_macro': 0.65,
            'f1_macro': 0.67,
            'val_samples': 8,
            'val_present_classes': 2,
            'val_total_classes': 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = save_metric_history(
                directory, 'method_target', [(1, metrics)], {'target': 'target'}
            )
            with open(csv_path, encoding='utf-8-sig', newline='') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                {'ACC', 'Precision_macro', 'Recall_macro', 'F1_macro'},
                set(row).intersection(('ACC', 'Precision_macro', 'Recall_macro', 'F1_macro')),
            )
            payload = json.loads(Path(json_path).read_text(encoding='utf-8'))
            self.assertEqual(payload['final']['ACC'], 0.75)
            self.assertEqual(payload['best_by_ACC']['F1_macro'], 0.67)

    def test_dsne_backbone_loading_is_all_or_nothing(self):
        script = Path(__file__).resolve().parents[1] / 'd-SNE' / 'train_dsne.py'
        spec = importlib.util.spec_from_file_location('tested_dsne_train', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.DSNE(num_classes=2, nlayers=18)
        model.features = nn.Sequential(nn.Linear(2, 2))
        exact = {
            f'features.{key}': value.detach().clone()
            for key, value in model.features.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            exact_path = Path(directory) / 'exact.pth'
            partial_path = Path(directory) / 'partial.pth'
            torch.save({'state_dict': exact}, exact_path)
            torch.save({'state_dict': {'features.0.weight': exact['features.0.weight']}}, partial_path)
            model.load_backbone_checkpoint(exact_path)
            with self.assertRaisesRegex(RuntimeError, 'not structurally compatible'):
                model.load_backbone_checkpoint(partial_path)

    def test_jmda_dsne_defaults_to_complete_target_training_list(self):
        script = Path(__file__).resolve().parents[1] / 'd-SNE' / 'train_dsne.py'
        spec = importlib.util.spec_from_file_location('tested_dsne_target_mode', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.DEFAULT_TARGET_LABELS_PER_CLASS, 0)
        self.assertEqual(
            module.target_training_list_path('lists', 'dark', 0, 42),
            str(Path('lists') / 'dark_train.txt'),
        )
        self.assertEqual(
            module.target_training_list_path('lists', 'dark', 10, 42),
            str(Path('lists') / 'dark_sda_10shot_seed42.txt'),
        )
        with self.assertRaisesRegex(ValueError, 'cannot be negative'):
            module.target_training_list_path('lists', 'dark', -1, 42)


if __name__ == '__main__':
    unittest.main()
