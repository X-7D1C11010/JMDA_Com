import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == '__main__':
    unittest.main()
