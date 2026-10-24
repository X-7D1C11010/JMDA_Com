import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    from tools.domain_pairing import CrossDomainPairDataset
    from tools.metrics_io import save_metric_history
except ImportError:  # Direct execution from the tools directory.
    from domain_pairing import CrossDomainPairDataset
    from metrics_io import save_metric_history


class _ToyDataset:
    def __init__(self, labels):
        self.samples = [(f'image-{index}', label) for index, label in enumerate(labels)]

    def __getitem__(self, index):
        return self.samples[index]


class OfficialMethodSupportTests(unittest.TestCase):
    def test_pair_table_keeps_all_positives_and_requested_negatives(self):
        source = _ToyDataset([0, 0, 1])
        target = _ToyDataset([0, 1])
        pairs = CrossDomainPairDataset(source, target, negative_ratio=1, seed=7)
        self.assertEqual(pairs.positive_pairs, 3)
        self.assertEqual(pairs.negative_pairs, 3)
        self.assertEqual(len(pairs), 6)

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
                set(('ACC', 'Precision_macro', 'Recall_macro', 'F1_macro')),
                set(row).intersection(('ACC', 'Precision_macro', 'Recall_macro', 'F1_macro')),
            )
            payload = json.loads(Path(json_path).read_text(encoding='utf-8'))
            self.assertEqual(payload['final']['ACC'], 0.75)
            self.assertEqual(payload['best_by_ACC']['F1_macro'], 0.67)


if __name__ == '__main__':
    unittest.main()
