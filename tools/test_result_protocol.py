import tempfile
import unittest
from pathlib import Path

from class_names import load_class_names
from classification_metrics import classification_metrics
from collect_run_result import parse_metrics
from summarize_run_results import summarize_rows


class ResultProtocolTests(unittest.TestCase):
    def test_macro_metrics_use_present_target_classes(self):
        metrics = classification_metrics(
            labels=[0, 0, 1, 1], predictions=[0, 1, 1, 1], num_classes=4
        )
        self.assertEqual(metrics['accuracy'], 0.75)
        self.assertEqual(metrics['val_present_classes'], 2)
        self.assertEqual(metrics['val_class_coverage'], 0.5)
        self.assertAlmostEqual(metrics['recall_macro'], 0.75)

    def test_explicit_final_is_not_replaced_by_diagnostic_peak(self):
        metrics = parse_metrics(
            "Epoch [1/2] target val accuracy: 0.20\n"
            "Epoch [2/2] target val accuracy: 0.30\n"
            "Final target val accuracy (backlight): 0.30\n"
            "Diagnostic peak target val accuracy (not used for model selection): 0.80\n"
        )
        self.assertEqual(metrics['final_acc'], 0.30)
        self.assertEqual(metrics['best_acc'], 0.80)
        self.assertEqual(metrics['accuracy_values_count'], 2)

    def test_jmda_summary_uses_seed_mean_and_std(self):
        text = (
            'Val Acc: 0.1\nVal Acc: 0.9\n'
            'JMDA_RUN_SUMMARY_JSON: '
            '{"num_iterations": 2, "seeds": [42, 43], "metrics": '
            '{"accuracy": {"values": [0.2, 0.4], "mean": 0.3, "std": 0.1}}}\n'
        )
        metrics = parse_metrics(text)
        self.assertEqual(metrics['final_acc'], 0.3)
        self.assertEqual(metrics['final_acc_std'], 0.1)
        self.assertEqual(metrics['num_iterations'], 2)
        self.assertEqual(metrics['seed_values'], '42 43')

    def test_class_names_reject_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'classes.txt'
            path.write_text(
                '\n'.join(f'class_{index}' for index in range(1, 15)),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'placeholder'):
                load_class_names(path)

    def test_class_names_require_english_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'classes.txt'
            path.write_text(
                '\n'.join(['船舶'] + [f'vessel type {index}' for index in range(2, 15)]),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'English'):
                load_class_names(path)

    def test_per_seed_final_results_are_aggregated(self):
        rows = [
            {
                'method': 'dann', 'target': 'backlight',
                'input_modalities': 'VIS',
                'label_space_relation': 'partial-set',
                'model_selection': 'final_checkpoint',
                'final_acc': value, 'final_acc_std': '',
                'num_iterations': '1', 'seed': seed, 'seed_values': '',
            }
            for seed, value in [('42', '0.2'), ('43', '0.4')]
        ]
        summary = summarize_rows(rows)[0]
        self.assertAlmostEqual(summary['final_acc_mean'], 0.3)
        self.assertAlmostEqual(summary['final_acc_std'], 0.1)
        self.assertEqual(summary['num_repeats'], 2)


if __name__ == '__main__':
    unittest.main()
