import tempfile
import unittest
from pathlib import Path

from class_names import load_class_names
from classification_metrics import classification_metrics
from collect_run_result import parse_metrics
from summarize_run_results import summarize_rows
from peak_timing import diagnose_peak_timing


class ResultProtocolTests(unittest.TestCase):
    def test_peak_timing_flags_unrecovered_early_peak(self):
        diagnostic = diagnose_peak_timing(
            [(1, 0.6), (2, 0.7), (3, 0.65), (10, 0.6)],
            total_budget=10,
        )
        self.assertEqual(diagnostic['status'], 'early_peak_not_recovered')
        self.assertEqual(diagnostic['peak_steps'], [2])

    def test_peak_timing_accepts_late_recovery_of_early_peak(self):
        diagnostic = diagnose_peak_timing(
            [(1, 0.7), (5, 0.6), (7, 0.7), (10, 0.65)],
            total_budget=10,
        )
        self.assertEqual(diagnostic['status'], 'late_peak_or_recovered')
        self.assertEqual(diagnostic['peak_steps'], [1, 7])

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

    def test_calibrated_metrics_preserve_the_paired_raw_result(self):
        metrics = parse_metrics(
            'Best target val accuracy (backlight, selected): 0.8056 at epoch 8\n'
            'Final target val accuracy (backlight, selected epoch 8): 0.8056\n'
            'UDA_COMPLEXITY: {"prediction_protocol": "target_train_BN_calibrated", '
            '"raw_best_acc": 0.75, "raw_best_position": 6, '
            '"bn_fit_partition": "target_train", "bn_fit_pairs": 113}\n'
        )
        self.assertEqual(metrics['final_acc'], 0.8056)
        self.assertEqual(metrics['raw_best_acc'], 0.75)
        self.assertEqual(metrics['bn_fit_pairs'], 113)
        self.assertEqual(metrics['prediction_protocol'], 'target_train_BN_calibrated')

    def test_selected_weather_weights_are_collected_with_accuracy(self):
        metrics = parse_metrics(
            'Final target val accuracy (dark, selected epoch 8): 0.75\n'
            'UDA_COMPLEXITY: {"fusion_weather": "dark", "fusion_vis_weight": 0.35, '
            '"fusion_ir_weight": 0.65, "fusion_weights_learnable": true, '
            '"source_selected_accuracy": 0.95}\n'
        )
        self.assertEqual(metrics['final_acc'], 0.75)
        self.assertEqual(metrics['fusion_weather'], 'dark')
        self.assertEqual(metrics['fusion_ir_weight'], 0.65)
        self.assertTrue(metrics['fusion_weights_learnable'])

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

    def test_class_names_accept_original_one_based_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'classes.txt'
            expected = [f'vessel type alpha {index}' for index in range(1, 15)]
            path.write_text(
                '\n'.join(
                    f'{index} {name}'
                    for index, name in enumerate(expected, start=1)
                ),
                encoding='utf-8',
            )
            self.assertEqual(load_class_names(path), expected)

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
