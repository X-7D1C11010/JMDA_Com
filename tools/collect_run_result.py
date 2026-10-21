"""Collect one training log into a structured JSON record and master CSV.

This is deliberately independent of individual training implementations: the
server launcher redirects stdout/stderr to a log, then invokes this collector.
The raw log remains the source of truth while the CSV provides a convenient
paper-table index.
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime


TARGET_NAMES = ('backlight', 'rainy', 'foggy', 'dark')


CSV_FIELDNAMES = [
    'timestamp', 'suite_id', 'run_tag', 'method', 'protocol', 'target', 'seed', 'epochs',
    'input_modalities', 'label_space_relation', 'model_selection',
    'status', 'exit_code', 'duration_seconds', 'final_acc', 'best_acc',
    'final_acc_std', 'precision_macro', 'precision_macro_std',
    'recall_macro', 'recall_macro_std', 'f1_macro', 'f1_macro_std',
    'num_iterations', 'seed_values', 'accuracy_values_count', 'log_path',
    'efficiency_dir',
]


def _failure_summary(text, status, max_lines=30, max_chars=6000):
    """Keep a compact traceback tail in JSON when remote logs are not copied."""
    if status != 'failed':
        return None, None
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    tail = '\n'.join(lines[-max_lines:])[-max_chars:]
    matches = re.findall(
        r'(?m)^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt)):\s*(.*)$',
        text,
    )
    if matches:
        error_type, message = matches[-1]
        return error_type, (message.strip() or tail)
    return 'ProcessError', tail or 'Process failed without captured output.'


def _last_float(text, patterns):
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def _infer_method_target(run_tag):
    for target in TARGET_NAMES:
        suffix = '_' + target
        if run_tag.endswith(suffix):
            return run_tag[:-len(suffix)], target
    return run_tag, ''


def _infer_input_modalities(method):
    if method.startswith(('jmda_net', 'mcft')):
        return 'VIS+IR'
    return 'VIS'


def parse_metrics(text):
    epoch_acc_patterns = [
        r'(?m)^(?![^\n]*(?:best|peak|diagnostic|final))[^\n]*target\s+val\s+accuracy[^:\n]*:\s*([0-9]*\.?[0-9]+)',
        r'(?m)^(?![^\n]*(?:best|最佳))[^\n]*Val\s+Acc\s*:\s*([0-9]*\.?[0-9]+)',
        r'reported_acc\s*=\s*([0-9]*\.?[0-9]+)',
    ]
    best_acc_patterns = [
        r'Best\s+target\s+val\s+accuracy[^:\n]*:\s*([0-9]*\.?[0-9]+)',
        r'Diagnostic\s+peak\s+target\s+val\s+accuracy[^:\n]*:\s*([0-9]*\.?[0-9]+)',
        r'best_acc\s*=\s*([0-9]*\.?[0-9]+)',
        r'New\s+Best\s+Val\s+Acc\s*:\s*([0-9]*\.?[0-9]+)',
        r'最佳验证准确率\s*[:：]\s*([0-9]*\.?[0-9]+)',
    ]
    epoch_values = []
    for pattern in epoch_acc_patterns:
        epoch_values.extend(
            float(value) for value in re.findall(pattern, text, flags=re.IGNORECASE)
        )
    explicit_final = _last_float(text, [
        r'Final\s+target\s+val\s+accuracy[^:\n]*:\s*([0-9]*\.?[0-9]+)',
        r'reported_acc\s*=\s*([0-9]*\.?[0-9]+)',
    ])
    final_acc = (
        explicit_final if explicit_final is not None
        else (epoch_values[-1] if epoch_values else None)
    )
    best_acc = _last_float(text, best_acc_patterns)
    if epoch_values:
        best_acc = max(max(epoch_values), best_acc) if best_acc is not None else max(epoch_values)

    metrics = {
        'final_acc': final_acc,
        'best_acc': best_acc,
        'final_acc_std': None,
        'accuracy_values_count': len(epoch_values),
        'precision_macro': _last_float(text, [
            r'Val\s+Precision\s*\(Macro\)\s*:\s*([0-9]*\.?[0-9]+)',
            r'Precision\s*\(Macro\)\s*:\s*([0-9]*\.?[0-9]+)',
        ]),
        'recall_macro': _last_float(text, [
            r'Val\s+Recall\s*\(Macro\)\s*:\s*([0-9]*\.?[0-9]+)',
            r'Recall\s*\(Macro\)\s*:\s*([0-9]*\.?[0-9]+)',
        ]),
        'f1_macro': _last_float(text, [
            r'Val\s+F1\s*\(Macro\)\s*:\s*([0-9]*\.?[0-9]+)',
            r'F1\s*\(Macro\)\s*:\s*([0-9]*\.?[0-9]+)',
        ]),
        'precision_macro_std': None,
        'recall_macro_std': None,
        'f1_macro_std': None,
        'num_iterations': 1,
        'seed_values': '',
    }
    summaries = re.findall(
        r'(?m)^JMDA_RUN_SUMMARY_JSON:\s*(\{.*\})\s*$', text
    )
    if summaries:
        try:
            summary = json.loads(summaries[-1])
            summary_metrics = summary.get('metrics', {})

            def apply_summary(name, output_name, std_name):
                values = summary_metrics.get(name, {})
                if values.get('mean') is not None:
                    metrics[output_name] = float(values['mean'])
                if values.get('std') is not None:
                    metrics[std_name] = float(values['std'])

            apply_summary('accuracy', 'final_acc', 'final_acc_std')
            accuracy_values = summary_metrics.get('accuracy', {}).get('values')
            if isinstance(accuracy_values, list):
                # Completed JMDA logs contain every per-epoch Val Acc plus one
                # reported_acc per seed.  The result-table count must describe
                # independent repeated runs, not those 100x duplicated traces.
                metrics['accuracy_values_count'] = len(accuracy_values)
            apply_summary(
                'precision_macro', 'precision_macro', 'precision_macro_std'
            )
            apply_summary('recall_macro', 'recall_macro', 'recall_macro_std')
            apply_summary('f1_macro', 'f1_macro', 'f1_macro_std')
            seeds = [int(seed) for seed in summary.get('seeds', [])]
            metrics['num_iterations'] = int(
                summary.get('num_iterations', len(seeds) or 1)
            )
            metrics['seed_values'] = ' '.join(str(seed) for seed in seeds)
        except (TypeError, ValueError, json.JSONDecodeError):
            # The raw log remains authoritative. A malformed optional summary
            # must not hide otherwise parseable epoch metrics.
            pass
    return metrics


def apply_protocol_metric_policy(metrics, protocol):
    """Remove target-label-selected fields from UDA-family result records."""
    if protocol.lower() in ('uda', 'oracle_shared_uda'):
        metrics['best_acc'] = None
    return metrics


def append_csv_record(csv_path, record):
    """Append a record, migrating older result tables to the current schema."""
    existing_rows = []
    existing_fields = None
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames
            existing_rows = list(reader)
    if existing_fields != CSV_FIELDNAMES:
        temporary = csv_path + '.tmp'
        with open(temporary, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=CSV_FIELDNAMES, extrasaction='ignore'
            )
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({name: row.get(name, '') for name in CSV_FIELDNAMES})
        os.replace(temporary, csv_path)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDNAMES, extrasaction='ignore'
        )
        if write_header:
            writer.writeheader()
        writer.writerow({name: record.get(name) for name in CSV_FIELDNAMES})


def main():
    parser = argparse.ArgumentParser(description='Collect a training run result')
    parser.add_argument('--log', required=True)
    parser.add_argument('--suite-id', default='')
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--status', required=True, choices=['success', 'failed'])
    parser.add_argument('--exit-code', type=int, required=True)
    parser.add_argument('--duration-seconds', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--efficiency-dir', default='')
    parser.add_argument('--input-modalities', default='')
    parser.add_argument('--label-space-relation', default='')
    parser.add_argument('--model-selection', default='')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    with open(args.log, 'r', encoding='utf-8', errors='replace') as handle:
        text = handle.read()

    method, target = _infer_method_target(args.run_tag)
    metrics = apply_protocol_metric_policy(parse_metrics(text), args.protocol)
    error_type, error_summary = _failure_summary(text, args.status)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    record = {
        'timestamp': timestamp,
        'suite_id': args.suite_id,
        'run_tag': args.run_tag,
        'method': method,
        'protocol': args.protocol.upper(),
        'target': target,
        'seed': args.seed,
        'epochs': args.epochs,
        'input_modalities': (
            args.input_modalities or _infer_input_modalities(method)
        ),
        'label_space_relation': args.label_space_relation,
        'model_selection': (
            args.model_selection
            or ('final_checkpoint' if args.protocol.lower() in (
                    'uda', 'oracle_shared_uda', 'vlm'
                )
                else 'method_declared')
        ),
        'status': args.status,
        'exit_code': args.exit_code,
        'duration_seconds': round(args.duration_seconds, 3),
        'final_acc': metrics['final_acc'],
        'best_acc': metrics['best_acc'],
        'final_acc_std': metrics['final_acc_std'],
        'precision_macro': metrics['precision_macro'],
        'precision_macro_std': metrics['precision_macro_std'],
        'recall_macro': metrics['recall_macro'],
        'recall_macro_std': metrics['recall_macro_std'],
        'f1_macro': metrics['f1_macro'],
        'f1_macro_std': metrics['f1_macro_std'],
        'num_iterations': metrics['num_iterations'],
        'seed_values': metrics['seed_values'],
        'accuracy_values_count': metrics['accuracy_values_count'],
        'log_path': os.path.abspath(args.log),
        'efficiency_dir': os.path.abspath(args.efficiency_dir) if args.efficiency_dir else '',
        'error_type': error_type,
        'error_summary': error_summary,
    }

    protocol_dir = os.path.join(args.results_dir, args.protocol.lower())
    os.makedirs(protocol_dir, exist_ok=True)
    json_name = f"{args.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(protocol_dir, json_name), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.results_dir, 'runs.csv')
    append_csv_record(csv_path, record)

    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
