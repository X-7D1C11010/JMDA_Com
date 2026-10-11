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


def parse_metrics(text):
    epoch_acc_patterns = [
        r'(?m)^(?![^\n]*best)[^\n]*target\s+val\s+accuracy[^:\n]*:\s*([0-9]*\.?[0-9]+)',
        r'(?m)^(?![^\n]*(?:best|最佳))[^\n]*Val\s+Acc\s*:\s*([0-9]*\.?[0-9]+)',
        r'reported_acc\s*=\s*([0-9]*\.?[0-9]+)',
    ]
    best_acc_patterns = [
        r'Best\s+target\s+val\s+accuracy[^:\n]*:\s*([0-9]*\.?[0-9]+)',
        r'best_acc\s*=\s*([0-9]*\.?[0-9]+)',
        r'New\s+Best\s+Val\s+Acc\s*:\s*([0-9]*\.?[0-9]+)',
        r'最佳验证准确率\s*[:：]\s*([0-9]*\.?[0-9]+)',
    ]
    epoch_values = []
    for pattern in epoch_acc_patterns:
        epoch_values.extend(
            float(value) for value in re.findall(pattern, text, flags=re.IGNORECASE)
        )
    final_acc = epoch_values[-1] if epoch_values else None
    best_acc = _last_float(text, best_acc_patterns)
    if epoch_values:
        best_acc = max(max(epoch_values), best_acc) if best_acc is not None else max(epoch_values)

    return {
        'final_acc': final_acc,
        'best_acc': best_acc,
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
    }


def main():
    parser = argparse.ArgumentParser(description='Collect a training run result')
    parser.add_argument('--log', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--status', required=True, choices=['success', 'failed'])
    parser.add_argument('--exit-code', type=int, required=True)
    parser.add_argument('--duration-seconds', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--efficiency-dir', default='')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    with open(args.log, 'r', encoding='utf-8', errors='replace') as handle:
        text = handle.read()

    method, target = _infer_method_target(args.run_tag)
    metrics = parse_metrics(text)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    record = {
        'timestamp': timestamp,
        'run_tag': args.run_tag,
        'method': method,
        'protocol': args.protocol.upper(),
        'target': target,
        'seed': args.seed,
        'epochs': args.epochs,
        'status': args.status,
        'exit_code': args.exit_code,
        'duration_seconds': round(args.duration_seconds, 3),
        'final_acc': metrics['final_acc'],
        'best_acc': metrics['best_acc'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
        'accuracy_values_count': metrics['accuracy_values_count'],
        'log_path': os.path.abspath(args.log),
        'efficiency_dir': os.path.abspath(args.efficiency_dir) if args.efficiency_dir else '',
    }

    protocol_dir = os.path.join(args.results_dir, args.protocol.lower())
    os.makedirs(protocol_dir, exist_ok=True)
    json_name = f"{args.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(protocol_dir, json_name), 'w', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.results_dir, 'runs.csv')
    fieldnames = list(record.keys())
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
