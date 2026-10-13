"""Summarize final-checkpoint results without using target-best epochs."""

import argparse
import csv
import os
import statistics
from collections import defaultdict
from datetime import datetime


def _number(value):
    if value is None or str(value).strip() == '':
        return None
    return float(value)


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get('method', ''), row.get('target', ''),
            row.get('input_modalities', ''),
            row.get('label_space_relation', ''),
            row.get('model_selection', ''),
        )
        grouped[key].append(row)

    summaries = []
    metric_fields = (
        ('final_acc', 'final_acc_std', 'final_acc'),
        ('precision_macro', 'precision_macro_std', 'precision_macro'),
        ('recall_macro', 'recall_macro_std', 'recall_macro'),
        ('f1_macro', 'f1_macro_std', 'f1_macro'),
    )
    for key, group in sorted(grouped.items()):
        aggregated = [
            row for row in group if int(row.get('num_iterations') or 1) > 1
        ]
        raw = [row for row in group if int(row.get('num_iterations') or 1) == 1]
        if aggregated and raw:
            raise ValueError(
                f'Mixed aggregated and per-seed records for {key}; '
                'use one representation in a formal suite.'
            )
        if len(aggregated) > 1:
            raise ValueError(
                f'Multiple aggregated records for {key}; choose one suite/run.'
            )
        if aggregated:
            row = aggregated[0]
            repeats = int(row.get('num_iterations') or 1)
            seeds = row.get('seed_values', '')
        else:
            if not any(_number(row.get('final_acc')) is not None for row in raw):
                continue
            repeats = len(raw)
            seeds = ' '.join(row.get('seed', '') for row in raw)
        summary = {
            'method': key[0],
            'target': key[1],
            'input_modalities': key[2],
            'label_space_relation': key[3],
            'model_selection': key[4],
            'num_repeats': repeats,
            'seeds': seeds,
        }
        for value_field, std_field, output_prefix in metric_fields:
            if aggregated:
                mean = _number(row.get(value_field))
                std = _number(row.get(std_field))
            else:
                values = [
                    value for item in raw
                    if (value := _number(item.get(value_field))) is not None
                ]
                mean = statistics.fmean(values) if values else None
                std = (
                    statistics.pstdev(values) if len(values) > 1
                    else (0.0 if values else None)
                )
            summary[f'{output_prefix}_mean'] = mean
            summary[f'{output_prefix}_std'] = std
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-csv', required=True)
    parser.add_argument('--suite-id', required=True)
    parser.add_argument('--protocol', default='UDA')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    with open(args.runs_csv, 'r', encoding='utf-8-sig', newline='') as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get('suite_id') == args.suite_id
            and row.get('protocol', '').upper() == args.protocol.upper()
            and row.get('status') == 'success'
            and row.get('model_selection') == 'final_checkpoint'
        ]
    if not rows:
        raise RuntimeError(
            f'No successful final-checkpoint records for suite {args.suite_id!r}.'
        )

    summaries = summarize_rows(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = f'{args.suite_id}_{args.protocol.lower()}_final'
    csv_path = os.path.join(args.output_dir, stem + '.csv')
    fields = [
        'method', 'target', 'input_modalities', 'label_space_relation',
        'model_selection',
        'final_acc_mean', 'final_acc_std',
        'precision_macro_mean', 'precision_macro_std',
        'recall_macro_mean', 'recall_macro_std',
        'f1_macro_mean', 'f1_macro_std',
        'num_repeats', 'seeds',
    ]
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    markdown_path = os.path.join(args.output_dir, stem + '.md')
    with open(markdown_path, 'w', encoding='utf-8') as handle:
        handle.write(
            f'# {args.protocol.upper()} final-checkpoint summary\n\n'
            f'- Suite: `{args.suite_id}`\n'
            f'- Generated: {datetime.now().isoformat(timespec="seconds")}\n'
            '- Model selection: final checkpoint only; target-domain peaks are excluded.\n'
            '- Input modalities are separated below and must not be merged into one fair main table.\n\n'
        )
        modalities = sorted({row['input_modalities'] for row in summaries})
        for modality in modalities:
            handle.write(
                f"## {modality or 'unspecified modalities'}\n\n"
                '| Method | Target | Label space | Final accuracy (mean ± std) | Macro-F1 (mean ± std) | Repeats |\n'
                '|---|---|---|---:|---:|---:|\n'
            )
            for row in summaries:
                if row['input_modalities'] != modality:
                    continue
                handle.write(
                    f"| {row['method']} | {row['target']} | "
                    f"{row['label_space_relation']} | "
                    f"{row['final_acc_mean']:.4f} ± "
                    f"{row['final_acc_std']:.4f} | "
                    f"{row['f1_macro_mean']:.4f} ± "
                    f"{row['f1_macro_std']:.4f} | "
                    f"{row['num_repeats']} |\n"
                )
            handle.write('\n')
    print(csv_path)
    print(markdown_path)


if __name__ == '__main__':
    main()
