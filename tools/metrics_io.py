"""Persist comparable classification metrics for experiment entry points."""

import csv
import json
import os


CSV_FIELDS = (
    'epoch', 'ACC', 'Precision_macro', 'Recall_macro', 'F1_macro',
    'val_samples', 'val_present_classes', 'val_total_classes',
)


def _metric_row(epoch, metrics):
    return {
        'epoch': int(epoch),
        'ACC': float(metrics['accuracy']),
        'Precision_macro': float(metrics['precision_macro']),
        'Recall_macro': float(metrics['recall_macro']),
        'F1_macro': float(metrics['f1_macro']),
        'val_samples': int(metrics['val_samples']),
        'val_present_classes': int(metrics['val_present_classes']),
        'val_total_classes': int(metrics['val_total_classes']),
    }


def save_metric_history(save_dir, stem, history, metadata=None):
    """Write per-evaluation CSV and final/best JSON, returning both paths.

    Precision, recall and F1 are macro averages over classes present in the
    evaluation set, matching :mod:`classification_metrics` and the console log.
    Files are replaced atomically so an interrupted rerun does not leave a
    partially written result file.
    """
    if not history:
        raise ValueError('metric history must contain at least one evaluation')
    os.makedirs(save_dir, exist_ok=True)
    rows = [_metric_row(epoch, metrics) for epoch, metrics in history]
    final_row = rows[-1]
    best_row = max(rows, key=lambda row: row['ACC'])

    csv_path = os.path.join(save_dir, f'{stem}_metrics.csv')
    csv_tmp = csv_path + '.tmp'
    with open(csv_tmp, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, csv_path)

    json_path = os.path.join(save_dir, f'{stem}_metrics.json')
    json_tmp = json_path + '.tmp'
    payload = {
        'metric_definition': {
            'ACC': 'classification accuracy',
            'Precision_macro': 'macro precision over evaluation classes',
            'Recall_macro': 'macro recall over evaluation classes',
            'F1_macro': 'macro F1 over evaluation classes',
        },
        'metadata': metadata or {},
        'final': final_row,
        'best_by_ACC': best_row,
        'history': rows,
    }
    with open(json_tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(json_tmp, json_path)
    return csv_path, json_path


__all__ = ['save_metric_history']
