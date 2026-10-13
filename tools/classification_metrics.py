"""Dependency-light classification metrics shared by comparison scripts."""


def classification_metrics(labels, predictions, num_classes=14):
    labels = [int(value) for value in labels]
    predictions = [int(value) for value in predictions]
    if len(labels) != len(predictions):
        raise ValueError('labels and predictions must have the same length')
    if not labels:
        raise ValueError('classification metrics require at least one sample')
    present = sorted(set(labels))
    if min(present) < 0 or max(present) >= num_classes:
        raise ValueError('validation labels exceed the configured class range')
    if min(predictions) < 0 or max(predictions) >= num_classes:
        raise ValueError('predictions exceed the configured class range')

    precision_values = []
    recall_values = []
    f1_values = []
    for class_id in present:
        true_positive = sum(
            truth == class_id and pred == class_id
            for truth, pred in zip(labels, predictions)
        )
        false_positive = sum(
            truth != class_id and pred == class_id
            for truth, pred in zip(labels, predictions)
        )
        false_negative = sum(
            truth == class_id and pred != class_id
            for truth, pred in zip(labels, predictions)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    correct = sum(truth == pred for truth, pred in zip(labels, predictions))
    return {
        'accuracy': correct / len(labels),
        'precision_macro': sum(precision_values) / len(precision_values),
        'recall_macro': sum(recall_values) / len(recall_values),
        'f1_macro': sum(f1_values) / len(f1_values),
        'val_present_classes': len(present),
        'val_total_classes': num_classes,
        'val_class_coverage': len(present) / num_classes,
        'val_samples': len(labels),
    }


def format_classification_metrics(metrics):
    return (
        f"Val Precision (Macro): {metrics['precision_macro']:.4f} | "
        f"Val Recall (Macro): {metrics['recall_macro']:.4f} | "
        f"Val F1 (Macro): {metrics['f1_macro']:.4f} | "
        f"Val class coverage: {metrics['val_present_classes']}/"
        f"{metrics['val_total_classes']} | Val samples: {metrics['val_samples']}"
    )


__all__ = ['classification_metrics', 'format_classification_metrics']
