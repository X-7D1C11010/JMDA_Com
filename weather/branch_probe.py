"""Read-only source-fitted branch diagnostics; never replace UDA method results."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from tools.classification_metrics import classification_metrics


def unit(values):
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def source_branch_probe(model, source, validation, device, batch_size=32, seed=42):
    if source.partition != "source_train" or validation.partition != "target_val":
        raise ValueError("Branch probe only accepts source_train and target_val")
    model.eval()
    collections = []
    for dataset in (source, validation):
        if getattr(model, "domain_running_stats", False):
            model.set_bn_domain("source" if dataset.partition == "source_train" else "target")
        branches, labels = None, []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False,
                                    generator=torch.Generator().manual_seed(0)):
                images = {key: batch[key].to(device) for key in ("rgb", "nir")}
                model.encode(images)
                if branches is None:
                    branches = [[] for _ in model.last_features]
                for rows, features in zip(branches, model.last_features):
                    rows.append(features.detach().cpu().numpy())
                labels.extend(batch["label"].tolist())
        collections.append(([unit(np.concatenate(rows)) for rows in branches], np.asarray(labels)))
    (source_features, source_labels), (val_features, val_labels) = collections
    classes = model.num_classes
    counts = np.bincount(source_labels, minlength=classes)
    if not bool((counts > 0).all()):
        raise ValueError("Source probe is missing a declared class")
    folds = min(3, int(counts.min()))
    result = {"kind": "source_only_fitted_branch_linear_probe", "formal_UDA_result": False,
              "classifier_C": 1., "fit_partition": "source_train", "target_training_labels_used": False,
              "validation_used_for_fit_or_C_selection": False, "source_CV_folds": folds if folds > 1 else None,
              "source_CV_scope": "linear_classifier_only_with_a_fixed_encoder",
              "encoder_may_have_seen_held_out_source_labels": True,
              "branches": {}}
    names = list(model.feature_names)
    modes = [(name, a, b) for name, a, b in zip(names, source_features, val_features)]
    modes.append(("equal_concat", unit(np.concatenate(source_features, 1)), unit(np.concatenate(val_features, 1))))
    with threadpool_limits(limits=2):
        for name, a, b in modes:
            def classifier():
                return LogisticRegression(C=1., class_weight="balanced", max_iter=1000, random_state=seed)
            fit = classifier().fit(a, source_labels)
            source_prediction, val_prediction = fit.predict(a), fit.predict(b)
            cv = None
            if folds > 1:
                cv_prediction = np.empty_like(source_labels)
                for training, held_out in StratifiedKFold(folds, shuffle=True, random_state=seed).split(a, source_labels):
                    cv_prediction[held_out] = classifier().fit(a[training], source_labels[training]).predict(a[held_out])
                cv = classification_metrics(source_labels.tolist(), cv_prediction.tolist(), classes)
            result["branches"][name] = {
                "source_fit": classification_metrics(source_labels.tolist(), source_prediction.tolist(), classes),
                "source_CV": cv, "target_val": classification_metrics(val_labels.tolist(), val_prediction.tolist(), classes),
                "target_val_predictions": val_prediction.tolist(),
            }
    return result
