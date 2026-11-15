"""Posterior-guided target-only spherical prototypes, without target labels.

Inspired by SHOT's soft-assignment initialization and hard refinement:
https://github.com/tim-learn/SHOT/blob/master/object/image_target.py
This optional diagnostic component is not a reimplementation of SHOT training.
"""
import numpy as np


def _unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class PosteriorTargetPrototypes:
    def __init__(self, refinements=2, center=False, append_bias=True):
        if refinements < 1:
            raise ValueError("At least one assignment refinement is required")
        self.refinements = int(refinements)
        self.center = bool(center)
        self.append_bias = bool(append_bias)

    def _features(self, features):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_width or not np.isfinite(values).all():
            raise ValueError("Invalid target feature matrix")
        values = values - self.mean
        if self.append_bias:
            values = np.concatenate((values, np.ones((len(values), 1))), axis=1)
        return _unit(values)

    def fit(self, target_features, target_probabilities):
        x = np.asarray(target_features, dtype=np.float64)
        p = np.asarray(target_probabilities, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or p.ndim != 2 or len(x) != len(p):
            raise ValueError("Expected aligned nonempty feature and probability matrices")
        if not np.isfinite(x).all() or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Nonfinite features or invalid probabilities")
        if not np.allclose(p.sum(1), 1., atol=1e-5):
            raise ValueError("Target probabilities must sum to one")
        self.input_width = x.shape[1]
        self.mean = x.mean(0, keepdims=True) if self.center else np.zeros((1, x.shape[1]))
        x = self._features(x)
        self.class_count = p.shape[1]
        assignments = p.copy()
        predictions = assignments.argmax(1)
        self.trace = []
        for _ in range(self.refinements):
            counts = np.bincount(predictions, minlength=self.class_count)
            self.active_classes = np.flatnonzero(counts > 0)
            weighted = assignments.T @ x
            centers = weighted / np.maximum(assignments.sum(0)[:, None], 1e-12)
            self.centers = _unit(centers[self.active_classes])
            next_predictions = self.active_classes[(x @ self.centers.T).argmax(1)]
            self.trace.append({"active_classes": self.active_classes.tolist(),
                               "changed_assignments": int((next_predictions != predictions).sum()),
                               "assigned_counts": np.bincount(next_predictions, minlength=self.class_count).tolist()})
            predictions = next_predictions
            assignments = np.eye(self.class_count)[predictions]
        self.training_predictions = predictions
        return self

    def predict(self, target_features):
        x = self._features(target_features)
        return self.active_classes[(x @ self.centers.T).argmax(1)]
