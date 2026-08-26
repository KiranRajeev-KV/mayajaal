"""Typed wrappers around the supported scikit-learn metrics and calibration APIs.

scikit-learn's public API is the source of truth for these calculations.  This
module deliberately confines its incomplete static typing to one boundary so
the rest of Mayajaal can keep strict project typing without reimplementing
well-tested numerical routines.
"""

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np

_calibration: Any = import_module("sklearn.calibration")
_linear_model: Any = import_module("sklearn.linear_model")
_metrics: Any = import_module("sklearn.metrics")


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return scikit-learn's non-interpolated Average Precision."""
    return float(_metrics.average_precision_score(labels, scores))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return scikit-learn's tie-aware ROC-AUC."""
    return float(_metrics.roc_auc_score(labels, scores))


def precision_recall_points(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the scikit-learn precision-recall curve points."""
    precision, recall, _ = _metrics.precision_recall_curve(labels, scores)
    return tuple(float(value) for value in precision), tuple(
        float(value) for value in recall
    )


def roc_points(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return all scikit-learn ROC curve points, including tie boundaries."""
    false_positive_rate, true_positive_rate, _ = _metrics.roc_curve(
        labels, scores, drop_intermediate=False
    )
    return tuple(float(value) for value in false_positive_rate), tuple(
        float(value) for value in true_positive_rate
    )


def classification_metrics(
    labels: Sequence[int], predicted: Sequence[int]
) -> tuple[float, float, float, tuple[int, int, int, int]]:
    """Return precision, recall, F1 and TP/FP/FN/TN from scikit-learn."""
    precision = float(_metrics.precision_score(labels, predicted, zero_division=0))
    recall = float(_metrics.recall_score(labels, predicted, zero_division=0))
    f1 = float(_metrics.f1_score(labels, predicted, zero_division=0))
    matrix = _metrics.confusion_matrix(labels, predicted, labels=(0, 1))
    return (
        precision,
        recall,
        f1,
        (
            int(matrix[1, 1]),
            int(matrix[0, 1]),
            int(matrix[1, 0]),
            int(matrix[0, 0]),
        ),
    )


def fit_sigmoid_logistic(
    scores: Sequence[float], labels: Sequence[int], maximum_iterations: int
) -> tuple[float, float]:
    """Fit a deterministic one-dimensional scikit-learn logistic mapping."""
    model = _linear_model.LogisticRegression(
        solver="lbfgs",
        max_iter=maximum_iterations,
        random_state=0,
    )
    model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return the binary Brier score with its conventional [0, 1] scale."""
    return float(_metrics.brier_score_loss(labels, probabilities, scale_by_half=True))


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return binary cross-entropy with explicit class order."""
    return float(_metrics.log_loss(labels, probabilities, labels=(0, 1)))


def quantile_calibration_points(
    labels: Sequence[int], probabilities: Sequence[float], bin_count: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return observed and mean-probability points from sklearn calibration_curve."""
    observed, mean_predicted = _calibration.calibration_curve(
        labels, probabilities, n_bins=bin_count, strategy="quantile"
    )
    return tuple(float(value) for value in observed), tuple(
        float(value) for value in mean_predicted
    )
