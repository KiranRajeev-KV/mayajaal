"""Deterministic, model-neutral sigmoid calibration and probability diagnostics."""

from collections.abc import Mapping, Sequence
from math import exp, isfinite
from typing import Any

import numpy as np

from mayajaal.evaluation import EvaluationSplit, PredictionRecord
from mayajaal.interop.sklearn import (
    average_precision as sklearn_average_precision,
)
from mayajaal.interop.sklearn import (
    brier_score,
    fit_sigmoid_logistic,
    quantile_calibration_points,
)
from mayajaal.interop.sklearn import (
    log_loss as sklearn_log_loss,
)
from mayajaal.interop.sklearn import (
    roc_auc as sklearn_roc_auc,
)

from .models import (
    CalibrationBin,
    CalibrationConfig,
    CalibrationEvaluation,
    CalibrationFit,
    CalibrationMethod,
    CalibrationPrediction,
    CalibrationStatus,
    ProbabilityMetrics,
    SigmoidCalibrator,
)


def fit(
    scores: Sequence[float], labels: Sequence[bool], config: CalibrationConfig
) -> CalibrationFit:
    """Fit a strictly increasing Platt sigmoid if validation support is adequate.

    The two-parameter logistic mapping is optimized only over the supplied raw
    margins. Its positive slope constraint preserves the classifier's ranking,
    which keeps AP and ROC-AUC invariant apart from harmless floating-point
    ties. The caller owns enforcing that inputs are validation-only.
    """
    _validate_scores_labels(scores, labels)
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    reasons = _support_reasons(positive_count, negative_count, config)
    if reasons:
        return CalibrationFit(
            status=CalibrationStatus.INVALID,
            calibrator=None,
            validation_sample_count=len(labels),
            validation_positive_count=positive_count,
            validation_negative_count=negative_count,
            reasons=reasons,
        )
    if config.method is not CalibrationMethod.SIGMOID:
        raise ValueError(f"unsupported calibration method: {config.method}")
    coefficient, intercept = fit_sigmoid_logistic(
        scores, [int(label) for label in labels], config.maximum_iterations
    )
    if not isfinite(coefficient) or not isfinite(intercept) or coefficient <= 0.0:
        return CalibrationFit(
            status=CalibrationStatus.INVALID,
            calibrator=None,
            validation_sample_count=len(labels),
            validation_positive_count=positive_count,
            validation_negative_count=negative_count,
            reasons=(
                "validation: scikit-learn sigmoid coefficient was not strictly increasing",
            ),
        )
    return CalibrationFit(
        status=CalibrationStatus.VALID,
        calibrator=SigmoidCalibrator(coefficient=coefficient, intercept=intercept),
        validation_sample_count=len(labels),
        validation_positive_count=positive_count,
        validation_negative_count=negative_count,
    )


def predict_probability(
    calibrator: SigmoidCalibrator, scores: Sequence[float]
) -> tuple[float, ...]:
    """Map raw margins to finite probabilities with a monotonic sigmoid."""
    if calibrator.coefficient <= 0.0:
        raise ValueError("sigmoid calibrator coefficient must be positive")
    if not isfinite(calibrator.coefficient) or not isfinite(calibrator.intercept):
        raise ValueError("sigmoid calibrator parameters must be finite")
    if any(not isfinite(score) for score in scores):
        raise ValueError("raw model scores must be finite")
    return tuple(
        _expit(calibrator.coefficient * score + calibrator.intercept)
        for score in scores
    )


def calibrate_records(
    records: Sequence[PredictionRecord],
    raw_scores: Mapping[str, float],
    config: CalibrationConfig,
) -> tuple[tuple[CalibrationPrediction, ...], CalibrationEvaluation]:
    """Fit on validation only, then evaluate calibrated probabilities once on test."""
    _validate_record_scores(records, raw_scores)
    validation = tuple(
        record for record in records if record.split is EvaluationSplit.VALIDATION
    )
    test = tuple(record for record in records if record.split is EvaluationSplit.TEST)
    if not validation:
        raise ValueError("calibration requires validation prediction records")
    if not test:
        raise ValueError("calibration requires held-out test prediction records")
    validation_raw_scores = tuple(raw_scores[record.sample_id] for record in validation)
    fit_result = fit(
        validation_raw_scores,
        tuple(record.y_true for record in validation),
        config,
    )
    calibrated_by_sample: dict[str, float] = {}
    if fit_result.calibrator is not None:
        calibrated_by_sample = dict(
            zip(
                (record.sample_id for record in records),
                predict_probability(
                    fit_result.calibrator,
                    tuple(raw_scores[record.sample_id] for record in records),
                ),
                strict=True,
            )
        )
    predictions = tuple(
        CalibrationPrediction(
            sample_id=record.sample_id,
            account_id=record.account_id,
            decision_time=record.decision_time,
            split=record.split.value,
            y_true=record.y_true,
            raw_model_score=raw_scores[record.sample_id],
            uncalibrated_probability=record.score,
            calibrated_probability=calibrated_by_sample.get(record.sample_id),
        )
        for record in sorted(
            records, key=lambda item: (item.decision_time, item.sample_id)
        )
    )
    validation_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.split == EvaluationSplit.VALIDATION.value
    )
    test_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.split == EvaluationSplit.TEST.value
    )
    validation_uncalibrated = probability_metrics(
        validation_predictions, probability_field="uncalibrated", config=config
    )
    test_uncalibrated = probability_metrics(
        test_predictions, probability_field="uncalibrated", config=config
    )
    validation_calibrated = (
        probability_metrics(
            validation_predictions, probability_field="calibrated", config=config
        )
        if fit_result.calibrator is not None
        else None
    )
    test_calibrated = (
        probability_metrics(
            test_predictions, probability_field="calibrated", config=config
        )
        if fit_result.calibrator is not None
        else None
    )
    return predictions, CalibrationEvaluation(
        fit=fit_result,
        validation_uncalibrated=validation_uncalibrated,
        validation_calibrated=validation_calibrated,
        test_uncalibrated=test_uncalibrated,
        test_calibrated=test_calibrated,
    )


def probability_metrics(
    predictions: Sequence[CalibrationPrediction],
    *,
    probability_field: str,
    config: CalibrationConfig,
) -> ProbabilityMetrics:
    """Compute deterministic proper scores, ranking diagnostics, and ECE bins."""
    if not predictions:
        raise ValueError("probability metrics require at least one prediction")
    labels = tuple(int(item.y_true) for item in predictions)
    probabilities = tuple(
        _probability_value(item, probability_field) for item in predictions
    )
    if any(
        not isfinite(probability) or probability < 0.0 or probability > 1.0
        for probability in probabilities
    ):
        raise ValueError("probabilities must be finite and within [0, 1]")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    warnings: list[str] = []
    if positive_count == 0:
        average_precision = None
        warnings.append("Average Precision is undefined without positives")
    else:
        average_precision = sklearn_average_precision(labels, probabilities)
    if positive_count == 0 or negative_count == 0:
        roc_auc = None
        warnings.append("ROC-AUC is undefined without both classes")
    else:
        roc_auc = sklearn_roc_auc(labels, probabilities)
    bins = quantile_bins(labels, probabilities, config.quantile_bin_count)
    return ProbabilityMetrics(
        sample_count=len(labels),
        positive_count=positive_count,
        negative_count=negative_count,
        observed_prevalence=positive_count / len(labels),
        mean_predicted_probability=sum(probabilities) / len(probabilities),
        brier_score=brier_score(labels, probabilities),
        log_loss=sklearn_log_loss(labels, probabilities),
        expected_calibration_error=sum(
            item.sample_count
            / len(labels)
            * abs(item.observed_prevalence - item.mean_predicted_probability)
            for item in bins
        ),
        average_precision=average_precision,
        roc_auc=roc_auc,
        reliability_bins=bins,
        warnings=tuple(warnings),
    )


def quantile_bins(
    labels: Sequence[int], probabilities: Sequence[float], bin_count: int
) -> tuple[CalibrationBin, ...]:
    """Attach deterministic count/range metadata to sklearn quantile points.

    ``sklearn.calibration.calibration_curve`` supplies the observed and mean
    probability values.  Its public API deliberately does not expose point-to-
    bin membership, so the compact metadata below mirrors its quantile bin
    assignment solely to report counts, ranges, and the ECE-style summary.
    """
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("quantile bins require aligned, non-empty sequences")
    observed, mean_predicted = quantile_calibration_points(
        labels, probabilities, bin_count
    )
    values = np.asarray(probabilities, dtype=float)
    bins = np.percentile(values, np.linspace(0.0, 100.0, bin_count + 1))
    bin_ids = np.searchsorted(bins[1:-1], values)
    metadata: list[tuple[int, Any]] = []
    for index in range(bin_count):
        members = values[bin_ids == index]
        if not len(members):
            continue
        metadata.append((index, members))
    result: list[CalibrationBin] = []
    for (index, members), observed_value, mean_value in zip(
        metadata, observed, mean_predicted, strict=True
    ):
        result.append(
            CalibrationBin(
                index=index,
                sample_count=len(members),
                lower_probability=float(np.min(members)),
                upper_probability=float(np.max(members)),
                mean_predicted_probability=mean_value,
                observed_prevalence=observed_value,
            )
        )
    return tuple(result)


def _validate_scores_labels(scores: Sequence[float], labels: Sequence[bool]) -> None:
    if len(scores) != len(labels) or not scores:
        raise ValueError("calibration needs aligned, non-empty scores and labels")
    if any(not isfinite(score) for score in scores):
        raise ValueError("raw model scores must be finite")


def _validate_record_scores(
    records: Sequence[PredictionRecord], raw_scores: Mapping[str, float]
) -> None:
    if not records:
        raise ValueError("calibration requires prediction records")
    if {record.sample_id for record in records} != set(raw_scores):
        raise ValueError("raw scores must map exactly to prediction record sample IDs")
    if any(not isfinite(raw_scores[record.sample_id]) for record in records):
        raise ValueError("raw model scores must be finite")


def _support_reasons(
    positive_count: int, negative_count: int, config: CalibrationConfig
) -> tuple[str, ...]:
    reasons: list[str] = []
    if positive_count < config.minimum_positive_samples:
        reasons.append(
            f"validation: positive support {positive_count} is below configured calibration minimum {config.minimum_positive_samples}"
        )
    if negative_count < config.minimum_negative_samples:
        reasons.append(
            f"validation: negative support {negative_count} is below configured calibration minimum {config.minimum_negative_samples}"
        )
    return tuple(reasons)


def _probability_value(prediction: CalibrationPrediction, field: str) -> float:
    if field == "uncalibrated":
        return prediction.uncalibrated_probability
    if field == "calibrated" and prediction.calibrated_probability is not None:
        return prediction.calibrated_probability
    raise ValueError(f"prediction has no {field} probability")


def _expit(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)
