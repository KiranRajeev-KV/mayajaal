"""Reusable metric, threshold, table, and plot routines for score records."""

from collections.abc import Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import polars as pl

from .models import (
    EvaluationConfig,
    EvaluationSplit,
    PredictionRecord,
    SplitMetrics,
    ThresholdRule,
    ThresholdSelection,
)

plt: Any = import_module("matplotlib.pyplot")


def select_threshold(
    records: Sequence[PredictionRecord], config: EvaluationConfig
) -> ThresholdSelection:
    """Freeze the configured threshold using validation records only."""
    if any(record.split is not EvaluationSplit.VALIDATION for record in records):
        raise ValueError("threshold selection accepts validation records only")
    labels, scores = _labels_scores(records)
    positive_count = sum(labels)
    if not labels or positive_count == 0 or positive_count == len(labels):
        return ThresholdSelection(
            threshold=0.5,
            rule=config.threshold_rule,
            validation_sample_count=len(records),
            validation_positive_count=positive_count,
            warnings=(
                "validation: threshold defaulted to 0.5 because validation lacks both classes",
            ),
        )
    if config.threshold_rule is not ThresholdRule.MAXIMIZE_F1:
        raise ValueError(f"unsupported threshold rule: {config.threshold_rule}")
    candidates = sorted(set(scores), reverse=True)
    ranked: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        metrics = _threshold_counts(labels, scores, threshold)
        precision, recall, f1 = _classification_metrics(metrics)
        ranked.append((f1, precision, threshold, recall))
    _, _, threshold, _ = max(ranked)
    return ThresholdSelection(
        threshold=threshold,
        rule=config.threshold_rule,
        validation_sample_count=len(records),
        validation_positive_count=positive_count,
    )


def evaluate_predictions(
    records: Sequence[PredictionRecord],
    threshold: float,
    config: EvaluationConfig,
) -> SplitMetrics:
    """Evaluate one already-scored split at a validation-frozen threshold."""
    if not records:
        raise ValueError("cannot evaluate an empty prediction split")
    split = records[0].split
    if any(record.split is not split for record in records):
        raise ValueError("metrics require records from exactly one split")
    labels, scores = _labels_scores(records)
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    warnings: list[str] = []
    if positive_count < config.minimum_positive_samples:
        warnings.append(
            f"{split.value}: positive support {positive_count} is below "
            f"configured minimum {config.minimum_positive_samples}"
        )
    if negative_count < config.minimum_negative_samples:
        warnings.append(
            f"{split.value}: negative support {negative_count} is below "
            f"configured minimum {config.minimum_negative_samples}"
        )
    if positive_count == 0:
        average_precision = None
        warnings.append(
            f"{split.value}: Average Precision is undefined without positives"
        )
    else:
        average_precision = _average_precision(labels, scores)
    if positive_count == 0 or negative_count == 0:
        roc_auc = None
        warnings.append(f"{split.value}: ROC-AUC is undefined without both classes")
    else:
        roc_auc = _roc_auc(labels, scores)
    counts = _threshold_counts(labels, scores, threshold)
    precision, recall, f1 = _classification_metrics(counts)
    fp_cost, prevented = _fixed_assumption_values(counts, config)
    return SplitMetrics(
        split=split,
        sample_count=len(labels),
        positive_count=positive_count,
        negative_count=negative_count,
        prevalence=float(positive_count / len(labels)) if labels else None,
        average_precision=average_precision,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=counts[0],
        false_positive=counts[1],
        false_negative=counts[2],
        true_negative=counts[3],
        warnings=tuple(warnings),
        estimated_false_positive_review_cost_paise=fp_cost,
        estimated_prevented_fraud_exposure_paise=prevented,
    )


def prediction_frame(records: Iterable[PredictionRecord]) -> pl.DataFrame:
    """Serialize the stable cross-model prediction-record contract."""
    rows = [
        {
            "sample_id": record.sample_id,
            "account_id": record.account_id,
            "decision_time": record.decision_time,
            "split": record.split.value,
            "y_true": record.y_true,
            "score": record.score,
            "model_variant": record.model_variant,
        }
        for record in sorted(
            records,
            key=lambda item: (item.model_variant, item.decision_time, item.sample_id),
        )
    ]
    return pl.DataFrame(
        rows,
        schema={
            "sample_id": pl.String,
            "account_id": pl.String,
            "decision_time": pl.Datetime(time_zone="UTC"),
            "split": pl.String,
            "y_true": pl.Boolean,
            "score": pl.Float64,
            "model_variant": pl.String,
        },
        strict=True,
    )


def save_curve_plots(
    records_by_variant: dict[str, Sequence[PredictionRecord]], output_directory: Path
) -> tuple[Path, Path]:
    """Write deterministic PR and ROC comparison plots for held-out test scores."""
    output_directory.mkdir(parents=True, exist_ok=True)
    pr_path = output_directory / "pr_curve.png"
    roc_path = output_directory / "roc_curve.png"
    figure, axes = plt.subplots()
    for name, records in sorted(records_by_variant.items()):
        labels, scores = _labels_scores(records)
        if not labels or not any(labels):
            continue
        precision, recall = _precision_recall_points(labels, scores)
        axes.plot(recall, precision, label=name)
    axes.set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall")
    axes.legend()
    figure.savefig(pr_path, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots()
    for name, records in sorted(records_by_variant.items()):
        labels, scores = _labels_scores(records)
        if not labels or not any(labels) or all(labels):
            continue
        false_positive_rate, true_positive_rate = _roc_points(labels, scores)
        axes.plot(false_positive_rate, true_positive_rate, label=name)
    axes.plot((0.0, 1.0), (0.0, 1.0), linestyle="--", color="black", label="chance")
    axes.set(
        xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC"
    )
    axes.legend()
    figure.savefig(roc_path, bbox_inches="tight")
    plt.close(figure)
    return pr_path, roc_path


def _labels_scores(
    records: Sequence[PredictionRecord],
) -> tuple[list[int], list[float]]:
    return [int(record.y_true) for record in records], [
        record.score for record in records
    ]


def _precision_recall_points(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Return score-threshold PR points with the AP-compatible initial point."""
    positives = sum(labels)
    if positives == 0:
        return [1.0], [0.0]
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: -item[0])
    precision = [1.0]
    recall = [0.0]
    true_positive = false_positive = 0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        precision.append(true_positive / (true_positive + false_positive))
        recall.append(true_positive / positives)
    return precision, recall


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Weighted precision increments, matching scikit-learn's AP definition."""
    precision, recall = _precision_recall_points(labels, scores)
    return sum(
        (recall[index] - recall[index - 1]) * precision[index]
        for index in range(1, len(recall))
    )


def _roc_points(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Return threshold ROC points, grouping ties before each step."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return [0.0], [0.0]
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: -item[0])
    false_positive_rate = [0.0]
    true_positive_rate = [0.0]
    true_positive = false_positive = 0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        false_positive_rate.append(false_positive / negatives)
        true_positive_rate.append(true_positive / positives)
    return false_positive_rate, true_positive_rate


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Trapezoidal area over grouped-tie ROC points."""
    false_positive_rate, true_positive_rate = _roc_points(labels, scores)
    return sum(
        (false_positive_rate[index] - false_positive_rate[index - 1])
        * (true_positive_rate[index] + true_positive_rate[index - 1])
        / 2.0
        for index in range(1, len(false_positive_rate))
    )


def _threshold_counts(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> tuple[int, int, int, int]:
    true_positive = false_positive = false_negative = true_negative = 0
    for label, score in zip(labels, scores, strict=True):
        predicted = score >= threshold
        if label and predicted:
            true_positive += 1
        elif not label and predicted:
            false_positive += 1
        elif label:
            false_negative += 1
        else:
            true_negative += 1
    return true_positive, false_positive, false_negative, true_negative


def _classification_metrics(
    counts: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    true_positive, false_positive, false_negative, _ = counts
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _fixed_assumption_values(
    counts: tuple[int, int, int, int], config: EvaluationConfig
) -> tuple[int | None, int | None]:
    if config.false_positive_review_cost_paise is None:
        return None, None
    return (
        counts[1] * config.false_positive_review_cost_paise,
        counts[0] * config.true_positive_exposure_paise,  # type: ignore[operator]
    )
