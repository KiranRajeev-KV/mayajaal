"""Reusable metric, threshold, table, and plot routines for score records."""

from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import polars as pl

from mayajaal.interop.sklearn import (
    average_precision as sklearn_average_precision,
)
from mayajaal.interop.sklearn import (
    classification_metrics,
    precision_recall_points,
    roc_points,
)
from mayajaal.interop.sklearn import (
    roc_auc as sklearn_roc_auc,
)

from .models import (
    BenchmarkStatus,
    BenchmarkValidity,
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
    negative_count = len(labels) - positive_count
    warnings = _support_warnings(
        EvaluationSplit.VALIDATION, positive_count, negative_count, config
    )
    if warnings:
        return ThresholdSelection(
            threshold=None,
            rule=config.threshold_rule,
            validation_sample_count=len(records),
            validation_positive_count=positive_count,
            validation_negative_count=negative_count,
            is_valid=False,
            warnings=warnings,
        )
    if config.threshold_rule is not ThresholdRule.MAXIMIZE_F1:
        raise ValueError(f"unsupported threshold rule: {config.threshold_rule}")
    candidates = sorted(set(scores), reverse=True)
    ranked: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        precision, recall, f1, _ = _threshold_metrics(labels, scores, threshold)
        ranked.append((f1, precision, threshold, recall))
    _, _, threshold, _ = max(ranked)
    return ThresholdSelection(
        threshold=threshold,
        rule=config.threshold_rule,
        validation_sample_count=len(records),
        validation_positive_count=positive_count,
        validation_negative_count=negative_count,
        is_valid=True,
    )


def evaluate_predictions(
    records: Sequence[PredictionRecord],
    threshold: float | None,
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
    warnings = list(_support_warnings(split, positive_count, negative_count, config))
    if positive_count == 0:
        average_precision = None
        warnings.append(
            f"{split.value}: Average Precision is undefined without positives"
        )
    else:
        average_precision = sklearn_average_precision(labels, scores)
    if positive_count == 0 or negative_count == 0:
        roc_auc = None
        warnings.append(f"{split.value}: ROC-AUC is undefined without both classes")
    else:
        roc_auc = sklearn_roc_auc(labels, scores)
    threshold_metrics = (
        _threshold_metrics(labels, scores, threshold) if threshold is not None else None
    )
    if threshold_metrics is None:
        precision = recall = f1 = None
        counts = None
        warnings.append(
            f"{split.value}: threshold metrics are unavailable because validation threshold selection was invalid"
        )
    else:
        precision, recall, f1, counts = threshold_metrics
    fp_cost, prevented = _fixed_assumption_values(counts, config)
    return SplitMetrics(
        split=split,
        sample_count=len(labels),
        positive_count=positive_count,
        negative_count=negative_count,
        support_is_sufficient=not _support_warnings(
            split, positive_count, negative_count, config
        ),
        prevalence=float(positive_count / len(labels)) if labels else None,
        average_precision=average_precision,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=counts[0] if counts is not None else None,
        false_positive=counts[1] if counts is not None else None,
        false_negative=counts[2] if counts is not None else None,
        true_negative=counts[3] if counts is not None else None,
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
    pr_curve_count = 0
    for name, records in sorted(records_by_variant.items()):
        labels, scores = _labels_scores(records)
        if not labels or not any(labels):
            continue
        precision, recall = precision_recall_points(labels, scores)
        axes.plot(recall, precision, label=name)
        pr_curve_count += 1
    axes.set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall")
    if pr_curve_count:
        axes.legend()
    else:
        axes.text(0.5, 0.5, "No valid held-out PR curves", ha="center", va="center")
    figure.savefig(pr_path, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots()
    roc_curve_count = 0
    for name, records in sorted(records_by_variant.items()):
        labels, scores = _labels_scores(records)
        if not labels or not any(labels) or all(labels):
            continue
        false_positive_rate, true_positive_rate = roc_points(labels, scores)
        axes.plot(false_positive_rate, true_positive_rate, label=name)
        roc_curve_count += 1
    if roc_curve_count:
        axes.plot((0.0, 1.0), (0.0, 1.0), linestyle="--", color="black", label="chance")
    axes.set(
        xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC"
    )
    if roc_curve_count:
        axes.legend()
    else:
        axes.text(0.5, 0.5, "No valid held-out ROC curves", ha="center", va="center")
    figure.savefig(roc_path, bbox_inches="tight")
    plt.close(figure)
    return pr_path, roc_path


def _labels_scores(
    records: Sequence[PredictionRecord],
) -> tuple[list[int], list[float]]:
    return [int(record.y_true) for record in records], [
        record.score for record in records
    ]


def _threshold_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> tuple[float, float, float, tuple[int, int, int, int]]:
    predicted = tuple(int(score >= threshold) for score in scores)
    return classification_metrics(labels, predicted)


def held_out_validity(
    thresholds: Mapping[str, ThresholdSelection],
    metrics: Mapping[str, Mapping[EvaluationSplit, SplitMetrics]],
) -> BenchmarkValidity:
    """Return one strict status for a multi-model held-out benchmark.

    Support depends only on the shared manifest labels, but each adapter is
    checked so another model can reuse this reporting contract safely.
    """
    reasons: list[str] = []
    for name in sorted(thresholds):
        if not thresholds[name].is_valid:
            reasons.append(
                f"{name}: validation support is insufficient for threshold selection"
            )
        test_metrics = metrics[name][EvaluationSplit.TEST]
        if not test_metrics.support_is_sufficient:
            reasons.append(
                f"{name}: test support is insufficient for held-out reporting"
            )
    return BenchmarkValidity(
        status=BenchmarkStatus.INVALID if reasons else BenchmarkStatus.VALID,
        reasons=tuple(reasons),
    )


def _support_warnings(
    split: EvaluationSplit,
    positive_count: int,
    negative_count: int,
    config: EvaluationConfig,
) -> tuple[str, ...]:
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
    return tuple(warnings)


def _fixed_assumption_values(
    counts: tuple[int, int, int, int] | None, config: EvaluationConfig
) -> tuple[int | None, int | None]:
    if counts is None or config.false_positive_review_cost_paise is None:
        return None, None
    return (
        counts[1] * config.false_positive_review_cost_paise,
        counts[0] * config.true_positive_exposure_paise,  # type: ignore[operator]
    )
