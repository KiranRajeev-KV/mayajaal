"""Focused tests for validation-only, model-neutral probability calibration."""

import json
import math
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl

from mayajaal.calibration import (
    CalibrationConfig,
    CalibrationPrediction,
    CalibrationStatus,
    calibrate_records,
    fit,
    predict_probability,
    probability_metrics,
    quantile_bins,
    save_calibration_artifacts,
)
from mayajaal.evaluation import EvaluationSplit, PredictionRecord

sklearn_calibration: Any = import_module("sklearn.calibration")
sklearn_linear_model: Any = import_module("sklearn.linear_model")
sklearn_metrics: Any = import_module("sklearn.metrics")


def probability(score: float) -> float:
    """Create a stable uncalibrated probability for a raw-margin fixture."""
    return 1.0 / (1.0 + math.exp(-score))


def record(
    sample_id: str, split: EvaluationSplit, label: bool, raw_score: float
) -> PredictionRecord:
    """Create one model-neutral decision record with a raw-score counterpart."""
    return PredictionRecord(
        sample_id=sample_id,
        account_id=f"account-{sample_id}",
        decision_time=datetime(2026, 2, 1, tzinfo=UTC),
        split=split,
        y_true=label,
        score=probability(raw_score),
        model_variant="full",
    )


def fixture_records() -> tuple[PredictionRecord, ...]:
    """Use a deliberately ordered score fixture with all chronological splits."""
    return (
        record("train-negative", EvaluationSplit.TRAIN, False, -3.0),
        record("train-positive", EvaluationSplit.TRAIN, True, 3.0),
        record("validation-negative-1", EvaluationSplit.VALIDATION, False, -2.0),
        record("validation-negative-2", EvaluationSplit.VALIDATION, False, -1.0),
        record("validation-positive-1", EvaluationSplit.VALIDATION, True, 1.0),
        record("validation-positive-2", EvaluationSplit.VALIDATION, True, 2.0),
        record("test-negative-1", EvaluationSplit.TEST, False, -1.5),
        record("test-negative-2", EvaluationSplit.TEST, False, -0.5),
        record("test-positive-1", EvaluationSplit.TEST, True, 0.5),
        record("test-positive-2", EvaluationSplit.TEST, True, 1.5),
    )


def raw_scores(records: tuple[PredictionRecord, ...]) -> dict[str, float]:
    """Recover fixture margins by an explicit, non-label-dependent mapping."""
    return {
        item.sample_id: math.log(item.score / (1.0 - item.score)) for item in records
    }


class CalibrationTests(unittest.TestCase):
    def test_sigmoid_is_validation_only_monotonic_and_preserves_rank_metrics(
        self,
    ) -> None:
        records = fixture_records()
        scores = raw_scores(records)
        config = CalibrationConfig(
            minimum_positive_samples=2,
            minimum_negative_samples=2,
            quantile_bin_count=3,
        )
        predictions, evaluation = calibrate_records(records, scores, config)

        self.assertEqual(evaluation.fit.status, CalibrationStatus.VALID)
        assert evaluation.fit.calibrator is not None
        mapped = predict_probability(evaluation.fit.calibrator, (-2.0, 0.0, 2.0))
        self.assertEqual(tuple(sorted(mapped)), mapped)
        self.assertTrue(
            all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in mapped)
        )
        self.assertEqual(
            evaluation.test_uncalibrated.average_precision,
            evaluation.test_calibrated.average_precision
            if evaluation.test_calibrated
            else None,
        )
        self.assertEqual(
            evaluation.test_uncalibrated.roc_auc,
            evaluation.test_calibrated.roc_auc if evaluation.test_calibrated else None,
        )
        self.assertEqual(
            [item.sample_id for item in predictions],
            sorted(
                item.sample_id
                for item in predictions
                if item.decision_time == datetime(2026, 2, 1, tzinfo=UTC)
            ),
        )
        self.assertTrue(
            all(item.calibrated_probability is not None for item in predictions)
        )

    def test_train_and_test_labels_cannot_affect_validation_calibrator(self) -> None:
        records = fixture_records()
        scores = raw_scores(records)
        config = CalibrationConfig(
            minimum_positive_samples=2, minimum_negative_samples=2
        )
        _, first = calibrate_records(records, scores, config)
        altered = tuple(
            replace(item, y_true=not item.y_true)
            if item.split is not EvaluationSplit.VALIDATION
            else item
            for item in records
        )
        _, second = calibrate_records(altered, scores, config)
        self.assertEqual(first.fit, second.fit)

    def test_insufficient_validation_support_is_invalid_and_does_not_fit(self) -> None:
        records = fixture_records()
        scores = raw_scores(records)
        predictions, evaluation = calibrate_records(
            records,
            scores,
            CalibrationConfig(minimum_positive_samples=3, minimum_negative_samples=2),
        )
        self.assertEqual(evaluation.fit.status, CalibrationStatus.INVALID)
        self.assertIsNone(evaluation.fit.calibrator)
        self.assertIsNone(evaluation.validation_calibrated)
        self.assertIsNone(evaluation.test_calibrated)
        self.assertTrue(
            all(
                "validation: positive support" in reason
                for reason in evaluation.fit.reasons
            )
        )
        self.assertTrue(
            all(item.calibrated_probability is None for item in predictions)
        )

    def test_fit_is_deterministic_and_requires_aligned_finite_validation_inputs(
        self,
    ) -> None:
        config = CalibrationConfig(
            minimum_positive_samples=2, minimum_negative_samples=2
        )
        first = fit((-2.0, -1.0, 1.0, 2.0), (False, False, True, True), config)
        second = fit((-2.0, -1.0, 1.0, 2.0), (False, False, True, True), config)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "finite"):
            _ = fit((0.0, float("nan")), (False, True), config)

    def test_sigmoid_artifact_and_probability_metrics_match_sklearn(self) -> None:
        scores = (-2.0, -1.0, 1.0, 2.0)
        labels = (False, False, True, True)
        config = CalibrationConfig(
            minimum_positive_samples=2,
            minimum_negative_samples=2,
            quantile_bin_count=3,
        )
        result = fit(scores, labels, config)
        self.assertEqual(result.status, CalibrationStatus.VALID)
        assert result.calibrator is not None
        expected = sklearn_linear_model.LogisticRegression(
            solver="lbfgs", max_iter=config.maximum_iterations, random_state=0
        ).fit([[score] for score in scores], [int(label) for label in labels])
        self.assertAlmostEqual(
            result.calibrator.coefficient, float(expected.coef_[0, 0])
        )
        self.assertAlmostEqual(
            result.calibrator.intercept, float(expected.intercept_[0])
        )
        probabilities = predict_probability(result.calibrator, scores)
        expected_probabilities = expected.predict_proba([[score] for score in scores])[
            :, 1
        ]
        for actual, expected_probability in zip(
            probabilities, expected_probabilities, strict=True
        ):
            self.assertAlmostEqual(actual, float(expected_probability))

        records = tuple(
            record(str(index), EvaluationSplit.TEST, label, score)
            for index, (score, label) in enumerate(zip(scores, labels, strict=True))
        )
        report = probability_metrics(
            tuple(
                CalibrationPrediction(
                    sample_id=item.sample_id,
                    account_id=item.account_id,
                    decision_time=item.decision_time,
                    split=item.split.value,
                    y_true=item.y_true,
                    raw_model_score=scores[index],
                    uncalibrated_probability=item.score,
                    calibrated_probability=probabilities[index],
                )
                for index, item in enumerate(records)
            ),
            probability_field="calibrated",
            config=config,
        )
        numeric_labels = [int(label) for label in labels]
        self.assertAlmostEqual(
            report.brier_score,
            float(
                sklearn_metrics.brier_score_loss(
                    numeric_labels, probabilities, scale_by_half=True
                )
            ),
        )
        self.assertAlmostEqual(
            report.log_loss,
            float(
                sklearn_metrics.log_loss(numeric_labels, probabilities, labels=(0, 1))
            ),
        )
        observed, mean_predicted = sklearn_calibration.calibration_curve(
            numeric_labels,
            probabilities,
            n_bins=config.quantile_bin_count,
            strategy="quantile",
        )
        bins = quantile_bins(numeric_labels, probabilities, config.quantile_bin_count)
        self.assertEqual(
            [
                (item.observed_prevalence, item.mean_predicted_probability)
                for item in bins
            ],
            list(zip(observed.tolist(), mean_predicted.tolist(), strict=True)),
        )

    def test_artifacts_and_prediction_contract_are_reproducible(self) -> None:
        records = fixture_records()
        scores = raw_scores(records)
        config = CalibrationConfig(
            minimum_positive_samples=2, minimum_negative_samples=2
        )
        predictions, evaluation = calibrate_records(records, scores, config)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            first = save_calibration_artifacts(
                output,
                predictions,
                evaluation,
                metadata={"seed": 7, "leakage_policy": "validation only"},
            )
            first_evaluation = first["evaluation"].read_text(encoding="utf-8")
            first_calibrator = first["calibrator"].read_text(encoding="utf-8")
            second = save_calibration_artifacts(
                output,
                predictions,
                evaluation,
                metadata={"seed": 7, "leakage_policy": "validation only"},
            )
            self.assertEqual(
                first_evaluation, second["evaluation"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                first_calibrator, second["calibrator"].read_text(encoding="utf-8")
            )
            frame = pl.read_parquet(second["predictions"])
            self.assertEqual(
                frame.columns,
                [
                    "sample_id",
                    "account_id",
                    "decision_time",
                    "split",
                    "y_true",
                    "raw_model_score",
                    "uncalibrated_probability",
                    "calibrated_probability",
                ],
            )
            report = json.loads(second["evaluation"].read_text(encoding="utf-8"))
            self.assertEqual(report["calibration"]["fit"]["status"], "VALID")
            self.assertTrue(all(path.is_file() for path in second.values()))


if __name__ == "__main__":
    _ = unittest.main()
