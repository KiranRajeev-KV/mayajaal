"""Stable contracts for post-hoc probability calibration."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from mayajaal.schemas.common import AwareDatetime, SchemaModel


class CalibrationMethod(StrEnum):
    """Supported score-to-probability mappings."""

    SIGMOID = "sigmoid"


class CalibrationStatus(StrEnum):
    """Whether a validation-only calibrator was safe to fit."""

    VALID = "VALID"
    INVALID = "INVALID"


class CalibrationConfig(SchemaModel):
    """Validated deterministic settings for validation-only calibration."""

    method: CalibrationMethod = CalibrationMethod.SIGMOID
    minimum_positive_samples: int = Field(default=10, ge=1)
    minimum_negative_samples: int = Field(default=20, ge=1)
    quantile_bin_count: int = Field(default=10, ge=2)
    maximum_iterations: int = Field(default=1_000, ge=1)


@dataclass(frozen=True)
class SigmoidCalibrator:
    """A strictly increasing Platt-style raw-margin calibration mapping."""

    coefficient: float
    intercept: float
    method: CalibrationMethod = CalibrationMethod.SIGMOID


@dataclass(frozen=True)
class CalibrationFit:
    """Audit record of fitting a calibrator from validation records only."""

    status: CalibrationStatus
    calibrator: SigmoidCalibrator | None
    validation_sample_count: int
    validation_positive_count: int
    validation_negative_count: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationPrediction:
    """Model-neutral score and probability record for one decision sample."""

    sample_id: str
    account_id: str
    decision_time: AwareDatetime
    split: str
    y_true: bool
    raw_model_score: float
    uncalibrated_probability: float
    calibrated_probability: float | None


@dataclass(frozen=True)
class CalibrationBin:
    """One deterministic equal-frequency reliability bin."""

    index: int
    sample_count: int
    lower_probability: float
    upper_probability: float
    mean_predicted_probability: float
    observed_prevalence: float


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Probability quality and ranking summaries for one scored split."""

    sample_count: int
    positive_count: int
    negative_count: int
    observed_prevalence: float
    mean_predicted_probability: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    average_precision: float | None
    roc_auc: float | None
    reliability_bins: tuple[CalibrationBin, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationEvaluation:
    """Calibration-fit state plus validation/test probability diagnostics."""

    fit: CalibrationFit
    validation_uncalibrated: ProbabilityMetrics
    validation_calibrated: ProbabilityMetrics | None
    test_uncalibrated: ProbabilityMetrics
    test_calibrated: ProbabilityMetrics | None
