"""Stable contracts for post-hoc probability calibration."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

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
class ProbabilityEstimate:
    """One verified score-to-probability result with runtime lineage.

    This is intentionally distinct from :class:`ProbabilityModel`: a model is
    a reusable mapping, while an estimate identifies one semantic scoring
    input and the probability it produced.
    """

    base_model_id: str
    probability_model_id: str
    probability_estimate_id: str
    raw_model_score: float
    calibrated_probability: float
    scoring_context_id: str | None = None

    def __post_init__(self) -> None:
        if not self.base_model_id or not self.probability_model_id:
            raise ValueError("probability estimate requires non-empty model lineage")
        if not self.probability_estimate_id:
            raise ValueError("probability estimate requires probability_estimate_id")
        if not isfinite(self.raw_model_score):
            raise ValueError("raw model score must be finite")
        if (
            not isfinite(self.calibrated_probability)
            or self.calibrated_probability < 0.0
            or self.calibrated_probability > 1.0
        ):
            raise ValueError("calibrated probability must be finite and within [0, 1]")
        if self.scoring_context_id == "":
            raise ValueError("scoring_context_id must be non-empty when provided")


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
