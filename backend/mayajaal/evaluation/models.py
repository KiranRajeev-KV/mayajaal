"""Stable, model-independent contracts for chronological evaluation."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from mayajaal.schemas.common import AwareDatetime, SchemaModel


class EvaluationSplit(StrEnum):
    """The ordered partitions in the single chronological benchmark."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class BenchmarkStatus(StrEnum):
    """Whether a held-out report has adequate configured class support."""

    VALID = "VALID"
    INVALID = "INVALID"


class ThresholdRule(StrEnum):
    """Validation-only rule used to freeze an operating threshold."""

    MAXIMIZE_F1 = "maximize_f1_then_precision_then_threshold"


class EvaluationConfig(SchemaModel):
    """Validated, model-neutral configuration for one held-out benchmark run."""

    train_end_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    validation_end_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)
    minimum_positive_samples: int = Field(default=10, ge=1)
    minimum_negative_samples: int = Field(default=20, ge=1)
    threshold_rule: ThresholdRule = ThresholdRule.MAXIMIZE_F1
    false_positive_review_cost_paise: int | None = Field(default=None, ge=0)
    true_positive_exposure_paise: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_cutoffs_and_optional_costs(self) -> "EvaluationConfig":
        if self.train_end_fraction >= self.validation_end_fraction:
            raise ValueError("train_end_fraction must precede validation_end_fraction")
        if (self.false_positive_review_cost_paise is None) != (
            self.true_positive_exposure_paise is None
        ):
            raise ValueError(
                "set both fixed-assumption cost values or neither; they are evaluation-only"
            )
        return self


@dataclass(frozen=True)
class EvaluationSample:
    """One account review decision, independent of any feature model."""

    sample_id: str
    account_id: str
    decision_time: datetime
    split: EvaluationSplit
    y_true: bool
    campaign_group_id: str | None = None


@dataclass(frozen=True)
class PredictionRecord:
    """A model's score for one immutable evaluation decision."""

    sample_id: str
    account_id: str
    decision_time: AwareDatetime
    split: EvaluationSplit
    y_true: bool
    score: float
    model_variant: str


@dataclass(frozen=True)
class ThresholdSelection:
    """A threshold selected entirely from validation prediction records."""

    threshold: float | None
    rule: ThresholdRule
    validation_sample_count: int
    validation_positive_count: int
    validation_negative_count: int
    is_valid: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitManifest:
    """Serializable audit record of cutoffs, assignments, and purged groups."""

    train_cutoff: AwareDatetime
    validation_cutoff: AwareDatetime
    test_cutoff: AwareDatetime
    samples: tuple[EvaluationSample, ...]
    purged_campaign_group_ids: tuple[str, ...] = ()
    purged_campaign_groups: tuple["CampaignPurge", ...] = ()


@dataclass(frozen=True)
class CampaignPurge:
    """A hidden-truth campaign excluded only to preserve evaluation isolation."""

    campaign_group_id: str
    reason: str


@dataclass(frozen=True)
class BenchmarkValidity:
    """Audit-friendly validity state for a held-out benchmark report."""

    status: BenchmarkStatus
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitMetrics:
    """Metrics from one score set and one frozen threshold."""

    split: EvaluationSplit
    sample_count: int
    positive_count: int
    negative_count: int
    support_is_sufficient: bool
    prevalence: float | None
    average_precision: float | None
    roc_auc: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    true_positive: int | None
    false_positive: int | None
    false_negative: int | None
    true_negative: int | None
    warnings: tuple[str, ...] = ()
    estimated_false_positive_review_cost_paise: int | None = None
    estimated_prevented_fraud_exposure_paise: int | None = None
