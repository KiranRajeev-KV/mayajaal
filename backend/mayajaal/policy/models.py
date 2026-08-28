"""Validated, model-neutral cost-sensitive decision policy contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite

from pydantic import Field, field_validator, model_validator

from mayajaal.schemas.common import SchemaModel


class PolicyAction(StrEnum):
    """The initial merchant actions supported by the decision policy."""

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class ProbabilitySensitivityConfig(SchemaModel):
    """Explicit odds scenarios for review, never probability recalibration."""

    optimistic_odds_multiplier: float = Field(default=0.5, gt=0.0, le=1.0)
    stressed_odds_multiplier: float = Field(default=2.0, ge=1.0)

    @field_validator("optimistic_odds_multiplier", "stressed_odds_multiplier")
    @staticmethod
    def validate_finite_multiplier(value: float) -> float:
        if not isfinite(value):
            raise ValueError("odds multiplier must be finite")
        return value


class PolicyConfig(SchemaModel):
    """Merchant economics expressed in integer paise and loss fractions.

    Fixed monetary assumptions are integers in paise.  Fractional loss fields
    apply to the decision context's transaction exposure and represent the
    portion of that exposure expected to remain lost for the given outcome.
    """

    allow_operational_cost_paise: int = Field(default=0, ge=0)
    allow_legitimate_cost_paise: int = Field(default=0, ge=0)
    allow_fraud_exposure_loss_fraction: float = Field(default=1.0, ge=0.0, le=1.0)

    review_operational_cost_paise: int = Field(default=1_500, ge=0)
    review_legitimate_friction_cost_paise: int = Field(default=500, ge=0)
    review_fraud_residual_loss_fraction: float = Field(default=0.20, ge=0.0, le=1.0)

    block_operational_cost_paise: int = Field(default=200, ge=0)
    block_legitimate_margin_loss_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    block_legitimate_friction_cost_paise: int = Field(default=1_000, ge=0)
    block_fraud_residual_loss_fraction: float = Field(default=0.01, ge=0.0, le=1.0)

    tie_break_order: tuple[PolicyAction, ...] = (
        PolicyAction.ALLOW,
        PolicyAction.REVIEW,
        PolicyAction.BLOCK,
    )
    sensitivity: ProbabilitySensitivityConfig = Field(
        default_factory=ProbabilitySensitivityConfig
    )

    @model_validator(mode="after")
    def validate_tie_break_order(self) -> "PolicyConfig":
        if len(self.tie_break_order) != len(PolicyAction) or set(
            self.tie_break_order
        ) != set(PolicyAction):
            raise ValueError(
                "tie_break_order must contain ALLOW, REVIEW, and BLOCK exactly once"
            )
        return self


class DecisionContext(SchemaModel):
    """Known decision-time context consumed by the business policy only."""

    exposure_paise: int = Field(ge=0)
    context_id: str | None = Field(default=None, min_length=1)


@dataclass(frozen=True)
class ActionCost:
    """Conditional and probability-weighted costs for one policy action."""

    action: PolicyAction
    fraud_cost_paise: float
    legitimate_cost_paise: float
    expected_cost_paise: float
    delta_from_chosen_paise: float


@dataclass(frozen=True)
class ScenarioDecision:
    """A declared probability assumption, not a newly calibrated probability."""

    scenario: str
    odds_multiplier: float
    assumed_fraud_probability: float
    chosen_action: PolicyAction
    expected_costs: tuple[ActionCost, ...]
    decision_margin_paise: float


@dataclass(frozen=True)
class PolicyDecision:
    """A complete, auditable decision from a verified probability model."""

    policy_id: str
    base_model_id: str
    probability_model_id: str
    probability_estimate_id: str
    score_id: str
    subject_id: str
    feature_vector_id: str
    decision_id: str
    raw_model_score: float
    calibrated_fraud_probability: float
    scoring_context_id: str | None
    scoring_cutoff: datetime
    context: DecisionContext
    chosen_action: PolicyAction
    expected_costs: tuple[ActionCost, ...]
    decision_margin_paise: float
    scenarios: tuple[ScenarioDecision, ...]
    decision_is_stable_across_scenarios: bool


def validate_probability(value: float) -> float:
    """Reject non-finite values and keep policy inputs within probability bounds."""
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            "calibrated fraud probability must be finite and within [0, 1]"
        )
    return value
