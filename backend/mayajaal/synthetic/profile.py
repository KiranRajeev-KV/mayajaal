"""Configuration for a deterministic, persona-driven synthetic world run."""

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from mayajaal.schemas.common import AwareDatetime, SchemaModel


class DifficultyPreset(StrEnum):
    """Named overlap settings for internal benchmark scenarios."""

    EASY = "easy"
    STANDARD = "standard"
    HARD = "hard"
    DRIFT = "drift"


class PopulationProfile(SchemaModel):
    """Controls the mix of ordinary people and benign shared contexts."""

    benign_network_group_count: int = Field(default=4, ge=0)
    accounts_per_benign_network_group: int = Field(default=5, gt=1)
    persona_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "occasional": 0.25,
            "repeat": 0.25,
            "promo_sensitive": 0.15,
            "high_value": 0.10,
            "commuter": 0.15,
            "digital_first": 0.10,
        }
    )

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if not self.persona_weights or any(
            weight <= 0.0 for weight in self.persona_weights.values()
        ):
            raise ValueError("persona_weights must contain positive weights")
        return self


class IdentityLifecycleProfile(SchemaModel):
    """Controls ordinary identity replacement and mobility over a history."""

    additional_device_probability: float = Field(default=0.20, ge=0.0, le=1.0)
    additional_payment_probability: float = Field(default=0.16, ge=0.0, le=1.0)
    additional_address_probability: float = Field(default=0.10, ge=0.0, le=1.0)
    travel_ip_probability: float = Field(default=0.24, ge=0.0, le=1.0)


class CommerceProfile(SchemaModel):
    """Controls ordinary purchase, promotion, and refund behaviour."""

    normal_refund_probability: float = Field(default=0.055, ge=0.0, le=1.0)


class CalendarProfile(SchemaModel):
    """Controls ordinary seasonal concentration and optional temporal drift."""

    seasonal_window_center_fraction: float = Field(default=0.78, ge=0.0, le=1.0)
    seasonal_window_width_fraction: float = Field(default=0.18, gt=0.0, le=1.0)
    seasonal_activity_share: float = Field(default=0.22, ge=0.0, le=1.0)
    drift_late_activity_power: float = Field(default=0.72, gt=0.0, le=1.0)


class AbuseProfile(SchemaModel):
    """Controls hidden campaign composition without appearing in output records."""

    min_warmup_orders: int = Field(default=1, ge=0)
    max_warmup_orders: int = Field(default=3, ge=0)
    low_and_slow_probability: float = Field(default=0.45, ge=0.0, le=1.0)
    partial_identity_sharing_probability: float = Field(default=0.68, ge=0.0, le=1.0)
    ring_size_variation: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_warmup_orders(self) -> Self:
        if self.max_warmup_orders < self.min_warmup_orders:
            raise ValueError("max_warmup_orders cannot be below min_warmup_orders")
        return self


class DiagnosticProfile(SchemaModel):
    """Internal plausibility guardrails, not claims about a real merchant."""

    min_variable_numeric_features: int = Field(default=5, ge=1)
    max_single_feature_perfect_separators: int = Field(default=0, ge=0)


class GenerationProfile(SchemaModel):
    """Explicit control surface for world size, behaviour, and time range.

    The original count fields remain supported so existing callers can continue
    to construct profiles directly.  Nested profiles add behavioural diversity
    without exposing synthetic truth to downstream consumers.
    """

    seed: int = Field(ge=0)
    normal_account_count: int = Field(default=20, ge=0)
    shared_household_count: int = Field(default=3, ge=0)
    accounts_per_shared_household: int = Field(default=3, gt=0)
    promo_ring_count: int = Field(default=1, ge=0)
    refund_ring_count: int = Field(default=1, ge=0)
    mixed_ring_count: int = Field(default=1, ge=0)
    accounts_per_ring: int = Field(default=4, gt=1)
    difficulty: DifficultyPreset = DifficultyPreset.STANDARD
    population: PopulationProfile = Field(default_factory=PopulationProfile)
    identity_lifecycle: IdentityLifecycleProfile = Field(
        default_factory=IdentityLifecycleProfile
    )
    commerce: CommerceProfile = Field(default_factory=CommerceProfile)
    calendar: CalendarProfile = Field(default_factory=CalendarProfile)
    abuse: AbuseProfile = Field(default_factory=AbuseProfile)
    diagnostics: DiagnosticProfile = Field(default_factory=DiagnosticProfile)
    start_at: AwareDatetime = datetime(2026, 1, 1, tzinfo=UTC)
    end_at: AwareDatetime = datetime(2026, 4, 1, tzinfo=UTC)

    @model_validator(mode="after")
    def validate_time_window(self) -> Self:
        if self.end_at - self.start_at < timedelta(days=14):
            raise ValueError("generation window must be at least 14 days")
        return self
