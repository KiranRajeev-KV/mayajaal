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


class PrevalencePreset(StrEnum):
    """Benchmark abuse-rarity configurations, independent of difficulty."""

    DEVELOPMENT = "development"
    RARE_ABUSE = "rare_abuse"


class DifficultyBundle(SchemaModel):
    """One coherent behavioural/topological-overlap configuration."""

    persona_weight_temperature: float = Field(gt=0.0)
    benign_sharing_multiplier: float = Field(gt=0.0)
    identity_lifecycle_multiplier: float = Field(gt=0.0)
    campaign_sharing_multiplier: float = Field(gt=0.0)
    campaign_low_and_slow_multiplier: float = Field(gt=0.0)
    seasonal_activity_multiplier: float = Field(gt=0.0)
    burst_activity_spread_fraction: float = Field(gt=0.0, le=1.0)


class DifficultyProfiles(SchemaModel):
    """Validated bundles for each named Mayajaal benchmark difficulty."""

    easy: DifficultyBundle = Field(
        default_factory=lambda: DifficultyBundle(
            persona_weight_temperature=0.80,
            benign_sharing_multiplier=0.70,
            identity_lifecycle_multiplier=0.70,
            campaign_sharing_multiplier=1.30,
            campaign_low_and_slow_multiplier=0.56,
            seasonal_activity_multiplier=0.75,
            burst_activity_spread_fraction=0.04,
        )
    )
    standard: DifficultyBundle = Field(
        default_factory=lambda: DifficultyBundle(
            persona_weight_temperature=1.00,
            benign_sharing_multiplier=1.00,
            identity_lifecycle_multiplier=1.00,
            campaign_sharing_multiplier=1.00,
            campaign_low_and_slow_multiplier=1.00,
            seasonal_activity_multiplier=1.00,
            burst_activity_spread_fraction=0.06,
        )
    )
    hard: DifficultyBundle = Field(
        default_factory=lambda: DifficultyBundle(
            persona_weight_temperature=1.20,
            benign_sharing_multiplier=1.25,
            identity_lifecycle_multiplier=1.35,
            campaign_sharing_multiplier=0.72,
            campaign_low_and_slow_multiplier=1.33,
            seasonal_activity_multiplier=1.10,
            burst_activity_spread_fraction=0.12,
        )
    )
    drift: DifficultyBundle = Field(
        default_factory=lambda: DifficultyBundle(
            persona_weight_temperature=1.15,
            benign_sharing_multiplier=1.20,
            identity_lifecycle_multiplier=1.25,
            campaign_sharing_multiplier=0.78,
            campaign_low_and_slow_multiplier=1.22,
            seasonal_activity_multiplier=1.15,
            burst_activity_spread_fraction=0.14,
        )
    )

    def for_preset(self, preset: DifficultyPreset) -> DifficultyBundle:
        """Return the bundle selected by the public difficulty name."""
        return getattr(self, preset.value)


class PopulationProfile(SchemaModel):
    """Controls the mix of ordinary people and benign shared contexts."""

    # ``None`` means population-scaled; a number, including zero, is an
    # explicit scenario override retained for compact fixtures and callers.
    benign_network_group_count: int | None = Field(default=None, ge=0)
    accounts_per_benign_network_group: int = Field(default=5, gt=1)
    households_per_thousand_ordinary_accounts: float = Field(default=35.0, ge=0.0)
    benign_network_groups_per_thousand_ordinary_accounts: float = Field(
        default=10.0, ge=0.0
    )
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

    def resolved_benign_network_group_count(self, ordinary_account_count: int) -> int:
        """Scale office/campus contexts unless an explicit count was requested."""
        if self.benign_network_group_count is not None:
            return self.benign_network_group_count
        return _scaled_context_count(
            ordinary_account_count,
            self.benign_network_groups_per_thousand_ordinary_accounts,
        )


def _scaled_context_count(population: int, per_thousand: float) -> int:
    """Round deterministic context counts while retaining small non-zero worlds."""
    if population <= 0 or per_thousand <= 0.0:
        return 0
    return max(1, round(population * per_thousand / 1_000.0))


class BenignSharingProfile(SchemaModel):
    """Base sharing rates for label-free household and network contexts."""

    household_device_probability: float = Field(default=0.55, ge=0.0, le=1.0)
    household_payment_probability: float = Field(default=0.28, ge=0.0, le=1.0)
    office_device_probability: float = Field(default=0.03, ge=0.0, le=1.0)
    office_payment_probability: float = Field(default=0.01, ge=0.0, le=1.0)


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

    low_and_slow_probability: float = Field(default=0.45, ge=0.0, le=1.0)
    partial_identity_sharing_probability: float = Field(default=0.68, ge=0.0, le=1.0)
    ring_size_variation: int = Field(default=2, ge=0)


class PrevalenceProfile(SchemaModel):
    """Mayajaal benchmark rarity, deliberately separate from behaviour difficulty."""

    preset: PrevalencePreset = PrevalencePreset.DEVELOPMENT
    target_labelled_account_rate: float | None = Field(default=None, gt=0.0, lt=1.0)
    target_tolerance: float = Field(default=0.01, ge=0.0, lt=1.0)
    strategy_weights: dict[str, float] = Field(
        default_factory=lambda: {"promo": 0.40, "refund": 0.30, "mixed": 0.30}
    )
    ring_sizes: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
    ring_size_weights: tuple[float, ...] = (0.10, 0.20, 0.28, 0.22, 0.14, 0.06)
    timeline_weights: dict[str, float] = Field(
        default_factory=lambda: {"early": 0.34, "middle": 0.33, "late": 0.33}
    )
    minimum_campaigns_per_timeline_bucket: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_strategy_weights(self) -> Self:
        allowed = {"promo", "refund", "mixed"}
        if set(self.strategy_weights) != allowed or any(
            weight <= 0.0 for weight in self.strategy_weights.values()
        ):
            raise ValueError(
                "strategy_weights must contain positive promo/refund/mixed weights"
            )
        if (
            len(self.ring_sizes) < 2
            or len(self.ring_sizes) != len(self.ring_size_weights)
            or any(size < 2 for size in self.ring_sizes)
            or any(weight <= 0.0 for weight in self.ring_size_weights)
            or not {2, 3}.issubset(self.ring_sizes)
        ):
            raise ValueError(
                "ring_sizes must include 2 and 3, with matching positive weights"
            )
        if set(self.timeline_weights) != {"early", "middle", "late"} or any(
            weight <= 0.0 for weight in self.timeline_weights.values()
        ):
            raise ValueError(
                "timeline_weights must contain positive early/middle/late weights"
            )
        return self

    def resolved_target_rate(self) -> float | None:
        """Return the explicit rate, or the rare benchmark default when requested."""
        if self.preset is PrevalencePreset.RARE_ABUSE:
            return self.target_labelled_account_rate or 0.0075
        return self.target_labelled_account_rate


class DiagnosticProfile(SchemaModel):
    """Internal plausibility guardrails, not claims about a real merchant."""

    min_variable_numeric_features: int = Field(default=5, ge=1)
    max_single_feature_perfect_separators: int = Field(default=0, ge=0)
    cutoff_fractions: tuple[float, float, float] = (0.25, 0.50, 1.00)
    expected_active_numeric_features: tuple[str, ...] = (
        "account_age_hours",
        "device_count",
        "ip_address_count",
        "payment_identity_count",
        "address_count",
        "shared_device_account_count",
        "shared_ip_account_count",
        "shared_payment_account_count",
        "shared_address_account_count",
        "max_identity_reuse_count",
        "identity_neighbour_count",
        "identity_component_account_count",
        "order_count",
        "total_order_value_paise",
        "promotion_redemption_count",
        "shared_promotion_account_count",
        "refund_requested_count",
        "refund_resolved_count",
        "refund_requested_order_rate",
    )
    intentionally_sparse_numeric_features: tuple[str, ...] = (
        "recent_shared_account_creation_count",
        "recent_shared_identity_event_count",
    )
    max_single_feature_auc: float = Field(default=0.95, ge=0.5, le=1.0)
    min_class_histogram_overlap: float = Field(default=0.05, ge=0.0, le=1.0)
    shap_top_feature_share_warning: float = Field(default=0.60, gt=0.0, le=1.0)
    min_cutoff_positive_samples: int = Field(default=50, ge=1)
    min_cutoff_negative_samples: int = Field(default=20, ge=1)
    max_identity_sharing_component_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    max_labelled_accounts_in_single_component_fraction: float = Field(
        default=0.25, gt=0.0, le=1.0
    )

    @model_validator(mode="after")
    def validate_feature_expectations(self) -> Self:
        expected = set(self.expected_active_numeric_features)
        sparse = set(self.intentionally_sparse_numeric_features)
        if expected & sparse:
            raise ValueError(
                "expected-active and intentionally sparse features must not overlap"
            )
        if (
            any(fraction <= 0.0 or fraction > 1.0 for fraction in self.cutoff_fractions)
            or tuple(sorted(self.cutoff_fractions)) != self.cutoff_fractions
            or self.cutoff_fractions[-1] != 1.0
        ):
            raise ValueError("cutoff_fractions must increase in (0, 1] and end at 1.0")
        return self


class ValidationProfile(SchemaModel):
    """Reproducible benchmark-run sizes; not production calibration settings."""

    multi_seed_count: int = Field(default=5, ge=1)
    small_account_count: int = Field(default=2_000, ge=20)
    full_account_count: int = Field(default=10_000, ge=100)
    shap_sample_count: int = Field(default=1_000, ge=1)


class GenerationProfile(SchemaModel):
    """Explicit control surface for world size, behaviour, and time range.

    The original count fields remain supported so existing callers can continue
    to construct profiles directly.  Nested profiles add behavioural diversity
    without exposing synthetic truth to downstream consumers.
    """

    seed: int = Field(ge=0)
    normal_account_count: int = Field(default=20, ge=0)
    shared_household_count: int | None = Field(default=None, ge=0)
    accounts_per_shared_household: int = Field(default=3, gt=0)
    promo_ring_count: int = Field(default=1, ge=0)
    refund_ring_count: int = Field(default=1, ge=0)
    mixed_ring_count: int = Field(default=1, ge=0)
    accounts_per_ring: int = Field(default=4, gt=1)
    difficulty: DifficultyPreset = DifficultyPreset.STANDARD
    difficulty_profiles: DifficultyProfiles = Field(default_factory=DifficultyProfiles)
    population: PopulationProfile = Field(default_factory=PopulationProfile)
    benign_sharing: BenignSharingProfile = Field(default_factory=BenignSharingProfile)
    identity_lifecycle: IdentityLifecycleProfile = Field(
        default_factory=IdentityLifecycleProfile
    )
    commerce: CommerceProfile = Field(default_factory=CommerceProfile)
    calendar: CalendarProfile = Field(default_factory=CalendarProfile)
    abuse: AbuseProfile = Field(default_factory=AbuseProfile)
    prevalence: PrevalenceProfile = Field(default_factory=PrevalenceProfile)
    diagnostics: DiagnosticProfile = Field(default_factory=DiagnosticProfile)
    validation: ValidationProfile = Field(default_factory=ValidationProfile)
    start_at: AwareDatetime = datetime(2026, 1, 1, tzinfo=UTC)
    end_at: AwareDatetime = datetime(2026, 4, 1, tzinfo=UTC)

    @model_validator(mode="after")
    def validate_time_window(self) -> Self:
        if self.end_at - self.start_at < timedelta(days=14):
            raise ValueError("generation window must be at least 14 days")
        return self

    @property
    def active_difficulty(self) -> DifficultyBundle:
        """Resolved behavioural overlap settings for this deterministic run."""
        return self.difficulty_profiles.for_preset(self.difficulty)

    @property
    def effective_persona_weights(self) -> dict[str, float]:
        """Flatten or sharpen the configured persona mix by difficulty only."""
        temperature = self.active_difficulty.persona_weight_temperature
        return {
            name: weight ** (1.0 / temperature)
            for name, weight in self.population.persona_weights.items()
        }

    def resolved_shared_household_count(self) -> int:
        """Scale households with ordinary population unless explicitly overridden."""
        if self.shared_household_count is not None:
            return self.shared_household_count
        return _scaled_context_count(
            self.normal_account_count,
            self.population.households_per_thousand_ordinary_accounts,
        )
