"""Hidden abuse-campaign plans with heterogeneous, partial identity sharing."""

from dataclasses import dataclass
from enum import StrEnum

from numpy.random import Generator

from mayajaal.schemas import AbuseType

from .profile import AbuseProfile, DifficultyBundle


class AbuseStrategy(StrEnum):
    """Campaign tactics observable through the existing event vocabulary."""

    PROMO_FARM = "promo"
    REFUND_ABUSE = "refund"
    MIXED = "mixed"


@dataclass(frozen=True)
class CampaignPlan:
    """Private campaign parameters; they are never exported with the world."""

    strategy: AbuseStrategy
    cluster_id: str
    shared_device: bool
    shared_ip: bool
    shared_payment: bool
    shared_address: bool
    low_and_slow: bool
    timeline_bucket: str | None
    activity_center_fraction: float
    activity_spread_fraction: float

    @property
    def abuse_types(self) -> tuple[AbuseType, ...]:
        if self.strategy is AbuseStrategy.PROMO_FARM:
            return (AbuseType.PROMOTION_ABUSE, AbuseType.ACCOUNT_FARMING)
        if self.strategy is AbuseStrategy.REFUND_ABUSE:
            return (AbuseType.REFUND_ABUSE, AbuseType.ADDRESS_REUSE)
        return (AbuseType.PROMOTION_ABUSE, AbuseType.REFUND_ABUSE)


def plan_campaign(
    strategy: AbuseStrategy,
    cluster_id: str,
    profile: AbuseProfile,
    difficulty: DifficultyBundle,
    rng: Generator,
    *,
    timeline_bucket: str | None = None,
    activity_center_fraction: float | None = None,
) -> CampaignPlan:
    """Create one partial-sharing campaign without creating an obvious clique."""
    chance = min(
        profile.partial_identity_sharing_probability
        * difficulty.campaign_sharing_multiplier,
        1.0,
    )
    preferred: dict[AbuseStrategy, tuple[float, float, float, float]] = {
        AbuseStrategy.PROMO_FARM: (0.78, 0.48, 0.88, 0.28),
        AbuseStrategy.REFUND_ABUSE: (0.34, 0.42, 0.36, 0.82),
        AbuseStrategy.MIXED: (0.55, 0.52, 0.62, 0.55),
    }
    device, ip_address, payment, address = preferred[strategy]
    probabilities = tuple(
        min(value * chance / 0.68, 0.96)
        for value in (
            device,
            ip_address,
            payment,
            address,
        )
    )
    selected = tuple(bool(rng.random() < value) for value in probabilities)
    if not any(selected):
        selected = (False, False, True, False)
    return CampaignPlan(
        strategy=strategy,
        cluster_id=cluster_id,
        shared_device=selected[0],
        shared_ip=selected[1],
        shared_payment=selected[2],
        shared_address=selected[3],
        low_and_slow=bool(
            rng.random()
            < min(
                profile.low_and_slow_probability
                * difficulty.campaign_low_and_slow_multiplier,
                1.0,
            )
        ),
        timeline_bucket=timeline_bucket,
        activity_center_fraction=(
            activity_center_fraction
            if activity_center_fraction is not None
            else float(rng.uniform(0.18, 0.82))
        ),
        activity_spread_fraction=(difficulty.burst_activity_spread_fraction),
    )
