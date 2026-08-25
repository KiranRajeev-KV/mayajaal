"""Hidden abuse-campaign plans with heterogeneous, partial identity sharing."""

from dataclasses import dataclass
from enum import StrEnum

from numpy.random import Generator

from mayajaal.schemas import AbuseType

from .profile import AbuseProfile, DifficultyPreset


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
    warmup_orders: int
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
    difficulty: DifficultyPreset,
    rng: Generator,
) -> CampaignPlan:
    """Create one partial-sharing campaign without creating an obvious clique."""
    chance = profile.partial_identity_sharing_probability
    if difficulty is DifficultyPreset.EASY:
        chance = min(chance + 0.22, 1.0)
    elif difficulty in {DifficultyPreset.HARD, DifficultyPreset.DRIFT}:
        chance = max(chance - 0.18, 0.15)
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
        low_and_slow=bool(rng.random() < profile.low_and_slow_probability),
        warmup_orders=int(
            rng.integers(profile.min_warmup_orders, profile.max_warmup_orders + 1)
        ),
        activity_center_fraction=float(rng.uniform(0.18, 0.82)),
        activity_spread_fraction=(
            0.34
            if difficulty is DifficultyPreset.DRIFT
            else 0.18
            if difficulty is DifficultyPreset.HARD
            else 0.06
        ),
    )
