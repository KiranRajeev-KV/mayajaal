"""Label-independent contexts that create realistic benign sharing."""

from dataclasses import dataclass
from enum import StrEnum

from numpy.random import Generator

from .profile import BenignSharingProfile


class ContextKind(StrEnum):
    """Hidden social or network contexts, independent of abuse campaigns."""

    HOUSEHOLD = "household"
    OFFICE_NETWORK = "office_network"


@dataclass(frozen=True)
class ContextSharingPlan:
    """Which canonical identity types a benign context may share."""

    kind: ContextKind
    share_address: bool
    share_ip: bool
    share_device_probability: float
    share_payment_probability: float


def household_plan(
    profile: BenignSharingProfile, multiplier: float
) -> ContextSharingPlan:
    """Families commonly share home address/network, less often devices/cards."""
    return ContextSharingPlan(
        ContextKind.HOUSEHOLD,
        True,
        True,
        min(profile.household_device_probability * multiplier, 1.0),
        min(profile.household_payment_probability * multiplier, 1.0),
    )


def office_network_plan(
    profile: BenignSharingProfile, multiplier: float
) -> ContextSharingPlan:
    """Office/campus members share NAT IPs but retain personal identities."""
    return ContextSharingPlan(
        ContextKind.OFFICE_NETWORK,
        False,
        True,
        min(profile.office_device_probability * multiplier, 1.0),
        min(profile.office_payment_probability * multiplier, 1.0),
    )


def shares(probability: float, rng: Generator) -> bool:
    """Make one deterministic context-membership decision."""
    return bool(rng.random() < probability)
