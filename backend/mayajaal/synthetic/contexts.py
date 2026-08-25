"""Label-independent contexts that create realistic benign sharing."""

from dataclasses import dataclass
from enum import StrEnum

from numpy.random import Generator


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


def household_plan() -> ContextSharingPlan:
    """Families commonly share home address/network, less often devices/cards."""
    return ContextSharingPlan(ContextKind.HOUSEHOLD, True, True, 0.55, 0.28)


def office_network_plan() -> ContextSharingPlan:
    """Office/campus members share NAT IPs but retain personal identities."""
    return ContextSharingPlan(ContextKind.OFFICE_NETWORK, False, True, 0.03, 0.01)


def shares(probability: float, rng: Generator) -> bool:
    """Make one deterministic context-membership decision."""
    return bool(rng.random() < probability)
