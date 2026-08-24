"""The canonical event contract shared by graph, ML, and simulation pipelines."""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from .common import AwareDatetime, SchemaModel
from .ids import (
    AccountId,
    AddressId,
    DeviceId,
    EventId,
    IPAddressId,
    OrderId,
    PaymentIdentityId,
    PromotionId,
    RefundId,
)


class EventType(StrEnum):
    """Supported facts that can create or update identity-graph relationships."""

    ACCOUNT_CREATED = "account_created"
    DEVICE_SEEN = "device_seen"
    IP_SEEN = "ip_seen"
    PAYMENT_ATTACHED = "payment_attached"
    ORDER_PLACED = "order_placed"
    PROMOTION_REDEEMED = "promotion_redeemed"
    REFUND_REQUESTED = "refund_requested"
    REFUND_RESOLVED = "refund_resolved"


class AbuseType(StrEnum):
    """Synthetic abuse categories used solely for offline evaluation."""

    PROMOTION_ABUSE = "promotion_abuse"
    REFUND_ABUSE = "refund_abuse"
    ACCOUNT_FARMING = "account_farming"
    PAYMENT_CYCLING = "payment_cycling"
    ADDRESS_REUSE = "address_reuse"


class SyntheticEventLabels(SchemaModel):
    """Optional synthetic ground truth; never part of production entity models."""

    is_coordinated_abuse: bool
    abuse_types: tuple[AbuseType, ...] = ()
    coordination_cluster_id: str | None = Field(
        default=None, min_length=1, max_length=128
    )

    @model_validator(mode="after")
    def validate_label_consistency(self) -> Self:
        if self.is_coordinated_abuse and not self.abuse_types:
            raise ValueError("abusive events must include at least one abuse type")
        if not self.is_coordinated_abuse and (
            self.abuse_types or self.coordination_cluster_id is not None
        ):
            raise ValueError("non-abusive events cannot have abuse labels or a cluster")
        return self


class Event(SchemaModel):
    """One immutable observed fact, with explicit graph IDs and optional synthetic labels."""

    id: EventId
    event_type: EventType
    occurred_at: AwareDatetime
    ingested_at: AwareDatetime
    account_id: AccountId
    device_id: DeviceId | None = None
    ip_address_id: IPAddressId | None = None
    payment_identity_id: PaymentIdentityId | None = None
    order_id: OrderId | None = None
    address_id: AddressId | None = None
    promotion_id: PromotionId | None = None
    refund_id: RefundId | None = None
    synthetic_labels: SyntheticEventLabels | None = None

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        if self.ingested_at < self.occurred_at:
            raise ValueError("ingested_at cannot be earlier than occurred_at")

        required_ids: dict[EventType, tuple[str, ...]] = {
            EventType.ACCOUNT_CREATED: (),
            EventType.DEVICE_SEEN: ("device_id",),
            EventType.IP_SEEN: ("ip_address_id",),
            EventType.PAYMENT_ATTACHED: ("payment_identity_id",),
            EventType.ORDER_PLACED: ("order_id", "address_id"),
            EventType.PROMOTION_REDEEMED: ("order_id", "promotion_id"),
            EventType.REFUND_REQUESTED: ("order_id", "refund_id"),
            EventType.REFUND_RESOLVED: ("order_id", "refund_id"),
        }
        missing = [
            name
            for name in required_ids[self.event_type]
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"{self.event_type.value} requires {', '.join(missing)}")
        return self
