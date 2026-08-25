"""Small composable feature extractors over a temporal graph snapshot."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from mayajaal.graph import GraphNodeType, GraphRelationshipType
from mayajaal.schemas import EventType

from .models import FeatureDefinition, FeatureKind, FeatureValue
from .temporal import AccountGraphIndex, IdentityKey, TemporalFeatureSnapshot

MISSING_CATEGORY = "__missing__"
RECENT_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class FeatureContext:
    """Inputs shared by every extractor for one account and cutoff."""

    account_id: str
    snapshot: TemporalFeatureSnapshot
    index: AccountGraphIndex

    @property
    def identities(self) -> set[IdentityKey]:
        return self.index.account_identities.get(self.account_id, set())

    @property
    def orders(self) -> set[str]:
        return self.index.account_orders.get(self.account_id, set())


class FeatureExtractor(Protocol):
    """A family of namespaced, independent feature definitions."""

    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]: ...

    def extract(self, context: FeatureContext) -> dict[str, FeatureValue]: ...


def _identity_count(
    context: FeatureContext, relationship_type: GraphRelationshipType
) -> int:
    return sum(1 for kind, _ in context.identities if kind is relationship_type)


def _peer_accounts(
    context: FeatureContext, relationship_type: GraphRelationshipType
) -> set[str]:
    peers: set[str] = set()
    for identity in context.identities:
        if identity[0] is relationship_type:
            peers.update(context.index.identity_accounts.get(identity, set()))
    peers.discard(context.account_id)
    return peers


@dataclass(frozen=True)
class AccountAgeExtractor:
    definitions: tuple[FeatureDefinition, ...] = (
        FeatureDefinition(
            "account_age_hours",
            FeatureKind.NUMERIC,
            "Hours since account creation, using the account creation fact at cutoff.",
        ),
    )

    def extract(self, context: FeatureContext) -> dict[str, FeatureValue]:
        age = context.snapshot.cutoff - context.snapshot.account_created_at(
            context.account_id
        )
        return {"account_age_hours": max(age.total_seconds() / 3600.0, 0.0)}


@dataclass(frozen=True)
class IdentityReuseExtractor:
    definitions: tuple[FeatureDefinition, ...] = (
        FeatureDefinition(
            "device_count",
            FeatureKind.NUMERIC,
            "Distinct devices used by the account by cutoff.",
        ),
        FeatureDefinition(
            "ip_address_count",
            FeatureKind.NUMERIC,
            "Distinct IP addresses seen for the account by cutoff.",
        ),
        FeatureDefinition(
            "payment_identity_count",
            FeatureKind.NUMERIC,
            "Distinct payment identities attached by cutoff.",
        ),
        FeatureDefinition(
            "address_count",
            FeatureKind.NUMERIC,
            "Distinct order shipping addresses used by cutoff.",
        ),
        FeatureDefinition(
            "shared_device_account_count",
            FeatureKind.NUMERIC,
            "Other accounts sharing any observed device by cutoff.",
        ),
        FeatureDefinition(
            "shared_ip_account_count",
            FeatureKind.NUMERIC,
            "Other accounts sharing any observed IP address by cutoff.",
        ),
        FeatureDefinition(
            "shared_payment_account_count",
            FeatureKind.NUMERIC,
            "Other accounts sharing any payment identity by cutoff.",
        ),
        FeatureDefinition(
            "shared_address_account_count",
            FeatureKind.NUMERIC,
            "Other accounts sharing any order shipping address by cutoff.",
        ),
        FeatureDefinition(
            "max_identity_reuse_count",
            FeatureKind.NUMERIC,
            "Largest account count on one device, IP, payment, or address identity by cutoff.",
        ),
        FeatureDefinition(
            "identity_neighbour_count",
            FeatureKind.NUMERIC,
            "Unique other accounts one identity hop away by cutoff.",
        ),
        FeatureDefinition(
            "identity_component_account_count",
            FeatureKind.NUMERIC,
            "Accounts in the identity-connected component by cutoff.",
        ),
    )

    def extract(self, context: FeatureContext) -> dict[str, FeatureValue]:
        peers_by_type = {
            "shared_device_account_count": _peer_accounts(
                context, GraphRelationshipType.USED_DEVICE
            ),
            "shared_ip_account_count": _peer_accounts(
                context, GraphRelationshipType.SEEN_FROM
            ),
            "shared_payment_account_count": _peer_accounts(
                context, GraphRelationshipType.PAID_WITH
            ),
            "shared_address_account_count": _peer_accounts(
                context, GraphRelationshipType.SHIPPED_TO
            ),
        }
        all_peers: set[str] = set()
        for peers in peers_by_type.values():
            all_peers.update(peers)
        max_reuse = max(
            (
                len(context.index.identity_accounts[identity])
                for identity in context.identities
            ),
            default=0,
        )
        return {
            "device_count": float(
                _identity_count(context, GraphRelationshipType.USED_DEVICE)
            ),
            "ip_address_count": float(
                _identity_count(context, GraphRelationshipType.SEEN_FROM)
            ),
            "payment_identity_count": float(
                _identity_count(context, GraphRelationshipType.PAID_WITH)
            ),
            "address_count": float(
                _identity_count(context, GraphRelationshipType.SHIPPED_TO)
            ),
            **{name: float(len(peers)) for name, peers in peers_by_type.items()},
            "max_identity_reuse_count": float(max_reuse),
            "identity_neighbour_count": float(len(all_peers)),
            "identity_component_account_count": float(
                len(context.index.connected_accounts(context.account_id))
            ),
        }


@dataclass(frozen=True)
class CommerceExtractor:
    definitions: tuple[FeatureDefinition, ...] = (
        FeatureDefinition(
            "order_count",
            FeatureKind.NUMERIC,
            "Orders placed by the account by cutoff.",
        ),
        FeatureDefinition(
            "total_order_value_paise",
            FeatureKind.NUMERIC,
            "Sum of placed order totals by cutoff.",
        ),
        FeatureDefinition(
            "promotion_redemption_count",
            FeatureKind.NUMERIC,
            "Promotion redemption events for the account's orders by cutoff.",
        ),
        FeatureDefinition(
            "shared_promotion_account_count",
            FeatureKind.NUMERIC,
            "Other accounts using a promotion also used by this account by cutoff.",
        ),
        FeatureDefinition(
            "refund_requested_count",
            FeatureKind.NUMERIC,
            "Refund request events for the account's orders by cutoff.",
        ),
        FeatureDefinition(
            "refund_resolved_count",
            FeatureKind.NUMERIC,
            "Refund resolution events already observed by cutoff.",
        ),
        FeatureDefinition(
            "refund_requested_order_rate",
            FeatureKind.NUMERIC,
            "Fraction of placed orders with a refund request by cutoff.",
        ),
    )

    def extract(self, context: FeatureContext) -> dict[str, FeatureValue]:
        total_value = 0.0
        promotions: set[str] = set()
        refund_requested_orders: set[str] = set()
        requested_count = 0
        resolved_count = 0
        for order_id in context.orders:
            node = context.snapshot.nodes[(GraphNodeType.ORDER, order_id)]
            total_paise = node.properties["total_paise"]
            if not isinstance(total_paise, int):
                raise TypeError("Order.total_paise must be an int")
            total_value += float(total_paise)
            promotions.update(context.index.order_promotions.get(order_id, set()))
            for event in context.index.order_refund_events.get(order_id, ()):
                if event.event_type == EventType.REFUND_REQUESTED.value:
                    requested_count += 1
                    refund_requested_orders.add(order_id)
                elif event.event_type == EventType.REFUND_RESOLVED.value:
                    resolved_count += 1

        promotion_accounts: set[str] = set()
        for promotion_id in promotions:
            promotion_accounts.update(
                context.index.promotion_accounts.get(promotion_id, set())
            )
        promotion_accounts.discard(context.account_id)
        order_count = len(context.orders)
        return {
            "order_count": float(order_count),
            "total_order_value_paise": total_value,
            "promotion_redemption_count": float(
                sum(
                    len(context.index.order_promotions.get(order_id, set()))
                    for order_id in context.orders
                )
            ),
            "shared_promotion_account_count": float(len(promotion_accounts)),
            "refund_requested_count": float(requested_count),
            "refund_resolved_count": float(resolved_count),
            "refund_requested_order_rate": (
                float(len(refund_requested_orders)) / float(order_count)
                if order_count
                else 0.0
            ),
        }


@dataclass(frozen=True)
class VelocityExtractor:
    definitions: tuple[FeatureDefinition, ...] = (
        FeatureDefinition(
            "recent_shared_account_creation_count",
            FeatureKind.NUMERIC,
            "Identity-connected peer accounts created in the preceding 24 hours.",
        ),
        FeatureDefinition(
            "recent_shared_identity_event_count",
            FeatureKind.NUMERIC,
            "Peer identity-link events on this account's identities in the preceding 24 hours.",
        ),
    )

    def extract(self, context: FeatureContext) -> dict[str, FeatureValue]:
        window_start = context.snapshot.cutoff - RECENT_WINDOW
        peers = context.index.connected_accounts(context.account_id) - {
            context.account_id
        }
        recent_created = sum(
            1
            for peer_id in peers
            if window_start
            <= context.snapshot.account_created_at(peer_id)
            <= context.snapshot.cutoff
        )
        recent_identity_events = sum(
            1
            for identity in context.identities
            for event in context.index.identity_events_by_identity.get(identity, ())
            if event.source_canonical_id != context.account_id
            and window_start <= event.event_time <= context.snapshot.cutoff
        )
        return {
            "recent_shared_account_creation_count": float(recent_created),
            "recent_shared_identity_event_count": float(recent_identity_events),
        }


def _latest_identity_property(
    context: FeatureContext,
    relationship_type: GraphRelationshipType,
    property_name: str,
) -> str:
    candidates = [
        event
        for event in context.index.identity_events
        if event.source_canonical_id == context.account_id
        and event.relationship_type is relationship_type
    ]
    if not candidates:
        return MISSING_CATEGORY
    latest = max(candidates, key=lambda event: (event.event_time, event.event_id))
    node_type = {
        GraphRelationshipType.USED_DEVICE: GraphNodeType.DEVICE,
        GraphRelationshipType.PAID_WITH: GraphNodeType.PAYMENT_IDENTITY,
    }[relationship_type]
    value = context.snapshot.nodes[
        (node_type, latest.target_canonical_id)
    ].properties.get(property_name)
    return str(value) if value is not None else MISSING_CATEGORY


@dataclass(frozen=True)
class CategoricalContextExtractor:
    definitions: tuple[FeatureDefinition, ...] = (
        FeatureDefinition(
            "latest_device_platform",
            FeatureKind.CATEGORICAL,
            "Platform on the latest device observation by cutoff.",
        ),
        FeatureDefinition(
            "latest_device_type",
            FeatureKind.CATEGORICAL,
            "Device type on the latest device observation by cutoff.",
        ),
        FeatureDefinition(
            "latest_payment_method",
            FeatureKind.CATEGORICAL,
            "Payment method on the latest attached identity by cutoff.",
        ),
        FeatureDefinition(
            "latest_promotion_code",
            FeatureKind.CATEGORICAL,
            "Promotion code on the latest redemption by cutoff.",
        ),
        FeatureDefinition(
            "latest_shipping_country_code",
            FeatureKind.CATEGORICAL,
            "Country code on the latest placed order's shipping address by cutoff.",
        ),
    )

    def extract(self, context: FeatureContext) -> dict[str, FeatureValue]:
        promotion_events = (
            event
            for order_id in context.orders
            for event in context.index.order_promotion_events.get(order_id, ())
        )
        latest_promotion = max(
            promotion_events,
            key=lambda event: (event.event_time, event.event_id),
            default=None,
        )
        latest_order = max(
            context.index.account_placed_events.get(context.account_id, ()),
            key=lambda event: (event.event_time, event.event_id),
            default=None,
        )

        promotion_code = MISSING_CATEGORY
        if latest_promotion is not None:
            promotion_code = str(
                context.snapshot.nodes[
                    (GraphNodeType.PROMOTION, latest_promotion.target_canonical_id)
                ].properties["code"]
            )
        country_code = MISSING_CATEGORY
        if latest_order is not None:
            addresses = context.index.order_addresses.get(
                latest_order.target_canonical_id, set()
            )
            if addresses:
                address_id = min(addresses)
                country_code = str(
                    context.snapshot.nodes[
                        (GraphNodeType.ADDRESS, address_id)
                    ].properties["country_code"]
                )
        return {
            "latest_device_platform": _latest_identity_property(
                context, GraphRelationshipType.USED_DEVICE, "platform"
            ),
            "latest_device_type": _latest_identity_property(
                context, GraphRelationshipType.USED_DEVICE, "device_type"
            ),
            "latest_payment_method": _latest_identity_property(
                context, GraphRelationshipType.PAID_WITH, "method"
            ),
            "latest_promotion_code": promotion_code,
            "latest_shipping_country_code": country_code,
        }


DEFAULT_EXTRACTORS: tuple[FeatureExtractor, ...] = (
    AccountAgeExtractor(),
    IdentityReuseExtractor(),
    CommerceExtractor(),
    VelocityExtractor(),
    CategoricalContextExtractor(),
)
