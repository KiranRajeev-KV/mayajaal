"""In-memory cutoff views over the storage-independent graph projection."""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

from mayajaal.graph import (
    GraphNode,
    GraphNodeType,
    GraphProjection,
    GraphRelationship,
    GraphRelationshipType,
)

IdentityKey = tuple[GraphRelationshipType, str]


@dataclass(frozen=True)
class TemporalFeatureSnapshot:
    """A projection view containing only event facts known by a cutoff."""

    cutoff: datetime
    nodes: dict[tuple[GraphNodeType, str], GraphNode]
    relationships: tuple[GraphRelationship, ...]

    def account_created_at(self, account_id: str) -> datetime:
        """Return a target account's creation fact, rejecting future accounts."""
        node = self.nodes[(GraphNodeType.ACCOUNT, account_id)]
        created_at = node.properties["created_at"]
        if not isinstance(created_at, datetime):
            raise TypeError("Account.created_at must be a datetime")
        if created_at > self.cutoff:
            raise ValueError(
                "cannot extract features for an account not known at cutoff"
            )
        return created_at


class TemporalFeatureGraph:
    """Create repeatable cutoff snapshots without querying mutable graph state."""

    def __init__(self, projection: GraphProjection) -> None:
        self._nodes = {
            (node.node_type, node.canonical_id): node for node in projection.nodes
        }
        self._relationships = tuple(
            sorted(
                projection.relationships,
                key=lambda item: (
                    item.event_time,
                    item.event_id,
                    item.relationship_type,
                ),
            )
        )

    def snapshot_at(self, cutoff: datetime) -> TemporalFeatureSnapshot:
        """Match repository cutoff semantics: include only ``event_time <= cutoff``."""
        return TemporalFeatureSnapshot(
            cutoff=cutoff,
            nodes=self._nodes,
            relationships=tuple(
                relationship
                for relationship in self._relationships
                if relationship.event_time <= cutoff
            ),
        )


@dataclass(frozen=True)
class AccountGraphIndex:
    """Account/identity and commerce indexes derived solely from a snapshot."""

    account_identities: dict[str, set[IdentityKey]]
    identity_accounts: dict[IdentityKey, set[str]]
    account_orders: dict[str, set[str]]
    order_addresses: dict[str, set[str]]
    order_promotions: dict[str, set[str]]
    promotion_accounts: dict[str, set[str]]
    order_refund_events: dict[str, tuple[GraphRelationship, ...]]
    identity_events: tuple[GraphRelationship, ...]
    identity_events_by_identity: dict[IdentityKey, tuple[GraphRelationship, ...]]
    account_placed_events: dict[str, tuple[GraphRelationship, ...]]
    order_promotion_events: dict[str, tuple[GraphRelationship, ...]]

    def connected_accounts(self, account_id: str) -> set[str]:
        """Return the identity-connected account component present at cutoff."""
        visited = {account_id}
        pending: deque[str] = deque([account_id])
        while pending:
            current = pending.popleft()
            for identity in self.account_identities.get(current, set()):
                for peer in self.identity_accounts[identity]:
                    if peer not in visited:
                        visited.add(peer)
                        pending.append(peer)
        return visited


def build_account_graph_index(snapshot: TemporalFeatureSnapshot) -> AccountGraphIndex:
    """Build feature indexes from immutable relationships in one cutoff view."""
    account_identities: defaultdict[str, set[IdentityKey]] = defaultdict(set)
    identity_accounts: defaultdict[IdentityKey, set[str]] = defaultdict(set)
    account_orders: defaultdict[str, set[str]] = defaultdict(set)
    order_addresses: defaultdict[str, set[str]] = defaultdict(set)
    order_promotions: defaultdict[str, set[str]] = defaultdict(set)
    refund_events: defaultdict[str, list[GraphRelationship]] = defaultdict(list)
    identity_events: list[GraphRelationship] = []
    identity_events_by_identity: defaultdict[IdentityKey, list[GraphRelationship]] = (
        defaultdict(list)
    )
    account_placed_events: defaultdict[str, list[GraphRelationship]] = defaultdict(list)
    order_promotion_events: defaultdict[str, list[GraphRelationship]] = defaultdict(
        list
    )

    identity_relationships = {
        GraphRelationshipType.USED_DEVICE,
        GraphRelationshipType.SEEN_FROM,
        GraphRelationshipType.PAID_WITH,
    }
    for relationship in snapshot.relationships:
        if relationship.relationship_type in identity_relationships:
            identity = (
                relationship.relationship_type,
                relationship.target_canonical_id,
            )
            account_identities[relationship.source_canonical_id].add(identity)
            identity_accounts[identity].add(relationship.source_canonical_id)
            identity_events.append(relationship)
            identity_events_by_identity[identity].append(relationship)
        elif relationship.relationship_type is GraphRelationshipType.PLACED:
            account_orders[relationship.source_canonical_id].add(
                relationship.target_canonical_id
            )
            account_placed_events[relationship.source_canonical_id].append(relationship)
        elif relationship.relationship_type is GraphRelationshipType.SHIPPED_TO:
            order_addresses[relationship.source_canonical_id].add(
                relationship.target_canonical_id
            )
        elif relationship.relationship_type is GraphRelationshipType.USED_PROMO:
            order_promotions[relationship.source_canonical_id].add(
                relationship.target_canonical_id
            )
            order_promotion_events[relationship.source_canonical_id].append(
                relationship
            )
        elif relationship.relationship_type is GraphRelationshipType.HAS_REFUND:
            refund_events[relationship.source_canonical_id].append(relationship)

    # Shipping identities are event-backed as part of ORDER_PLACED facts.
    for account_id, orders in account_orders.items():
        for order_id in orders:
            for address_id in order_addresses.get(order_id, set()):
                identity = (GraphRelationshipType.SHIPPED_TO, address_id)
                account_identities[account_id].add(identity)
                identity_accounts[identity].add(account_id)

    promotion_accounts: defaultdict[str, set[str]] = defaultdict(set)
    for account_id, orders in account_orders.items():
        for order_id in orders:
            for promotion_id in order_promotions.get(order_id, set()):
                promotion_accounts[promotion_id].add(account_id)

    return AccountGraphIndex(
        account_identities=dict(account_identities),
        identity_accounts=dict(identity_accounts),
        account_orders=dict(account_orders),
        order_addresses=dict(order_addresses),
        order_promotions=dict(order_promotions),
        promotion_accounts=dict(promotion_accounts),
        order_refund_events={
            order_id: tuple(
                sorted(events, key=lambda event: (event.event_time, event.event_id))
            )
            for order_id, events in refund_events.items()
        },
        identity_events=tuple(identity_events),
        identity_events_by_identity={
            identity: tuple(
                sorted(events, key=lambda event: (event.event_time, event.event_id))
            )
            for identity, events in identity_events_by_identity.items()
        },
        account_placed_events={
            account_id: tuple(
                sorted(events, key=lambda event: (event.event_time, event.event_id))
            )
            for account_id, events in account_placed_events.items()
        },
        order_promotion_events={
            order_id: tuple(
                sorted(events, key=lambda event: (event.event_time, event.event_id))
            )
            for order_id, events in order_promotion_events.items()
        },
    )
