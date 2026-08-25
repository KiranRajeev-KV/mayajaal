"""Storage-independent representation of the identity graph."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class GraphNodeType(StrEnum):
    """Neo4j labels used by the derived identity graph."""

    ACCOUNT = "Account"
    DEVICE = "Device"
    IP_ADDRESS = "IPAddress"
    PAYMENT_IDENTITY = "PaymentIdentity"
    ADDRESS = "Address"
    ORDER = "Order"
    PROMOTION = "Promotion"
    REFUND = "Refund"


class GraphRelationshipType(StrEnum):
    """The supported event-backed relationship types."""

    USED_DEVICE = "USED_DEVICE"
    SEEN_FROM = "SEEN_FROM"
    PAID_WITH = "PAID_WITH"
    PLACED = "PLACED"
    SHIPPED_TO = "SHIPPED_TO"
    USED_PROMO = "USED_PROMO"
    HAS_REFUND = "HAS_REFUND"


@dataclass(frozen=True)
class GraphNode:
    """One canonical Neo4j node, independent of driver-specific values."""

    node_type: GraphNodeType
    canonical_id: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class GraphRelationship:
    """One immutable event fact represented as a Neo4j relationship."""

    relationship_type: GraphRelationshipType
    source_type: GraphNodeType
    source_canonical_id: str
    target_type: GraphNodeType
    target_canonical_id: str
    event_id: str
    event_type: str
    event_time: datetime


@dataclass(frozen=True)
class GraphProjection:
    """The deterministic graph payload supplied to a graph repository."""

    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]
