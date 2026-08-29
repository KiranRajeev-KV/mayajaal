"""Build the graph payload without importing Neo4j or fraud labels."""

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from mayajaal.resolution import ResolutionBundle, ResolutionEntityType
from mayajaal.schemas import Event, EventType
from mayajaal.synthetic.world import SyntheticWorld

from .models import (
    GraphNode,
    GraphNodeType,
    GraphProjection,
    GraphRelationship,
    GraphRelationshipType,
)


def _canonical_id_map(
    resolution: ResolutionBundle,
) -> dict[ResolutionEntityType, dict[UUID, UUID]]:
    return {
        entity_type: {
            result.raw_entity_id: result.canonical_entity_id
            for result in resolution.results
            if result.entity_type is entity_type
        }
        for entity_type in ResolutionEntityType
    }


def _canonical(
    raw_id: UUID,
    entity_type: ResolutionEntityType,
    resolution: dict[ResolutionEntityType, dict[UUID, UUID]],
) -> str:
    """Return a resolved ID, retaining the raw ID if no result exists."""
    return str(resolution[entity_type].get(raw_id, raw_id))


def _properties(**values: object) -> dict[str, object]:
    """Neo4j properties cannot be null; omit optional values instead."""
    return {key: value for key, value in values.items() if value is not None}


class _HasId(Protocol):
    @property
    def id(self) -> UUID: ...


def _canonical_nodes[EntityT: _HasId](
    entities: Iterable[EntityT],
    entity_type: ResolutionEntityType,
    node_type: GraphNodeType,
    resolution: dict[ResolutionEntityType, dict[UUID, UUID]],
    properties: Callable[[EntityT], dict[str, object]],
) -> list[GraphNode]:
    """Choose the stable lowest raw record for each canonical identity node."""
    selected: dict[str, EntityT] = {}
    for entity in sorted(entities, key=lambda item: str(item.id)):
        canonical_id = _canonical(entity.id, entity_type, resolution)
        selected.setdefault(canonical_id, entity)
    return [
        GraphNode(
            node_type=node_type,
            canonical_id=canonical_id,
            properties={"canonical_id": canonical_id, **properties(entity)},
        )
        for canonical_id, entity in sorted(selected.items())
    ]


def _relationship(
    relationship_type: GraphRelationshipType,
    source_type: GraphNodeType,
    source_id: str,
    target_type: GraphNodeType,
    target_id: str,
    event_id: UUID,
    event_type: EventType,
    event_time: datetime,
) -> GraphRelationship:
    return GraphRelationship(
        relationship_type=relationship_type,
        source_type=source_type,
        source_canonical_id=source_id,
        target_type=target_type,
        target_canonical_id=target_id,
        event_id=str(event_id),
        event_type=event_type.value,
        event_time=event_time,
    )


def build_graph_projection(
    world: SyntheticWorld, resolution: ResolutionBundle
) -> GraphProjection:
    """Project resolved entities and immutable event facts into the graph.

    ``Event.synthetic_labels`` is deliberately not read.  Each relationship is
    keyed by its source event ID so it can be merged idempotently and queried
    safely with ``event_time <= T``.
    """
    resolved = _canonical_id_map(resolution)
    nodes: list[GraphNode] = [
        *[
            GraphNode(
                node_type=GraphNodeType.ACCOUNT,
                canonical_id=str(account.id),
                properties=_properties(
                    canonical_id=str(account.id),
                    created_at=account.created_at,
                ),
            )
            for account in sorted(world.accounts, key=lambda item: str(item.id))
        ],
        *_canonical_nodes(
            world.devices,
            ResolutionEntityType.DEVICE,
            GraphNodeType.DEVICE,
            resolved,
            lambda device: _properties(
                fingerprint=device.fingerprint,
                device_type=str(device.device_type),
                platform=str(device.platform),
                is_emulator=device.is_emulator,
            ),
        ),
        *_canonical_nodes(
            world.ip_addresses,
            ResolutionEntityType.IP_ADDRESS,
            GraphNodeType.IP_ADDRESS,
            resolved,
            lambda ip_address: _properties(address=str(ip_address.address)),
        ),
        *_canonical_nodes(
            world.payment_identities,
            ResolutionEntityType.PAYMENT_IDENTITY,
            GraphNodeType.PAYMENT_IDENTITY,
            resolved,
            lambda payment: _properties(
                fingerprint=payment.fingerprint,
                method=str(payment.method),
                issuer_country_code=payment.issuer_country_code,
            ),
        ),
        *_canonical_nodes(
            world.addresses,
            ResolutionEntityType.ADDRESS,
            GraphNodeType.ADDRESS,
            resolved,
            lambda address: _properties(
                recipient_name=address.recipient_name,
                line1=address.line1,
                line2=address.line2,
                city=address.city,
                region=address.region,
                postal_code=address.postal_code,
                country_code=address.country_code,
            ),
        ),
        *[
            GraphNode(
                node_type=GraphNodeType.ORDER,
                canonical_id=str(order.id),
                properties=_properties(
                    canonical_id=str(order.id),
                    placed_at=order.placed_at,
                    subtotal_paise=order.subtotal_paise,
                    discount_paise=order.discount_paise,
                    total_paise=order.total_paise,
                    item_count=order.item_count,
                ),
            )
            for order in sorted(world.orders, key=lambda item: str(item.id))
        ],
        *[
            GraphNode(
                node_type=GraphNodeType.PROMOTION,
                canonical_id=str(promotion.id),
                properties=_properties(
                    canonical_id=str(promotion.id),
                    code=promotion.code,
                    campaign_name=promotion.campaign_name,
                    discount_type=str(promotion.discount_type),
                    discount_value=promotion.discount_value,
                    valid_from=promotion.valid_from,
                    valid_until=promotion.valid_until,
                ),
            )
            for promotion in sorted(world.promotions, key=lambda item: str(item.id))
        ],
        *[
            GraphNode(
                node_type=GraphNodeType.REFUND,
                canonical_id=str(refund.id),
                properties=_properties(
                    canonical_id=str(refund.id),
                    amount_paise=refund.amount_paise,
                    requested_at=refund.requested_at,
                    reason_code=refund.reason_code,
                ),
            )
            for refund in sorted(world.refunds, key=lambda item: str(item.id))
        ],
    ]

    relationships: list[GraphRelationship] = []
    for event in world.events:
        account_id = str(event.account_id)
        if event.event_type is EventType.DEVICE_SEEN and event.device_id is not None:
            relationships.append(
                _relationship(
                    GraphRelationshipType.USED_DEVICE,
                    GraphNodeType.ACCOUNT,
                    account_id,
                    GraphNodeType.DEVICE,
                    _canonical(event.device_id, ResolutionEntityType.DEVICE, resolved),
                    event.id,
                    event.event_type,
                    event.occurred_at,
                )
            )
        elif event.event_type is EventType.IP_SEEN and event.ip_address_id is not None:
            relationships.append(
                _relationship(
                    GraphRelationshipType.SEEN_FROM,
                    GraphNodeType.ACCOUNT,
                    account_id,
                    GraphNodeType.IP_ADDRESS,
                    _canonical(
                        event.ip_address_id, ResolutionEntityType.IP_ADDRESS, resolved
                    ),
                    event.id,
                    event.event_type,
                    event.occurred_at,
                )
            )
        elif (
            event.event_type is EventType.PAYMENT_ATTACHED
            and event.payment_identity_id is not None
        ):
            relationships.append(
                _relationship(
                    GraphRelationshipType.PAID_WITH,
                    GraphNodeType.ACCOUNT,
                    account_id,
                    GraphNodeType.PAYMENT_IDENTITY,
                    _canonical(
                        event.payment_identity_id,
                        ResolutionEntityType.PAYMENT_IDENTITY,
                        resolved,
                    ),
                    event.id,
                    event.event_type,
                    event.occurred_at,
                )
            )
        elif (
            event.event_type is EventType.ORDER_PLACED
            and event.order_id is not None
            and event.address_id is not None
        ):
            order_id = str(event.order_id)
            relationships.extend(
                (
                    _relationship(
                        GraphRelationshipType.PLACED,
                        GraphNodeType.ACCOUNT,
                        account_id,
                        GraphNodeType.ORDER,
                        order_id,
                        event.id,
                        event.event_type,
                        event.occurred_at,
                    ),
                    _relationship(
                        GraphRelationshipType.SHIPPED_TO,
                        GraphNodeType.ORDER,
                        order_id,
                        GraphNodeType.ADDRESS,
                        _canonical(
                            event.address_id, ResolutionEntityType.ADDRESS, resolved
                        ),
                        event.id,
                        event.event_type,
                        event.occurred_at,
                    ),
                )
            )
        elif (
            event.event_type is EventType.PROMOTION_REDEEMED
            and event.order_id is not None
            and event.promotion_id is not None
        ):
            relationships.append(
                _relationship(
                    GraphRelationshipType.USED_PROMO,
                    GraphNodeType.ORDER,
                    str(event.order_id),
                    GraphNodeType.PROMOTION,
                    str(event.promotion_id),
                    event.id,
                    event.event_type,
                    event.occurred_at,
                )
            )
        elif (
            event.event_type in {EventType.REFUND_REQUESTED, EventType.REFUND_RESOLVED}
            and event.order_id is not None
            and event.refund_id is not None
        ):
            relationships.append(
                _relationship(
                    GraphRelationshipType.HAS_REFUND,
                    GraphNodeType.ORDER,
                    str(event.order_id),
                    GraphNodeType.REFUND,
                    str(event.refund_id),
                    event.id,
                    event.event_type,
                    event.occurred_at,
                )
            )

    return GraphProjection(
        nodes=tuple(
            sorted(nodes, key=lambda node: (node.node_type, node.canonical_id))
        ),
        relationships=tuple(
            sorted(
                relationships,
                key=lambda relationship: (
                    relationship.event_time,
                    relationship.event_id,
                    relationship.relationship_type,
                ),
            )
        ),
    )


def build_incremental_graph_projection(event: Event) -> GraphProjection:
    """Project one already-canonical runtime fact using the offline graph schema.

    Runtime provider fixtures supply canonical UUIDs, so this narrow adapter uses
    the same exact-identifier semantics as resolution's stable-ID normalizer. It
    intentionally does not fabricate entity attributes from raw provider JSON.
    """
    account_id = str(event.account_id)
    nodes: list[GraphNode] = [
        GraphNode(
            node_type=GraphNodeType.ACCOUNT,
            canonical_id=account_id,
            properties=_properties(
                canonical_id=account_id,
                created_at=(
                    event.occurred_at
                    if event.event_type is EventType.ACCOUNT_CREATED
                    else None
                ),
            ),
        )
    ]
    relationships: list[GraphRelationship] = []

    def add_identity(
        node_type: GraphNodeType,
        identity_id: UUID,
        relationship_type: GraphRelationshipType,
    ) -> None:
        canonical_id = str(identity_id)
        nodes.append(
            GraphNode(
                node_type=node_type,
                canonical_id=canonical_id,
                properties={"canonical_id": canonical_id},
            )
        )
        relationships.append(
            _relationship(
                relationship_type,
                GraphNodeType.ACCOUNT,
                account_id,
                node_type,
                canonical_id,
                event.id,
                event.event_type,
                event.occurred_at,
            )
        )

    if event.event_type is EventType.DEVICE_SEEN and event.device_id is not None:
        add_identity(
            GraphNodeType.DEVICE, event.device_id, GraphRelationshipType.USED_DEVICE
        )
    elif event.event_type is EventType.IP_SEEN and event.ip_address_id is not None:
        add_identity(
            GraphNodeType.IP_ADDRESS,
            event.ip_address_id,
            GraphRelationshipType.SEEN_FROM,
        )
    elif (
        event.event_type is EventType.PAYMENT_ATTACHED
        and event.payment_identity_id is not None
    ):
        add_identity(
            GraphNodeType.PAYMENT_IDENTITY,
            event.payment_identity_id,
            GraphRelationshipType.PAID_WITH,
        )
    else:
        if event.event_type is not EventType.ACCOUNT_CREATED:
            raise ValueError(
                f"runtime incremental projection does not support {event.event_type.value}"
            )
    return GraphProjection(nodes=tuple(nodes), relationships=tuple(relationships))
