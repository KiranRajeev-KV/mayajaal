"""Neo4j persistence for the derived identity graph."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, LiteralString, cast

from neo4j import (
    Driver,
    GraphDatabase,
    ManagedTransaction,
    Query,
)
from neo4j import (
    Session as Neo4jSession,
)

from .cypher import (
    CLEAR_DERIVED_GRAPH,
    CONSTRAINTS,
    NODES_FOR_FEATURES,
    RELATIONSHIPS_KNOWN_AT,
    merge_nodes_query,
    merge_relationships_query,
)
from .models import (
    GraphNode,
    GraphNodeType,
    GraphProjection,
    GraphRelationship,
    GraphRelationshipType,
)


@dataclass(frozen=True)
class GraphLoadReport:
    """Counts submitted to Neo4j; stable across idempotent reloads."""

    node_count: int
    relationship_count: int


@dataclass(frozen=True)
class TemporalGraphRelationship:
    """An event-backed relationship returned by a cutoff-time graph query."""

    source_type: str
    source_canonical_id: str
    relationship_type: str
    event_id: str
    event_type: str
    event_time: datetime
    known_at: datetime
    target_type: str
    target_canonical_id: str


class Neo4jGraphRepository:
    """Load and query the derived graph with official Neo4j driver semantics."""

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        *,
        database: str = "neo4j",
        driver: Driver | None = None,
    ) -> None:
        self.database = database
        self.driver = driver or GraphDatabase.driver(uri, auth=auth)  # type: ignore[reportUnknownMemberType]

    def close(self) -> None:
        """Close the owned Neo4j driver connection pool."""
        self.driver.close()

    def verify_connectivity(self) -> None:
        """Fail fast when the long-lived driver cannot reach Neo4j."""
        self.driver.verify_connectivity()  # type: ignore[reportUnknownMemberType]

    def ensure_schema(self) -> None:
        """Create one canonical-ID uniqueness constraint per graph label."""
        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
            for query in CONSTRAINTS:
                session.run(Query(query)).consume()  # type: ignore[reportArgumentType]

    def clear(self) -> None:
        """Delete every node and relationship in the dedicated derived database."""
        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
            session.run(CLEAR_DERIVED_GRAPH).consume()

    def load(self, projection: GraphProjection) -> GraphLoadReport:
        """Idempotently merge canonical nodes and event-backed relationships."""
        self.ensure_schema()
        nodes_by_type, relationships_by_type = _projection_batches(projection)

        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
            _merge_batch_projection(session, nodes_by_type, relationships_by_type)
        return GraphLoadReport(
            node_count=len(projection.nodes),
            relationship_count=len(projection.relationships),
        )

    def load_incremental(self, projection: GraphProjection) -> GraphLoadReport:
        """Merge one event projection in one idempotent managed transaction.

        Neo4j managed transaction functions may retry, so every statement uses
        the existing canonical-ID/event-ID ``MERGE`` semantics and consumes its
        result before the function returns.
        """
        self.ensure_schema()
        nodes_by_type, relationships_by_type = _projection_batches(projection)
        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
            session.execute_write(
                _merge_incremental_projection,
                nodes_by_type,
                relationships_by_type,
            )
        return GraphLoadReport(
            node_count=len(projection.nodes),
            relationship_count=len(projection.relationships),
        )

    def relationships_known_at(
        self, cutoff: datetime
    ) -> tuple[TemporalGraphRelationship, ...]:
        """Return only facts whose event time was known by ``cutoff``.

        The graph stores no lifetime aggregate on these edges, preventing a
        post-cutoff ``last_seen`` or ``count`` value from leaking into a query.
        """
        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
            result = session.run(
                RELATIONSHIPS_KNOWN_AT,
                cutoff=cutoff,
                relationship_types=[item.value for item in GraphRelationshipType],
            )
            return tuple(
                TemporalGraphRelationship(
                    source_type=record["source_type"],
                    source_canonical_id=record["source_canonical_id"],
                    relationship_type=record["relationship_type"],
                    event_id=record["event_id"],
                    event_type=record["event_type"],
                    event_time=record["event_time"].to_native(),
                    known_at=record["known_at"].to_native(),
                    target_type=record["target_type"],
                    target_canonical_id=record["target_canonical_id"],
                )
                for record in result
            )

    def feature_projection_at(self, cutoff: datetime) -> GraphProjection:
        """Read a storage-neutral, knowledge-time-safe feature projection."""
        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
            return session.execute_read(_feature_projection_at, cutoff)


def _feature_projection_at(
    transaction: ManagedTransaction, cutoff: datetime
) -> GraphProjection:
    nodes = tuple(
        GraphNode(
            node_type=GraphNodeType(record["node_type"]),
            canonical_id=record["canonical_id"],
            properties={
                key: _native(value) for key, value in record["properties"].items()
            },
        )
        for record in transaction.run(
            NODES_FOR_FEATURES,
            cutoff=cutoff,
            node_types=[item.value for item in GraphNodeType],
        )
    )
    relationships = tuple(
        GraphRelationship(
            relationship_type=GraphRelationshipType(record["relationship_type"]),
            source_type=GraphNodeType(record["source_type"]),
            source_canonical_id=record["source_canonical_id"],
            target_type=GraphNodeType(record["target_type"]),
            target_canonical_id=record["target_canonical_id"],
            event_id=record["event_id"],
            event_type=record["event_type"],
            event_time=_native(record["event_time"]),
            known_at=_native(record["known_at"]),
        )
        for record in transaction.run(
            RELATIONSHIPS_KNOWN_AT,
            cutoff=cutoff,
            relationship_types=[item.value for item in GraphRelationshipType],
        )
    )
    return GraphProjection(nodes=nodes, relationships=relationships)


def _native(value: Any) -> Any:
    return value.to_native() if hasattr(value, "to_native") else value


def _relationship_row(relationship: GraphRelationship) -> dict[str, Any]:
    return {
        "source_canonical_id": relationship.source_canonical_id,
        "target_canonical_id": relationship.target_canonical_id,
        "event_id": relationship.event_id,
        "event_type": relationship.event_type,
        "event_time": relationship.event_time,
        "known_at": relationship.known_at,
    }


type _NodeBatches = defaultdict[GraphNodeType, list[dict[str, Any]]]
type _RelationshipKey = tuple[GraphRelationshipType, GraphNodeType, GraphNodeType]
type _RelationshipBatches = defaultdict[_RelationshipKey, list[dict[str, Any]]]


def _projection_batches(
    projection: GraphProjection,
) -> tuple[_NodeBatches, _RelationshipBatches]:
    nodes_by_type: _NodeBatches = defaultdict(list)
    for node in projection.nodes:
        nodes_by_type[node.node_type].append(
            {"canonical_id": node.canonical_id, "properties": node.properties}
        )

    relationships_by_type: _RelationshipBatches = defaultdict(list)
    for relationship in projection.relationships:
        relationships_by_type[
            (
                relationship.relationship_type,
                relationship.source_type,
                relationship.target_type,
            )
        ].append(_relationship_row(relationship))
    return nodes_by_type, relationships_by_type


def _merge_batch_projection(
    session: Neo4jSession,
    nodes_by_type: _NodeBatches,
    relationships_by_type: _RelationshipBatches,
) -> None:
    """Preserve the batch loader's established auto-commit behavior."""
    for node_type in sorted(nodes_by_type):
        session.run(
            Query(merge_nodes_query(node_type)),  # type: ignore[reportArgumentType]
            rows=nodes_by_type[node_type],
        ).consume()
    for key in sorted(relationships_by_type):
        relationship_type, source_type, target_type = key
        session.run(
            Query(
                cast(
                    LiteralString,
                    merge_relationships_query(
                        relationship_type, source_type, target_type
                    ),
                )
            ),
            rows=relationships_by_type[key],
        ).consume()


def _merge_incremental_projection(
    transaction: ManagedTransaction,
    nodes_by_type: _NodeBatches,
    relationships_by_type: _RelationshipBatches,
) -> None:
    """Run all projection mutations within the supplied write boundary."""
    for node_type in sorted(nodes_by_type):
        transaction.run(
            cast(LiteralString, merge_nodes_query(node_type)),
            rows=nodes_by_type[node_type],
        ).consume()
    for key in sorted(relationships_by_type):
        relationship_type, source_type, target_type = key
        transaction.run(
            cast(
                LiteralString,
                merge_relationships_query(relationship_type, source_type, target_type),
            ),
            rows=relationships_by_type[key],
        ).consume()
