"""Neo4j persistence for the derived identity graph."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, LiteralString, cast

from neo4j import Driver, GraphDatabase, Query

from .cypher import (
    CONSTRAINTS,
    RELATIONSHIPS_KNOWN_AT,
    merge_nodes_query,
    merge_relationships_query,
)
from .models import (
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
    event_time: datetime
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

    def ensure_schema(self) -> None:
        """Create one canonical-ID uniqueness constraint per graph label."""
        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
            for query in CONSTRAINTS:
                session.run(Query(query)).consume()  # type: ignore[reportArgumentType]

    def load(self, projection: GraphProjection) -> GraphLoadReport:
        """Idempotently merge canonical nodes and event-backed relationships."""
        self.ensure_schema()
        nodes_by_type: defaultdict[GraphNodeType, list[dict[str, Any]]] = defaultdict(
            list
        )
        for node in projection.nodes:
            nodes_by_type[node.node_type].append(
                {"canonical_id": node.canonical_id, "properties": node.properties}
            )

        relationships_by_type: defaultdict[
            tuple[GraphRelationshipType, GraphNodeType, GraphNodeType],
            list[dict[str, Any]],
        ] = defaultdict(list)
        for relationship in projection.relationships:
            relationships_by_type[
                (
                    relationship.relationship_type,
                    relationship.source_type,
                    relationship.target_type,
                )
            ].append(_relationship_row(relationship))

        with self.driver.session(database=self.database) as session:  # type: ignore[reportUnknownMemberType]
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
                    event_time=record["event_time"].to_native(),
                    target_type=record["target_type"],
                    target_canonical_id=record["target_canonical_id"],
                )
                for record in result
            )


def _relationship_row(relationship: GraphRelationship) -> dict[str, Any]:
    return {
        "source_canonical_id": relationship.source_canonical_id,
        "target_canonical_id": relationship.target_canonical_id,
        "event_id": relationship.event_id,
        "event_time": relationship.event_time,
    }
