"""Cypher statements for the Neo4j-derived identity graph."""

from .models import GraphNodeType, GraphRelationshipType

CONSTRAINTS = tuple(
    f"CREATE CONSTRAINT {node_type.value.lower()}_canonical_id_unique IF NOT EXISTS "
    f"FOR (node:{node_type.value}) REQUIRE node.canonical_id IS UNIQUE"
    for node_type in GraphNodeType
)


def merge_nodes_query(node_type: GraphNodeType) -> str:
    """Return a label-specific, parameterized canonical-node upsert."""
    return (
        f"UNWIND $rows AS row MERGE (node:{node_type.value} "
        "{canonical_id: row.canonical_id}) SET node += row.properties"
    )


def merge_relationships_query(
    relationship_type: GraphRelationshipType,
    source_type: GraphNodeType,
    target_type: GraphNodeType,
) -> str:
    """Return an event-keyed relationship upsert for one typed edge batch."""
    return (
        "UNWIND $rows AS row "
        f"MATCH (source:{source_type.value} {{canonical_id: row.source_canonical_id}}) "
        f"MATCH (target:{target_type.value} {{canonical_id: row.target_canonical_id}}) "
        f"MERGE (source)-[relationship:{relationship_type.value} "
        "{event_id: row.event_id}]->(target) "
        "SET relationship.event_time = row.event_time"
    )


RELATIONSHIPS_KNOWN_AT = """
MATCH (source)-[relationship]->(target)
WHERE type(relationship) IN $relationship_types
  AND relationship.event_time <= datetime($cutoff)
RETURN labels(source)[0] AS source_type,
       source.canonical_id AS source_canonical_id,
       type(relationship) AS relationship_type,
       relationship.event_id AS event_id,
       relationship.event_time AS event_time,
       labels(target)[0] AS target_type,
       target.canonical_id AS target_canonical_id
ORDER BY relationship.event_time, relationship.event_id, relationship_type
""".strip()
