"""Tests for the temporal, heterogeneous graph projection."""

import unittest
from datetime import UTC, datetime, timedelta
from typing import Any

from neo4j import Query

from mayajaal.graph import (
    GraphNodeType,
    GraphRelationshipType,
    Neo4jGraphRepository,
    build_graph_projection,
)
from mayajaal.graph.cypher import RELATIONSHIPS_KNOWN_AT
from mayajaal.resolution import ResolutionEntityType, resolve_all
from mayajaal.schemas import EventType
from mayajaal.synthetic import GenerationProfile, generate_world


def profile() -> GenerationProfile:
    """Build a compact world that includes sharing and every graph edge kind."""
    return GenerationProfile(
        seed=73,
        normal_account_count=1,
        shared_household_count=1,
        accounts_per_shared_household=3,
        promo_ring_count=1,
        refund_ring_count=1,
        mixed_ring_count=1,
        accounts_per_ring=3,
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 2, 1, tzinfo=UTC),
    )


def projection():  # type: ignore[no-untyped-def]
    """Resolve and project a deterministic world using the public API."""
    world = generate_world(profile())
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    return world, resolution, build_graph_projection(world, resolution)


class _Result:
    def consume(self) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(())


class _Session:
    def __init__(self, calls: list[tuple[Query | str, dict[str, Any]]]) -> None:
        self.calls = calls

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_arguments: object) -> None:
        return None

    def run(self, query: Query | str, **parameters: Any) -> _Result:
        self.calls.append((query, parameters))
        return _Result()


class _Driver:
    def __init__(self) -> None:
        self.calls: list[tuple[Query | str, dict[str, Any]]] = []

    def close(self) -> None:
        return None

    def session(self, **_: object) -> _Session:
        return _Session(self.calls)


class GraphProjectionTests(unittest.TestCase):
    def test_expected_shared_and_ring_connections_exist(self) -> None:
        world, _, graph = projection()
        used_device = [
            relationship
            for relationship in graph.relationships
            if relationship.relationship_type is GraphRelationshipType.USED_DEVICE
        ]
        by_device: dict[str, set[str]] = {}
        for relationship in used_device:
            by_device.setdefault(relationship.target_canonical_id, set()).add(
                relationship.source_canonical_id
            )
        self.assertTrue(any(len(accounts) >= 2 for accounts in by_device.values()))

        ring_events = [
            event
            for event in world.events
            if event.synthetic_labels is not None
            and event.synthetic_labels.coordination_cluster_id == "promo-ring-000"
            and event.device_id is not None
        ]
        ring_accounts = {str(event.account_id) for event in ring_events}
        ring_device = str(ring_events[0].device_id)
        self.assertTrue(
            any(
                relationship.target_canonical_id == ring_device
                and relationship.source_canonical_id in ring_accounts
                for relationship in used_device
            )
        )

    def test_resolved_identity_edges_target_canonical_nodes(self) -> None:
        world, resolution, graph = projection()
        canonical_devices = {
            str(result.raw_entity_id): str(result.canonical_entity_id)
            for result in resolution.results
            if result.entity_type is ResolutionEntityType.DEVICE
        }
        device_nodes = {
            node.canonical_id
            for node in graph.nodes
            if node.node_type is GraphNodeType.DEVICE
        }
        for event in world.events:
            if event.device_id is None:
                continue
            expected = canonical_devices[str(event.device_id)]
            self.assertIn(expected, device_nodes)
            self.assertTrue(
                any(
                    edge.event_id == str(event.id)
                    and edge.target_canonical_id == expected
                    for edge in graph.relationships
                )
                if event.event_type.value == "device_seen"
                else True
            )

    def test_projection_is_deterministic_and_excludes_fraud_labels(self) -> None:
        _, _, first = projection()
        _, _, second = projection()
        self.assertEqual(first, second)
        properties = [node.properties for node in first.nodes]
        self.assertTrue(
            all(
                "synthetic_labels" not in node_properties
                and "coordination_cluster_id" not in node_properties
                for node_properties in properties
            )
        )

    def test_refund_cutoff_does_not_reveal_completion(self) -> None:
        world, _, graph = projection()
        requested = next(
            event
            for event in world.events
            if event.event_type is EventType.REFUND_REQUESTED
        )
        resolved = next(
            event
            for event in world.events
            if event.event_type is EventType.REFUND_RESOLVED
            and event.refund_id == requested.refund_id
        )
        cutoff = requested.occurred_at + timedelta(minutes=1)
        refund_edges = [
            edge
            for edge in graph.relationships
            if edge.relationship_type is GraphRelationshipType.HAS_REFUND
            and edge.target_canonical_id == str(requested.refund_id)
            and edge.event_time <= cutoff
        ]
        self.assertEqual(
            [edge.event_type for edge in refund_edges], ["refund_requested"]
        )
        self.assertGreater(resolved.occurred_at, cutoff)

        refund_node = next(
            node
            for node in graph.nodes
            if node.node_type is GraphNodeType.REFUND
            and node.canonical_id == str(requested.refund_id)
        )
        self.assertNotIn("state", refund_node.properties)
        self.assertNotIn("resolved_at", refund_node.properties)
        self.assertTrue(
            all(
                "status" not in node.properties
                for node in graph.nodes
                if node.node_type in {GraphNodeType.ACCOUNT, GraphNodeType.ORDER}
            )
        )

    def test_every_relationship_retains_its_source_event_type(self) -> None:
        world, _, graph = projection()
        event_types = {str(event.id): event.event_type.value for event in world.events}
        self.assertTrue(
            all(
                relationship.event_type == event_types[relationship.event_id]
                for relationship in graph.relationships
            )
        )

    def test_cutoff_filter_excludes_later_event_facts(self) -> None:
        _, _, graph = projection()
        cutoff = min(edge.event_time for edge in graph.relationships) + timedelta(
            minutes=1
        )
        known = tuple(edge for edge in graph.relationships if edge.event_time <= cutoff)
        self.assertLess(len(known), len(graph.relationships))
        self.assertTrue(all(edge.event_time <= cutoff for edge in known))
        self.assertIn(
            "relationship.event_time <= datetime($cutoff)", RELATIONSHIPS_KNOWN_AT
        )

    def test_repository_uses_event_keyed_merges_idempotently(self) -> None:
        _, _, graph = projection()
        driver = _Driver()
        repository = Neo4jGraphRepository(
            "bolt://unused",
            ("unused", "unused"),
            driver=driver,  # type: ignore[arg-type]
        )
        self.assertEqual(repository.load(graph), repository.load(graph))
        merge_queries = [
            query.text if isinstance(query, Query) else query
            for query, _ in driver.calls
            if "MERGE" in (query.text if isinstance(query, Query) else query)
        ]
        self.assertTrue(
            any("{event_id: row.event_id}" in query for query in merge_queries)
        )
        self.assertTrue(
            any("{canonical_id: row.canonical_id}" in query for query in merge_queries)
        )
        self.assertTrue(
            any("relationship.event_type" in query for query in merge_queries)
        )


if __name__ == "__main__":
    _ = unittest.main()
