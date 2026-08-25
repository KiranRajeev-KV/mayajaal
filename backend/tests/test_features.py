"""Tests for cutoff-safe, model-independent graph feature extraction."""

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from mayajaal.features import FeatureService
from mayajaal.graph import GraphProjection, build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.schemas import EventType
from mayajaal.synthetic import (
    GenerationProfile,
    SyntheticWorld,
    cutoff_times,
    diagnose_feature_health,
    generate_world,
)
from mayajaal.synthetic.profile import PopulationProfile


def profile() -> GenerationProfile:
    """Create a small population with household sharing and every ring pattern."""
    return GenerationProfile(
        seed=991,
        normal_account_count=2,
        shared_household_count=1,
        accounts_per_shared_household=3,
        promo_ring_count=1,
        refund_ring_count=1,
        mixed_ring_count=1,
        accounts_per_ring=3,
        population=PopulationProfile(benign_network_group_count=0),
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 2, 1, tzinfo=UTC),
    )


def prepared() -> tuple[SyntheticWorld, FeatureService]:
    """Generate, resolve, and expose the public feature service."""
    world = generate_world(profile())
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    return world, FeatureService(build_graph_projection(world, resolution))


class FeatureServiceTests(unittest.TestCase):
    def test_shared_household_is_a_shared_identity_not_a_fraud_feature(self) -> None:
        world, service = prepared()
        fraud_accounts = {
            str(event.account_id)
            for event in world.events
            if event.synthetic_labels is not None
            and event.synthetic_labels.is_coordinated_abuse
        }
        vectors = service.extract_many(
            (str(account.id) for account in world.accounts), profile().end_at
        )
        household_vectors = [
            vector
            for vector in vectors
            if vector.account_id not in fraud_accounts
            and vector.values["shared_ip_account_count"] == 2.0
        ]
        self.assertEqual(len(household_vectors), 3)
        self.assertTrue(
            all(
                vector.values["identity_component_account_count"] == 3.0
                for vector in household_vectors
            )
        )

    def test_future_facts_cannot_change_any_feature_family_at_cutoff(self) -> None:
        world, service = prepared()
        target = min(world.accounts, key=lambda account: str(account.id))
        cutoff = target.created_at + timedelta(minutes=1)
        original = service.extract(str(target.id), cutoff)

        resolution = resolve_all(
            accounts=world.accounts,
            addresses=world.addresses,
            ip_addresses=world.ip_addresses,
            payment_identities=world.payment_identities,
            devices=world.devices,
        )
        projection = build_graph_projection(world, resolution)
        future_projection = GraphProjection(
            nodes=projection.nodes,
            relationships=(
                *projection.relationships,
                *(
                    replace(
                        relationship,
                        event_id=f"future-{relationship.event_id}",
                        event_time=cutoff + timedelta(days=90),
                    )
                    for relationship in projection.relationships
                ),
            ),
        )
        augmented = FeatureService(future_projection).extract(str(target.id), cutoff)
        self.assertEqual(original, augmented)

    def test_refund_resolution_is_not_visible_before_its_event(self) -> None:
        world, service = prepared()
        request = next(
            event
            for event in world.events
            if event.event_type is EventType.REFUND_REQUESTED
        )
        resolved = next(
            event
            for event in world.events
            if event.event_type is EventType.REFUND_RESOLVED
            and event.refund_id == request.refund_id
        )
        cutoff = request.occurred_at + timedelta(minutes=1)
        vector = service.extract(str(request.account_id), cutoff)
        self.assertEqual(vector.values["refund_requested_count"], 1.0)
        self.assertEqual(vector.values["refund_resolved_count"], 0.0)
        self.assertGreater(resolved.occurred_at, cutoff)

    def test_labels_cannot_influence_graph_feature_extraction(self) -> None:
        world, service = prepared()
        label_free_world = replace(
            world,
            events=tuple(
                event.model_copy(update={"synthetic_labels": None})
                for event in world.events
            ),
        )
        resolution = resolve_all(
            accounts=label_free_world.accounts,
            addresses=label_free_world.addresses,
            ip_addresses=label_free_world.ip_addresses,
            payment_identities=label_free_world.payment_identities,
            devices=label_free_world.devices,
        )
        label_free_service = FeatureService(
            build_graph_projection(label_free_world, resolution)
        )
        target = str(world.accounts[0].id)
        self.assertEqual(
            service.extract(target, profile().end_at),
            label_free_service.extract(target, profile().end_at),
        )

    def test_extraction_and_schema_are_deterministic(self) -> None:
        world, service = prepared()
        account_ids = tuple(str(account.id) for account in world.accounts)
        first = service.extract_many(account_ids, profile().end_at)
        second = service.extract_many(reversed(account_ids), profile().end_at)
        self.assertEqual(first, second)
        self.assertEqual(len(service.schema.names), len(set(service.schema.names)))
        self.assertIn("latest_payment_method", service.schema.categorical_names)

    def test_feature_health_is_deterministic_at_early_middle_and_late_cutoffs(
        self,
    ) -> None:
        world, service = prepared()
        reports = {}
        for name, cutoff in cutoff_times(profile()).items():
            vectors = service.extract_many(
                (
                    str(account.id)
                    for account in world.accounts
                    if account.created_at <= cutoff
                ),
                cutoff,
            )
            first = diagnose_feature_health(
                vectors, service.schema, world, cutoff, profile().diagnostics
            )
            second = diagnose_feature_health(
                vectors, service.schema, world, cutoff, profile().diagnostics
            )
            self.assertEqual(first, second)
            reports[name] = first
        self.assertEqual(set(reports), {"early", "middle", "late"})
        self.assertFalse(reports["late"].inactive_expected_numeric_features)
        self.assertEqual(
            reports["late"].intentionally_sparse_numeric_features,
            (
                "recent_shared_account_creation_count",
                "recent_shared_identity_event_count",
            ),
        )
