"""Tests for deterministic, scenario-driven synthetic-world generation."""

import unittest
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from pydantic import BaseModel

from mayajaal.schemas import Event, EventType
from mayajaal.synthetic import (
    GenerationProfile,
    SyntheticWorld,
    diagnose_world,
    export_parquet,
    generate_world,
    guardrail_failures,
    to_tables,
)
from mayajaal.synthetic.profile import (
    AbuseProfile,
    PopulationProfile,
    PrevalenceProfile,
)


def profile(seed: int) -> GenerationProfile:
    """Build a compact profile that exercises every population type."""
    return GenerationProfile(
        seed=seed,
        normal_account_count=3,
        shared_household_count=1,
        accounts_per_shared_household=3,
        promo_ring_count=1,
        refund_ring_count=1,
        mixed_ring_count=1,
        accounts_per_ring=3,
        population=PopulationProfile(benign_network_group_count=0),
        abuse=AbuseProfile(ring_size_variation=0),
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 2, 1, tzinfo=UTC),
    )


def table_records(seed: int) -> dict[str, list[dict[str, object]]]:
    """Return a comparison-friendly representation of one generated world."""
    return {
        name: table.to_dicts()
        for name, table in to_tables(generate_world(profile(seed))).items()
    }


class SyntheticWorldTests(unittest.TestCase):
    def test_same_seed_and_profile_produce_identical_tables(self) -> None:
        self.assertEqual(table_records(73), table_records(73))

    def test_different_seed_produces_different_output(self) -> None:
        self.assertNotEqual(table_records(73), table_records(74))

    def test_all_generated_records_revalidate(self) -> None:
        world = generate_world(profile(73))
        for record in world.all_models():
            self.assertIsInstance(record, BaseModel)
            _ = type(record).model_validate(record.model_dump())

    def test_fraud_campaigns_share_partial_identities_and_label_only_abuse_events(
        self,
    ) -> None:
        world = generate_world(profile(73))
        relationships: defaultdict[str, list[Event]] = defaultdict(list)
        for event in world.events:
            labels = event.synthetic_labels
            if labels is not None:
                relationships[labels.coordination_cluster_id or ""].append(event)

        self.assertEqual(
            set(relationships), {"promo-ring-000", "refund-ring-000", "mixed-ring-000"}
        )
        for events in relationships.values():
            account_ids = {event.account_id for event in events}
            self.assertEqual(len(account_ids), 3)
            self.assertTrue(
                {event.event_type for event in events}.issubset(
                    {
                        EventType.PROMOTION_REDEEMED,
                        EventType.REFUND_REQUESTED,
                        EventType.REFUND_RESOLVED,
                    }
                )
            )
            account_histories = [
                [event for event in world.events if event.account_id == account_id]
                for account_id in account_ids
            ]
            identity_sets: list[set[tuple[str, str]]] = [
                {
                    ("device", str(event.device_id))
                    for event in history
                    if event.device_id is not None
                }
                | {
                    ("ip", str(event.ip_address_id))
                    for event in history
                    if event.ip_address_id is not None
                }
                | {
                    ("payment", str(event.payment_identity_id))
                    for event in history
                    if event.payment_identity_id is not None
                }
                | {
                    ("address", str(event.address_id))
                    for event in history
                    if event.address_id is not None
                }
                for history in account_histories
            ]
            self.assertTrue(
                any(
                    left & right
                    for index, left in enumerate(identity_sets)
                    for right in identity_sets[index + 1 :]
                )
            )
            shared_by_every_member = identity_sets[0].copy()
            for identity_set in identity_sets[1:]:
                shared_by_every_member.intersection_update(identity_set)
            self.assertLess(len(shared_by_every_member), 4)

    def test_shared_household_is_unlabelled(self) -> None:
        household_only = GenerationProfile(
            seed=73,
            normal_account_count=0,
            shared_household_count=1,
            accounts_per_shared_household=3,
            promo_ring_count=0,
            refund_ring_count=0,
            mixed_ring_count=0,
            population=PopulationProfile(
                benign_network_group_count=0,
                persona_weights={"repeat": 1.0},
            ),
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
        world = generate_world(household_only)
        first_order_addresses = {
            min(
                (
                    event
                    for event in world.events
                    if event.account_id == account.id
                    and event.event_type is EventType.ORDER_PLACED
                ),
                key=lambda event: event.occurred_at,
            ).address_id
            for account in world.accounts
        }
        first_ip_addresses = {
            min(
                (
                    event
                    for event in world.events
                    if event.account_id == account.id
                    and event.event_type is EventType.IP_SEEN
                ),
                key=lambda event: event.occurred_at,
            ).ip_address_id
            for account in world.accounts
        }
        self.assertEqual(len(first_order_addresses), 1)
        self.assertEqual(len(first_ip_addresses), 1)
        self.assertTrue(all(event.synthetic_labels is None for event in world.events))

    def test_office_network_shares_ip_without_shared_shipping_address(self) -> None:
        office_only = GenerationProfile(
            seed=174,
            normal_account_count=0,
            shared_household_count=0,
            promo_ring_count=0,
            refund_ring_count=0,
            mixed_ring_count=0,
            population=PopulationProfile(
                benign_network_group_count=1,
                accounts_per_benign_network_group=3,
                persona_weights={"repeat": 1.0},
            ),
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
        world = generate_world(office_only)
        first_ip_addresses = {
            min(
                (
                    event
                    for event in world.events
                    if event.account_id == account.id
                    and event.event_type is EventType.IP_SEEN
                ),
                key=lambda event: event.occurred_at,
            ).ip_address_id
            for account in world.accounts
        }
        first_order_addresses = {
            min(
                (
                    event
                    for event in world.events
                    if event.account_id == account.id
                    and event.event_type is EventType.ORDER_PLACED
                ),
                key=lambda event: event.occurred_at,
            ).address_id
            for account in world.accounts
        }
        self.assertEqual(len(first_ip_addresses), 1)
        self.assertEqual(len(first_order_addresses), 3)
        self.assertTrue(all(event.synthetic_labels is None for event in world.events))

    def test_repeat_personas_create_multi_order_identity_lifecycles(self) -> None:
        repeat_only = GenerationProfile(
            seed=616,
            normal_account_count=8,
            shared_household_count=0,
            promo_ring_count=0,
            refund_ring_count=0,
            mixed_ring_count=0,
            population=PopulationProfile(
                benign_network_group_count=0,
                persona_weights={"repeat": 1.0},
            ),
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 4, 1, tzinfo=UTC),
        )
        world = generate_world(repeat_only)
        account_orders: defaultdict[object, int] = defaultdict(int)
        account_devices: defaultdict[object, set[object]] = defaultdict(set)
        for event in world.events:
            if event.event_type is EventType.ORDER_PLACED:
                account_orders[event.account_id] += 1
            if event.event_type is EventType.DEVICE_SEEN:
                account_devices[event.account_id].add(event.device_id)
        self.assertGreater(max(account_orders.values()), 1)
        self.assertTrue(any(len(devices) > 1 for devices in account_devices.values()))

    def test_internal_diagnostics_are_deterministic_and_standard_has_overlap(
        self,
    ) -> None:
        diagnostic_profile = GenerationProfile(
            seed=901,
            normal_account_count=80,
            shared_household_count=8,
            accounts_per_shared_household=3,
            promo_ring_count=2,
            refund_ring_count=2,
            mixed_ring_count=2,
            accounts_per_ring=5,
            population=PopulationProfile(
                benign_network_group_count=4,
                accounts_per_benign_network_group=5,
            ),
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 4, 1, tzinfo=UTC),
        )
        first = diagnose_world(generate_world(diagnostic_profile))
        second = diagnose_world(generate_world(diagnostic_profile))
        self.assertEqual(first, second)
        self.assertGreater(
            first.graph["identity_sharing_subgraph_largest_component_account_count"],
            1.0,
        )
        self.assertEqual(
            first.graph["full_account_projection_account_count"],
            float(first.account_count),
        )
        self.assertGreater(first.temporal["active_day_count"], 20.0)
        self.assertFalse(first.perfect_single_feature_separators)
        self.assertFalse(guardrail_failures(first, diagnostic_profile))
        world = generate_world(diagnostic_profile)
        promotion_codes = {
            str(promotion.id): promotion.code for promotion in world.promotions
        }
        self.assertTrue(
            any(
                event.synthetic_labels is None
                and event.event_type is EventType.PROMOTION_REDEEMED
                and event.promotion_id is not None
                and promotion_codes[str(event.promotion_id)] == "FLASH50"
                for event in world.events
            )
        )

    def test_full_projection_includes_isolated_accounts(self) -> None:
        isolated_profile = GenerationProfile(
            seed=603,
            normal_account_count=6,
            shared_household_count=0,
            promo_ring_count=0,
            refund_ring_count=0,
            mixed_ring_count=0,
            population=PopulationProfile(benign_network_group_count=0),
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
        diagnostics = diagnose_world(generate_world(isolated_profile))
        self.assertEqual(
            diagnostics.graph["full_account_projection_isolated_account_count"], 6.0
        )
        self.assertEqual(
            diagnostics.graph["full_account_projection_component_count"], 6.0
        )
        self.assertEqual(
            diagnostics.graph["identity_sharing_subgraph_account_count"], 0.0
        )

    def test_difficulty_and_prevalence_are_orthogonal(self) -> None:
        profile_with_rare_abuse = GenerationProfile(
            seed=817,
            difficulty="hard",
            prevalence=PrevalenceProfile(preset="rare_abuse"),
        )
        self.assertEqual(
            profile_with_rare_abuse.prevalence.resolved_target_rate(), 0.0075
        )
        self.assertGreater(
            profile_with_rare_abuse.active_difficulty.benign_sharing_multiplier,
            1.0,
        )
        self.assertEqual(
            profile_with_rare_abuse.active_difficulty.campaign_sharing_multiplier,
            0.72,
        )

    def test_target_prevalence_stays_within_configured_tolerance(self) -> None:
        target_profile = GenerationProfile(
            seed=337,
            normal_account_count=100,
            shared_household_count=0,
            promo_ring_count=2,
            refund_ring_count=1,
            mixed_ring_count=1,
            population=PopulationProfile(benign_network_group_count=0),
            prevalence=PrevalenceProfile(
                target_labelled_account_rate=0.10,
                target_tolerance=0.01,
            ),
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
        diagnostics = diagnose_world(generate_world(target_profile))
        self.assertNotIn(
            "labelled account prevalence is outside configured tolerance",
            guardrail_failures(diagnostics, target_profile),
        )

    def test_campaign_schedule_supports_bursty_and_low_and_slow_activity(
        self,
    ) -> None:
        def campaign_profile(abuse: AbuseProfile) -> GenerationProfile:
            return GenerationProfile(
                seed=218,
                normal_account_count=0,
                shared_household_count=0,
                promo_ring_count=1,
                refund_ring_count=0,
                mixed_ring_count=0,
                accounts_per_ring=8,
                population=PopulationProfile(benign_network_group_count=0),
                abuse=abuse,
                start_at=datetime(2026, 1, 1, tzinfo=UTC),
                end_at=datetime(2026, 4, 1, tzinfo=UTC),
            )

        bursty = generate_world(
            campaign_profile(AbuseProfile(low_and_slow_probability=0.0))
        )
        low_and_slow = generate_world(
            campaign_profile(AbuseProfile(low_and_slow_probability=1.0))
        )

        def labelled_span_days(world: SyntheticWorld) -> float:
            times = [
                event.occurred_at
                for event in world.events
                if event.synthetic_labels is not None
            ]
            return (max(times) - min(times)).total_seconds() / 86_400.0

        self.assertLess(labelled_span_days(bursty), labelled_span_days(low_and_slow))

    def test_events_are_chronological_and_ingested_after_occurrence(self) -> None:
        events = generate_world(profile(73)).events
        self.assertEqual(
            list(events),
            sorted(events, key=lambda event: (event.occurred_at, str(event.id))),
        )
        self.assertTrue(all(event.ingested_at >= event.occurred_at for event in events))

    def test_export_writes_one_parquet_file_per_table(self) -> None:
        world = generate_world(profile(73))
        with TemporaryDirectory() as directory:
            paths = export_parquet(world, Path(directory))
            self.assertEqual(set(paths), set(to_tables(world)))
            self.assertEqual(pl.read_parquet(paths["events"]).height, len(world.events))


if __name__ == "__main__":
    _ = unittest.main()
