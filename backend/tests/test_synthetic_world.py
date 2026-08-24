"""Tests for deterministic, scenario-driven synthetic-world generation."""

import unittest
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from pydantic import BaseModel

from mayajaal.schemas import Event
from mayajaal.synthetic import (
    GenerationProfile,
    export_parquet,
    generate_world,
    to_tables,
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

    def test_fraud_rings_share_intended_identity_relationships(self) -> None:
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
            self.assertEqual(len({event.device_id for event in events}), 1)
            self.assertEqual(len({event.ip_address_id for event in events}), 1)
            self.assertEqual(len({event.address_id for event in events}), 1)
            self.assertEqual(len({event.payment_identity_id for event in events}), 1)

    def test_shared_household_is_unlabelled(self) -> None:
        household_only = GenerationProfile(
            seed=73,
            normal_account_count=0,
            shared_household_count=1,
            accounts_per_shared_household=3,
            promo_ring_count=0,
            refund_ring_count=0,
            mixed_ring_count=0,
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
        world = generate_world(household_only)
        self.assertEqual(len({order.shipping_address_id for order in world.orders}), 1)
        self.assertEqual(len({event.device_id for event in world.events}), 1)
        self.assertTrue(all(event.synthetic_labels is None for event in world.events))

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
