"""Polars table conversion and Parquet export for synthetic-world records."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from .world import SyntheticWorld


def _table(records: Sequence[BaseModel]) -> pl.DataFrame:
    """Convert validated models into a portable, JSON-compatible Polars table."""
    return pl.DataFrame(
        [record.model_dump(mode="json") for record in records],
        infer_schema_length=None,
    )


def to_tables(world: SyntheticWorld) -> dict[str, pl.DataFrame]:
    """Return one Polars table per canonical entity and a separate event table."""
    return {
        "accounts": _table(world.accounts),
        "devices": _table(world.devices),
        "ip_addresses": _table(world.ip_addresses),
        "addresses": _table(world.addresses),
        "payment_identities": _table(world.payment_identities),
        "orders": _table(world.orders),
        "promotions": _table(world.promotions),
        "refunds": _table(world.refunds),
        "events": _table(world.events),
    }


def export_parquet(world: SyntheticWorld, output_directory: Path) -> dict[str, Path]:
    """Write each generated table as a deterministic-named Parquet file."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in to_tables(world).items():
        path = output_directory / f"{name}.parquet"
        table.write_parquet(path, compression="zstd")
        paths[name] = path
    return paths
