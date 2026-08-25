"""Feature service that composes registered extractors without model imports."""

from collections.abc import Iterable, Sequence
from datetime import datetime

import polars as pl

from mayajaal.graph import GraphProjection

from .extractors import DEFAULT_EXTRACTORS, FeatureContext, FeatureExtractor
from .models import FeatureDefinition, FeatureSchema, FeatureValue, FeatureVector
from .temporal import (
    AccountGraphIndex,
    TemporalFeatureGraph,
    TemporalFeatureSnapshot,
    build_account_graph_index,
)


class FeatureService:
    """Extract stable account vectors from the resolved graph at a cutoff time."""

    def __init__(
        self,
        projection: GraphProjection,
        *,
        extractors: Sequence[FeatureExtractor] = DEFAULT_EXTRACTORS,
    ) -> None:
        self._graph = TemporalFeatureGraph(projection)
        self._extractors = tuple(extractors)
        definitions = tuple(
            definition
            for extractor in self._extractors
            for definition in extractor.definitions
        )
        _validate_definitions(definitions)
        self.schema = FeatureSchema(definitions)

    def extract(self, account_id: str, cutoff: datetime) -> FeatureVector:
        """Compute one account vector using no relationship after ``cutoff``."""
        snapshot = self._graph.snapshot_at(cutoff)
        return self._extract_from_snapshot(
            account_id, snapshot, build_account_graph_index(snapshot)
        )

    def _extract_from_snapshot(
        self,
        account_id: str,
        snapshot: TemporalFeatureSnapshot,
        index: AccountGraphIndex,
    ) -> FeatureVector:
        """Extract one vector from caller-provided, cutoff-safe graph indexes."""
        snapshot.account_created_at(account_id)
        context = FeatureContext(
            account_id=account_id,
            snapshot=snapshot,
            index=index,
        )
        values: dict[str, FeatureValue] = {}
        for extractor in self._extractors:
            values.update(extractor.extract(context))
        self.schema.validate(values)
        return FeatureVector(
            account_id=account_id, cutoff=snapshot.cutoff, values=values
        )

    def extract_many(
        self, account_ids: Iterable[str], cutoff: datetime
    ) -> tuple[FeatureVector, ...]:
        """Extract a deterministically account-ID-sorted batch at one cutoff."""
        snapshot = self._graph.snapshot_at(cutoff)
        index = build_account_graph_index(snapshot)
        return tuple(
            self._extract_from_snapshot(account_id, snapshot, index)
            for account_id in sorted(account_ids)
        )


def feature_frame(
    vectors: Sequence[FeatureVector], schema: FeatureSchema
) -> pl.DataFrame:
    """Return a stable Polars table suitable for offline export or inspection."""
    return pl.DataFrame(
        [
            {"account_id": vector.account_id, "cutoff": vector.cutoff, **vector.values}
            for vector in vectors
        ],
        schema={
            "account_id": pl.String,
            "cutoff": pl.Datetime(time_zone="UTC"),
            **{
                definition.name: (
                    pl.Float64 if definition.kind.value == "numeric" else pl.String
                )
                for definition in schema.definitions
            },
        },
        strict=True,
    )


def _validate_definitions(definitions: tuple[FeatureDefinition, ...]) -> None:
    names = tuple(definition.name for definition in definitions)
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
