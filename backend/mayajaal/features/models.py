"""Stable, model-independent contracts for cutoff-safe feature values."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class FeatureKind(StrEnum):
    """Supported feature value kinds."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


type FeatureValue = float | str


@dataclass(frozen=True)
class FeatureDefinition:
    """One versioned feature name, type, and leakage-safe interpretation."""

    name: str
    kind: FeatureKind
    description: str


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered feature definitions shared by extraction, training, and serving."""

    definitions: tuple[FeatureDefinition, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions)

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(
            definition.name
            for definition in self.definitions
            if definition.kind is FeatureKind.CATEGORICAL
        )

    @property
    def numeric_names(self) -> tuple[str, ...]:
        return tuple(
            definition.name
            for definition in self.definitions
            if definition.kind is FeatureKind.NUMERIC
        )

    def validate(self, values: dict[str, FeatureValue]) -> None:
        """Reject incomplete or ill-typed vectors before a model consumes them."""
        if tuple(values) != self.names:
            raise ValueError("feature vector keys must exactly match schema order")
        for definition in self.definitions:
            value = values[definition.name]
            if definition.kind is FeatureKind.NUMERIC and not isinstance(value, float):
                raise TypeError(f"{definition.name} must be numeric")
            if definition.kind is FeatureKind.CATEGORICAL and not isinstance(
                value, str
            ):
                raise TypeError(f"{definition.name} must be categorical")


@dataclass(frozen=True)
class FeatureVector:
    """Features for one account as the graph was known at ``cutoff``."""

    account_id: str
    cutoff: datetime
    values: dict[str, FeatureValue]


@dataclass(frozen=True)
class LabeledFeatureVector:
    """A supervised target kept separate from the graph-derived feature values."""

    vector: FeatureVector
    is_fraud: bool
