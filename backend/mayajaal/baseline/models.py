"""Serializable contracts for the deterministic CatBoost baseline."""

from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostClassifier  # type: ignore[reportMissingTypeStubs]

from mayajaal.features import FeatureSchema


@dataclass(frozen=True)
class BaselineConfig:
    """Deliberate deterministic and imbalance-aware CatBoost settings."""

    random_seed: int = 20260825
    iterations: int = 80
    depth: int = 4
    learning_rate: float = 0.1


@dataclass(frozen=True)
class TrainedBaseline:
    """A model paired with the exact feature schema it was trained to consume."""

    model: CatBoostClassifier
    schema: FeatureSchema
    config: BaselineConfig


@dataclass(frozen=True)
class FeatureContribution:
    """One signed SHAP contribution to a fraud probability prediction."""

    feature_name: str
    feature_value: float | str
    shap_value: float


@dataclass(frozen=True)
class PredictionExplanation:
    """Reusable local explanation with the strongest signed feature effects."""

    fraud_probability: float
    base_value: float
    positive: tuple[FeatureContribution, ...]
    negative: tuple[FeatureContribution, ...]


@dataclass(frozen=True)
class GlobalFeatureImportance:
    """Mean absolute SHAP contribution for one feature across a sample set."""

    feature_name: str
    mean_absolute_shap: float


@dataclass(frozen=True)
class BaselineArtifacts:
    """Paths produced by an offline training run."""

    model_path: Path
    metadata_path: Path
    shap_summary_path: Path
