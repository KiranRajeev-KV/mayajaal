"""Deterministic CatBoost training kept separate from graph feature extraction."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import shap  # type: ignore[reportMissingTypeStubs]
from catboost import CatBoostClassifier, Pool  # type: ignore[reportMissingTypeStubs]

from mayajaal.features import (
    FeatureSchema,
    FeatureVector,
    LabeledFeatureVector,
)
from mayajaal.synthetic import SyntheticWorld

from .models import (
    BaselineArtifacts,
    BaselineConfig,
    FeatureContribution,
    GlobalFeatureImportance,
    PredictionExplanation,
    TrainedBaseline,
)


def labels_at_cutoff(world: SyntheticWorld, cutoff: datetime) -> dict[str, bool]:
    """Create synthetic training targets without exposing labels to features.

    The target is whether an account has an abuse-labelled event *already
    observed* by the cutoff. This function is intentionally the sole baseline
    code path that reads ``synthetic_labels``.
    """
    labels = {
        str(account.id): False
        for account in world.accounts
        if account.created_at <= cutoff
    }
    for event in world.events:
        if event.occurred_at > cutoff or str(event.account_id) not in labels:
            continue
        if (
            event.synthetic_labels is not None
            and event.synthetic_labels.is_coordinated_abuse
        ):
            labels[str(event.account_id)] = True
    return labels


def label_vectors(
    vectors: tuple[FeatureVector, ...], world: SyntheticWorld, cutoff: datetime
) -> tuple[LabeledFeatureVector, ...]:
    """Attach evaluation/training truth after, never during, feature extraction."""
    labels = labels_at_cutoff(world, cutoff)
    return tuple(
        LabeledFeatureVector(vector=vector, is_fraud=labels[vector.account_id])
        for vector in vectors
        if vector.account_id in labels
    )


def train_baseline(
    examples: tuple[LabeledFeatureVector, ...],
    schema: FeatureSchema,
    config: BaselineConfig | None = None,
) -> TrainedBaseline:
    """Fit a single-threaded, class-balanced deterministic CatBoost classifier."""
    if len(examples) < 2:
        raise ValueError("at least two labelled examples are required")
    labels = [int(example.is_fraud) for example in examples]
    if len(set(labels)) != 2:
        raise ValueError("training examples must contain both target classes")
    chosen_config = config or BaselineConfig()
    pool = _pool(tuple(example.vector for example in examples), schema, labels)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=chosen_config.iterations,
        depth=chosen_config.depth,
        learning_rate=chosen_config.learning_rate,
        random_seed=chosen_config.random_seed,
        random_strength=0.0,
        bootstrap_type="No",
        auto_class_weights="Balanced",
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )
    cast(Any, model).fit(pool)
    return TrainedBaseline(model=model, schema=schema, config=chosen_config)


def predict_fraud_probability(
    baseline: TrainedBaseline, vector: FeatureVector
) -> float:
    """Return the positive-class probability for a schema-compatible vector."""
    pool = _pool((vector,), baseline.schema)
    probabilities = cast(Any, baseline.model).predict_proba(pool)
    return float(probabilities[0][1])


def explain_prediction(
    baseline: TrainedBaseline,
    vector: FeatureVector,
    *,
    limit: int = 5,
) -> PredictionExplanation:
    """Return strongest positive and negative exact TreeSHAP contributions."""
    if limit < 1:
        raise ValueError("limit must be positive")
    contributions, base_values = _shap_values(baseline, (vector,))
    ranked = sorted(
        (
            FeatureContribution(
                feature_name=name,
                feature_value=vector.values[name],
                shap_value=float(contributions[0, index]),
            )
            for index, name in enumerate(baseline.schema.names)
        ),
        key=lambda item: (-abs(item.shap_value), item.feature_name),
    )
    return PredictionExplanation(
        fraud_probability=predict_fraud_probability(baseline, vector),
        base_value=float(base_values[0]),
        positive=tuple(item for item in ranked if item.shap_value > 0.0)[:limit],
        negative=tuple(item for item in ranked if item.shap_value < 0.0)[:limit],
    )


def global_shap_importance(
    baseline: TrainedBaseline, vectors: tuple[FeatureVector, ...]
) -> tuple[GlobalFeatureImportance, ...]:
    """Rank features by mean absolute exact TreeSHAP contribution."""
    if not vectors:
        raise ValueError("at least one vector is required")
    contributions, _ = _shap_values(baseline, vectors)
    return tuple(
        sorted(
            (
                GlobalFeatureImportance(
                    feature_name=name,
                    mean_absolute_shap=float(np.mean(np.abs(contributions[:, index]))),
                )
                for index, name in enumerate(baseline.schema.names)
            ),
            key=lambda item: (-item.mean_absolute_shap, item.feature_name),
        )
    )


def save_baseline(
    baseline: TrainedBaseline,
    vectors: tuple[FeatureVector, ...],
    output_directory: Path,
) -> BaselineArtifacts:
    """Persist the CatBoost model, schema metadata, and offline SHAP bar plot."""
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / "catboost_fraud_baseline.cbm"
    metadata_path = output_directory / "feature_metadata.json"
    shap_summary_path = output_directory / "shap_summary.png"
    cast(Any, baseline.model).save_model(str(model_path))
    metadata_path.write_text(
        json.dumps(
            {
                "feature_definitions": [
                    asdict(definition) for definition in baseline.schema.definitions
                ],
                "categorical_feature_names": baseline.schema.categorical_names,
                "training": asdict(baseline.config),
                "target": "synthetic event-labelled coordinated abuse at cutoff",
                "leakage_policy": "features use only immutable graph events at or before cutoff",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_shap_summary(baseline, vectors, shap_summary_path)
    return BaselineArtifacts(model_path, metadata_path, shap_summary_path)


def save_shap_summary(
    baseline: TrainedBaseline, vectors: tuple[FeatureVector, ...], output_path: Path
) -> None:
    """Use SHAP to write a global mean-absolute-contribution bar summary."""
    contributions, base_values = _shap_values(baseline, vectors)
    values = np.asarray(
        [[vector.values[name] for name in baseline.schema.names] for vector in vectors],
        dtype=object,
    )
    explanation = shap.Explanation(
        values=contributions,
        base_values=base_values,
        data=values,
        feature_names=list(baseline.schema.names),
    )
    axes = shap.plots.bar(
        explanation, max_display=len(baseline.schema.names), show=False
    )
    axes.figure.savefig(output_path, bbox_inches="tight")
    axes.figure.clf()


def _pool(
    vectors: tuple[FeatureVector, ...],
    schema: FeatureSchema,
    labels: list[int] | None = None,
) -> Pool:
    for vector in vectors:
        schema.validate(vector.values)
    rows = [[vector.values[name] for name in schema.names] for vector in vectors]
    categorical_indices = [
        index
        for index, name in enumerate(schema.names)
        if name in schema.categorical_names
    ]
    return Pool(
        data=rows,
        label=labels,
        cat_features=categorical_indices,
        feature_names=list(schema.names),
    )


def _shap_values(
    baseline: TrainedBaseline, vectors: tuple[FeatureVector, ...]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Obtain SHAP TreeExplainer values in stable schema column order."""
    if not vectors:
        raise ValueError("at least one vector is required")
    explainer = cast(Any, shap.TreeExplainer(baseline.model))
    matrix = np.asarray(
        explainer.shap_values(_pool(vectors, baseline.schema)), dtype=float
    )
    expected_value = float(explainer.expected_value)
    return matrix, np.full(matrix.shape[0], expected_value)
