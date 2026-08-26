"""Model-neutral evaluation orchestration plus the CatBoost adapter boundary."""

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mayajaal.baseline import (
    BaselineArtifacts,
    BaselineConfig,
    TrainedBaseline,
    predict_fraud_probability,
    predict_raw_score,
    save_baseline,
    train_baseline,
)
from mayajaal.features import (
    FeatureSchema,
    FeatureService,
    FeatureVector,
    LabeledFeatureVector,
)

from .metrics import (
    evaluate_predictions,
    held_out_validity,
    prediction_frame,
    save_curve_plots,
    select_threshold,
)
from .models import (
    EvaluationConfig,
    EvaluationSample,
    EvaluationSplit,
    PredictionRecord,
    SplitManifest,
    SplitMetrics,
    ThresholdSelection,
)
from .provenance import (
    FrozenFullArtifactInput,
    provenance_base_model_id,
    write_frozen_full_artifacts,
)

LOCAL_IDENTITY_FEATURE_NAMES = frozenset(
    {
        "device_count",
        "ip_address_count",
        "payment_identity_count",
        "address_count",
    }
)

RELATIONAL_GRAPH_FEATURE_NAMES = frozenset(
    {
        "shared_device_account_count",
        "shared_ip_account_count",
        "shared_payment_account_count",
        "shared_address_account_count",
        "max_identity_reuse_count",
        "identity_neighbour_count",
        "identity_component_account_count",
        "shared_promotion_account_count",
        "recent_shared_account_creation_count",
        "recent_shared_identity_event_count",
    }
)

GRAPH_IDENTITY_FEATURE_NAMES = (
    LOCAL_IDENTITY_FEATURE_NAMES | RELATIONAL_GRAPH_FEATURE_NAMES
)


def vectors_for_manifest(
    service: FeatureService, manifest: SplitManifest
) -> dict[str, FeatureVector]:
    """Extract one cutoff-safe vector per review sample, batched by time."""
    by_decision_time: defaultdict[datetime, list[str]] = defaultdict(list)
    for sample in manifest.samples:
        by_decision_time[sample.decision_time].append(sample.account_id)
    vectors: dict[str, FeatureVector] = {}
    for decision_time, account_ids in sorted(
        by_decision_time.items(), key=lambda item: item[0]
    ):
        extracted = {
            vector.account_id: vector
            for vector in service.extract_many(sorted(account_ids), decision_time)
        }
        vectors.update(
            {
                sample.sample_id: extracted[sample.account_id]
                for sample in manifest.samples
                if sample.decision_time == decision_time
            }
        )
    return vectors


def evaluate_catboost(
    service: FeatureService,
    manifest: SplitManifest,
    config: EvaluationConfig,
    *,
    baseline_config: BaselineConfig | None = None,
) -> tuple[
    dict[str, tuple[PredictionRecord, ...]],
    dict[str, ThresholdSelection],
    dict[str, dict[EvaluationSplit, SplitMetrics]],
    dict[str, FeatureSchema],
    dict[str, TrainedBaseline],
]:
    """Run the CatBoost ablations on an identical split manifest.

    This is deliberately an adapter: sample construction, score records, metric
    calculation, and artifact formats have no CatBoost-specific dependency.
    """
    vectors = vectors_for_manifest(service, manifest)
    full_schema = service.schema
    no_graph_schema = _without_graph_schema(full_schema)
    no_relational_schema = _without_relational_graph_schema(full_schema)
    variants = {
        "full": (full_schema, _identity),
        "no_graph_identity": (no_graph_schema, _without_graph_vector),
        "no_relational_graph": (
            no_relational_schema,
            _without_relational_graph_vector,
        ),
    }
    records_by_variant: dict[str, tuple[PredictionRecord, ...]] = {}
    thresholds: dict[str, ThresholdSelection] = {}
    metrics: dict[str, dict[EvaluationSplit, SplitMetrics]] = {}
    models: dict[str, TrainedBaseline] = {}
    for name, (schema, transform) in variants.items():
        examples_by_split = _examples_by_split(manifest.samples, vectors, transform)
        model = train_baseline(
            examples_by_split[EvaluationSplit.TRAIN], schema, baseline_config
        )
        models[name] = model
        records = tuple(
            PredictionRecord(
                sample_id=sample.sample_id,
                account_id=sample.account_id,
                decision_time=sample.decision_time,
                split=sample.split,
                y_true=sample.y_true,
                score=predict_fraud_probability(
                    model, transform(vectors[sample.sample_id])
                ),
                model_variant=name,
            )
            for sample in manifest.samples
        )
        records_by_variant[name] = records
        validation_records = tuple(
            record for record in records if record.split is EvaluationSplit.VALIDATION
        )
        threshold = select_threshold(validation_records, config)
        thresholds[name] = threshold
        metrics[name] = {
            split: evaluate_predictions(
                tuple(record for record in records if record.split is split),
                threshold.threshold,
                config,
            )
            for split in EvaluationSplit
        }
    return (
        records_by_variant,
        thresholds,
        metrics,
        {
            "full": full_schema,
            "no_graph_identity": no_graph_schema,
            "no_relational_graph": no_relational_schema,
        },
        models,
    )


def fit_full_catboost_scores(
    service: FeatureService,
    manifest: SplitManifest,
    *,
    baseline_config: BaselineConfig | None = None,
) -> tuple[
    tuple[PredictionRecord, ...],
    dict[str, float],
    FeatureSchema,
    TrainedBaseline,
]:
    """Train and score only the full CatBoost adapter without selecting a threshold.

    Post-model consumers such as calibration need frozen raw scores but must not
    introduce an operating-threshold decision into their workflow.
    """
    vectors = vectors_for_manifest(service, manifest)
    schema = service.schema
    examples = _examples_by_split(manifest.samples, vectors, _identity)
    model = train_baseline(examples[EvaluationSplit.TRAIN], schema, baseline_config)
    records = tuple(
        PredictionRecord(
            sample_id=sample.sample_id,
            account_id=sample.account_id,
            decision_time=sample.decision_time,
            split=sample.split,
            y_true=sample.y_true,
            score=predict_fraud_probability(model, vectors[sample.sample_id]),
            model_variant="full",
        )
        for sample in manifest.samples
    )
    raw_scores = {
        sample.sample_id: predict_raw_score(model, vectors[sample.sample_id])
        for sample in manifest.samples
    }
    return records, raw_scores, schema, model


def save_catboost_evaluation_models(
    models: dict[str, TrainedBaseline],
    service: FeatureService,
    manifest: SplitManifest,
    output_directory: Path,
    *,
    shap_sample_count: int,
) -> dict[str, BaselineArtifacts]:
    """Save each adapter model and bounded deterministic training SHAP sample."""
    if shap_sample_count < 1:
        raise ValueError("shap_sample_count must be positive")
    vectors = vectors_for_manifest(service, manifest)
    train_samples = tuple(
        sorted(
            (
                sample
                for sample in manifest.samples
                if sample.split is EvaluationSplit.TRAIN
            ),
            key=lambda sample: sample.account_id,
        )[:shap_sample_count]
    )
    return {
        name: save_baseline(
            model,
            tuple(
                _transform_for_variant(name, vectors[sample.sample_id])
                for sample in train_samples
            ),
            output_directory / name,
        )
        for name, model in sorted(models.items())
    }


def write_evaluation_artifacts(
    output_directory: Path,
    manifest: SplitManifest,
    records_by_variant: dict[str, tuple[PredictionRecord, ...]],
    thresholds: dict[str, ThresholdSelection],
    metrics: dict[str, dict[EvaluationSplit, SplitMetrics]],
    schemas: dict[str, FeatureSchema],
    config: EvaluationConfig,
    *,
    seed: int,
    frozen_full: FrozenFullArtifactInput | None = None,
) -> dict[str, Path]:
    """Persist the reusable manifest, prediction contract, reports, and curves."""
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "split_manifest.json"
    predictions_path = output_directory / "predictions.parquet"
    evaluation_path = output_directory / "evaluation.json"
    manifest_path.write_text(
        json.dumps(_manifest_json(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frozen_artifacts = (
        write_frozen_full_artifacts(
            output_directory,
            manifest_path=manifest_path,
            full_records=frozen_full.records,
            raw_scores=frozen_full.raw_scores,
            schema=frozen_full.schema,
            model_artifacts=frozen_full.model_artifacts,
            training_config=frozen_full.training_config,
            evaluation_config=config,
            generation_profile=frozen_full.generation_profile,
        )
        if frozen_full is not None
        else {}
    )
    frozen_base_model_id = (
        provenance_base_model_id(frozen_artifacts["full_provenance"])
        if frozen_artifacts
        else None
    )
    prediction_frame(
        record for records in records_by_variant.values() for record in records
    ).write_parquet(predictions_path, compression="zstd")
    pr_path, roc_path = save_curve_plots(
        {
            name: tuple(
                record for record in records if record.split is EvaluationSplit.TEST
            )
            for name, records in records_by_variant.items()
        },
        output_directory,
    )
    evaluation_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "name": "fixed decision-time incident-abuse evaluation",
                    "primary_ranking_metric": "Average Precision (AP)",
                    "leakage_policy": "features use immutable facts at or before each decision_time; labels mark newly observable abuse in that interval; campaigns spanning intervals are purged and known campaigns are excluded from later windows",
                    "threshold_policy": "selected on validation only and frozen for test only with sufficient validation class support",
                    "seed": seed,
                    "config": config.model_dump(mode="json"),
                    "cutoffs": {
                        "train": manifest.train_cutoff.isoformat(),
                        "validation": manifest.validation_cutoff.isoformat(),
                        "test": manifest.test_cutoff.isoformat(),
                    },
                },
                "benchmark": asdict(held_out_validity(thresholds, metrics)),
                "frozen_full": (
                    {"base_model_id": frozen_base_model_id}
                    if frozen_artifacts
                    else None
                ),
                "variants": {
                    name: {
                        "threshold": asdict(thresholds[name]),
                        "feature_schema": [
                            asdict(item) for item in schemas[name].definitions
                        ],
                        "metrics": {
                            split.value: asdict(report)
                            for split, report in metrics[name].items()
                        },
                    }
                    for name in sorted(metrics)
                },
                "artifacts": {
                    "split_manifest": str(manifest_path),
                    "predictions": str(predictions_path),
                    "pr_curve": str(pr_path),
                    "roc_curve": str(roc_path),
                    **{name: str(path) for name, path in frozen_artifacts.items()},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "split_manifest": manifest_path,
        "predictions": predictions_path,
        "evaluation": evaluation_path,
        "pr_curve": pr_path,
        "roc_curve": roc_path,
        **frozen_artifacts,
    }


def _examples_by_split(
    samples: tuple[EvaluationSample, ...],
    vectors: dict[str, FeatureVector],
    transform: Callable[[FeatureVector], FeatureVector],
) -> dict[EvaluationSplit, tuple[LabeledFeatureVector, ...]]:
    grouped: defaultdict[EvaluationSplit, list[LabeledFeatureVector]] = defaultdict(
        list
    )
    for sample in samples:
        grouped[sample.split].append(
            LabeledFeatureVector(
                vector=transform(vectors[sample.sample_id]), is_fraud=sample.y_true
            )
        )
    return {split: tuple(grouped[split]) for split in EvaluationSplit}


def _without_graph_schema(schema: FeatureSchema) -> FeatureSchema:
    return _without_features_schema(schema, GRAPH_IDENTITY_FEATURE_NAMES)


def _without_relational_graph_schema(schema: FeatureSchema) -> FeatureSchema:
    return _without_features_schema(schema, RELATIONAL_GRAPH_FEATURE_NAMES)


def _without_features_schema(
    schema: FeatureSchema, feature_names: frozenset[str]
) -> FeatureSchema:
    return FeatureSchema(
        tuple(
            definition
            for definition in schema.definitions
            if definition.name not in feature_names
        )
    )


def _without_graph_vector(vector: FeatureVector) -> FeatureVector:
    return _without_features_vector(vector, GRAPH_IDENTITY_FEATURE_NAMES)


def _without_relational_graph_vector(vector: FeatureVector) -> FeatureVector:
    return _without_features_vector(vector, RELATIONAL_GRAPH_FEATURE_NAMES)


def _without_features_vector(
    vector: FeatureVector, feature_names: frozenset[str]
) -> FeatureVector:
    return FeatureVector(
        account_id=vector.account_id,
        cutoff=vector.cutoff,
        values={
            name: value
            for name, value in vector.values.items()
            if name not in feature_names
        },
    )


def _identity(vector: FeatureVector) -> FeatureVector:
    return vector


def _transform_for_variant(name: str, vector: FeatureVector) -> FeatureVector:
    transforms: dict[str, Callable[[FeatureVector], FeatureVector]] = {
        "full": _identity,
        "no_graph_identity": _without_graph_vector,
        "no_relational_graph": _without_relational_graph_vector,
    }
    try:
        return transforms[name](vector)
    except KeyError as error:
        raise ValueError(f"unknown evaluation model variant: {name}") from error


def _manifest_json(manifest: SplitManifest) -> dict[str, object]:
    return {
        "train_cutoff": manifest.train_cutoff.isoformat(),
        "validation_cutoff": manifest.validation_cutoff.isoformat(),
        "test_cutoff": manifest.test_cutoff.isoformat(),
        "purged_campaign_group_ids": list(manifest.purged_campaign_group_ids),
        "purged_campaign_groups": [
            asdict(item) for item in manifest.purged_campaign_groups
        ],
        "samples": [
            {
                "sample_id": sample.sample_id,
                "account_id": sample.account_id,
                "decision_time": sample.decision_time.isoformat(),
                "split": sample.split.value,
                "y_true": sample.y_true,
                "campaign_group_id": sample.campaign_group_id,
            }
            for sample in manifest.samples
        ],
    }
