"""Frozen full-model artifact contracts for downstream post-model consumers."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isclose
from pathlib import Path
from typing import Any, cast

import polars as pl

from mayajaal.baseline import (
    BaselineArtifacts,
    BaselineConfig,
    TrainedBaseline,
    load_baseline,
    model_semantic_hash,
)
from mayajaal.features import (
    FeatureDefinition,
    FeatureKind,
    FeatureSchema,
    FeatureService,
)
from mayajaal.synthetic import GenerationProfile

from .models import (
    CampaignPurge,
    EvaluationConfig,
    EvaluationSample,
    EvaluationSplit,
    PredictionRecord,
    SplitManifest,
)

CONTRACT_VERSION = 1
FULL_MODEL_PROVENANCE_FILENAME = "full_model_provenance.json"
FULL_PREDICTIONS_FILENAME = "full_predictions.parquet"


@dataclass(frozen=True)
class FrozenFullEvaluation:
    """Verified full-model evaluation inputs available to post-model stages."""

    evaluation_directory: Path
    manifest: SplitManifest
    records: tuple[PredictionRecord, ...]
    raw_scores: dict[str, float]
    baseline: TrainedBaseline
    provenance: dict[str, Any]

    @property
    def base_model_id(self) -> str:
        """Return the deterministic identifier binding model and evaluation inputs."""
        return str(self.provenance["base_model_id"])


@dataclass(frozen=True)
class FrozenFullArtifactInput:
    """Exact full-model inputs to persist beside a held-out evaluation."""

    records: tuple[PredictionRecord, ...]
    raw_scores: dict[str, float]
    schema: FeatureSchema
    model_artifacts: BaselineArtifacts
    training_config: BaselineConfig
    generation_profile: GenerationProfile


def canonical_hash(value: object) -> str:
    """Hash canonical JSON rather than platform-dependent object representations."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    """Return a streaming SHA-256 checksum for one persisted artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_base_model_id(provenance_path: Path) -> str:
    """Read the stable identifier from a freshly written contract."""
    provenance = _validate_provenance_shape(
        json.loads(provenance_path.read_text(encoding="utf-8"))
    )
    base_model_id = provenance["base_model_id"]
    if not isinstance(base_model_id, str):
        raise ValueError("invalid frozen evaluation base-model identifier")
    return base_model_id


def write_frozen_full_artifacts(
    output_directory: Path,
    *,
    manifest_path: Path,
    full_records: tuple[PredictionRecord, ...],
    raw_scores: dict[str, float],
    schema: FeatureSchema,
    model_artifacts: BaselineArtifacts,
    training_config: BaselineConfig,
    evaluation_config: EvaluationConfig,
    generation_profile: GenerationProfile,
) -> dict[str, Path]:
    """Persist an immutable full-model hand-off bound by deterministic hashes."""
    if any(record.model_variant != "full" for record in full_records):
        raise ValueError("frozen full artifacts require only full-model records")
    if {record.sample_id for record in full_records} != set(raw_scores):
        raise ValueError("frozen full raw scores must exactly match full records")
    predictions_path = output_directory / FULL_PREDICTIONS_FILENAME
    _full_prediction_frame(full_records, raw_scores).write_parquet(
        predictions_path, compression="zstd"
    )
    schema_payload = _schema_payload(schema)
    training_payload = asdict(training_config)
    evaluation_payload = evaluation_config.model_dump(mode="json")
    profile_payload = generation_profile.model_dump(mode="json")
    artifact_paths = {
        "model": model_artifacts.model_path,
        "feature_metadata": model_artifacts.metadata_path,
        "split_manifest": manifest_path,
        "full_predictions": predictions_path,
    }
    model_semantic_sha256 = model_semantic_hash(
        load_baseline(model_artifacts.model_path, model_artifacts.metadata_path)
    )
    relative_artifacts = {
        name: {
            "path": str(path.relative_to(output_directory)),
            "sha256": file_hash(path),
        }
        for name, path in artifact_paths.items()
    }
    identity_payload = {
        "model_semantic_sha256": model_semantic_sha256,
        "split_manifest_sha256": relative_artifacts["split_manifest"]["sha256"],
        "feature_schema_sha256": canonical_hash(schema_payload),
        "training_config_sha256": canonical_hash(training_payload),
        "evaluation_config_sha256": canonical_hash(evaluation_payload),
        "generation_profile_sha256": canonical_hash(profile_payload),
    }
    provenance = {
        "contract_version": CONTRACT_VERSION,
        "model_variant": "full",
        "base_model_id": canonical_hash(identity_payload),
        "identity": identity_payload,
        "artifacts": relative_artifacts,
        "feature_schema": schema_payload,
        "training_config": training_payload,
        "evaluation_config": evaluation_payload,
        "generation_profile": profile_payload,
    }
    provenance_path = output_directory / FULL_MODEL_PROVENANCE_FILENAME
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"full_predictions": predictions_path, "full_provenance": provenance_path}


def load_frozen_full_evaluation(
    evaluation_directory: Path,
    *,
    expected_profile: GenerationProfile,
    expected_evaluation_config: EvaluationConfig,
) -> FrozenFullEvaluation:
    """Verify and load the exact persisted full model without retraining it."""
    provenance_path = evaluation_directory / FULL_MODEL_PROVENANCE_FILENAME
    try:
        untrusted_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"missing frozen evaluation provenance: {provenance_path}; run held-out evaluation first"
        ) from error
    provenance = _validate_provenance_shape(untrusted_provenance)
    if provenance["contract_version"] != CONTRACT_VERSION:
        raise ValueError("unsupported frozen evaluation provenance contract version")
    if provenance["model_variant"] != "full":
        raise ValueError("frozen evaluation provenance must reference the full model")
    _verify_expected_configuration(
        provenance,
        expected_profile=expected_profile,
        expected_evaluation_config=expected_evaluation_config,
    )
    artifacts = {
        name: _verified_artifact_path(evaluation_directory, name, value)
        for name, value in provenance["artifacts"].items()
    }
    schema = _schema_from_payload(provenance["feature_schema"])
    if (
        canonical_hash(provenance["feature_schema"])
        != provenance["identity"]["feature_schema_sha256"]
    ):
        raise ValueError("feature schema provenance hash mismatch")
    metadata = json.loads(artifacts["feature_metadata"].read_text(encoding="utf-8"))
    if (
        metadata.get("feature_definitions")
        != provenance["feature_schema"]["definitions"]
    ):
        raise ValueError(
            "feature schema mismatch between model metadata and provenance"
        )
    if metadata.get("training") != provenance["training_config"]:
        raise ValueError(
            "training config mismatch between model metadata and provenance"
        )
    manifest = _load_manifest(artifacts["split_manifest"])
    records, raw_scores = _load_full_predictions(artifacts["full_predictions"])
    _validate_records_against_manifest(records, manifest)
    baseline = load_baseline(artifacts["model"], artifacts["feature_metadata"])
    if baseline.schema != schema:
        raise ValueError("feature schema mismatch between loaded model and provenance")
    if asdict(baseline.config) != provenance["training_config"]:
        raise ValueError("training config mismatch between loaded model and provenance")
    expected_identity = {
        "model_semantic_sha256": model_semantic_hash(baseline),
        "split_manifest_sha256": provenance["artifacts"]["split_manifest"]["sha256"],
        "feature_schema_sha256": canonical_hash(provenance["feature_schema"]),
        "training_config_sha256": canonical_hash(provenance["training_config"]),
        "evaluation_config_sha256": canonical_hash(provenance["evaluation_config"]),
        "generation_profile_sha256": canonical_hash(provenance["generation_profile"]),
    }
    if (
        expected_identity != provenance["identity"]
        or canonical_hash(expected_identity) != provenance["base_model_id"]
    ):
        raise ValueError("base-model provenance identifier mismatch")
    return FrozenFullEvaluation(
        evaluation_directory=evaluation_directory,
        manifest=manifest,
        records=records,
        raw_scores=raw_scores,
        baseline=baseline,
        provenance=provenance,
    )


def verify_frozen_full_predictions(
    frozen: FrozenFullEvaluation, service: FeatureService
) -> None:
    """Prove persisted scores/probabilities reproduce from the loaded model.

    The world and cutoff-safe vectors are reconstructed solely for verification;
    this function never fits a model and never reads labels for scoring.
    """
    from mayajaal.baseline import predict_fraud_probability, predict_raw_score

    from .runner import vectors_for_manifest

    vectors = vectors_for_manifest(service, frozen.manifest)
    for record in frozen.records:
        vector = vectors.get(record.sample_id)
        if vector is None:
            raise ValueError(f"missing reconstructed vector for {record.sample_id}")
        if tuple(vector.values) != frozen.baseline.schema.names:
            raise ValueError(
                "feature schema mismatch between reconstructed vectors and frozen model"
            )
        raw_score = predict_raw_score(frozen.baseline, vector)
        probability = predict_fraud_probability(frozen.baseline, vector)
        if not isclose(raw_score, frozen.raw_scores[record.sample_id], abs_tol=1e-12):
            raise ValueError(f"raw model score mismatch for {record.sample_id}")
        if not isclose(probability, record.score, abs_tol=1e-12):
            raise ValueError(
                f"uncalibrated probability mismatch for {record.sample_id}"
            )


def _full_prediction_frame(
    records: tuple[PredictionRecord, ...], raw_scores: dict[str, float]
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "sample_id": record.sample_id,
                "account_id": record.account_id,
                "decision_time": record.decision_time,
                "split": record.split.value,
                "y_true": record.y_true,
                "raw_model_score": raw_scores[record.sample_id],
                "uncalibrated_probability": record.score,
            }
            for record in sorted(
                records, key=lambda item: (item.decision_time, item.sample_id)
            )
        ],
        schema={
            "sample_id": pl.String,
            "account_id": pl.String,
            "decision_time": pl.Datetime(time_zone="UTC"),
            "split": pl.String,
            "y_true": pl.Boolean,
            "raw_model_score": pl.Float64,
            "uncalibrated_probability": pl.Float64,
        },
        strict=True,
    )


def _schema_payload(schema: FeatureSchema) -> dict[str, object]:
    return {"definitions": [asdict(item) for item in schema.definitions]}


def _schema_from_payload(payload: object) -> FeatureSchema:
    try:
        if not isinstance(payload, dict):
            raise TypeError("schema payload must be an object")
        payload_object = cast(dict[str, object], payload)
        values = payload_object["definitions"]
        if not isinstance(values, list):
            raise TypeError("schema definitions must be a list")
        definition_values = cast(list[object], values)
        definitions = tuple(
            FeatureDefinition(
                name=str(_schema_item(item, "name")),
                kind=FeatureKind(str(_schema_item(item, "kind"))),
                description=str(_schema_item(item, "description")),
            )
            for item in definition_values
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "invalid feature schema in frozen evaluation provenance"
        ) from error
    return FeatureSchema(definitions)


def _schema_item(item: object, key: str) -> object:
    if not isinstance(item, dict) or key not in item:
        raise TypeError("schema definition is invalid")
    return cast(dict[str, object], item)[key]


def _validate_provenance_shape(provenance: object) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError("invalid frozen evaluation provenance")
    required = {
        "contract_version",
        "model_variant",
        "base_model_id",
        "identity",
        "artifacts",
        "feature_schema",
        "training_config",
        "evaluation_config",
        "generation_profile",
    }
    typed = cast(dict[str, Any], provenance)
    if not required.issubset(typed):
        raise ValueError("frozen evaluation provenance is missing required fields")
    return typed


def _verify_expected_configuration(
    provenance: dict[str, Any],
    *,
    expected_profile: GenerationProfile,
    expected_evaluation_config: EvaluationConfig,
) -> None:
    if (
        canonical_hash(expected_profile.model_dump(mode="json"))
        != provenance["identity"]["generation_profile_sha256"]
    ):
        raise ValueError("generation profile config mismatch with frozen evaluation")
    if (
        canonical_hash(expected_evaluation_config.model_dump(mode="json"))
        != provenance["identity"]["evaluation_config_sha256"]
    ):
        raise ValueError("evaluation config mismatch with frozen evaluation")


def _verified_artifact_path(
    evaluation_directory: Path, name: str, descriptor: object
) -> Path:
    try:
        relative_path = Path(str(descriptor["path"]))  # type: ignore[index]
        expected_hash = str(descriptor["sha256"])  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid {name} artifact provenance") from error
    root = evaluation_directory.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"invalid {name} artifact path outside evaluation directory")
    if not path.is_file():
        raise ValueError(f"missing {name} artifact: {path}")
    if file_hash(path) != expected_hash:
        raise ValueError(f"{name} artifact hash mismatch")
    return path


def _load_manifest(path: Path) -> SplitManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        samples = tuple(
            EvaluationSample(
                sample_id=str(item["sample_id"]),
                account_id=str(item["account_id"]),
                decision_time=datetime.fromisoformat(str(item["decision_time"])),
                split=EvaluationSplit(str(item["split"])),
                y_true=bool(item["y_true"]),
                campaign_group_id=(
                    str(item["campaign_group_id"])
                    if item["campaign_group_id"] is not None
                    else None
                ),
            )
            for item in data["samples"]
        )
        purges = tuple(
            CampaignPurge(
                campaign_group_id=str(item["campaign_group_id"]),
                reason=str(item["reason"]),
            )
            for item in data.get("purged_campaign_groups", [])
        )
        return SplitManifest(
            train_cutoff=datetime.fromisoformat(str(data["train_cutoff"])),
            validation_cutoff=datetime.fromisoformat(str(data["validation_cutoff"])),
            test_cutoff=datetime.fromisoformat(str(data["test_cutoff"])),
            samples=samples,
            purged_campaign_group_ids=tuple(data.get("purged_campaign_group_ids", [])),
            purged_campaign_groups=purges,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid frozen split manifest") from error


def _load_full_predictions(
    path: Path,
) -> tuple[tuple[PredictionRecord, ...], dict[str, float]]:
    frame = pl.read_parquet(path)
    expected_columns = {
        "sample_id",
        "account_id",
        "decision_time",
        "split",
        "y_true",
        "raw_model_score",
        "uncalibrated_probability",
    }
    if set(frame.columns) != expected_columns:
        raise ValueError("frozen full predictions have an incompatible schema")
    rows = frame.sort("decision_time", "sample_id").to_dicts()
    records = tuple(
        PredictionRecord(
            sample_id=str(item["sample_id"]),
            account_id=str(item["account_id"]),
            decision_time=item["decision_time"],
            split=EvaluationSplit(str(item["split"])),
            y_true=bool(item["y_true"]),
            score=float(item["uncalibrated_probability"]),
            model_variant="full",
        )
        for item in rows
    )
    raw_scores = {
        str(item["sample_id"]): float(item["raw_model_score"]) for item in rows
    }
    if len(raw_scores) != len(records):
        raise ValueError("frozen full predictions contain duplicate sample IDs")
    return records, raw_scores


def _validate_records_against_manifest(
    records: tuple[PredictionRecord, ...], manifest: SplitManifest
) -> None:
    expected = {
        sample.sample_id: (
            sample.account_id,
            sample.decision_time,
            sample.split,
            sample.y_true,
        )
        for sample in manifest.samples
    }
    actual = {
        record.sample_id: (
            record.account_id,
            record.decision_time,
            record.split,
            record.y_true,
        )
        for record in records
    }
    if actual != expected:
        raise ValueError("frozen full predictions do not match split manifest")
