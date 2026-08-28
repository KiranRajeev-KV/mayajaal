"""Deterministic semantic identities for feature-vector score observations."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime

from mayajaal.features import FeatureSchema, FeatureVector

from .models import ScoreObservation

SCORE_OBSERVATION_CONTRACT_VERSION = 1


def canonical_hash(value: object) -> str:
    """Return SHA-256 over stable JSON semantics, not storage presentation."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def feature_vector_id(schema: FeatureSchema, vector: FeatureVector) -> str:
    """Identify the exact schema-validated account feature input to a scorer."""
    schema.validate(vector.values)
    _require_aware_cutoff(vector.cutoff)
    if not vector.account_id:
        raise ValueError("feature vector account_id must be non-empty")
    return canonical_hash(
        {
            "feature_schema": [asdict(item) for item in schema.definitions],
            "account_id": vector.account_id,
            "cutoff": vector.cutoff.isoformat(),
            "values": {name: vector.values[name] for name in schema.names},
        }
    )


def score_observation_semantics(
    *,
    score_contract_version: int,
    base_model_id: str,
    subject_id: str,
    scoring_cutoff: datetime,
    raw_model_score: float,
    feature_vector_id: str,
) -> dict[str, object]:
    """Return every semantic input that defines one raw score observation."""
    cutoff = _require_aware_cutoff(scoring_cutoff)
    return {
        "score_contract_version": score_contract_version,
        "base_model_id": base_model_id,
        "subject_id": subject_id,
        "scoring_cutoff": cutoff.isoformat(),
        "raw_model_score": raw_model_score,
        "feature_vector_id": feature_vector_id,
    }


def score_id(**semantics: object) -> str:
    """Hash one model-bound score observation independently of file location."""
    return canonical_hash(semantics)


def verify_score_observation(observation: ScoreObservation) -> ScoreObservation:
    """Validate score semantic identity before a downstream contract consumes it."""
    semantics = score_observation_semantics(
        score_contract_version=SCORE_OBSERVATION_CONTRACT_VERSION,
        base_model_id=observation.base_model_id,
        subject_id=observation.subject_id,
        scoring_cutoff=observation.scoring_cutoff,
        raw_model_score=observation.raw_model_score,
        feature_vector_id=observation.feature_vector_id,
    )
    if observation.score_id != score_id(**semantics):
        raise ValueError("score observation semantics or raw score mismatch")
    return observation


def _require_aware_cutoff(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scoring_cutoff must include a timezone offset")
    return value
