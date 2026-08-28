"""Verified feature-vector scoring through the frozen full-model boundary."""

from mayajaal.baseline import predict_raw_score
from mayajaal.evaluation.provenance import FrozenFullEvaluation
from mayajaal.features import FeatureVector

from .models import ScoreObservation
from .provenance import (
    SCORE_OBSERVATION_CONTRACT_VERSION,
    feature_vector_id,
    score_id,
    score_observation_semantics,
)


def score_feature_vector(
    frozen_evaluation: FrozenFullEvaluation,
    vector: FeatureVector,
) -> ScoreObservation:
    """Score one vector through the verified frozen model and exact schema."""
    raw_model_score = predict_raw_score(frozen_evaluation.baseline, vector)
    vector_id = feature_vector_id(frozen_evaluation.baseline.schema, vector)
    semantics = score_observation_semantics(
        score_contract_version=SCORE_OBSERVATION_CONTRACT_VERSION,
        base_model_id=frozen_evaluation.base_model_id,
        subject_id=vector.account_id,
        scoring_cutoff=vector.cutoff,
        raw_model_score=raw_model_score,
        feature_vector_id=vector_id,
    )
    return ScoreObservation(
        score_id=score_id(**semantics),
        base_model_id=frozen_evaluation.base_model_id,
        subject_id=vector.account_id,
        scoring_cutoff=vector.cutoff,
        raw_model_score=raw_model_score,
        feature_vector_id=vector_id,
    )


def verify_score_from_feature_vector(
    observation: ScoreObservation,
    frozen_evaluation: FrozenFullEvaluation,
    vector: FeatureVector,
) -> ScoreObservation:
    """Rescore a vector through trusted artifacts and reject any forged observation."""
    expected = score_feature_vector(frozen_evaluation, vector)
    if observation != expected:
        raise ValueError("score observation does not match verified feature scoring")
    return expected
