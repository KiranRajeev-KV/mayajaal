"""Reusable chronological held-out evaluation for Mayajaal model adapters."""

from .metrics import (
    evaluate_predictions,
    held_out_validity,
    prediction_frame,
    select_threshold,
)
from .models import (
    BenchmarkStatus,
    BenchmarkValidity,
    CampaignPurge,
    EvaluationConfig,
    EvaluationSample,
    EvaluationSplit,
    PredictionRecord,
    SplitManifest,
    SplitMetrics,
    ThresholdRule,
    ThresholdSelection,
)
from .provenance import (
    FrozenFullArtifactInput,
    FrozenFullEvaluation,
    canonical_hash,
    load_frozen_full_evaluation,
    provenance_base_model_id,
    verify_frozen_full_predictions,
)
from .runner import (
    GRAPH_IDENTITY_FEATURE_NAMES,
    LOCAL_IDENTITY_FEATURE_NAMES,
    RELATIONAL_GRAPH_FEATURE_NAMES,
    evaluate_catboost,
    fit_full_catboost_scores,
    save_catboost_evaluation_models,
    vectors_for_manifest,
    write_evaluation_artifacts,
)
from .sampling import build_split_manifest

__all__ = [
    "GRAPH_IDENTITY_FEATURE_NAMES",
    "LOCAL_IDENTITY_FEATURE_NAMES",
    "RELATIONAL_GRAPH_FEATURE_NAMES",
    "BenchmarkStatus",
    "BenchmarkValidity",
    "CampaignPurge",
    "EvaluationConfig",
    "EvaluationSample",
    "EvaluationSplit",
    "FrozenFullArtifactInput",
    "FrozenFullEvaluation",
    "PredictionRecord",
    "SplitManifest",
    "SplitMetrics",
    "ThresholdRule",
    "ThresholdSelection",
    "build_split_manifest",
    "canonical_hash",
    "evaluate_catboost",
    "evaluate_predictions",
    "fit_full_catboost_scores",
    "held_out_validity",
    "load_frozen_full_evaluation",
    "prediction_frame",
    "provenance_base_model_id",
    "save_catboost_evaluation_models",
    "select_threshold",
    "vectors_for_manifest",
    "verify_frozen_full_predictions",
    "write_evaluation_artifacts",
]
