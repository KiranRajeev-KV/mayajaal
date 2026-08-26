"""Reusable chronological held-out evaluation for Mayajaal model adapters."""

from .metrics import evaluate_predictions, prediction_frame, select_threshold
from .models import (
    EvaluationConfig,
    EvaluationSample,
    EvaluationSplit,
    PredictionRecord,
    SplitManifest,
    SplitMetrics,
    ThresholdRule,
    ThresholdSelection,
)
from .runner import (
    GRAPH_IDENTITY_FEATURE_NAMES,
    evaluate_catboost,
    save_catboost_evaluation_models,
    vectors_for_manifest,
    write_evaluation_artifacts,
)
from .sampling import build_split_manifest

__all__ = [
    "GRAPH_IDENTITY_FEATURE_NAMES",
    "EvaluationConfig",
    "EvaluationSample",
    "EvaluationSplit",
    "PredictionRecord",
    "SplitManifest",
    "SplitMetrics",
    "ThresholdRule",
    "ThresholdSelection",
    "build_split_manifest",
    "evaluate_catboost",
    "evaluate_predictions",
    "prediction_frame",
    "save_catboost_evaluation_models",
    "select_threshold",
    "vectors_for_manifest",
    "write_evaluation_artifacts",
]
