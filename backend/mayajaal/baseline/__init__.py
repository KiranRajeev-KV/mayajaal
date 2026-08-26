"""Deterministic CatBoost baseline and SHAP explanations."""

from .models import (
    BaselineArtifacts,
    BaselineConfig,
    FeatureContribution,
    GlobalFeatureImportance,
    PredictionExplanation,
    TrainedBaseline,
)
from .training import (
    explain_prediction,
    global_shap_importance,
    label_vectors,
    labels_at_cutoff,
    load_baseline,
    model_semantic_hash,
    predict_fraud_probability,
    predict_raw_score,
    save_baseline,
    save_shap_summary,
    train_baseline,
)

__all__ = [
    "BaselineArtifacts",
    "BaselineConfig",
    "FeatureContribution",
    "GlobalFeatureImportance",
    "PredictionExplanation",
    "TrainedBaseline",
    "explain_prediction",
    "global_shap_importance",
    "label_vectors",
    "labels_at_cutoff",
    "load_baseline",
    "model_semantic_hash",
    "predict_fraud_probability",
    "predict_raw_score",
    "save_baseline",
    "save_shap_summary",
    "train_baseline",
]
