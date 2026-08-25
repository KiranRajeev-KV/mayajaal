"""Tests for deterministic CatBoost fitting and SHAP explanations."""

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from mayajaal.baseline import (
    BaselineConfig,
    explain_prediction,
    global_shap_importance,
    label_vectors,
    predict_fraud_probability,
    save_baseline,
    train_baseline,
)
from mayajaal.features import FeatureService, FeatureVector, LabeledFeatureVector
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import GenerationProfile, generate_world


def trained_inputs() -> tuple[
    FeatureService, tuple[tuple[FeatureVector, ...], tuple[LabeledFeatureVector, ...]]
]:
    """Build enough deterministic normal and abuse examples for the baseline."""
    profile = GenerationProfile(
        seed=446,
        normal_account_count=3,
        shared_household_count=1,
        accounts_per_shared_household=3,
        promo_ring_count=1,
        refund_ring_count=1,
        mixed_ring_count=1,
        accounts_per_ring=3,
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    world = generate_world(profile)
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    service = FeatureService(build_graph_projection(world, resolution))
    vectors = service.extract_many(
        (str(account.id) for account in world.accounts), profile.end_at
    )
    return service, (vectors, label_vectors(vectors, world, profile.end_at))


class BaselineTests(unittest.TestCase):
    def test_training_predictions_and_shap_are_deterministic(self) -> None:
        service, (vectors, examples) = trained_inputs()
        config = BaselineConfig(iterations=12)
        first = train_baseline(examples, service.schema, config)
        second = train_baseline(examples, service.schema, config)
        first_probabilities = [
            predict_fraud_probability(first, vector) for vector in vectors
        ]
        second_probabilities = [
            predict_fraud_probability(second, vector) for vector in vectors
        ]
        self.assertEqual(first_probabilities, second_probabilities)

        explanation = explain_prediction(first, vectors[0])
        self.assertGreaterEqual(explanation.fraud_probability, 0.0)
        self.assertLessEqual(explanation.fraud_probability, 1.0)
        self.assertTrue(explanation.positive or explanation.negative)
        importance = global_shap_importance(first, vectors)
        self.assertEqual(len(importance), len(service.schema.names))

    def test_saves_model_metadata_and_shap_summary(self) -> None:
        service, (vectors, examples) = trained_inputs()
        baseline = train_baseline(
            examples, service.schema, BaselineConfig(iterations=8)
        )
        with TemporaryDirectory() as directory:
            artifacts = save_baseline(baseline, vectors, Path(directory))
            self.assertTrue(artifacts.model_path.is_file())
            self.assertTrue(artifacts.metadata_path.is_file())
            self.assertTrue(artifacts.shap_summary_path.is_file())
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["categorical_feature_names"],
                list(service.schema.categorical_names),
            )
