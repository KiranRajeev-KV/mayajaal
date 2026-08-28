"""Focused tests for verified feature-vector score observations."""

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from mayajaal.baseline import BaselineConfig, train_baseline
from mayajaal.evaluation.models import SplitManifest
from mayajaal.evaluation.provenance import FrozenFullEvaluation
from mayajaal.features import (
    FeatureDefinition,
    FeatureKind,
    FeatureSchema,
    FeatureVector,
    LabeledFeatureVector,
)
from mayajaal.scoring import (
    ScoreObservation,
    score_id,
    score_observation_semantics,
)
from mayajaal.scoring.service import (
    score_feature_vector,
    verify_score_from_feature_vector,
)


def cutoff() -> datetime:
    """Return an explicit point-in-time score cutoff."""
    return datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def schema() -> FeatureSchema:
    """Build a small schema suitable for a real CatBoost adapter fixture."""
    return FeatureSchema(
        (FeatureDefinition("order_count", FeatureKind.NUMERIC, "Order count."),)
    )


def vector(
    account_id: str, count: float, *, at: datetime | None = None
) -> FeatureVector:
    """Build one feature vector with an account-scoped cutoff."""
    return FeatureVector(
        account_id=account_id,
        cutoff=at or cutoff(),
        values={"order_count": count},
    )


def frozen_evaluation() -> FrozenFullEvaluation:
    """Return a minimal verified-model wrapper around a trained baseline."""
    examples = tuple(
        LabeledFeatureVector(vector(account_id, count), label)
        for account_id, count, label in (
            ("account-1", 0.0, False),
            ("account-2", 1.0, False),
            ("account-3", 4.0, True),
            ("account-4", 5.0, True),
        )
    )
    baseline = train_baseline(
        examples,
        schema(),
        BaselineConfig(iterations=4, depth=2, learning_rate=0.1),
    )
    with TemporaryDirectory() as directory:
        return FrozenFullEvaluation(
            evaluation_directory=Path(directory),
            manifest=SplitManifest(
                train_cutoff=cutoff(),
                validation_cutoff=cutoff(),
                test_cutoff=cutoff(),
                samples=(),
            ),
            records=(),
            raw_scores={},
            baseline=baseline,
            provenance={"base_model_id": "base-model-fixture"},
        )


class ScoreObservationTests(unittest.TestCase):
    def test_score_derives_account_and_cutoff_from_actual_feature_vector(self) -> None:
        frozen = frozen_evaluation()
        feature_vector = vector("account-123", 2.0)
        observation = score_feature_vector(frozen, feature_vector)
        self.assertEqual(observation.base_model_id, frozen.base_model_id)
        self.assertEqual(observation.subject_id, feature_vector.account_id)
        self.assertEqual(observation.scoring_cutoff, feature_vector.cutoff)
        self.assertEqual(
            verify_score_from_feature_vector(observation, frozen, feature_vector),
            observation,
        )

    def test_account_cutoff_and_forged_raw_score_change_or_fail_verification(
        self,
    ) -> None:
        frozen = frozen_evaluation()
        original_vector = vector("account-123", 2.0)
        original = score_feature_vector(frozen, original_vector)
        changed_account = score_feature_vector(frozen, vector("account-456", 2.0))
        changed_cutoff = score_feature_vector(
            frozen,
            vector("account-123", 2.0, at=datetime(2026, 5, 2, 12, 0, tzinfo=UTC)),
        )
        self.assertNotEqual(original.score_id, changed_account.score_id)
        self.assertNotEqual(original.score_id, changed_cutoff.score_id)

        forged_semantics = score_observation_semantics(
            score_contract_version=1,
            base_model_id=frozen.base_model_id,
            subject_id=original.subject_id,
            scoring_cutoff=original.scoring_cutoff,
            raw_model_score=original.raw_model_score + 1.0,
            feature_vector_id=original.feature_vector_id,
        )
        forged = ScoreObservation(
            score_id=score_id(**forged_semantics),
            base_model_id=frozen.base_model_id,
            subject_id=original.subject_id,
            scoring_cutoff=original.scoring_cutoff,
            raw_model_score=original.raw_model_score + 1.0,
            feature_vector_id=original.feature_vector_id,
        )
        with self.assertRaisesRegex(
            ValueError, "does not match verified feature scoring"
        ):
            _ = verify_score_from_feature_vector(forged, frozen, original_vector)


if __name__ == "__main__":
    _ = unittest.main()
