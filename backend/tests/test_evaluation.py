"""Focused tests for chronological, reusable held-out evaluation contracts."""

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from mayajaal.baseline import BaselineConfig
from mayajaal.evaluation import (
    EvaluationConfig,
    EvaluationSplit,
    PredictionRecord,
    build_split_manifest,
    evaluate_catboost,
    evaluate_predictions,
    select_threshold,
    save_catboost_evaluation_models,
    write_evaluation_artifacts,
)
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import GenerationProfile, generate_world
from mayajaal.synthetic.profile import PopulationProfile, PrevalenceProfile
from mayajaal.synthetic.world import SyntheticWorld


def profile() -> GenerationProfile:
    """A small target-rate world with campaigns in all chronological windows."""
    return GenerationProfile(
        seed=1821,
        normal_account_count=36,
        shared_household_count=0,
        promo_ring_count=1,
        refund_ring_count=1,
        mixed_ring_count=1,
        accounts_per_ring=3,
        population=PopulationProfile(benign_network_group_count=0),
        prevalence=PrevalenceProfile(
            target_labelled_account_rate=0.22,
            minimum_campaigns_per_timeline_bucket=1,
        ),
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 4, 1, tzinfo=UTC),
    )


def prepared() -> tuple[GenerationProfile, SyntheticWorld, FeatureService]:
    """Generate one resolved world behind the public feature-service API."""
    current = profile()
    world = generate_world(current)
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    return current, world, FeatureService(build_graph_projection(world, resolution))


def record(
    sample_id: str,
    split: EvaluationSplit,
    positive: bool,
    score: float,
) -> PredictionRecord:
    """Build a compact model-neutral prediction record fixture."""
    return PredictionRecord(
        sample_id=sample_id,
        account_id=sample_id,
        decision_time=datetime(2026, 1, 1, tzinfo=UTC),
        split=split,
        y_true=positive,
        score=score,
        model_variant="fixture",
    )


class ChronologicalEvaluationTests(unittest.TestCase):
    def test_manifest_is_strictly_chronological_group_disjoint_and_deterministic(
        self,
    ) -> None:
        current, world, _ = prepared()
        config = EvaluationConfig(
            minimum_positive_samples=1, minimum_negative_samples=1
        )
        first = build_split_manifest(
            world,
            config,
            start_at=current.start_at,
            end_at=current.end_at,
        )
        second = build_split_manifest(
            world,
            config,
            start_at=current.start_at,
            end_at=current.end_at,
        )
        self.assertEqual(first, second)
        by_split = {
            split: [sample for sample in first.samples if sample.split is split]
            for split in EvaluationSplit
        }
        self.assertTrue(by_split[EvaluationSplit.TRAIN])
        self.assertTrue(by_split[EvaluationSplit.VALIDATION])
        self.assertTrue(by_split[EvaluationSplit.TEST])
        self.assertTrue(
            all(
                sample.decision_time <= first.train_cutoff
                for sample in by_split[EvaluationSplit.TRAIN]
            )
        )
        self.assertTrue(
            all(
                first.train_cutoff < sample.decision_time <= first.validation_cutoff
                for sample in by_split[EvaluationSplit.VALIDATION]
            )
        )
        self.assertTrue(
            all(
                first.validation_cutoff < sample.decision_time <= first.test_cutoff
                for sample in by_split[EvaluationSplit.TEST]
            )
        )
        campaign_splits: dict[str, set[EvaluationSplit]] = {}
        for sample in first.samples:
            if sample.campaign_group_id is not None:
                campaign_splits.setdefault(sample.campaign_group_id, set()).add(
                    sample.split
                )
        self.assertTrue(campaign_splits)
        self.assertTrue(all(len(splits) == 1 for splits in campaign_splits.values()))
        self.assertEqual(
            len({sample.account_id for sample in first.samples}), len(first.samples)
        )

    def test_label_at_decision_time_never_reads_future_abuse(self) -> None:
        current, world, _ = prepared()
        manifest = build_split_manifest(
            world,
            EvaluationConfig(),
            start_at=current.start_at,
            end_at=current.end_at,
        )
        for sample in manifest.samples:
            expected = any(
                str(event.account_id) == sample.account_id
                and event.occurred_at <= sample.decision_time
                and event.synthetic_labels is not None
                and event.synthetic_labels.is_coordinated_abuse
                for event in world.events
            )
            self.assertEqual(sample.y_true, expected)

    def test_threshold_is_validation_only_and_metrics_are_correct(self) -> None:
        validation = (
            record("v0", EvaluationSplit.VALIDATION, True, 0.9),
            record("v1", EvaluationSplit.VALIDATION, False, 0.8),
            record("v2", EvaluationSplit.VALIDATION, False, 0.1),
        )
        selection = select_threshold(
            validation,
            EvaluationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
        )
        self.assertEqual(selection.threshold, 0.9)
        with self.assertRaisesRegex(ValueError, "validation records only"):
            _ = select_threshold(
                (*validation, record("t", EvaluationSplit.TEST, True, 0.7)),
                EvaluationConfig(),
            )
        report = evaluate_predictions(
            (
                record("t0", EvaluationSplit.TEST, True, 0.9),
                record("t1", EvaluationSplit.TEST, False, 0.8),
                record("t2", EvaluationSplit.TEST, False, 0.1),
            ),
            selection.threshold,
            EvaluationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
        )
        self.assertEqual(
            (
                report.true_positive,
                report.false_positive,
                report.false_negative,
                report.true_negative,
            ),
            (1, 0, 0, 2),
        )
        self.assertEqual(report.average_precision, 1.0)
        self.assertEqual(report.roc_auc, 1.0)
        self.assertEqual(report.f1, 1.0)

    def test_insufficient_support_is_reported_without_discarding_available_metrics(
        self,
    ) -> None:
        report = evaluate_predictions(
            (
                record("t0", EvaluationSplit.TEST, True, 0.9),
                record("t1", EvaluationSplit.TEST, False, 0.2),
            ),
            0.5,
            EvaluationConfig(minimum_positive_samples=5, minimum_negative_samples=5),
        )
        self.assertEqual(report.average_precision, 1.0)
        self.assertEqual(report.roc_auc, 1.0)
        self.assertTrue(all(warning.startswith("test:") for warning in report.warnings))

    def test_ablation_uses_identical_samples_and_cutoff_safe_vectors(self) -> None:
        current, world, service = prepared()
        manifest = build_split_manifest(
            world,
            EvaluationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
            start_at=current.start_at,
            end_at=current.end_at,
        )
        records, thresholds, reports, schemas, models = evaluate_catboost(
            service,
            manifest,
            EvaluationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
            baseline_config=BaselineConfig(iterations=4),
        )
        full = records["full"]
        no_graph = records["no_graph_identity"]
        self.assertEqual(
            [(item.sample_id, item.split, item.y_true) for item in full],
            [(item.sample_id, item.split, item.y_true) for item in no_graph],
        )
        self.assertLess(
            len(schemas["no_graph_identity"].names), len(schemas["full"].names)
        )
        self.assertTrue(all(item.decision_time <= current.end_at for item in full))
        repeated_records, repeated_thresholds, repeated_reports, _, _ = (
            evaluate_catboost(
                service,
                manifest,
                EvaluationConfig(
                    minimum_positive_samples=1, minimum_negative_samples=1
                ),
                baseline_config=BaselineConfig(iterations=4),
            )
        )
        self.assertEqual(records, repeated_records)
        self.assertEqual(thresholds, repeated_thresholds)
        self.assertEqual(reports, repeated_reports)
        with TemporaryDirectory() as directory:
            artifacts = write_evaluation_artifacts(
                Path(directory),
                manifest,
                records,
                thresholds,
                reports,
                schemas,
                EvaluationConfig(
                    minimum_positive_samples=1, minimum_negative_samples=1
                ),
                seed=current.seed,
            )
            self.assertTrue(all(path.is_file() for path in artifacts.values()))
            model_artifacts = save_catboost_evaluation_models(
                models,
                service,
                manifest,
                Path(directory) / "models",
                shap_sample_count=1,
            )
            self.assertTrue(
                all(artifact.shap_summary_path.is_file() for artifact in model_artifacts.values())
            )


if __name__ == "__main__":
    _ = unittest.main()
