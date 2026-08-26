"""Focused tests for chronological, reusable held-out evaluation contracts."""

import unittest
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mayajaal.baseline import BaselineConfig, predict_raw_score
from mayajaal.calibration import CalibrationConfig, CalibrationStatus, calibrate_records
from mayajaal.evaluation import (
    EvaluationConfig,
    EvaluationSample,
    EvaluationSplit,
    PredictionRecord,
    build_split_manifest,
    evaluate_catboost,
    evaluate_predictions,
    fit_full_catboost_scores,
    held_out_validity,
    save_catboost_evaluation_models,
    select_threshold,
    vectors_for_manifest,
    write_evaluation_artifacts,
)
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import GenerationProfile, generate_world
from mayajaal.synthetic.profile import PopulationProfile, PrevalenceProfile
from mayajaal.synthetic.world import SyntheticWorld

sklearn_metrics: Any = import_module("sklearn.metrics")


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


def campaign_accounts(world: SyntheticWorld) -> dict[str, tuple[str, ...]]:
    """Return deterministic hidden campaign membership for evaluation fixtures."""
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for event in world.events:
        labels = event.synthetic_labels
        if labels is not None and labels.is_coordinated_abuse:
            assert labels.coordination_cluster_id is not None
            groups[labels.coordination_cluster_id].add(str(event.account_id))
    return {
        group_id: tuple(sorted(account_ids))
        for group_id, account_ids in sorted(groups.items())
    }


def with_spanning_campaign(
    world: SyntheticWorld, current: GenerationProfile
) -> tuple[SyntheticWorld, str]:
    """Move labelled facts for one campaign into two target intervals."""
    validation_cutoff = current.start_at + (current.end_at - current.start_at) * 0.50
    groups = campaign_accounts(world)
    group_id = next(
        group_id for group_id, members in groups.items() if len(members) >= 2
    )
    members = set(groups[group_id][:2])
    updated_events = tuple(
        event.model_copy(
            update={
                "occurred_at": (
                    current.start_at + timedelta(days=1)
                    if str(event.account_id) == groups[group_id][0]
                    else validation_cutoff + timedelta(days=1)
                ),
                "ingested_at": (
                    current.start_at + timedelta(days=1, seconds=1)
                    if str(event.account_id) == groups[group_id][0]
                    else validation_cutoff + timedelta(days=1, seconds=1)
                ),
            }
        )
        if event.synthetic_labels is not None
        and event.synthetic_labels.coordination_cluster_id == group_id
        and str(event.account_id) in members
        else event
        for event in world.events
    )
    return replace(world, events=updated_events), group_id


class ChronologicalEvaluationTests(unittest.TestCase):
    def test_manifest_has_repeated_fixed_time_samples_and_is_deterministic(
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
                sample.decision_time == first.train_cutoff
                for sample in by_split[EvaluationSplit.TRAIN]
            )
        )
        self.assertTrue(
            all(
                sample.decision_time == first.validation_cutoff
                for sample in by_split[EvaluationSplit.VALIDATION]
            )
        )
        self.assertTrue(
            all(
                sample.decision_time == first.test_cutoff
                for sample in by_split[EvaluationSplit.TEST]
            )
        )
        accounts_by_id = {str(account.id): account for account in world.accounts}
        for samples in by_split.values():
            for sample in samples:
                created_at = accounts_by_id[sample.account_id].created_at
                self.assertLessEqual(created_at, sample.decision_time)
        samples_by_account: defaultdict[str, list[EvaluationSample]] = defaultdict(list)
        for sample in first.samples:
            samples_by_account[sample.account_id].append(sample)
        self.assertTrue(
            any(len(samples) > 1 for samples in samples_by_account.values())
        )
        self.assertTrue(
            all(
                [sample.decision_time for sample in samples]
                == sorted(sample.decision_time for sample in samples)
                for samples in samples_by_account.values()
            )
        )

        campaign_positive_splits: defaultdict[str, set[EvaluationSplit]] = defaultdict(
            set
        )
        for sample in first.samples:
            if sample.campaign_group_id is not None and sample.y_true:
                campaign_positive_splits[sample.campaign_group_id].add(sample.split)
        self.assertTrue(campaign_positive_splits)
        self.assertTrue(
            all(len(splits) == 1 for splits in campaign_positive_splits.values())
        )

    def test_account_can_transition_from_negative_to_newly_positive(self) -> None:
        current, world, _ = prepared()
        manifest = build_split_manifest(
            world,
            EvaluationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
            start_at=current.start_at,
            end_at=current.end_at,
        )
        samples_by_account: defaultdict[str, list[EvaluationSample]] = defaultdict(list)
        for sample in manifest.samples:
            samples_by_account[sample.account_id].append(sample)
        self.assertTrue(
            any(
                any(not sample.y_true for sample in samples)
                and any(sample.y_true for sample in samples)
                for samples in samples_by_account.values()
            )
        )

    def test_campaigns_spanning_target_intervals_are_purged_with_reason(self) -> None:
        current, world, _ = prepared()
        changed_world, group_id = with_spanning_campaign(world, current)
        config = EvaluationConfig(
            minimum_positive_samples=1, minimum_negative_samples=1
        )
        first = build_split_manifest(
            changed_world, config, start_at=current.start_at, end_at=current.end_at
        )
        second = build_split_manifest(
            changed_world, config, start_at=current.start_at, end_at=current.end_at
        )
        self.assertEqual(first, second)
        self.assertIn(group_id, first.purged_campaign_group_ids)
        self.assertFalse(
            any(sample.campaign_group_id == group_id for sample in first.samples)
        )
        self.assertEqual(
            [
                item.reason
                for item in first.purged_campaign_groups
                if item.campaign_group_id == group_id
            ],
            ["labelled_abuse_spans_multiple_target_intervals"],
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
                and (
                    sample.split is EvaluationSplit.TRAIN
                    or event.occurred_at
                    > {
                        EvaluationSplit.VALIDATION: manifest.train_cutoff,
                        EvaluationSplit.TEST: manifest.validation_cutoff,
                    }[sample.split]
                )
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
        self.assertTrue(selection.is_valid)
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
        self.assertFalse(report.support_is_sufficient)
        self.assertTrue(all(warning.startswith("test:") for warning in report.warnings))

    def test_metrics_match_sklearn_for_tied_scores(self) -> None:
        records = (
            record("t0", EvaluationSplit.TEST, True, 0.8),
            record("t1", EvaluationSplit.TEST, False, 0.8),
            record("t2", EvaluationSplit.TEST, True, 0.4),
            record("t3", EvaluationSplit.TEST, False, 0.1),
        )
        labels = [1, 0, 1, 0]
        scores = [0.8, 0.8, 0.4, 0.1]
        report = evaluate_predictions(
            records,
            0.4,
            EvaluationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
        )
        assert report.average_precision is not None
        assert report.roc_auc is not None
        assert report.precision is not None
        assert report.recall is not None
        assert report.f1 is not None
        predicted = [int(score >= 0.4) for score in scores]
        expected_confusion = sklearn_metrics.confusion_matrix(
            labels, predicted, labels=(0, 1)
        )
        self.assertAlmostEqual(
            report.average_precision,
            float(sklearn_metrics.average_precision_score(labels, scores)),
        )
        self.assertAlmostEqual(
            report.roc_auc,
            float(sklearn_metrics.roc_auc_score(labels, scores)),
        )
        self.assertAlmostEqual(
            report.precision,
            float(sklearn_metrics.precision_score(labels, predicted, zero_division=0)),
        )
        self.assertAlmostEqual(
            report.recall,
            float(sklearn_metrics.recall_score(labels, predicted, zero_division=0)),
        )
        self.assertAlmostEqual(
            report.f1,
            float(sklearn_metrics.f1_score(labels, predicted, zero_division=0)),
        )
        self.assertEqual(
            (
                report.true_positive,
                report.false_positive,
                report.false_negative,
                report.true_negative,
            ),
            (
                int(expected_confusion[1, 1]),
                int(expected_confusion[0, 1]),
                int(expected_confusion[1, 0]),
                int(expected_confusion[0, 0]),
            ),
        )

    def test_degenerate_class_support_preserves_explicit_null_metrics(self) -> None:
        report = evaluate_predictions(
            (
                record("t0", EvaluationSplit.TEST, False, 0.8),
                record("t1", EvaluationSplit.TEST, False, 0.1),
            ),
            0.5,
            EvaluationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
        )
        self.assertIsNone(report.average_precision)
        self.assertIsNone(report.roc_auc)
        self.assertTrue(
            any(
                "Average Precision is undefined" in warning
                for warning in report.warnings
            )
        )
        self.assertTrue(
            any("ROC-AUC is undefined" in warning for warning in report.warnings)
        )

    def test_insufficient_validation_support_does_not_select_threshold(self) -> None:
        config = EvaluationConfig(
            minimum_positive_samples=2, minimum_negative_samples=2
        )
        selection = select_threshold(
            (
                record("v0", EvaluationSplit.VALIDATION, True, 0.9),
                record("v1", EvaluationSplit.VALIDATION, False, 0.2),
            ),
            config,
        )
        self.assertIsNone(selection.threshold)
        self.assertFalse(selection.is_valid)
        report = evaluate_predictions(
            (
                record("t0", EvaluationSplit.TEST, True, 0.9),
                record("t1", EvaluationSplit.TEST, False, 0.2),
            ),
            selection.threshold,
            config,
        )
        self.assertIsNone(report.f1)
        self.assertIsNone(report.true_positive)
        self.assertTrue(all(warning.startswith("test:") for warning in report.warnings))

    def test_insufficient_test_support_marks_benchmark_invalid(self) -> None:
        config = EvaluationConfig(
            minimum_positive_samples=2, minimum_negative_samples=2
        )
        threshold = select_threshold(
            (
                record("v0", EvaluationSplit.VALIDATION, True, 0.9),
                record("v1", EvaluationSplit.VALIDATION, True, 0.8),
                record("v2", EvaluationSplit.VALIDATION, False, 0.2),
                record("v3", EvaluationSplit.VALIDATION, False, 0.1),
            ),
            config,
        )
        train = evaluate_predictions(
            (
                record("r0", EvaluationSplit.TRAIN, True, 0.9),
                record("r1", EvaluationSplit.TRAIN, True, 0.8),
                record("r2", EvaluationSplit.TRAIN, False, 0.2),
                record("r3", EvaluationSplit.TRAIN, False, 0.1),
            ),
            threshold.threshold,
            config,
        )
        validation = evaluate_predictions(
            (
                record("v0", EvaluationSplit.VALIDATION, True, 0.9),
                record("v1", EvaluationSplit.VALIDATION, True, 0.8),
                record("v2", EvaluationSplit.VALIDATION, False, 0.2),
                record("v3", EvaluationSplit.VALIDATION, False, 0.1),
            ),
            threshold.threshold,
            config,
        )
        test = evaluate_predictions(
            (
                record("t0", EvaluationSplit.TEST, True, 0.9),
                record("t1", EvaluationSplit.TEST, False, 0.2),
            ),
            threshold.threshold,
            config,
        )
        validity = held_out_validity(
            {"fixture": threshold},
            {
                "fixture": {
                    EvaluationSplit.TRAIN: train,
                    EvaluationSplit.VALIDATION: validation,
                    EvaluationSplit.TEST: test,
                }
            },
        )
        self.assertEqual(validity.status.value, "INVALID")
        self.assertTrue(all("test support" in reason for reason in validity.reasons))

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
        no_relational = records["no_relational_graph"]
        self.assertEqual(
            [(item.sample_id, item.split, item.y_true) for item in full],
            [(item.sample_id, item.split, item.y_true) for item in no_graph],
        )
        self.assertEqual(
            [(item.sample_id, item.split, item.y_true) for item in full],
            [(item.sample_id, item.split, item.y_true) for item in no_relational],
        )
        self.assertLess(
            len(schemas["no_graph_identity"].names), len(schemas["full"].names)
        )
        self.assertTrue(
            {
                "device_count",
                "ip_address_count",
                "payment_identity_count",
                "address_count",
            }.issubset(schemas["no_relational_graph"].names)
        )
        self.assertFalse(
            {
                "shared_device_account_count",
                "max_identity_reuse_count",
                "identity_component_account_count",
                "shared_promotion_account_count",
                "recent_shared_identity_event_count",
            }.intersection(schemas["no_relational_graph"].names)
        )
        self.assertTrue(all(item.decision_time <= current.end_at for item in full))
        vectors = vectors_for_manifest(service, manifest)
        self.assertTrue(
            all(
                vectors[sample.sample_id].cutoff == sample.decision_time
                for sample in manifest.samples
            )
        )
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
                all(
                    artifact.shap_summary_path.is_file()
                    for artifact in model_artifacts.values()
                )
            )

    def test_test_labels_do_not_affect_model_or_threshold_selection(self) -> None:
        current, world, service = prepared()
        config = EvaluationConfig(
            minimum_positive_samples=1, minimum_negative_samples=1
        )
        manifest = build_split_manifest(
            world, config, start_at=current.start_at, end_at=current.end_at
        )
        altered_manifest = replace(
            manifest,
            samples=tuple(
                replace(sample, y_true=not sample.y_true)
                if sample.split is EvaluationSplit.TEST
                else sample
                for sample in manifest.samples
            ),
        )
        first_records, first_thresholds, _, _, _ = evaluate_catboost(
            service, manifest, config, baseline_config=BaselineConfig(iterations=4)
        )
        second_records, second_thresholds, _, _, _ = evaluate_catboost(
            service,
            altered_manifest,
            config,
            baseline_config=BaselineConfig(iterations=4),
        )
        self.assertEqual(first_thresholds, second_thresholds)
        for name in first_records:
            self.assertEqual(
                [
                    (record.sample_id, record.score)
                    for record in first_records[name]
                    if record.split is not EvaluationSplit.TEST
                ],
                [
                    (record.sample_id, record.score)
                    for record in second_records[name]
                    if record.split is not EvaluationSplit.TEST
                ],
            )

    def test_frozen_full_catboost_raw_scores_are_calibrated_from_validation_only(
        self,
    ) -> None:
        current, world, service = prepared()
        evaluation_config = EvaluationConfig(
            minimum_positive_samples=1, minimum_negative_samples=1
        )
        manifest = build_split_manifest(
            world,
            evaluation_config,
            start_at=current.start_at,
            end_at=current.end_at,
        )
        records, raw_scores, _, model = fit_full_catboost_scores(
            service,
            manifest,
            baseline_config=BaselineConfig(iterations=4),
        )
        vectors = vectors_for_manifest(service, manifest)
        before = dict(raw_scores)
        _, calibrated = calibrate_records(
            records,
            raw_scores,
            CalibrationConfig(minimum_positive_samples=1, minimum_negative_samples=1),
        )
        after = {
            sample.sample_id: predict_raw_score(model, vectors[sample.sample_id])
            for sample in manifest.samples
        }
        self.assertEqual(before, after)
        self.assertEqual(calibrated.fit.status, CalibrationStatus.VALID)


if __name__ == "__main__":
    _ = unittest.main()
