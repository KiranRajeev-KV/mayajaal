"""High-value tests for immutable operational database persistence."""

import unittest
from dataclasses import replace
from datetime import UTC, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mayajaal.api.db import (
    Base,
    DatabaseConfig,
    ImmutablePersistenceConflict,
    InvestigationReportRepository,
    InvestigationRequestRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    ScoreObservationRepository,
    session_scope,
)
from mayajaal.api.db.base import NAMING_CONVENTION
from mayajaal.api.db.models import ScoreObservationRecord
from mayajaal.calibration import (
    CalibrationConfig,
    ProbabilityEstimate,
    ProbabilityModel,
    SigmoidCalibrator,
    estimate_probability,
)
from mayajaal.investigation import (
    InvestigationPattern,
    InvestigationReport,
    InvestigationRequest,
    InvestigationStatus,
)
from mayajaal.policy import (
    DecisionContext,
    PolicyConfig,
    PolicyDecision,
    build_policy_model,
    decide,
)
from mayajaal.scoring import ScoreObservation, score_id, score_observation_semantics


class DatabaseFoundationTests(unittest.TestCase):
    """Protect the secret boundary, metadata, and immutable lineage behavior."""

    def test_database_url_is_environment_only_and_requires_sync_psycopg(self) -> None:
        with self.assertRaisesRegex(ValueError, "MAYAJAAL_DATABASE_URL must be set"):
            DatabaseConfig.from_environment({})

        with self.assertRaisesRegex(ValueError, "postgresql\\+psycopg"):
            DatabaseConfig.from_environment(
                {"MAYAJAAL_DATABASE_URL": "postgresql://user:password@localhost/db"}
            )

        config = DatabaseConfig.from_environment(
            {
                "MAYAJAAL_DATABASE_URL": (
                    "postgresql+psycopg://user:password@localhost:5433/mayajaal"
                )
            }
        )
        self.assertNotIn("password", repr(config))

    def test_session_scope_rolls_back_on_failure(self) -> None:
        engine = create_engine("sqlite://")
        sessions = sessionmaker(bind=engine)
        session = None
        try:
            with (
                self.assertRaisesRegex(RuntimeError, "rollback"),
                session_scope(sessions) as session,
            ):
                session.execute(text("SELECT 1"))
                raise RuntimeError("rollback")

            self.assertIsNotNone(session)
            self.assertFalse(session.in_transaction())
        finally:
            engine.dispose()

    def test_metadata_uses_named_constraints_and_registers_operational_tables(
        self,
    ) -> None:
        self.assertEqual(Base.metadata.naming_convention, NAMING_CONVENTION)
        self.assertEqual(
            set(Base.metadata.tables),
            {
                "score_observations",
                "probability_estimates",
                "policy_decisions",
                "investigation_requests",
                "investigation_reports",
            },
        )

    def test_immutable_lineage_round_trips_and_rejects_conflicting_score(
        self,
    ) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        score, estimate, decision, request, report = _lineage()
        try:
            with session_scope(sessions) as session:
                ScoreObservationRepository(session).persist(score)
                ProbabilityEstimateRepository(session).persist(estimate)
                PolicyDecisionRepository(session).persist(decision)
                InvestigationRequestRepository(session).persist(request)
                InvestigationReportRepository(session).persist(report)

            with session_scope(sessions) as session:
                self.assertEqual(
                    ScoreObservationRepository(session).get(score.score_id), score
                )
                self.assertEqual(
                    ProbabilityEstimateRepository(session).get(
                        estimate.probability_estimate_id
                    ),
                    estimate,
                )
                self.assertEqual(
                    PolicyDecisionRepository(session).get(decision.decision_id),
                    decision,
                )
                self.assertEqual(
                    InvestigationRequestRepository(session).get(decision.decision_id),
                    request,
                )
                self.assertEqual(
                    InvestigationReportRepository(session).get(decision.decision_id),
                    report,
                )
                with self.assertRaises(ImmutablePersistenceConflict):
                    ScoreObservationRepository(session).persist(
                        replace(score, raw_model_score=score.raw_model_score + 1.0)
                    )
        finally:
            engine.dispose()

    def test_corrupted_payload_fails_closed_during_repository_reload(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        score, _, _, _, _ = _lineage()
        try:
            with session_scope(sessions) as session:
                ScoreObservationRepository(session).persist(score)
            with session_scope(sessions) as session:
                row = session.get(ScoreObservationRecord, score.score_id)
                self.assertIsNotNone(row)
                assert row is not None
                row.payload = {"unexpected": "corruption"}
            with (
                session_scope(sessions) as session,
                self.assertRaisesRegex(ValueError, "authoritative validation"),
            ):
                _ = ScoreObservationRepository(session).get(score.score_id)
        finally:
            engine.dispose()


def _lineage() -> tuple[
    ScoreObservation,
    ProbabilityEstimate,
    PolicyDecision,
    InvestigationRequest,
    InvestigationReport,
]:
    cutoff = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    score_semantics = score_observation_semantics(
        score_contract_version=1,
        base_model_id="base-model-fixture",
        subject_id="account-fixture",
        scoring_cutoff=cutoff,
        raw_model_score=0.5,
        feature_vector_id="feature-vector-fixture",
    )
    score = ScoreObservation(
        score_id=score_id(**score_semantics),
        base_model_id="base-model-fixture",
        subject_id="account-fixture",
        scoring_cutoff=cutoff,
        raw_model_score=0.5,
        feature_vector_id="feature-vector-fixture",
    )
    probability_model = ProbabilityModel(
        base_model_id=score.base_model_id,
        probability_model_id="probability-model-fixture",
        calibration_config=CalibrationConfig(
            minimum_positive_samples=1, minimum_negative_samples=1
        ),
        calibrator=SigmoidCalibrator(coefficient=1.0, intercept=0.0),
        frozen_provenance=None,
    )
    estimate = estimate_probability(
        probability_model, score, scoring_context_id="context-fixture"
    )
    policy_model = build_policy_model(probability_model, PolicyConfig())
    decision = decide(
        policy_model,
        probability_model,
        score,
        estimate,
        DecisionContext(exposure_paise=10_000, context_id="context-fixture"),
    )
    request = InvestigationRequest.from_policy_decision(
        decision, probability_model, score, estimate
    )
    report = InvestigationReport(
        request=request,
        policy_action=request.policy_action,
        status=InvestigationStatus.COMPLETED,
        pattern=InvestigationPattern.INCONCLUSIVE,
        summary="No factual investigation evidence was supplied.",
    )
    return score, estimate, decision, request, report
