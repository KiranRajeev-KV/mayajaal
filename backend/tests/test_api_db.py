"""High-value tests for immutable operational database persistence."""

import unittest
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.requests import Request

from mayajaal.api.app import (
    CaseListResponse,
    HealthResponse,
    InvestigationReportResponse,
    InvestigationRunListResponse,
    create_app,
    get_session,
)
from mayajaal.api.contracts import RiskCase
from mayajaal.api.db import (
    Base,
    DatabaseConfig,
    DatabaseRuntime,
    ImmutablePersistenceConflict,
    InvestigationReportRepository,
    InvestigationRequestRepository,
    InvestigationRunRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    RiskCaseRepository,
    ScoreObservationRepository,
    session_scope,
)
from mayajaal.api.db.base import NAMING_CONVENTION
from mayajaal.api.db.models import ScoreObservationRecord
from mayajaal.api.orchestration import RuntimeLineagePersistenceService
from mayajaal.calibration import (
    CalibrationConfig,
    ProbabilityEstimate,
    ProbabilityModel,
    SigmoidCalibrator,
    estimate_probability,
)
from mayajaal.investigation import (
    InvestigationConfig,
    InvestigationExecution,
    InvestigationPattern,
    InvestigationReport,
    InvestigationRequest,
    InvestigationStatus,
    InvestigationSubjectType,
    investigation_id,
    report_id,
)
from mayajaal.investigation.ledger import EvidenceLedgerSnapshot
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
                "feature_vectors",
                "score_observations",
                "probability_estimates",
                "policy_decisions",
                "investigation_requests",
                "investigation_reports",
                "investigation_runs",
                "risk_cases",
                "risk_case_decisions",
                "webhook_events",
                "normalized_events",
                "risk_evaluations",
            },
        )

    def test_immutable_lineage_round_trips_and_rejects_conflicting_score(
        self,
    ) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        score, estimate, decision, request, execution = _lineage()
        case = _case(decision)
        try:
            with session_scope(sessions) as session:
                persisted = RuntimeLineagePersistenceService(session).persist_execution(
                    score_observation=score,
                    probability_estimate=estimate,
                    policy_decision=decision,
                    investigation_request=request,
                    execution=execution,
                    run_id="run-fixture-001",
                    started_at=datetime(2026, 8, 29, 12, 1, tzinfo=UTC),
                    completed_at=datetime(2026, 8, 29, 12, 2, tzinfo=UTC),
                    risk_case=case,
                )

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
                    InvestigationRunRepository(session).get("run-fixture-001"),
                    persisted.investigation_run,
                )
                self.assertEqual(
                    InvestigationReportRepository(session).get(
                        persisted.investigation_report.report_id
                    ),
                    persisted.investigation_report,
                )
                expected_investigation_id = investigation_id(
                    request=request,
                    config=execution.config,
                    agent_model_id=execution.agent_model_id,
                    snapshot=execution.snapshot,
                )
                self.assertEqual(
                    persisted.investigation_run.investigation_id,
                    expected_investigation_id,
                )
                self.assertEqual(
                    persisted.investigation_report.report_id,
                    report_id(expected_investigation_id, execution.report),
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

    def test_risk_case_allows_only_idempotent_open_to_closed_transition(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        _, _, decision, _, _ = _lineage()
        risk_case = _case(decision)
        closed_at = datetime(2026, 8, 29, 13, tzinfo=UTC)
        try:
            with session_scope(sessions) as session:
                repository = RiskCaseRepository(session)
                repository.persist(risk_case)
                closed = repository.close_case(risk_case.case_id, closed_at)
                self.assertEqual(closed.status.value, "CLOSED")
                self.assertEqual(closed.closed_at, closed_at)
                self.assertEqual(
                    repository.close_case(risk_case.case_id, closed_at), closed
                )
                with self.assertRaisesRegex(ValueError, "already closed"):
                    repository.close_case(
                        risk_case.case_id, datetime(2026, 8, 29, 14, tzinfo=UTC)
                    )
            with session_scope(sessions) as session:
                restored = RiskCaseRepository(session).get(risk_case.case_id)
                self.assertEqual(restored, closed)
        finally:
            engine.dispose()

    def test_one_decision_supports_multiple_runs_reports_and_read_api(self) -> None:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        score, estimate, decision, request, first_execution = _lineage()
        second_execution = replace(first_execution, agent_model_id="injected:second")
        case = _case(decision)
        try:
            with session_scope(sessions) as session:
                service = RuntimeLineagePersistenceService(session)
                first = service.persist_execution(
                    score_observation=score,
                    probability_estimate=estimate,
                    policy_decision=decision,
                    investigation_request=request,
                    execution=first_execution,
                    run_id="run-fixture-001",
                    started_at=datetime(2026, 8, 29, 12, 1, tzinfo=UTC),
                    risk_case=case,
                )
                second = service.persist_execution(
                    score_observation=score,
                    probability_estimate=estimate,
                    policy_decision=decision,
                    investigation_request=request,
                    execution=second_execution,
                    run_id="run-fixture-002",
                    started_at=datetime(2026, 8, 29, 12, 3, tzinfo=UTC),
                    risk_case=case,
                )
                self.assertNotEqual(
                    first.investigation_report.report_id,
                    second.investigation_report.report_id,
                )
                self.assertEqual(
                    len(
                        InvestigationRunRepository(session).list_for_case(
                            case.case_id, limit=10
                        )
                    ),
                    2,
                )

            runtime = _runtime(engine, sessions)
            app = create_app(runtime)
            app.state.database_runtime = runtime
            request = Request({"type": "http", "app": app})
            session_dependency = get_session(request)
            request_session = next(session_dependency)
            try:
                health = cast(
                    Callable[[Request], HealthResponse], _endpoint(app, "/health")
                )
                self.assertEqual(health(request).status, "ready")
                list_cases = cast(
                    Callable[..., CaseListResponse], _endpoint(app, "/cases")
                )
                cases = list_cases(request_session, limit=50, offset=0)
                self.assertEqual([item.case_id for item in cases.items], [case.case_id])
                list_runs = cast(
                    Callable[..., InvestigationRunListResponse],
                    _endpoint(app, "/cases/{case_id}/investigations"),
                )
                runs = list_runs(case.case_id, request_session, limit=50, offset=0)
                self.assertEqual(
                    [item.run_id for item in runs.items],
                    ["run-fixture-002", "run-fixture-001"],
                )
                get_report = cast(
                    Callable[..., InvestigationReportResponse],
                    _endpoint(app, "/investigations/{run_id}/report"),
                )
                report = get_report("run-fixture-001", request_session)
                self.assertEqual(report.report_id, first.investigation_report.report_id)
                get_case = cast(
                    Callable[..., object], _endpoint(app, "/cases/{case_id}")
                )
                with self.assertRaises(HTTPException) as error:
                    get_case("missing", request_session)
                self.assertEqual(error.exception.status_code, 404)
            finally:
                session_dependency.close()
        finally:
            engine.dispose()

    def test_orchestration_rolls_back_complete_lineage_on_late_conflict(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        score, estimate, decision, request, execution = _lineage()
        try:
            with (
                self.assertRaises(ImmutablePersistenceConflict),
                session_scope(sessions) as session,
            ):
                service = RuntimeLineagePersistenceService(session)
                service.persist_execution(
                    score_observation=score,
                    probability_estimate=estimate,
                    policy_decision=decision,
                    investigation_request=request,
                    execution=execution,
                    run_id="run-rollback-001",
                    started_at=datetime(2026, 8, 29, 12, 1, tzinfo=UTC),
                )
                service.persist_execution(
                    score_observation=score,
                    probability_estimate=estimate,
                    policy_decision=decision,
                    investigation_request=request,
                    execution=execution,
                    run_id="run-rollback-002",
                    started_at=datetime(2026, 8, 29, 12, 2, tzinfo=UTC),
                )
            with session_scope(sessions) as session:
                self.assertIsNone(
                    ScoreObservationRepository(session).get(score.score_id)
                )
        finally:
            engine.dispose()


def _lineage() -> tuple[
    ScoreObservation,
    ProbabilityEstimate,
    PolicyDecision,
    InvestigationRequest,
    InvestigationExecution,
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
    return (
        score,
        estimate,
        decision,
        request,
        InvestigationExecution(
            report=report,
            snapshot=EvidenceLedgerSnapshot(evidence=(), tool_trace=()),
            agent_model_id="injected:fixture",
            config=InvestigationConfig(),
        ),
    )


def _case(decision: PolicyDecision) -> RiskCase:
    return RiskCase(
        case_id="case-fixture-001",
        subject_type=InvestigationSubjectType.ACCOUNT,
        subject_id=decision.subject_id,
        opened_at=datetime(2026, 8, 29, 12, 0, tzinfo=UTC),
        opening_decision_id=decision.decision_id,
    )


def _runtime(
    engine: Engine,
    sessions: sessionmaker[Session],
) -> DatabaseRuntime:
    return DatabaseRuntime(
        config=DatabaseConfig.from_environment(
            {
                "MAYAJAAL_DATABASE_URL": (
                    "postgresql+psycopg://user:password@localhost:5433/mayajaal"
                )
            }
        ),
        engine=engine,
        sessions=sessions,
    )


def _endpoint(app: object, path: str) -> object:
    """Locate one registered endpoint without a live HTTP client dependency."""
    routes = cast(FastAPI, app).routes
    for route in cast(list[object], routes):
        if isinstance(route, APIRoute) and route.path == path:
            return route.endpoint
    raise AssertionError(f"missing route: {path}")
