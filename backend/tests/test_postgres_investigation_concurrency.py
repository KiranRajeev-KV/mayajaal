"""PostgreSQL-only Stage 12E trigger-idempotency race contract."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier
from unittest import TestCase, skipUnless
from uuid import uuid4

from sqlalchemy import func, select

from mayajaal.api.contracts import InvestigationJob, InvestigationJobStatus, RiskCase
from mayajaal.api.db import (
    DatabaseConfig,
    InvestigationJobRepository,
    InvestigationRequestRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    RiskCaseRepository,
    ScoreObservationRepository,
    create_database_runtime,
)
from mayajaal.api.db.models import InvestigationJobRecord
from mayajaal.api.env import load_environment
from mayajaal.calibration import (
    CalibrationConfig,
    ProbabilityEstimate,
    ProbabilityModel,
    SigmoidCalibrator,
    estimate_probability,
)
from mayajaal.investigation import InvestigationRequest, InvestigationSubjectType
from mayajaal.policy import (
    DecisionContext,
    PolicyConfig,
    PolicyDecision,
    build_policy_model,
    decide,
)
from mayajaal.scoring import ScoreObservation, score_id, score_observation_semantics


@skipUnless(
    os.environ.get("MAYAJAAL_RUN_POSTGRES_TESTS") == "1",
    "set MAYAJAAL_RUN_POSTGRES_TESTS=1 to run PostgreSQL concurrency contracts",
)
class PostgreSQLInvestigationConcurrencyTests(TestCase):
    def setUp(self) -> None:
        load_environment()
        self.runtime = create_database_runtime(DatabaseConfig.from_environment())
        self.score, self.estimate, self.decision, self.request, self.case = _lineage()
        with self.runtime.sessions.begin() as session:
            ScoreObservationRepository(session).persist(self.score)
            ProbabilityEstimateRepository(session).persist(self.estimate)
            PolicyDecisionRepository(session).persist(self.decision)
            InvestigationRequestRepository(session).persist(self.request)
            RiskCaseRepository(session).persist(self.case)

    def tearDown(self) -> None:
        self.runtime.dispose()

    def test_concurrent_duplicate_trigger_creates_one_job(self) -> None:
        barrier = Barrier(2)

        def enqueue(run_id: str) -> tuple[str, bool]:
            with self.runtime.sessions.begin() as session:
                barrier.wait()
                job, created = InvestigationJobRepository(session).enqueue(
                    InvestigationJob(
                        run_id=run_id,
                        decision_id=self.decision.decision_id,
                        case_id=self.case.case_id,
                        idempotency_key="same-client-request",
                        status=InvestigationJobStatus.QUEUED,
                        created_at=datetime.now(tz=UTC),
                    )
                )
                return job.run_id, created

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(enqueue, (str(uuid4()), str(uuid4()))))
        self.assertEqual({result[0] for result in results}, {results[0][0]})
        self.assertEqual(sum(created for _, created in results), 1)
        with self.runtime.sessions() as session:
            self.assertEqual(
                session.scalar(
                    select(func.count(InvestigationJobRecord.run_id)).where(
                        InvestigationJobRecord.case_id == self.case.case_id,
                        InvestigationJobRecord.decision_id == self.decision.decision_id,
                        InvestigationJobRecord.idempotency_key == "same-client-request",
                    )
                ),
                1,
            )


def _lineage() -> tuple[
    ScoreObservation,
    ProbabilityEstimate,
    PolicyDecision,
    InvestigationRequest,
    RiskCase,
]:
    cutoff = datetime.now(tz=UTC)
    account = str(uuid4())
    vector_id = str(uuid4()).replace("-", "")
    score_semantics = score_observation_semantics(
        score_contract_version=1,
        base_model_id="base-model-fixture",
        subject_id=account,
        scoring_cutoff=cutoff,
        raw_model_score=0.5,
        feature_vector_id=vector_id,
    )
    score = ScoreObservation(
        score_id=score_id(**score_semantics),
        base_model_id="base-model-fixture",
        subject_id=account,
        scoring_cutoff=cutoff,
        raw_model_score=0.5,
        feature_vector_id=vector_id,
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
    estimate = estimate_probability(probability_model, score, scoring_context_id="ctx")
    decision = decide(
        build_policy_model(probability_model, PolicyConfig()),
        probability_model,
        score,
        estimate,
        DecisionContext(exposure_paise=10_000, context_id="ctx"),
    )
    request = InvestigationRequest.from_policy_decision(
        decision, probability_model, score, estimate
    )
    case = RiskCase(
        case_id=str(uuid4()),
        subject_type=InvestigationSubjectType.ACCOUNT,
        subject_id=account,
        opened_at=cutoff,
        opening_decision_id=decision.decision_id,
    )
    return score, estimate, decision, request, case
