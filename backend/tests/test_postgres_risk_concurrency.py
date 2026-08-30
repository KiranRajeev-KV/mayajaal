"""PostgreSQL-only Stage 12C race contracts (opt in with MAYAJAAL_RUN_POSTGRES_TESTS)."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from types import SimpleNamespace
from typing import cast
from unittest import TestCase, skipUnless
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import func, select

from mayajaal.api.db import (
    DatabaseConfig,
    RiskCaseDecisionRecord,
    RiskCaseRecord,
    RiskCaseRepository,
    RiskEvaluationRecord,
    create_database_runtime,
)
from mayajaal.api.db.models import (
    FeatureVectorRecord,
    PolicyDecisionRecord,
    ProbabilityEstimateRecord,
    ScoreObservationRecord,
)
from mayajaal.api.env import load_environment
from mayajaal.api.event_processing import WebhookEventProcessor
from mayajaal.api.risk_scoring import RuntimeRiskScoringService
from mayajaal.api.webhooks import RazorpayWebhookEnvelope, WebhookInboxService
from mayajaal.calibration import ProbabilityEstimate, ProbabilityModel
from mayajaal.evaluation import FrozenFullEvaluation
from mayajaal.features import FeatureService, FeatureVector
from mayajaal.graph import (
    GraphLoadReport,
    GraphNode,
    GraphNodeType,
    GraphProjection,
    GraphRelationship,
    GraphRelationshipType,
)
from mayajaal.policy import DecisionContext, PolicyAction, PolicyDecision, PolicyModel
from mayajaal.scoring import ScoreObservation, feature_vector_id


@skipUnless(
    os.environ.get("MAYAJAAL_RUN_POSTGRES_TESTS") == "1",
    "set MAYAJAAL_RUN_POSTGRES_TESTS=1 to run PostgreSQL concurrency contracts",
)
class PostgreSQLRiskConcurrencyTests(TestCase):
    def setUp(self) -> None:
        load_environment()
        self.runtime = create_database_runtime(DatabaseConfig.from_environment())
        self.account = str(uuid4())
        self.now = datetime.now(tz=UTC) - timedelta(seconds=1)
        self.graph = _BarrierGraph(self.account, self.now)

    def tearDown(self) -> None:
        self.runtime.dispose()

    def test_same_event_and_open_case_races_converge(self) -> None:
        same_event = self._seed("same", "same-context")
        with self._patched_runtime(), ThreadPoolExecutor(max_workers=2) as pool:
            same_results = list(pool.map(self._service().process, (same_event,) * 2))
        self.assertEqual(
            {result.decision_id for result in same_results},
            {same_results[0].decision_id},
        )
        with self.runtime.sessions() as session:
            self.assertEqual(
                session.scalar(
                    select(func.count(RiskEvaluationRecord.provider_event_id)).where(
                        RiskEvaluationRecord.provider_event_id == same_event
                    )
                ),
                1,
            )
            self.assertEqual(
                session.scalar(
                    select(func.count(FeatureVectorRecord.feature_vector_id)).where(
                        FeatureVectorRecord.account_id == self.account
                    )
                ),
                1,
            )
            self.assertEqual(
                session.scalar(
                    select(func.count(ScoreObservationRecord.score_id)).where(
                        ScoreObservationRecord.subject_id == self.account
                    )
                ),
                1,
            )
            self.assertEqual(
                session.scalar(
                    select(
                        func.count(ProbabilityEstimateRecord.probability_estimate_id)
                    ).where(ProbabilityEstimateRecord.subject_id == self.account)
                ),
                1,
            )
            self.assertEqual(
                session.scalar(
                    select(func.count(PolicyDecisionRecord.decision_id)).where(
                        PolicyDecisionRecord.subject_id == self.account
                    )
                ),
                1,
            )
        with self.runtime.sessions.begin() as session:
            RiskCaseRepository(session).close_case(
                same_results[0].case_id or "", datetime.now(tz=UTC)
            )

        first = self._seed("case-a", "case-a")
        second = self._seed("case-b", "case-b")
        self.graph.reset()
        with self._patched_runtime(), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(self._service().process, (first, second)))
        case_ids = {result.case_id for result in results}
        self.assertEqual(len(case_ids), 1)
        case_id = next(iter(case_ids))
        assert case_id is not None
        with self.runtime.sessions() as session:
            self.assertEqual(
                session.scalar(
                    select(func.count(RiskCaseRecord.case_id)).where(
                        RiskCaseRecord.subject_id == self.account,
                        RiskCaseRecord.status == "OPEN",
                    )
                ),
                1,
            )
            self.assertEqual(
                session.scalar(
                    select(func.count(RiskCaseDecisionRecord.decision_id)).where(
                        RiskCaseDecisionRecord.case_id == case_id
                    )
                ),
                2,
            )

    def _seed(self, suffix: str, context_id: str) -> str:
        provider_event_id = f"postgres-risk-{suffix}-{uuid4()}"
        envelope = RazorpayWebhookEnvelope.model_validate(
            {
                "entity": "event",
                "event": "mayajaal.device.seen",
                "contains": ["payment"],
                "payload": {
                    "mayajaal": {
                        "account_id": self.account,
                        "device_id": str(uuid4()),
                        "exposure_paise": 1_000_000_000,
                        "context_id": f"{context_id}-{self.account}",
                    }
                },
                "created_at": int(self.now.timestamp()),
            }
        )
        with self.runtime.sessions.begin() as session:
            WebhookInboxService(session).accept(
                provider_event_id=provider_event_id,
                envelope=envelope,
                raw_body=provider_event_id.encode(),
                received_at=self.now,
            )
        WebhookEventProcessor(self.runtime.sessions, _Writer()).process(
            provider_event_id
        )
        return provider_event_id

    def _service(self) -> RuntimeRiskScoringService:
        return RuntimeRiskScoringService(
            self.runtime.sessions,
            self.graph,  # type: ignore[arg-type]
            cast(FrozenFullEvaluation, SimpleNamespace(base_model_id="base")),
            cast(
                ProbabilityModel,
                SimpleNamespace(base_model_id="base", probability_model_id="prob"),
            ),
            cast(
                PolicyModel,
                SimpleNamespace(
                    base_model_id="base",
                    probability_model_id="prob",
                    policy_id="policy",
                ),
            ),
        )

    def _patched_runtime(self):
        return patch.multiple(
            "mayajaal.api.risk_scoring",
            score_feature_vector=self._score,
            estimate_probability=self._estimate,
            decide=self._decide,
        )

    @staticmethod
    def _score(_frozen: object, vector: FeatureVector) -> ScoreObservation:
        schema = FeatureService(GraphProjection((), ())).schema
        vector_id = feature_vector_id(schema, vector)
        return ScoreObservation(
            "score-" + vector_id[:20],
            "base",
            vector.account_id,
            vector.cutoff,
            0.5,
            vector_id,
        )

    @staticmethod
    def _estimate(
        _model: object, score: ScoreObservation, *, scoring_context_id: str | None
    ) -> ProbabilityEstimate:
        return ProbabilityEstimate(
            "base",
            "prob",
            str(
                uuid5(NAMESPACE_URL, f"estimate:{score.score_id}:{scoring_context_id}")
            ),
            score.score_id,
            score.subject_id,
            score.feature_vector_id,
            score.raw_model_score,
            0.5,
            score.scoring_cutoff,
            scoring_context_id,
        )

    @staticmethod
    def _decide(
        _policy: object,
        _probability: object,
        score: ScoreObservation,
        estimate: ProbabilityEstimate,
        context: DecisionContext,
    ) -> PolicyDecision:
        return PolicyDecision(
            "policy",
            "base",
            "prob",
            estimate.probability_estimate_id,
            score.score_id,
            score.subject_id,
            score.feature_vector_id,
            "decision-" + (context.context_id or score.score_id),
            0.5,
            0.5,
            context.context_id,
            score.scoring_cutoff,
            context,
            PolicyAction.REVIEW,
            (),
            1.0,
            (),
            True,
        )


class _Writer:
    def load_incremental(self, projection: object) -> GraphLoadReport:
        assert isinstance(projection, GraphProjection)
        return GraphLoadReport(len(projection.nodes), len(projection.relationships))


class _BarrierGraph:
    def __init__(self, account_id: str, now: datetime) -> None:
        self._account_id, self._now = account_id, now
        self.reset()

    def reset(self) -> None:
        self._barrier = Barrier(2)

    def feature_projection_at(self, _cutoff: datetime) -> GraphProjection:
        self._barrier.wait(timeout=10)
        return GraphProjection(
            (
                GraphNode(
                    GraphNodeType.ACCOUNT,
                    self._account_id,
                    {"created_at": self._now, "created_known_at": self._now},
                ),
                GraphNode(GraphNodeType.DEVICE, "runtime-device", {}),
            ),
            (
                GraphRelationship(
                    GraphRelationshipType.USED_DEVICE,
                    GraphNodeType.ACCOUNT,
                    self._account_id,
                    GraphNodeType.DEVICE,
                    "runtime-device",
                    "runtime-edge",
                    "device_seen",
                    self._now,
                    self._now,
                ),
            ),
        )
