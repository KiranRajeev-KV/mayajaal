"""High-value Stage 12C runtime scoring and episode contracts."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest import TestCase
from unittest.mock import patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from mayajaal.api.db import Base, RiskCaseRepository
from mayajaal.api.db.models import (
    FeatureVectorRecord,
    PolicyDecisionRecord,
    ProbabilityEstimateRecord,
    RiskCaseRecord,
    ScoreObservationRecord,
)
from mayajaal.api.event_processing import WebhookEventProcessor
from mayajaal.api.risk_scoring import RuntimeRiskScoringService
from mayajaal.api.webhooks import RazorpayWebhookEnvelope, WebhookInboxService
from mayajaal.calibration import ProbabilityEstimate, ProbabilityModel
from mayajaal.evaluation import FrozenFullEvaluation
from mayajaal.features import FeatureService, FeatureVector
from mayajaal.graph import (
    GraphLoadReport,
    GraphNode,
    GraphProjection,
    GraphRelationship,
)
from mayajaal.policy import DecisionContext, PolicyAction, PolicyDecision, PolicyModel
from mayajaal.scoring import ScoreObservation, feature_vector_id

ACCOUNT = "00000000-0000-0000-0000-000000000001"
DEVICE = "00000000-0000-0000-0000-0000000000d1"
NOW = datetime(2026, 6, 1, tzinfo=UTC)


class _Graph:
    def __init__(self) -> None:
        self._nodes: dict[tuple[object, str], GraphNode] = {}
        self._relationships: list[GraphRelationship] = []

    def load_incremental(self, projection: object) -> GraphLoadReport:
        assert isinstance(projection, GraphProjection)
        for node in projection.nodes:
            key = (node.node_type, node.canonical_id)
            previous = self._nodes.get(key)
            properties = {} if previous is None else dict(previous.properties)
            properties.update(node.properties)
            self._nodes[key] = GraphNode(node.node_type, node.canonical_id, properties)
        self._relationships.extend(projection.relationships)
        return GraphLoadReport(len(projection.nodes), len(projection.relationships))

    def feature_projection_at(self, cutoff: datetime) -> GraphProjection:
        return GraphProjection(tuple(self._nodes.values()), tuple(self._relationships))


class RuntimeRiskScoringTests(TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(bind=engine, expire_on_commit=False)
        self.graph = _Graph()

    def test_fresh_setup_then_identity_extracts_and_retries_immutable_lineage(
        self,
    ) -> None:
        self._accept("account", "mayajaal.account.created", {"account_id": ACCOUNT})
        self._accept(
            "device",
            "mayajaal.device.seen",
            {
                "account_id": ACCOUNT,
                "device_id": DEVICE,
                "exposure_paise": 250_000,
                "context_id": "device-1",
            },
        )
        processor = WebhookEventProcessor(self.sessions, self.graph)
        self.assertEqual(processor.process("account").status.value, "PROCESSED")
        self.assertEqual(processor.process("device").status.value, "PROCESSED")
        service = self._service(PolicyAction.REVIEW)
        with (
            patch("mayajaal.api.risk_scoring.score_feature_vector", self._score),
            patch("mayajaal.api.risk_scoring.estimate_probability", self._estimate),
            patch(
                "mayajaal.api.risk_scoring.decide", self._decision(PolicyAction.REVIEW)
            ),
        ):
            first = service.process("device")
            replay = service.process("device")
        self.assertFalse(first.reused)
        self.assertTrue(replay.reused)
        self.assertEqual(first.decision_id, replay.decision_id)
        self.assertEqual(first.case_id, replay.case_id)
        with self.sessions() as session:
            self.assertEqual(
                session.scalar(
                    select(func.count(FeatureVectorRecord.feature_vector_id))
                ),
                1,
            )
            self.assertEqual(
                session.scalar(select(func.count(ScoreObservationRecord.score_id))), 1
            )
            self.assertEqual(
                session.scalar(
                    select(
                        func.count(ProbabilityEstimateRecord.probability_estimate_id)
                    )
                ),
                1,
            )
            self.assertEqual(
                session.scalar(select(func.count(PolicyDecisionRecord.decision_id))), 1
            )
            self.assertEqual(
                session.scalar(select(func.count(RiskCaseRecord.case_id))), 1
            )

    def test_scoring_failure_rolls_back_trusted_lineage(self) -> None:
        self._prepare_identity()
        with (
            patch(
                "mayajaal.api.risk_scoring.score_feature_vector",
                side_effect=RuntimeError("model failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "model failed"),
        ):
            self._service(PolicyAction.REVIEW).process("device")
        with self.sessions() as session:
            self.assertEqual(
                session.scalar(
                    select(func.count(FeatureVectorRecord.feature_vector_id))
                ),
                0,
            )
            self.assertEqual(
                session.scalar(select(func.count(ScoreObservationRecord.score_id))), 0
            )

    def test_open_case_attaches_then_closed_case_opens_new_episode(self) -> None:
        service = self._service(PolicyAction.REVIEW)
        first = self._policy_decision("first", PolicyAction.REVIEW)
        second = self._policy_decision("second", PolicyAction.BLOCK)
        third = self._policy_decision("third", PolicyAction.REVIEW)
        with self.sessions.begin() as session:
            first_case = service._link_case(session, first, NOW)  # pyright: ignore[reportPrivateUsage]
            self.assertEqual(service._link_case(session, second, NOW), first_case)  # pyright: ignore[reportPrivateUsage]
            RiskCaseRepository(session).close_case(
                first_case or "", NOW + timedelta(minutes=1)
            )
            second_case = service._link_case(session, third, NOW + timedelta(minutes=2))  # pyright: ignore[reportPrivateUsage]
        self.assertIsNotNone(first_case)
        self.assertNotEqual(first_case, second_case)

    def test_allow_creates_no_case(self) -> None:
        with self.sessions.begin() as session:
            case_id = self._service(PolicyAction.ALLOW)._link_case(  # pyright: ignore[reportPrivateUsage]
                session, self._policy_decision("allow", PolicyAction.ALLOW), NOW
            )
        self.assertIsNone(case_id)

    def test_runtime_identity_attributes_feed_existing_categorical_features(
        self,
    ) -> None:
        self._accept("account", "mayajaal.account.created", {"account_id": ACCOUNT})
        self._accept(
            "device",
            "mayajaal.device.seen",
            {
                "account_id": ACCOUNT,
                "device_id": DEVICE,
                "device_platform": "ANDROID",
                "device_type": "MOBILE",
            },
        )
        self._accept(
            "payment",
            "mayajaal.payment.attached",
            {
                "account_id": ACCOUNT,
                "payment_identity_id": "00000000-0000-0000-0000-0000000000aa",
                "payment_method": "CARD",
            },
        )
        processor = WebhookEventProcessor(self.sessions, self.graph)
        for provider_event_id in ("account", "device", "payment"):
            processor.process(provider_event_id)
        values = (
            FeatureService(self.graph.feature_projection_at(NOW))
            .extract(ACCOUNT, NOW)
            .values
        )
        self.assertEqual(values["latest_device_platform"], "android")
        self.assertEqual(values["latest_device_type"], "mobile")
        self.assertEqual(values["latest_payment_method"], "card")

    def test_missing_runtime_identity_attributes_remain_missing_features(self) -> None:
        self._prepare_identity()
        values = (
            FeatureService(self.graph.feature_projection_at(NOW))
            .extract(ACCOUNT, NOW)
            .values
        )
        self.assertEqual(values["latest_device_platform"], "__missing__")
        self.assertEqual(values["latest_device_type"], "__missing__")
        self.assertEqual(values["latest_payment_method"], "__missing__")

    def _prepare_identity(self) -> None:
        self._accept("account", "mayajaal.account.created", {"account_id": ACCOUNT})
        self._accept(
            "device",
            "mayajaal.device.seen",
            {"account_id": ACCOUNT, "device_id": DEVICE, "exposure_paise": 1},
        )
        processor = WebhookEventProcessor(self.sessions, self.graph)
        processor.process("account")
        processor.process("device")

    def _accept(self, event_id: str, event: str, metadata: dict[str, object]) -> None:
        envelope = RazorpayWebhookEnvelope.model_validate(
            {
                "entity": "event",
                "event": event,
                "contains": ["payment"],
                "payload": {"mayajaal": metadata},
                "created_at": 1_780_000_000,
            }
        )
        with self.sessions.begin() as session:
            WebhookInboxService(session).accept(
                provider_event_id=event_id,
                envelope=envelope,
                raw_body=event_id.encode(),
                received_at=NOW,
            )

    def _service(self, action: PolicyAction) -> RuntimeRiskScoringService:
        return RuntimeRiskScoringService(
            self.sessions,
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
                    action=action,
                ),
            ),
        )

    @staticmethod
    def _score(_frozen: object, vector: FeatureVector) -> ScoreObservation:
        schema = FeatureService(_Graph().feature_projection_at(NOW)).schema
        # The runtime repository verifies the same authoritative schema below.
        vector_id = feature_vector_id(schema, vector)
        return ScoreObservation(
            "score-" + vector_id[:12],
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
            "estimate-" + score.score_id,
            score.score_id,
            score.subject_id,
            score.feature_vector_id,
            score.raw_model_score,
            0.5,
            score.scoring_cutoff,
            scoring_context_id,
        )

    @staticmethod
    def _decision(action: PolicyAction):
        def decide(
            _policy: object,
            _probability: object,
            score: ScoreObservation,
            estimate: ProbabilityEstimate,
            context: DecisionContext,
        ) -> PolicyDecision:
            return RuntimeRiskScoringTests._policy_decision(
                score.score_id, action, score=score, estimate=estimate, context=context
            )

        return decide

    @staticmethod
    def _policy_decision(
        name: str,
        action: PolicyAction,
        *,
        score: ScoreObservation | None = None,
        estimate: ProbabilityEstimate | None = None,
        context: DecisionContext | None = None,
    ) -> PolicyDecision:
        score = score or ScoreObservation(
            "score-" + name, "base", ACCOUNT, NOW, 0.5, "vector-" + name
        )
        estimate = estimate or ProbabilityEstimate(
            "base",
            "prob",
            "estimate-" + name,
            score.score_id,
            ACCOUNT,
            score.feature_vector_id,
            0.5,
            0.5,
            NOW,
        )
        return PolicyDecision(
            "policy",
            "base",
            "prob",
            estimate.probability_estimate_id,
            score.score_id,
            ACCOUNT,
            score.feature_vector_id,
            "decision-" + name,
            0.5,
            0.5,
            None,
            NOW,
            context or DecisionContext(exposure_paise=1),
            action,
            (),
            1.0,
            (),
            True,
        )
