"""High-value Stage 12D composition and recovery contracts."""

import json
from asyncio import run
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi.routing import APIRoute
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from mayajaal.api.app import create_app
from mayajaal.api.db import (
    Base,
    RiskEvaluationRecord,
    RiskProcessingFailureRepository,
    WebhookEventRepository,
)
from mayajaal.api.event_processing import WebhookEventProcessor
from mayajaal.api.realtime_pipeline import (
    RealtimePipelineState,
    RealtimeRiskPipelineService,
)
from mayajaal.api.risk_scoring import RiskEvaluationResult
from mayajaal.api.runtime import create_realtime_application_runtime
from mayajaal.api.webhooks import (
    RazorpayWebhookEnvelope,
    WebhookConfig,
    WebhookInboxService,
)
from mayajaal.graph import GraphLoadReport, GraphProjection
from mayajaal.schemas import EventType

ACCOUNT = "00000000-0000-0000-0000-0000000000aa"
DEVICE = "00000000-0000-0000-0000-0000000000bb"


class _Graph:
    def __init__(self) -> None:
        self.loads = 0

    def load_incremental(self, projection: GraphProjection) -> GraphLoadReport:
        self.loads += 1
        return GraphLoadReport(len(projection.nodes), len(projection.relationships))


class _Scoring:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def process(self, provider_event_id: str) -> RiskEvaluationResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("frozen model unavailable")
        return RiskEvaluationResult(
            provider_event_id, "decision-1", "case-1", self.calls > 1
        )


class RealtimePipelineTests(TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(bind=engine, expire_on_commit=False)
        self.graph = _Graph()
        self.scoring = _Scoring()
        self.pipeline = RealtimeRiskPipelineService(
            self.sessions,
            WebhookEventProcessor(self.sessions, self.graph),
            self.scoring,  # type: ignore[arg-type]
        )

    def test_identity_flows_once_and_replay_uses_processed_event(self) -> None:
        self._accept("device", "mayajaal.device.seen", device_fixture())
        first = self.pipeline.process("device")
        replay = self.pipeline.process("device")
        self.assertEqual(first.state, RealtimePipelineState.SCORED)
        self.assertEqual(replay.state, RealtimePipelineState.REUSED)
        self.assertEqual(self.scoring.calls, 2)
        self.assertEqual(self.graph.loads, 1)

    def test_account_creation_is_setup_only_and_stage_b_failure_never_scores(
        self,
    ) -> None:
        self._accept("account", "mayajaal.account.created", {"account_id": ACCOUNT})
        self._accept("bad", "not.supported", {"account_id": ACCOUNT})
        setup = self.pipeline.process("account")
        failed = self.pipeline.process("bad")
        self.assertEqual(setup.state, RealtimePipelineState.SETUP)
        self.assertEqual(failed.state, RealtimePipelineState.WEBHOOK_FAILED)
        self.assertEqual(self.scoring.calls, 0)

    def test_commerce_fact_uses_existing_scoring_pipeline(self) -> None:
        self._accept(
            "order",
            "mayajaal.order.placed",
            {
                "account_id": ACCOUNT,
                "order_id": "00000000-0000-0000-0000-0000000000cc",
                "address_id": "00000000-0000-0000-0000-0000000000dd",
                "total_paise": 100,
                "shipping_country_code": "IN",
                "exposure_paise": 100,
                "context_id": "order-context",
            },
        )
        result = self.pipeline.process("order")
        self.assertEqual(result.state, RealtimePipelineState.SCORED)
        self.assertEqual(result.canonical_event_type, EventType.ORDER_PLACED)
        self.assertEqual(self.scoring.calls, 1)

    def test_scoring_failure_is_durable_skipped_by_catch_up_and_explicitly_recovers(
        self,
    ) -> None:
        self._accept("device", "mayajaal.device.seen", device_fixture())
        failing = RealtimeRiskPipelineService(
            self.sessions,
            WebhookEventProcessor(self.sessions, self.graph),
            _Scoring(fail=True),  # type: ignore[arg-type]
        )
        result = failing.process("device")
        self.assertEqual(result.state, RealtimePipelineState.SCORING_FAILED)
        with self.sessions() as session:
            record = WebhookEventRepository(session).get("device")
            assert record is not None
            self.assertEqual(record.status, "PROCESSED")
            self.assertEqual(
                session.scalar(
                    select(func.count(RiskEvaluationRecord.provider_event_id))
                ),
                0,
            )
            failure = RiskProcessingFailureRepository(session).get("device")
            assert failure is not None
            self.assertEqual(failure.status, "FAILED")
            self.assertIn("frozen model unavailable", failure.failure_detail)
        self.assertEqual(self.pipeline.process_next(limit=10), ())
        recovered = self.pipeline.process("device")
        self.assertEqual(recovered.state, RealtimePipelineState.SCORED)
        with self.sessions() as session:
            self.assertIsNone(RiskProcessingFailureRepository(session).get("device"))

    def test_runtime_factory_runs_once_per_application_lifecycle(self) -> None:
        application_runtime = SimpleNamespace(
            database=SimpleNamespace(sessions=self.sessions),
            pipeline=self.pipeline,
            dispose=lambda: None,
        )
        app = create_app(
            webhook_config=WebhookConfig.model_validate(
                {"razorpay_webhook_secret": "test-secret"}
            )
        )
        with patch(
            "mayajaal.api.app.create_realtime_application_runtime",
            return_value=application_runtime,
        ) as factory:

            async def exercise() -> None:
                async with app.router.lifespan_context(app):
                    self.assertIs(app.state.realtime_runtime, application_runtime)

            run(exercise())
        factory.assert_called_once()

    def test_runtime_startup_verifies_neo4j_once_and_health_rejects_unavailable_graph(
        self,
    ) -> None:
        graph = SimpleNamespace(verify_connectivity=lambda: None, close=lambda: None)
        database = SimpleNamespace(sessions=self.sessions, dispose=lambda: None)
        frozen = SimpleNamespace(base_model_id="base")
        probability = SimpleNamespace(base_model_id="base")
        policy = SimpleNamespace(
            base_model_id="base", probability_model_id="probability"
        )
        probability.probability_model_id = "probability"
        configuration = SimpleNamespace(
            synthetic_world=SimpleNamespace(
                validation=SimpleNamespace(full_account_count=1)
            ),
            evaluation=object(),
        )
        with (
            patch(
                "mayajaal.api.runtime.load_generation_config",
                return_value=configuration,
            ),
            patch(
                "mayajaal.api.runtime.profile_for_total_accounts", return_value=object()
            ),
            patch(
                "mayajaal.api.runtime.load_frozen_full_evaluation", return_value=frozen
            ),
            patch(
                "mayajaal.api.runtime.load_probability_model", return_value=probability
            ),
            patch("mayajaal.api.runtime.load_policy_model", return_value=policy),
            patch(
                "mayajaal.api.runtime.Neo4jRuntimeConfig.from_environment",
                return_value=SimpleNamespace(
                    uri="bolt://test", username="neo4j", password="test"
                ),
            ),
            patch("mayajaal.api.runtime.Neo4jGraphRepository", return_value=graph),
            patch("mayajaal.api.runtime.WebhookEventProcessor"),
            patch("mayajaal.api.runtime.RuntimeRiskScoringService"),
            patch.object(
                graph, "verify_connectivity", wraps=graph.verify_connectivity
            ) as verify,
        ):
            runtime = create_realtime_application_runtime(database=database)  # type: ignore[arg-type]
        self.assertIs(runtime.graph, graph)
        verify.assert_called_once()

        unavailable = create_app()
        unavailable.state.database_runtime = SimpleNamespace(
            engine=create_engine("sqlite://"), sessions=self.sessions
        )
        unavailable.state.realtime_runtime = SimpleNamespace(
            graph=SimpleNamespace(
                verify_connectivity=lambda: (_ for _ in ()).throw(RuntimeError("down"))
            )
        )
        endpoint = next(
            route.endpoint
            for route in unavailable.routes
            if isinstance(route, APIRoute) and route.path == "/health"
        )
        from fastapi import HTTPException
        from starlette.requests import Request

        with self.assertRaises(HTTPException) as failure:
            endpoint(Request({"type": "http", "app": unavailable}))
        self.assertEqual(failure.exception.status_code, 503)

    def test_result_endpoint_exposes_only_persisted_outcome(self) -> None:
        self._accept("received", "mayajaal.device.seen", device_fixture())
        app = create_app()
        endpoint = next(
            route.endpoint
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path == "/webhooks/events/{provider_event_id}/result"
        )
        with self.sessions() as session:
            response = endpoint("received", session)
        self.assertEqual(response.provider_event_id, "received")
        self.assertEqual(response.processing_status.value, "RECEIVED")
        self.assertIsNone(response.decision_id)
        self.assertIsNone(response.calibrated_probability)
        self.assertEqual(response.pipeline_state, RealtimePipelineState.PROCESSING)
        with self.sessions.begin() as session:
            RiskProcessingFailureRepository(session).persist_failed(
                "received", attempted_at=datetime.now(tz=UTC), detail="unavailable"
            )
        with self.sessions() as session:
            failed = endpoint("received", session)
        self.assertEqual(failed.pipeline_state, RealtimePipelineState.SCORING_FAILED)

    def _accept(
        self, event_id: str, event_type: str, fixture: dict[str, object]
    ) -> None:
        body = json.dumps(
            {
                "entity": "event",
                "event": event_type,
                "contains": ["payment"],
                "payload": {"mayajaal": fixture},
                "created_at": 1_780_000_000,
            },
            separators=(",", ":"),
        ).encode()
        with self.sessions.begin() as session:
            WebhookInboxService(session).accept(
                provider_event_id=event_id,
                envelope=RazorpayWebhookEnvelope.model_validate_json(body),
                raw_body=body,
                received_at=datetime(2026, 8, 30, tzinfo=UTC),
            )


def device_fixture() -> dict[str, object]:
    return {
        "account_id": ACCOUNT,
        "device_id": DEVICE,
        "exposure_paise": 100,
        "context_id": "context-1",
    }
