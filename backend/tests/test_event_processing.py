"""High-value contracts for durable normalization and one-event graph writes."""

from datetime import UTC, datetime
from unittest import TestCase

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mayajaal.api.db import Base, NormalizedEventRepository, WebhookEventRepository
from mayajaal.api.event_processing import RazorpayEventNormalizer, WebhookEventProcessor
from mayajaal.api.webhooks import (
    RazorpayWebhookEnvelope,
    WebhookInboxService,
    WebhookProcessingStatus,
)
from mayajaal.graph import (
    GraphLoadReport,
    GraphProjection,
    build_incremental_graph_projection,
)
from mayajaal.schemas import EventType

ACCOUNT = "00000000-0000-0000-0000-000000000001"
DEVICE = "00000000-0000-0000-0000-0000000000d1"
PAYMENT = "00000000-0000-0000-0000-0000000000aa"


class _Graph:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.projections: list[GraphProjection] = []

    def load(self, projection: object) -> GraphLoadReport:
        if not isinstance(projection, GraphProjection):
            raise TypeError("expected graph projection")
        if self.fail:
            raise RuntimeError("Neo4j unavailable")
        self.projections.append(projection)
        return GraphLoadReport(len(projection.nodes), len(projection.relationships))


class EventProcessingTests(TestCase):
    def setUp(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(bind=engine, expire_on_commit=False)

    def test_supported_fixture_is_deterministic_and_projects_one_event(self) -> None:
        self._accept(
            "evt_001",
            "mayajaal.device.seen",
            {"account_id": ACCOUNT, "device_id": DEVICE},
        )
        graph = _Graph()
        processor = WebhookEventProcessor(self.sessions, graph)  # type: ignore[arg-type]
        first = processor.process("evt_001")
        second = processor.process("evt_001")
        self.assertEqual(first.status, WebhookProcessingStatus.PROCESSED)
        self.assertEqual(first.canonical_event_id, second.canonical_event_id)
        self.assertEqual(first.canonical_event_type, EventType.DEVICE_SEEN)
        self.assertEqual(first.graph_relationships_written, 1)
        self.assertEqual(len(graph.projections), 2)
        self.assertEqual(graph.projections[0], graph.projections[1])
        with self.sessions() as session:
            event = NormalizedEventRepository(session).get_for_provider("evt_001")
            record = WebhookEventRepository(session).get("evt_001")
        assert event is not None and record is not None
        self.assertIsNone(event.synthetic_labels)
        self.assertEqual(record.status, WebhookProcessingStatus.PROCESSED.value)

    def test_unsupported_or_graph_failure_is_durable_failed_and_retryable(self) -> None:
        self._accept("evt_bad", "payment.captured", {"account_id": ACCOUNT})
        failed = WebhookEventProcessor(self.sessions, _Graph()).process("evt_bad")
        self.assertEqual(failed.status, WebhookProcessingStatus.FAILED)
        self._accept(
            "evt_retry",
            "mayajaal.payment.attached",
            {"account_id": ACCOUNT, "payment_identity_id": PAYMENT},
        )
        failing_graph = _Graph(fail=True)
        processor = WebhookEventProcessor(self.sessions, failing_graph)  # type: ignore[arg-type]
        self.assertEqual(
            processor.process("evt_retry").status, WebhookProcessingStatus.FAILED
        )
        recovered_graph = _Graph()
        recovered = WebhookEventProcessor(self.sessions, recovered_graph).process(
            "evt_retry"
        )  # type: ignore[arg-type]
        self.assertEqual(recovered.status, WebhookProcessingStatus.PROCESSED)
        self.assertEqual(recovered.graph_relationships_written, 1)

    def test_incremental_projection_preserves_event_occurrence_time(self) -> None:
        self._accept(
            "evt_time",
            "mayajaal.device.seen",
            {"account_id": ACCOUNT, "device_id": DEVICE},
        )
        with self.sessions() as session:
            record = WebhookEventRepository(session).get("evt_time")
            assert record is not None
            event = RazorpayEventNormalizer().normalize(record)
        projection = build_incremental_graph_projection(event)
        self.assertEqual(projection.relationships[0].event_time, event.occurred_at)

    def _accept(self, event_id: str, event_type: str, fixture: dict[str, str]) -> None:
        envelope = RazorpayWebhookEnvelope.model_validate(
            {
                "entity": "event",
                "event": event_type,
                "contains": ["payment"],
                "payload": {"mayajaal": fixture},
                "created_at": 1_780_000_000,
            }
        )
        with self.sessions.begin() as session:
            WebhookInboxService(session).accept(
                provider_event_id=event_id,
                envelope=envelope,
                raw_body=b"fixture",
                received_at=datetime(2026, 6, 1, tzinfo=UTC),
            )
