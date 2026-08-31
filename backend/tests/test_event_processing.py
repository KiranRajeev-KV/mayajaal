"""High-value contracts for durable normalization and one-event graph writes."""

from datetime import UTC, datetime, timedelta
from unittest import TestCase

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mayajaal.api.db import (
    Base,
    NormalizedEventRepository,
    WebhookClaimUnavailable,
    WebhookEventRepository,
)
from mayajaal.api.event_processing import (
    RazorpayEventNormalizer,
    WebhookEventProcessor,
)
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
ADDRESS = "00000000-0000-0000-0000-0000000000ab"
PROMOTION = "00000000-0000-0000-0000-0000000000ac"


class _Graph:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.projections: list[GraphProjection] = []

    def load_incremental(self, projection: object) -> GraphLoadReport:
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
        self._accept_payload(
            "evt_malformed_order",
            "order.paid",
            ["payment", "order"],
            {
                "order": {"entity": {"entity": "order", "id": "order_bad"}},
                "payment": {"entity": {"entity": "payment", "id": "pay_bad"}},
            },
        )
        self.assertEqual(
            WebhookEventProcessor(self.sessions, _Graph())
            .process("evt_malformed_order")
            .status,
            WebhookProcessingStatus.FAILED,
        )
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

    def test_commerce_handlers_project_offline_graph_semantics_idempotently(
        self,
    ) -> None:
        order_provider_id = "order_commerce_001"
        self._accept_payload(
            "evt_order",
            "order.paid",
            ["payment", "order"],
            {
                "order": {
                    "entity": {
                        "entity": "order",
                        "id": order_provider_id,
                        "amount": 12_500,
                    }
                },
                "payment": {
                    "entity": {
                        "entity": "payment",
                        "id": "pay_commerce_001",
                        "order_id": order_provider_id,
                    }
                },
                "mayajaal": {
                    "account_id": ACCOUNT,
                    "shipping_address_id": ADDRESS,
                    "shipping_country_code": "in",
                },
            },
        )
        self._accept(
            "evt_promotion",
            "mayajaal.promotion.redeemed",
            {
                "account_id": ACCOUNT,
                "order_id": "7dc33ba8-cff4-5259-a8e2-a91f6000e8ca",
                "promotion_id": PROMOTION,
                "promotion_code": "SAVE10",
            },
        )
        for event_id, event_type in (
            ("evt_refund_created", "refund.created"),
            ("evt_refund_processed", "refund.processed"),
        ):
            self._accept_payload(
                event_id,
                event_type,
                ["refund", "payment"],
                {
                    "refund": {
                        "entity": {
                            "entity": "refund",
                            "id": "rfnd_commerce_001",
                            "amount": 12_500,
                        }
                    },
                    "payment": {
                        "entity": {
                            "entity": "payment",
                            "id": "pay_commerce_001",
                            "order_id": order_provider_id,
                        }
                    },
                    "mayajaal": {"account_id": ACCOUNT},
                },
            )

        graph = _Graph()
        processor = WebhookEventProcessor(self.sessions, graph)  # type: ignore[arg-type]
        results = tuple(
            processor.process(event_id)
            for event_id in (
                "evt_order",
                "evt_promotion",
                "evt_refund_created",
                "evt_refund_processed",
            )
        )
        self.assertEqual(
            tuple(result.canonical_event_type for result in results),
            (
                EventType.ORDER_PLACED,
                EventType.PROMOTION_REDEEMED,
                EventType.REFUND_REQUESTED,
                EventType.REFUND_RESOLVED,
            ),
        )
        order_projection = graph.projections[0]
        self.assertEqual(order_projection.nodes[1].properties["total_paise"], 12_500)
        self.assertEqual(order_projection.nodes[2].properties["country_code"], "IN")
        self.assertEqual(len(order_projection.relationships), 2)
        self.assertTrue(
            all(
                relationship.known_at == datetime(2026, 6, 1, tzinfo=UTC)
                for relationship in order_projection.relationships
            )
        )
        self.assertEqual(
            processor.process("evt_order").canonical_event_id,
            results[0].canonical_event_id,
        )
        self.assertEqual(graph.projections[0], graph.projections[-1])

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

    def test_runtime_categorical_metadata_is_canonical_or_fails_closed(self) -> None:
        self._accept(
            "evt_categorical",
            "mayajaal.device.seen",
            {
                "account_id": ACCOUNT,
                "device_id": DEVICE,
                "device_platform": "ANDROID",
                "device_type": "MOBILE",
            },
        )
        graph = _Graph()
        self.assertEqual(
            WebhookEventProcessor(self.sessions, graph)
            .process("evt_categorical")
            .status,
            WebhookProcessingStatus.PROCESSED,
        )
        device = graph.projections[0].nodes[1]
        self.assertEqual(device.properties["platform"], "android")
        self.assertEqual(device.properties["device_type"], "mobile")

        self._accept(
            "evt_bad_categorical",
            "mayajaal.device.seen",
            {
                "account_id": ACCOUNT,
                "device_id": DEVICE,
                "device_platform": "ANDRIOD",
            },
        )
        self.assertEqual(
            WebhookEventProcessor(self.sessions, _Graph())
            .process("evt_bad_categorical")
            .status,
            WebhookProcessingStatus.FAILED,
        )

    def test_identity_projection_does_not_clear_account_creation_metadata(self) -> None:
        self._accept("evt_account", "mayajaal.account.created", {"account_id": ACCOUNT})
        self._accept(
            "evt_device",
            "mayajaal.device.seen",
            {"account_id": ACCOUNT, "device_id": DEVICE},
        )
        with self.sessions() as session:
            account_record = WebhookEventRepository(session).get("evt_account")
            device_record = WebhookEventRepository(session).get("evt_device")
        assert account_record is not None and device_record is not None
        account = RazorpayEventNormalizer().normalize(account_record)
        device = RazorpayEventNormalizer().normalize(device_record)
        account_node = build_incremental_graph_projection(account).nodes[0]
        device_node = build_incremental_graph_projection(device).nodes[0]
        merged = {**account_node.properties, **device_node.properties}
        self.assertEqual(merged["created_at"], account.occurred_at)
        self.assertEqual(merged["created_known_at"], account.ingested_at)

    def test_processing_lease_reclaims_only_abandoned_work(self) -> None:
        self._accept(
            "evt_active",
            "mayajaal.device.seen",
            {"account_id": ACCOUNT, "device_id": DEVICE},
        )
        claimed_at = datetime.now(tz=UTC)
        with self.sessions.begin() as session:
            repository = WebhookEventRepository(session)
            repository.claim(
                "evt_active",
                claimed_at=claimed_at,
                lease_timeout=timedelta(minutes=5),
            )
        active_processor = WebhookEventProcessor(
            self.sessions,
            _Graph(),
            processing_lease_timeout=timedelta(minutes=5),
        )
        self.assertEqual(active_processor.process_next(limit=1), ())
        with self.sessions.begin() as session:
            record = WebhookEventRepository(session).get("evt_active")
            assert record is not None
            record.claimed_at = datetime(2020, 1, 1, tzinfo=UTC)
        recovered = active_processor.process_next(limit=1)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].status, WebhookProcessingStatus.PROCESSED)

    def test_next_claims_are_distinct_for_competing_processors(self) -> None:
        for event_id in ("evt_first", "evt_second"):
            self._accept(
                event_id,
                "mayajaal.device.seen",
                {"account_id": ACCOUNT, "device_id": DEVICE},
            )
        with self.sessions.begin() as session:
            first = WebhookEventRepository(session).claim_next(
                claimed_at=datetime.now(tz=UTC), lease_timeout=timedelta(minutes=5)
            )
        with self.sessions.begin() as session:
            second = WebhookEventRepository(session).claim_next(
                claimed_at=datetime.now(tz=UTC), lease_timeout=timedelta(minutes=5)
            )
        assert first is not None and second is not None
        self.assertEqual(
            (first.provider_event_id, second.provider_event_id),
            ("evt_first", "evt_second"),
        )

    def test_stale_worker_cannot_finalize_reclaimed_event(self) -> None:
        old_claim, new_claim = self._reclaim("evt_fenced")
        with self.sessions.begin() as session:
            repository = WebhookEventRepository(session)
            with self.assertRaises(WebhookClaimUnavailable):
                repository.mark_processed(
                    "evt_fenced",
                    expected_claimed_at=old_claim,
                    processed_at=datetime.now(tz=UTC),
                )
            with self.assertRaises(WebhookClaimUnavailable):
                repository.mark_failed(
                    "evt_fenced",
                    expected_claimed_at=old_claim,
                    detail="stale failure",
                )
            repository.mark_processed(
                "evt_fenced",
                expected_claimed_at=new_claim,
                processed_at=datetime.now(tz=UTC),
            )
        with self.sessions() as session:
            record = WebhookEventRepository(session).get("evt_fenced")
        assert record is not None
        self.assertEqual(record.status, WebhookProcessingStatus.PROCESSED.value)

    def _reclaim(self, event_id: str) -> tuple[datetime, datetime]:
        self._accept(
            event_id,
            "mayajaal.device.seen",
            {"account_id": ACCOUNT, "device_id": DEVICE},
        )
        new_claim = datetime.now(tz=UTC)
        old_claim = new_claim - timedelta(minutes=10)
        with self.sessions.begin() as session:
            repository = WebhookEventRepository(session)
            repository.claim(
                event_id,
                claimed_at=old_claim,
                lease_timeout=timedelta(minutes=5),
            )
        with self.sessions.begin() as session:
            WebhookEventRepository(session).claim(
                event_id,
                claimed_at=new_claim,
                lease_timeout=timedelta(minutes=5),
            )
        return old_claim, new_claim

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

    def _accept_payload(
        self,
        event_id: str,
        event_type: str,
        contains: list[str],
        payload: dict[str, object],
    ) -> None:
        envelope = RazorpayWebhookEnvelope.model_validate(
            {
                "entity": "event",
                "event": event_type,
                "contains": contains,
                "payload": payload,
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
