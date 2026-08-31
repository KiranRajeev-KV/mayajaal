"""Durable webhook normalization and incremental, idempotent graph projection."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from os import environ
from typing import ClassVar, Protocol, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, field_validator

from mayajaal.graph import (
    GraphLoadReport,
    GraphProjection,
    RuntimeCommerceAttributes,
    RuntimeIdentityAttributes,
    build_incremental_graph_projection,
)
from mayajaal.resolution.normalizers import normalize_stable_identifier
from mayajaal.schemas import DevicePlatform, DeviceType, Event, EventType, PaymentMethod
from mayajaal.schemas.common import SchemaModel

from .db import (
    NormalizedEventRepository,
    WebhookClaimUnavailable,
    WebhookEventRecord,
    WebhookEventRepository,
)
from .db.session import SessionFactory
from .webhooks import RazorpayWebhookEnvelope, WebhookProcessingStatus

NEO4J_URI_ENVIRONMENT_VARIABLE = "MAYAJAAL_NEO4J_URI"
NEO4J_USERNAME_ENVIRONMENT_VARIABLE = "MAYAJAAL_NEO4J_USERNAME"
NEO4J_PASSWORD_ENVIRONMENT_VARIABLE = "MAYAJAAL_NEO4J_PASSWORD"
_EVENT_NAMESPACE = NAMESPACE_URL
DEFAULT_PROCESSING_LEASE_TIMEOUT = timedelta(minutes=5)
type _NormalizerHandler = Callable[[WebhookEventRecord, RazorpayWebhookEnvelope], Event]


class Neo4jRuntimeConfig(SchemaModel):
    """Environment-only connection boundary for the derived graph runtime."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = Field(min_length=1)

    @field_validator("uri", "username")
    @classmethod
    def require_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Neo4j connection setting cannot be empty")
        return value

    @classmethod
    def from_environment(cls) -> "Neo4jRuntimeConfig":
        return cls.model_validate(
            {
                "uri": environ.get(
                    NEO4J_URI_ENVIRONMENT_VARIABLE, "bolt://localhost:7687"
                ),
                "username": environ.get(NEO4J_USERNAME_ENVIRONMENT_VARIABLE, "neo4j"),
                "password": environ.get(
                    NEO4J_PASSWORD_ENVIRONMENT_VARIABLE, "mayajaal"
                ),
            }
        )


class UnsupportedProviderEvent(ValueError):
    """Raised for provider facts without an explicit truthful canonical mapping."""


class IncrementalGraphWriter(Protocol):
    """The narrow idempotent graph-write boundary used by the processor."""

    def load_incremental(self, projection: GraphProjection) -> GraphLoadReport: ...


class RazorpayEventNormalizer:
    """Dispatch truthful provider and namespaced simulator facts to Event.

    Razorpay does not provide a customer-account or shipping-address identity in
    these webhooks.  Those local graph bindings therefore stay in
    ``payload.mayajaal``; provider order/refund IDs and monetary amounts remain
    sourced from the provider entity payload.  Adding a supported delivery is a
    small handler, rather than another branch in a central conditional.
    """

    _types: ClassVar[dict[str, EventType]] = {
        "mayajaal.account.created": EventType.ACCOUNT_CREATED,
        "mayajaal.device.seen": EventType.DEVICE_SEEN,
        "mayajaal.ip.seen": EventType.IP_SEEN,
        "mayajaal.payment.attached": EventType.PAYMENT_ATTACHED,
        "mayajaal.order.placed": EventType.ORDER_PLACED,
        "mayajaal.promotion.redeemed": EventType.PROMOTION_REDEEMED,
        "mayajaal.refund.requested": EventType.REFUND_REQUESTED,
        "mayajaal.refund.resolved": EventType.REFUND_RESOLVED,
        "order.paid": EventType.ORDER_PLACED,
        "refund.created": EventType.REFUND_REQUESTED,
        "refund.processed": EventType.REFUND_RESOLVED,
    }

    def normalize(self, record: WebhookEventRecord) -> Event:
        envelope = RazorpayWebhookEnvelope.model_validate(record.payload)
        handler = self._handlers().get(record.event_type)
        if handler is None:
            raise UnsupportedProviderEvent(
                f"unsupported provider event type: {record.event_type}"
            )
        return handler(record, envelope)

    def _normalize_synthetic(
        self, record: WebhookEventRecord, envelope: RazorpayWebhookEnvelope
    ) -> Event:
        event_type = self._types[record.event_type]
        fixture = _mayajaal_metadata(envelope)
        values: dict[str, object] = self._event_values(record, event_type, fixture)
        identifiers = {
            EventType.DEVICE_SEEN: (("device_id", "device_id"),),
            EventType.IP_SEEN: (("ip_address_id", "ip_address_id"),),
            EventType.PAYMENT_ATTACHED: (
                ("payment_identity_id", "payment_identity_id"),
            ),
            EventType.ORDER_PLACED: (
                ("order_id", "order_id"),
                ("address_id", "address_id"),
            ),
            EventType.PROMOTION_REDEEMED: (
                ("order_id", "order_id"),
                ("promotion_id", "promotion_id"),
            ),
            EventType.REFUND_REQUESTED: (
                ("order_id", "order_id"),
                ("refund_id", "refund_id"),
            ),
            EventType.REFUND_RESOLVED: (
                ("order_id", "order_id"),
                ("refund_id", "refund_id"),
            ),
        }
        for event_key, fixture_key in identifiers.get(event_type, ()):
            values[event_key] = _fixture_uuid(fixture, fixture_key)
        return Event.model_validate(values)

    def _normalize_order_paid(
        self, record: WebhookEventRecord, envelope: RazorpayWebhookEnvelope
    ) -> Event:
        fixture = _mayajaal_metadata(envelope)
        order = _provider_entity(envelope.payload, "order")
        values = self._event_values(record, EventType.ORDER_PLACED, fixture)
        values["order_id"] = _provider_uuid(order, "id", "order")
        values["address_id"] = _fixture_uuid(fixture, "shipping_address_id")
        return Event.model_validate(values)

    def _normalize_refund(
        self, record: WebhookEventRecord, envelope: RazorpayWebhookEnvelope
    ) -> Event:
        fixture = _mayajaal_metadata(envelope)
        refund = _provider_entity(envelope.payload, "refund")
        payment = _provider_entity(envelope.payload, "payment")
        event_type = self._types[record.event_type]
        values = self._event_values(record, event_type, fixture)
        values["refund_id"] = _provider_uuid(refund, "id", "refund")
        values["order_id"] = _provider_uuid(payment, "order_id", "order")
        return Event.model_validate(values)

    def _event_values(
        self,
        record: WebhookEventRecord,
        event_type: EventType,
        fixture: dict[str, object],
    ) -> dict[str, object]:
        return {
            "id": uuid5(
                _EVENT_NAMESPACE, f"mayajaal:webhook:{record.provider_event_id}"
            ),
            "event_type": event_type,
            "occurred_at": _aware(record.provider_created_at),
            "ingested_at": _aware(record.received_at),
            "account_id": _fixture_uuid(fixture, "account_id"),
        }

    def runtime_identity_attributes(
        self, record: WebhookEventRecord
    ) -> RuntimeIdentityAttributes:
        """Map optional namespaced fixture facts to storage-neutral node metadata."""
        envelope = RazorpayWebhookEnvelope.model_validate(record.payload)
        fixture = envelope.payload.get("mayajaal")
        values = cast(dict[str, object], fixture) if isinstance(fixture, dict) else {}
        return RuntimeIdentityAttributes(
            device_platform=_optional_enum_value(
                values, "device_platform", DevicePlatform
            ),
            device_type=_optional_enum_value(values, "device_type", DeviceType),
            payment_method=_optional_enum_value(
                values, "payment_method", PaymentMethod
            ),
        )

    def runtime_commerce_attributes(
        self, record: WebhookEventRecord
    ) -> RuntimeCommerceAttributes:
        """Read only node properties required by existing commerce extractors."""
        envelope = RazorpayWebhookEnvelope.model_validate(record.payload)
        fixture = _mayajaal_metadata(envelope)
        if record.event_type == "order.paid":
            order = _provider_entity(envelope.payload, "order")
            return RuntimeCommerceAttributes(
                order_total_paise=_provider_int(order, "amount"),
                shipping_country_code=_country_code(fixture, "shipping_country_code"),
            )
        if record.event_type == "mayajaal.order.placed":
            return RuntimeCommerceAttributes(
                order_total_paise=_fixture_int(fixture, "total_paise"),
                shipping_country_code=_country_code(fixture, "shipping_country_code"),
            )
        if record.event_type == "mayajaal.promotion.redeemed":
            return RuntimeCommerceAttributes(
                promotion_code=_fixture_string(fixture, "promotion_code")
            )
        return RuntimeCommerceAttributes()

    def _handlers(self) -> dict[str, _NormalizerHandler]:
        return {
            "mayajaal.account.created": self._normalize_synthetic,
            "mayajaal.device.seen": self._normalize_synthetic,
            "mayajaal.ip.seen": self._normalize_synthetic,
            "mayajaal.payment.attached": self._normalize_synthetic,
            "mayajaal.order.placed": self._normalize_synthetic,
            "mayajaal.promotion.redeemed": self._normalize_synthetic,
            "mayajaal.refund.requested": self._normalize_synthetic,
            "mayajaal.refund.resolved": self._normalize_synthetic,
            "order.paid": self._normalize_order_paid,
            "refund.created": self._normalize_refund,
            "refund.processed": self._normalize_refund,
        }


def _mayajaal_metadata(envelope: RazorpayWebhookEnvelope) -> dict[str, object]:
    metadata = envelope.payload.get("mayajaal")
    if not isinstance(metadata, dict):
        raise UnsupportedProviderEvent("missing Mayajaal mapping metadata")
    return cast(dict[str, object], metadata)


def _provider_entity(payload: dict[str, object], key: str) -> dict[str, object]:
    resource = payload.get(key)
    if not isinstance(resource, dict):
        raise UnsupportedProviderEvent(f"missing Razorpay {key} payload")
    resource_values = cast(dict[str, object], resource)
    entity = resource_values.get("entity")
    if not isinstance(entity, dict):
        raise UnsupportedProviderEvent(f"missing Razorpay {key}.entity payload")
    values = cast(dict[str, object], entity)
    if values.get("entity") != key:
        raise UnsupportedProviderEvent(f"invalid Razorpay {key}.entity type")
    return values


def _provider_uuid(values: dict[str, object], key: str, resource_type: str) -> UUID:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise UnsupportedProviderEvent(f"missing Razorpay {resource_type} {key}")
    return uuid5(_EVENT_NAMESPACE, f"mayajaal:razorpay:{resource_type}:{value}")


def _provider_int(values: dict[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UnsupportedProviderEvent(f"invalid Razorpay numeric field {key}")
    return value


def _fixture_uuid(values: dict[str, object], key: str) -> UUID:
    value = values.get(key)
    if not isinstance(value, str):
        raise UnsupportedProviderEvent(f"missing synthetic fixture {key}")
    try:
        # Reuse the established exact stable-identifier rule. Address fuzzy
        # matching is deliberately inapplicable: these fixture references are
        # opaque canonical UUIDs, not raw identity observations.
        return UUID(normalize_stable_identifier(value))
    except ValueError as error:
        raise UnsupportedProviderEvent(f"invalid synthetic fixture {key}") from error


def _fixture_int(values: dict[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UnsupportedProviderEvent(f"invalid synthetic fixture {key}")
    return value


def _fixture_string(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise UnsupportedProviderEvent(f"invalid synthetic fixture {key}")
    return value


def _country_code(values: dict[str, object], key: str) -> str:
    value = _fixture_string(values, key)
    if len(value) != 2 or not value.isascii() or not value.isalpha():
        raise UnsupportedProviderEvent(f"invalid synthetic fixture {key}")
    return value.upper()


def _optional_enum_value[EnumT: DevicePlatform | DeviceType | PaymentMethod](
    values: dict[str, object], key: str, enum_type: type[EnumT]
) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise UnsupportedProviderEvent(f"invalid synthetic fixture {key}")
    try:
        return enum_type(value.lower()).value
    except ValueError as error:
        raise UnsupportedProviderEvent(f"invalid synthetic fixture {key}") from error


@dataclass(frozen=True)
class ProcessedWebhookEvent:
    """Concise deterministic result of one inbox processing attempt."""

    provider_event_id: str
    status: WebhookProcessingStatus
    canonical_event_id: str | None
    canonical_event_type: EventType | None
    graph_nodes_written: int
    graph_relationships_written: int


class WebhookEventProcessor:
    """Coordinate Postgres delivery state with idempotent derived Neo4j writes.

    PostgreSQL and Neo4j are deliberately not one transaction: canonical data is
    persisted before graph projection; only a successful idempotent projection
    advances the inbox to ``PROCESSED``. Failures remain durable and retryable.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        graph_repository: IncrementalGraphWriter,
        *,
        normalizer: RazorpayEventNormalizer | None = None,
        processing_lease_timeout: timedelta = DEFAULT_PROCESSING_LEASE_TIMEOUT,
    ) -> None:
        if processing_lease_timeout <= timedelta(0):
            raise ValueError("processing lease timeout must be positive")
        self._sessions = session_factory
        self._graph = graph_repository
        self._normalizer = normalizer or RazorpayEventNormalizer()
        self._processing_lease_timeout = processing_lease_timeout

    @property
    def processing_lease_timeout(self) -> timedelta:
        """Expose the existing claim lease to the bounded recovery coordinator."""
        return self._processing_lease_timeout

    def process(self, provider_event_id: str) -> ProcessedWebhookEvent:
        with self._sessions() as session:
            record = WebhookEventRepository(session).get(provider_event_id)
            if record is None:
                raise ValueError("webhook event does not exist")
            if record.status == WebhookProcessingStatus.PROCESSED.value:
                event = NormalizedEventRepository(session).get_for_provider(
                    provider_event_id
                )
                if event is None:
                    raise ValueError(
                        "processed webhook is missing its normalized event"
                    )
                return self._project_existing(record, event)
        with self._sessions.begin() as session:
            record = WebhookEventRepository(session).claim(
                provider_event_id,
                claimed_at=datetime.now(tz=UTC),
                lease_timeout=self._processing_lease_timeout,
            )
            claim_token = _claim_token(record)
            try:
                event = self._normalizer.normalize(record)
                attributes = self._normalizer.runtime_identity_attributes(record)
                commerce_attributes = self._normalizer.runtime_commerce_attributes(
                    record
                )
                NormalizedEventRepository(session).persist(
                    provider_event_id=provider_event_id, event=event
                )
            except Exception as error:
                WebhookEventRepository(session).mark_failed(
                    provider_event_id,
                    expected_claimed_at=claim_token,
                    detail=_failure_detail(error),
                )
                return ProcessedWebhookEvent(
                    provider_event_id, WebhookProcessingStatus.FAILED, None, None, 0, 0
                )
        try:
            projection = build_incremental_graph_projection(
                event, attributes, commerce_attributes
            )
            report = self._graph.load_incremental(projection)
        except Exception as error:
            with self._sessions.begin() as session:
                WebhookEventRepository(session).mark_failed(
                    provider_event_id,
                    expected_claimed_at=claim_token,
                    detail=_failure_detail(error),
                )
            return ProcessedWebhookEvent(
                provider_event_id,
                WebhookProcessingStatus.FAILED,
                str(event.id),
                event.event_type,
                0,
                0,
            )
        with self._sessions.begin() as session:
            WebhookEventRepository(session).mark_processed(
                provider_event_id,
                expected_claimed_at=claim_token,
                processed_at=datetime.now(tz=UTC),
            )
        return _result(provider_event_id, event, report)

    def process_next(self, *, limit: int) -> tuple[ProcessedWebhookEvent, ...]:
        results: list[ProcessedWebhookEvent] = []
        for _ in range(limit):
            with self._sessions.begin() as session:
                claimed = WebhookEventRepository(session).claim_next(
                    claimed_at=datetime.now(tz=UTC),
                    lease_timeout=self._processing_lease_timeout,
                )
            if claimed is None:
                break
            try:
                results.append(
                    self._process_claimed(
                        claimed.provider_event_id, _claim_token(claimed)
                    )
                )
            except WebhookClaimUnavailable:
                # A processor may have recovered or completed it after its lease.
                continue
        return tuple(results)

    def _project_existing(
        self, record: WebhookEventRecord, event: Event
    ) -> ProcessedWebhookEvent:
        report = self._graph.load_incremental(
            build_incremental_graph_projection(
                event,
                self._normalizer.runtime_identity_attributes(record),
                self._normalizer.runtime_commerce_attributes(record),
            )
        )
        return _result(record.provider_event_id, event, report)

    def _process_claimed(
        self, provider_event_id: str, claim_token: datetime
    ) -> ProcessedWebhookEvent:
        """Finish one row already atomically moved to ``PROCESSING``."""
        with self._sessions.begin() as session:
            record = WebhookEventRepository(session).get(provider_event_id)
            if (
                record is None
                or record.status != WebhookProcessingStatus.PROCESSING.value
                or record.claimed_at is None
                or _aware(record.claimed_at) != claim_token
            ):
                raise WebhookClaimUnavailable(
                    "webhook event is no longer claimed by this processor"
                )
            try:
                event = self._normalizer.normalize(record)
                attributes = self._normalizer.runtime_identity_attributes(record)
                commerce_attributes = self._normalizer.runtime_commerce_attributes(
                    record
                )
                NormalizedEventRepository(session).persist(
                    provider_event_id=provider_event_id, event=event
                )
            except Exception as error:
                WebhookEventRepository(session).mark_failed(
                    provider_event_id,
                    expected_claimed_at=claim_token,
                    detail=_failure_detail(error),
                )
                return ProcessedWebhookEvent(
                    provider_event_id, WebhookProcessingStatus.FAILED, None, None, 0, 0
                )
        try:
            projection = build_incremental_graph_projection(
                event, attributes, commerce_attributes
            )
            report = self._graph.load_incremental(projection)
        except Exception as error:
            with self._sessions.begin() as session:
                WebhookEventRepository(session).mark_failed(
                    provider_event_id,
                    expected_claimed_at=claim_token,
                    detail=_failure_detail(error),
                )
            return ProcessedWebhookEvent(
                provider_event_id,
                WebhookProcessingStatus.FAILED,
                str(event.id),
                event.event_type,
                0,
                0,
            )
        with self._sessions.begin() as session:
            WebhookEventRepository(session).mark_processed(
                provider_event_id,
                expected_claimed_at=claim_token,
                processed_at=datetime.now(tz=UTC),
            )
        return _result(provider_event_id, event, report)


def _result(
    provider_event_id: str, event: Event, report: GraphLoadReport
) -> ProcessedWebhookEvent:
    return ProcessedWebhookEvent(
        provider_event_id,
        WebhookProcessingStatus.PROCESSED,
        str(event.id),
        event.event_type,
        report.node_count,
        report.relationship_count,
    )


def _failure_detail(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:1000]


def _claim_token(record: WebhookEventRecord) -> datetime:
    if record.claimed_at is None:
        raise RuntimeError("claimed webhook event has no claim token")
    return _aware(record.claimed_at)


def _aware(value: datetime) -> datetime:
    """SQLite test storage strips offsets; PostgreSQL retains the UTC instant."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
