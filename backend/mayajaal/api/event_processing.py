"""Durable webhook normalization and incremental, idempotent graph projection."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from os import environ
from typing import ClassVar, Protocol, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, field_validator

from mayajaal.graph import (
    GraphLoadReport,
    GraphProjection,
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
    """Map only explicitly namespaced Mayajaal synthetic fixtures to Event facts."""

    _types: ClassVar[dict[str, EventType]] = {
        "mayajaal.account.created": EventType.ACCOUNT_CREATED,
        "mayajaal.device.seen": EventType.DEVICE_SEEN,
        "mayajaal.ip.seen": EventType.IP_SEEN,
        "mayajaal.payment.attached": EventType.PAYMENT_ATTACHED,
    }

    def normalize(self, record: WebhookEventRecord) -> Event:
        event_type = self._types.get(record.event_type)
        if event_type is None:
            raise UnsupportedProviderEvent(
                f"unsupported synthetic provider event type: {record.event_type}"
            )
        envelope = RazorpayWebhookEnvelope.model_validate(record.payload)
        fixture = envelope.payload.get("mayajaal")
        if not isinstance(fixture, dict):
            raise UnsupportedProviderEvent(
                "missing mayajaal synthetic fixture metadata"
            )
        fixture_values = cast(dict[str, object], fixture)
        account_id = _fixture_uuid(fixture_values, "account_id")
        values: dict[str, object] = {
            "id": uuid5(
                _EVENT_NAMESPACE, f"mayajaal:webhook:{record.provider_event_id}"
            ),
            "event_type": event_type,
            "occurred_at": _aware(record.provider_created_at),
            "ingested_at": _aware(record.received_at),
            "account_id": account_id,
        }
        identity_keys = {
            EventType.DEVICE_SEEN: ("device_id", "device_id"),
            EventType.IP_SEEN: ("ip_address_id", "ip_address_id"),
            EventType.PAYMENT_ATTACHED: ("payment_identity_id", "payment_identity_id"),
        }
        if identity := identity_keys.get(event_type):
            values[identity[0]] = _fixture_uuid(fixture_values, identity[1])
        return Event.model_validate(values)

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
            projection = build_incremental_graph_projection(event, attributes)
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
                event, self._normalizer.runtime_identity_attributes(record)
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
            projection = build_incremental_graph_projection(event, attributes)
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
