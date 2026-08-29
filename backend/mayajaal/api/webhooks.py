"""Framework-independent Razorpay-shaped webhook contracts and inbox service."""

import hashlib
import hmac
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from sqlalchemy.orm import Session

from mayajaal.api.db import WebhookEventRecord, WebhookEventRepository
from mayajaal.schemas.common import SchemaModel

RAZORPAY_WEBHOOK_SECRET_ENVIRONMENT_VARIABLE = "MAYAJAAL_RAZORPAY_WEBHOOK_SECRET"
RAZORPAY_PROVIDER = "RAZORPAY"
ProviderEventId = Annotated[str, Field(min_length=1, max_length=255)]


class WebhookConfig(SchemaModel):
    """Secret-bearing webhook boundary sourced solely from the environment."""

    razorpay_webhook_secret: SecretStr

    @field_validator("razorpay_webhook_secret")
    @classmethod
    def require_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("MAYAJAAL_RAZORPAY_WEBHOOK_SECRET must not be empty")
        return value

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "WebhookConfig":
        source = os.environ if environment is None else environment
        secret = source.get(RAZORPAY_WEBHOOK_SECRET_ENVIRONMENT_VARIABLE)
        if secret is None or not secret:
            raise ValueError(
                f"{RAZORPAY_WEBHOOK_SECRET_ENVIRONMENT_VARIABLE} must be set"
            )
        return cls.model_validate({"razorpay_webhook_secret": secret})


class WebhookProcessingStatus(StrEnum):
    """Minimal durable inbox lifecycle; Stage 12A accepts only RECEIVED rows."""

    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class RazorpayWebhookEnvelope(SchemaModel):
    """Extensible validated common envelope for Razorpay-shaped inputs."""

    entity: str
    event: Annotated[str, Field(min_length=1, max_length=255)]
    contains: tuple[Annotated[str, Field(min_length=1)], ...]
    payload: dict[str, object]
    created_at: Annotated[int, Field(ge=0)]

    @field_validator("entity")
    @classmethod
    def require_event_entity(cls, value: str) -> str:
        if value != "event":
            raise ValueError("webhook envelope entity must be 'event'")
        return value

    def provider_created_datetime(self) -> datetime:
        """Translate the provider Unix timestamp without imposing delivery order."""
        return datetime.fromtimestamp(self.created_at, tz=UTC)


class WebhookIngestResult(SchemaModel):
    """Small service result separating new persistence from safe duplicates."""

    provider_event_id: ProviderEventId
    accepted_new: bool
    status: WebhookProcessingStatus


def verify_razorpay_signature(
    *, raw_body: bytes, signature: str | None, secret: str
) -> bool:
    """Verify an HMAC-SHA256 over *exact* received bytes in constant time."""
    if signature is None or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class WebhookInboxService:
    """Thin durable-ingestion boundary with no business event processing."""

    def __init__(self, session: Session) -> None:
        self._repository = WebhookEventRepository(session)

    def accept(
        self,
        *,
        provider_event_id: str,
        envelope: RazorpayWebhookEnvelope,
        raw_body: bytes,
        received_at: datetime,
    ) -> WebhookIngestResult:
        digest = hashlib.sha256(raw_body).hexdigest()
        _, accepted_new = self._repository.persist_received(
            provider_event_id=provider_event_id,
            provider=RAZORPAY_PROVIDER,
            event_type=envelope.event,
            provider_created_at=envelope.provider_created_datetime(),
            received_at=received_at,
            raw_body=raw_body,
            raw_body_sha256=digest,
            payload=envelope.model_dump(mode="json"),
        )
        return WebhookIngestResult(
            provider_event_id=provider_event_id,
            accepted_new=accepted_new,
            status=WebhookProcessingStatus.RECEIVED,
        )


def webhook_record_status(record: WebhookEventRecord) -> WebhookProcessingStatus:
    """Validate the persisted stable enum value at the storage boundary."""
    return WebhookProcessingStatus(record.status)
