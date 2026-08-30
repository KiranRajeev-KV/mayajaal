"""Stage 12D composition of durable webhook processing and risk scoring."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from mayajaal.schemas import EventType

from .db import (
    NormalizedEventRepository,
    RiskProcessingFailureRepository,
    WebhookClaimUnavailable,
    WebhookEventRepository,
)
from .db.session import SessionFactory
from .event_processing import ProcessedWebhookEvent, WebhookEventProcessor
from .risk_scoring import RiskEvaluationResult, RuntimeRiskScoringService
from .webhooks import WebhookProcessingStatus


class RealtimePipelineState(StrEnum):
    """Concise outcome suitable for recovery commands and observability."""

    SCORED = "SCORED"
    REUSED = "REUSED"
    SETUP = "SETUP"
    WEBHOOK_FAILED = "WEBHOOK_FAILED"
    SCORING_FAILED = "SCORING_FAILED"
    PROCESSING = "PROCESSING"


@dataclass(frozen=True)
class RealtimePipelineResult:
    provider_event_id: str
    processing_status: WebhookProcessingStatus
    canonical_event_type: EventType | None
    decision_id: str | None
    case_id: str | None
    state: RealtimePipelineState


class RealtimeRiskPipelineService:
    """Compose Stage 12B and 12C without taking ownership of either's logic."""

    def __init__(
        self,
        sessions: SessionFactory,
        webhook_processor: WebhookEventProcessor,
        risk_scoring: RuntimeRiskScoringService,
    ) -> None:
        self._sessions = sessions
        self._webhook_processor = webhook_processor
        self._risk_scoring = risk_scoring

    def process(
        self, provider_event_id: str, *, retry_failed: bool = True
    ) -> RealtimePipelineResult:
        """Advance one durable delivery as far as its current state permits."""
        processed = self._existing_processed(provider_event_id)
        if processed is None:
            try:
                processed = self._webhook_processor.process(provider_event_id)
            except WebhookClaimUnavailable:
                return self._processing_result(provider_event_id)
        if processed.status is not WebhookProcessingStatus.PROCESSED:
            return RealtimePipelineResult(
                provider_event_id=provider_event_id,
                processing_status=processed.status,
                canonical_event_type=processed.canonical_event_type,
                decision_id=None,
                case_id=None,
                state=RealtimePipelineState.WEBHOOK_FAILED,
            )
        if processed.canonical_event_type is EventType.ACCOUNT_CREATED:
            return RealtimePipelineResult(
                provider_event_id,
                processed.status,
                processed.canonical_event_type,
                None,
                None,
                RealtimePipelineState.SETUP,
            )
        if not retry_failed and self._has_scoring_failure(provider_event_id):
            return RealtimePipelineResult(
                provider_event_id,
                processed.status,
                processed.canonical_event_type,
                None,
                None,
                RealtimePipelineState.SCORING_FAILED,
            )
        try:
            scored = self._risk_scoring.process(provider_event_id)
        except Exception as error:
            # Stage 12C has no webhook-state ownership. Its atomic persistence
            # boundary guarantees there is no partial trusted lineage to clean up.
            with self._sessions.begin() as session:
                RiskProcessingFailureRepository(session).persist_failed(
                    provider_event_id,
                    attempted_at=datetime.now(tz=UTC),
                    detail=_failure_detail(error),
                )
            return RealtimePipelineResult(
                provider_event_id,
                processed.status,
                processed.canonical_event_type,
                None,
                None,
                RealtimePipelineState.SCORING_FAILED,
            )
        with self._sessions.begin() as session:
            RiskProcessingFailureRepository(session).clear(provider_event_id)
        return self._scored_result(processed, scored)

    def process_next(self, *, limit: int) -> tuple[RealtimePipelineResult, ...]:
        """Bounded durable catch-up; Stage 12B claims remain the authority."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._sessions() as session:
            ready = WebhookEventRepository(session).ready_ids(
                limit=limit,
                now=_now(),
                lease_timeout=self._webhook_processor.processing_lease_timeout,
            )
        results = [
            self.process(provider_event_id, retry_failed=False)
            for provider_event_id in ready
        ]
        remaining = limit - len(results)
        if remaining:
            with self._sessions() as session:
                pending_scores = WebhookEventRepository(session).unscored_processed_ids(
                    limit=remaining
                )
            results.extend(
                self.process(provider_event_id, retry_failed=False)
                for provider_event_id in pending_scores
            )
        return tuple(results)

    def _existing_processed(
        self, provider_event_id: str
    ) -> ProcessedWebhookEvent | None:
        with self._sessions() as session:
            record = WebhookEventRepository(session).get(provider_event_id)
            if record is None:
                raise ValueError("webhook event does not exist")
            if record.status != WebhookProcessingStatus.PROCESSED.value:
                return None
            event = NormalizedEventRepository(session).get_for_provider(
                provider_event_id
            )
            if event is None:
                raise ValueError("processed webhook is missing its normalized event")
            return ProcessedWebhookEvent(
                provider_event_id,
                WebhookProcessingStatus.PROCESSED,
                str(event.id),
                event.event_type,
                0,
                0,
            )

    def _processing_result(self, provider_event_id: str) -> RealtimePipelineResult:
        with self._sessions() as session:
            record = WebhookEventRepository(session).get(provider_event_id)
            if record is None:
                raise ValueError("webhook event does not exist")
            event = NormalizedEventRepository(session).get_for_provider(
                provider_event_id
            )
        return RealtimePipelineResult(
            provider_event_id,
            WebhookProcessingStatus(record.status),
            None if event is None else event.event_type,
            None,
            None,
            RealtimePipelineState.PROCESSING,
        )

    def _has_scoring_failure(self, provider_event_id: str) -> bool:
        with self._sessions() as session:
            return (
                RiskProcessingFailureRepository(session).get(provider_event_id)
                is not None
            )

    @staticmethod
    def _scored_result(
        processed: ProcessedWebhookEvent, scored: RiskEvaluationResult
    ) -> RealtimePipelineResult:
        return RealtimePipelineResult(
            scored.provider_event_id,
            processed.status,
            processed.canonical_event_type,
            scored.decision_id,
            scored.case_id,
            RealtimePipelineState.REUSED
            if scored.reused
            else RealtimePipelineState.SCORED,
        )


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _failure_detail(error: Exception) -> str:
    """Persist only a bounded exception summary, never a traceback."""
    detail = f"{type(error).__name__}: {error}".strip()
    return detail[:1000] or "risk scoring failed"
