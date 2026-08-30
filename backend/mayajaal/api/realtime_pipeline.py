"""Stage 12D composition of durable webhook processing and risk scoring."""

from dataclasses import dataclass
from enum import StrEnum

from mayajaal.schemas import EventType

from .db import (
    NormalizedEventRepository,
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

    def process(self, provider_event_id: str) -> RealtimePipelineResult:
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
        try:
            scored = self._risk_scoring.process(provider_event_id)
        except Exception:
            # Stage 12C has no webhook-state ownership. Its atomic persistence
            # boundary guarantees there is no partial trusted lineage to clean up.
            return RealtimePipelineResult(
                provider_event_id,
                processed.status,
                processed.canonical_event_type,
                None,
                None,
                RealtimePipelineState.SCORING_FAILED,
            )
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
        results = [self.process(provider_event_id) for provider_event_id in ready]
        remaining = limit - len(results)
        if remaining:
            with self._sessions() as session:
                pending_scores = WebhookEventRepository(session).unscored_processed_ids(
                    limit=remaining
                )
            results.extend(
                self.process(provider_event_id) for provider_event_id in pending_scores
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


def _now():
    from datetime import UTC, datetime

    return datetime.now(tz=UTC)
