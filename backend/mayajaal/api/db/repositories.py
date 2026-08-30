"""Small session-owned repositories for immutable operational lineage."""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import cast

from sqlalchemy import Select, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from mayajaal.api.contracts import (
    InvestigationRun,
    PersistedInvestigationReport,
    RiskCase,
)
from mayajaal.calibration import ProbabilityEstimate
from mayajaal.features import FeatureSchema, FeatureVector
from mayajaal.investigation import InvestigationReport, InvestigationRequest, report_id
from mayajaal.policy import PolicyDecision
from mayajaal.schemas import Event
from mayajaal.scoring import ScoreObservation
from mayajaal.scoring.provenance import feature_vector_id

from .models import (
    FeatureVectorRecord,
    InvestigationReportRecord,
    InvestigationRequestRecord,
    InvestigationRunRecord,
    NormalizedEventRecord,
    PolicyDecisionRecord,
    ProbabilityEstimateRecord,
    RiskCaseDecisionRecord,
    RiskCaseRecord,
    RiskEvaluationRecord,
    ScoreObservationRecord,
    WebhookEventRecord,
)
from .serialization import from_payload, payload_for


class ImmutablePersistenceConflict(ValueError):
    """Raised when one authoritative ID is reused for different semantics."""


class WebhookPayloadConflict(ValueError):
    """Raised when a provider delivery ID is reused with changed raw bytes."""


class WebhookClaimUnavailable(ValueError):
    """Raised when an inbox event cannot be claimed by this processor."""


class _ImmutableRepository[
    Domain: (
        ScoreObservation,
        ProbabilityEstimate,
        PolicyDecision,
        InvestigationRequest,
        InvestigationRun,
        RiskCase,
    ),
    Record: (
        ScoreObservationRecord,
        ProbabilityEstimateRecord,
        PolicyDecisionRecord,
        InvestigationRequestRecord,
        InvestigationRunRecord,
        RiskCaseRecord,
    ),
]:
    """Shared immutable insert/get behavior; subclasses define their boundary."""

    def __init__(
        self,
        session: Session,
        *,
        record_type: type[Record],
        domain_type: type[Domain],
        key_for: Callable[[Domain], str],
        make_record: Callable[[Domain, dict[str, object]], Record],
    ) -> None:
        self._session = session
        self._record_type = record_type
        self._domain_type = domain_type
        self._key_for = key_for
        self._make_record = make_record

    def persist(self, value: Domain) -> Domain:
        """Insert once; accept byte-equivalent repeat writes and reject conflicts."""
        key = self._key_for(value)
        payload = payload_for(value)
        existing = self._session.get(self._record_type, key)
        if existing is None:
            self._session.add(self._make_record(value, payload))
            return value
        if existing.payload != payload:
            raise ImmutablePersistenceConflict(
                f"immutable object {key} already exists with different payload"
            )
        return cast(Domain, from_payload(self._domain_type, existing.payload))

    def get(self, object_id: str) -> Domain | None:
        """Load and revalidate one stored authoritative domain object."""
        record = self._session.get(self._record_type, object_id)
        if record is None:
            return None
        return cast(Domain, from_payload(self._domain_type, record.payload))


class ScoreObservationRepository(
    _ImmutableRepository[ScoreObservation, ScoreObservationRecord]
):
    """Persistence boundary for score observations."""

    def __init__(self, session: Session) -> None:
        super().__init__(
            session,
            record_type=ScoreObservationRecord,
            domain_type=ScoreObservation,
            key_for=lambda value: value.score_id,
            make_record=lambda value, payload: ScoreObservationRecord(
                score_id=value.score_id,
                base_model_id=value.base_model_id,
                subject_id=value.subject_id,
                scoring_cutoff=value.scoring_cutoff,
                feature_vector_id=value.feature_vector_id,
                raw_model_score=value.raw_model_score,
                payload=payload,
            ),
        )


class FeatureVectorRepository:
    """Persist the exact feature input independently from mutable graph state."""

    def __init__(self, session: Session, schema: FeatureSchema) -> None:
        self._session, self._schema = session, schema

    def persist(self, vector: FeatureVector) -> str:
        vector_id = feature_vector_id(self._schema, vector)
        payload = {
            "account_id": vector.account_id,
            "cutoff": vector.cutoff.isoformat(),
            "values": vector.values,
        }
        existing = self._session.get(FeatureVectorRecord, vector_id)
        if existing is None:
            self._session.add(
                FeatureVectorRecord(
                    feature_vector_id=vector_id,
                    account_id=vector.account_id,
                    scoring_cutoff=vector.cutoff,
                    payload=payload,
                )
            )
        elif existing.payload != payload:
            raise ImmutablePersistenceConflict(
                f"immutable feature vector {vector_id} already exists with different payload"
            )
        return vector_id


class RiskEvaluationRepository:
    """Replay key from one processed provider delivery to immutable decision lineage."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, provider_event_id: str) -> tuple[str, str | None] | None:
        row = self._session.get(RiskEvaluationRecord, provider_event_id)
        return None if row is None else (row.decision_id, row.case_id)

    def persist(
        self, provider_event_id: str, decision_id: str, case_id: str | None
    ) -> None:
        row = self._session.get(RiskEvaluationRecord, provider_event_id)
        if row is None:
            self._session.add(
                RiskEvaluationRecord(
                    provider_event_id=provider_event_id,
                    decision_id=decision_id,
                    case_id=case_id,
                )
            )
        elif row.decision_id != decision_id or row.case_id != case_id:
            raise ImmutablePersistenceConflict(
                "risk evaluation replay has different lineage"
            )


class ProbabilityEstimateRepository(
    _ImmutableRepository[ProbabilityEstimate, ProbabilityEstimateRecord]
):
    """Persistence boundary for calibrated probability estimates."""

    def __init__(self, session: Session) -> None:
        super().__init__(
            session,
            record_type=ProbabilityEstimateRecord,
            domain_type=ProbabilityEstimate,
            key_for=lambda value: value.probability_estimate_id,
            make_record=lambda value, payload: ProbabilityEstimateRecord(
                probability_estimate_id=value.probability_estimate_id,
                score_id=value.score_id,
                base_model_id=value.base_model_id,
                probability_model_id=value.probability_model_id,
                subject_id=value.subject_id,
                feature_vector_id=value.feature_vector_id,
                scoring_cutoff=value.scoring_cutoff,
                calibrated_probability=value.calibrated_probability,
                payload=payload,
            ),
        )


class PolicyDecisionRepository(
    _ImmutableRepository[PolicyDecision, PolicyDecisionRecord]
):
    """Persistence boundary for cost-aware policy decisions."""

    def __init__(self, session: Session) -> None:
        super().__init__(
            session,
            record_type=PolicyDecisionRecord,
            domain_type=PolicyDecision,
            key_for=lambda value: value.decision_id,
            make_record=lambda value, payload: PolicyDecisionRecord(
                decision_id=value.decision_id,
                probability_estimate_id=value.probability_estimate_id,
                score_id=value.score_id,
                policy_id=value.policy_id,
                base_model_id=value.base_model_id,
                probability_model_id=value.probability_model_id,
                subject_id=value.subject_id,
                feature_vector_id=value.feature_vector_id,
                scoring_cutoff=value.scoring_cutoff,
                context_id=value.context.context_id,
                chosen_action=value.chosen_action.value,
                payload=payload,
            ),
        )


class InvestigationRequestRepository(
    _ImmutableRepository[InvestigationRequest, InvestigationRequestRecord]
):
    """Persistence boundary for requests, keyed by their immutable decision ID."""

    def __init__(self, session: Session) -> None:
        super().__init__(
            session,
            record_type=InvestigationRequestRecord,
            domain_type=InvestigationRequest,
            key_for=lambda value: value.decision_id,
            make_record=lambda value, payload: InvestigationRequestRecord(
                decision_id=value.decision_id,
                policy_id=value.policy_id,
                probability_estimate_id=value.probability_estimate_id,
                score_id=value.score_id,
                subject_type=value.subject_type.value,
                subject_id=value.subject_id,
                cutoff_time=value.cutoff_time,
                context_id=value.context_id,
                policy_action=value.policy_action.value,
                payload=payload,
            ),
        )


class RiskCaseRepository(_ImmutableRepository[RiskCase, RiskCaseRecord]):
    """Persistence and bounded lookup boundary for operational risk cases."""

    def __init__(self, session: Session) -> None:
        super().__init__(
            session,
            record_type=RiskCaseRecord,
            domain_type=RiskCase,
            key_for=lambda value: value.case_id,
            make_record=lambda value, payload: RiskCaseRecord(
                case_id=value.case_id,
                subject_type=value.subject_type.value,
                subject_id=value.subject_id,
                status=value.status.value,
                opened_at=value.opened_at,
                closed_at=value.closed_at,
                opening_decision_id=value.opening_decision_id,
                payload=payload,
            ),
        )

    def persist(self, value: RiskCase) -> RiskCase:
        persisted = super().persist(value)
        self.attach_decision(value.case_id, value.opening_decision_id)
        return persisted

    def attach_decision(self, case_id: str, decision_id: str) -> None:
        """Attach an existing decision without granting it case lifecycle authority."""
        key = {"case_id": case_id, "decision_id": decision_id}
        if self._session.get(RiskCaseDecisionRecord, key) is None:
            self._session.add(RiskCaseDecisionRecord(**key))

    def open_for_subject(self, subject_id: str) -> RiskCase | None:
        row = self._session.scalar(
            select(RiskCaseRecord)
            .where(
                RiskCaseRecord.subject_id == subject_id, RiskCaseRecord.status == "OPEN"
            )
            .order_by(RiskCaseRecord.opened_at.asc())
        )
        return (
            None if row is None else cast(RiskCase, from_payload(RiskCase, row.payload))
        )

    def close_case(self, case_id: str, closed_at: datetime) -> RiskCase:
        """Apply the sole mutable case transition: OPEN to CLOSED.

        An identical repeated close is accepted for operational idempotency;
        a conflicting close or any other state is rejected.
        """
        record = self._session.get(RiskCaseRecord, case_id)
        if record is None:
            raise ValueError("risk case does not exist")
        existing = cast(RiskCase, from_payload(RiskCase, record.payload))
        if existing.status.value == "CLOSED":
            if existing.closed_at == closed_at:
                return existing
            raise ValueError("risk case is already closed with a different timestamp")
        closed = RiskCase.model_validate(
            {
                **existing.model_dump(mode="json"),
                "status": "CLOSED",
                "closed_at": closed_at,
            }
        )
        record.status = closed.status.value
        record.closed_at = closed.closed_at
        record.payload = payload_for(closed)
        return closed

    def list_recent(self, *, limit: int, offset: int = 0) -> tuple[RiskCase, ...]:
        return tuple(
            cast(RiskCase, from_payload(RiskCase, row.payload))
            for row in self._session.scalars(
                select(RiskCaseRecord)
                .order_by(RiskCaseRecord.opened_at.desc(), RiskCaseRecord.case_id.asc())
                .limit(_bounded_limit(limit))
                .offset(_bounded_offset(offset))
            )
        )

    def list_for_subject(
        self, subject_id: str, *, limit: int, offset: int = 0
    ) -> tuple[RiskCase, ...]:
        return tuple(
            cast(RiskCase, from_payload(RiskCase, row.payload))
            for row in self._session.scalars(
                select(RiskCaseRecord)
                .where(RiskCaseRecord.subject_id == subject_id)
                .order_by(RiskCaseRecord.opened_at.desc(), RiskCaseRecord.case_id.asc())
                .limit(_bounded_limit(limit))
                .offset(_bounded_offset(offset))
            )
        )


class InvestigationRunRepository(
    _ImmutableRepository[InvestigationRun, InvestigationRunRecord]
):
    """Persistence and bounded lookup boundary for execution attempts."""

    def __init__(self, session: Session) -> None:
        super().__init__(
            session,
            record_type=InvestigationRunRecord,
            domain_type=InvestigationRun,
            key_for=lambda value: value.run_id,
            make_record=lambda value, payload: InvestigationRunRecord(
                run_id=value.run_id,
                decision_id=value.decision_id,
                case_id=value.case_id,
                investigation_id=value.investigation_id,
                agent_model_id=value.agent_model_id,
                status=value.status.value,
                started_at=value.started_at,
                completed_at=value.completed_at,
                payload=payload,
            ),
        )

    def list_for_case(
        self, case_id: str, *, limit: int, offset: int = 0
    ) -> tuple[InvestigationRun, ...]:
        return self._list(
            select(InvestigationRunRecord)
            .where(InvestigationRunRecord.case_id == case_id)
            .order_by(
                InvestigationRunRecord.started_at.desc(),
                InvestigationRunRecord.run_id.asc(),
            )
            .limit(_bounded_limit(limit))
            .offset(_bounded_offset(offset))
        )

    def list_for_decision(
        self, decision_id: str, *, limit: int, offset: int = 0
    ) -> tuple[InvestigationRun, ...]:
        return self._list(
            select(InvestigationRunRecord)
            .where(InvestigationRunRecord.decision_id == decision_id)
            .order_by(
                InvestigationRunRecord.started_at.desc(),
                InvestigationRunRecord.run_id.asc(),
            )
            .limit(_bounded_limit(limit))
            .offset(_bounded_offset(offset))
        )

    def _list(
        self, statement: Select[tuple[InvestigationRunRecord]]
    ) -> tuple[InvestigationRun, ...]:
        return tuple(
            cast(InvestigationRun, from_payload(InvestigationRun, row.payload))
            for row in self._session.scalars(statement)
        )


class InvestigationReportRepository:
    """Persist trusted reports by report ID, with one report per run."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def persist(
        self, value: PersistedInvestigationReport
    ) -> PersistedInvestigationReport:
        expected_report_id = report_id(value.investigation_id, value.report)
        if value.report_id != expected_report_id:
            raise ValueError("report_id does not match trusted report provenance")
        run = self._session.get(InvestigationRunRecord, value.run_id)
        if run is None:
            raise ValueError("investigation report requires a persisted run")
        if (
            run.investigation_id != value.investigation_id
            or run.decision_id != value.report.request.decision_id
        ):
            raise ValueError("investigation report does not match its operational run")
        payload = payload_for(value.report)
        existing = self._session.get(InvestigationReportRecord, value.report_id)
        if existing is None:
            self._session.add(
                InvestigationReportRecord(
                    report_id=value.report_id,
                    run_id=value.run_id,
                    investigation_id=value.investigation_id,
                    policy_action=value.report.policy_action.value,
                    status=value.report.status.value,
                    pattern=value.report.pattern.value,
                    payload=payload,
                )
            )
            return value
        if (
            existing.run_id != value.run_id
            or existing.investigation_id != value.investigation_id
            or existing.payload != payload
        ):
            raise ImmutablePersistenceConflict(
                f"immutable object {value.report_id} already exists with different payload"
            )
        return self._from_record(existing)

    def get(self, report_id: str) -> PersistedInvestigationReport | None:
        record = self._session.get(InvestigationReportRecord, report_id)
        return None if record is None else self._from_record(record)

    def get_for_run(self, run_id: str) -> PersistedInvestigationReport | None:
        record = self._session.scalar(
            select(InvestigationReportRecord).where(
                InvestigationReportRecord.run_id == run_id
            )
        )
        return None if record is None else self._from_record(record)

    def _from_record(
        self, record: InvestigationReportRecord
    ) -> PersistedInvestigationReport:
        report = cast(
            InvestigationReport, from_payload(InvestigationReport, record.payload)
        )
        if record.report_id != report_id(record.investigation_id, report):
            raise ValueError("stored report_id failed provenance validation")
        run = self._session.get(InvestigationRunRecord, record.run_id)
        if run is None:
            raise ValueError("stored report references a missing investigation run")
        if (
            run.investigation_id != record.investigation_id
            or run.decision_id != report.request.decision_id
        ):
            raise ValueError("stored report does not match its operational run")
        return PersistedInvestigationReport(
            report_id=record.report_id,
            run_id=record.run_id,
            investigation_id=record.investigation_id,
            report=report,
        )


class WebhookEventRepository:
    """Atomic provider-delivery inbox persistence, owned by a caller session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def persist_received(
        self,
        *,
        provider_event_id: str,
        provider: str,
        event_type: str,
        provider_created_at: datetime,
        received_at: datetime,
        raw_body: bytes,
        raw_body_sha256: str,
        payload: dict[str, object],
    ) -> tuple[WebhookEventRecord, bool]:
        """Insert once atomically; reject delivery-ID reuse with changed bytes."""
        values = {
            "provider_event_id": provider_event_id,
            "provider": provider,
            "event_type": event_type,
            "provider_created_at": provider_created_at,
            "received_at": received_at,
            "raw_body": raw_body,
            "raw_body_sha256": raw_body_sha256,
            "payload": payload,
            "status": "RECEIVED",
            "claimed_at": None,
            "processed_at": None,
            "failure_detail": None,
        }
        bind = self._session.get_bind()
        if bind.dialect.name == "postgresql":
            statement = postgresql_insert(WebhookEventRecord).values(**values)
        elif bind.dialect.name == "sqlite":
            statement = sqlite_insert(WebhookEventRecord).values(**values)
        else:
            raise ValueError("webhook inbox requires PostgreSQL or SQLite test dialect")
        inserted_id = self._session.scalar(
            statement.on_conflict_do_nothing(
                index_elements=["provider_event_id"]
            ).returning(WebhookEventRecord.provider_event_id)
        )
        self._session.flush()
        record = self._session.get(WebhookEventRecord, provider_event_id)
        if record is None:
            raise RuntimeError("webhook inbox insert did not return a durable row")
        if inserted_id is None:
            if record.raw_body_sha256 != raw_body_sha256 or record.raw_body != raw_body:
                raise WebhookPayloadConflict(
                    "provider event ID already exists with a different raw payload"
                )
            return record, False
        return record, True

    def claim(
        self,
        provider_event_id: str,
        *,
        claimed_at: datetime,
        lease_timeout: timedelta,
    ) -> WebhookEventRecord:
        """Atomically claim an available or abandoned delivery for one processor."""
        lease_expires_at = _lease_expires_at(claimed_at, lease_timeout)
        result = self._session.execute(
            update(WebhookEventRecord)
            .where(
                WebhookEventRecord.provider_event_id == provider_event_id,
                _claimable_webhook_condition(lease_expires_at),
            )
            .values(status="PROCESSING", claimed_at=claimed_at, failure_detail=None)
        )
        if result.rowcount != 1:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            raise WebhookClaimUnavailable(
                "webhook event is not available for processing"
            )
        record = self._session.get(WebhookEventRecord, provider_event_id)
        if record is None:
            raise RuntimeError("claimed webhook event disappeared")
        return record

    def claim_next(
        self, *, claimed_at: datetime, lease_timeout: timedelta
    ) -> WebhookEventRecord | None:
        """Claim the oldest available row with ``SKIP LOCKED`` for concurrent workers."""
        lease_expires_at = _lease_expires_at(claimed_at, lease_timeout)
        record = self._session.scalar(
            select(WebhookEventRecord)
            .where(_claimable_webhook_condition(lease_expires_at))
            .order_by(
                WebhookEventRecord.received_at.asc(),
                WebhookEventRecord.provider_event_id.asc(),
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if record is None:
            return None
        record.status = "PROCESSING"
        record.claimed_at = claimed_at
        record.failure_detail = None
        return record

    def mark_processed(
        self,
        provider_event_id: str,
        *,
        expected_claimed_at: datetime,
        processed_at: datetime,
    ) -> None:
        """Finalize only the exact ``PROCESSING`` claim that owns the row."""
        self._finalize_claim(
            provider_event_id,
            expected_claimed_at=expected_claimed_at,
            values={
                "status": "PROCESSED",
                "processed_at": processed_at,
                "failure_detail": None,
            },
        )

    def mark_failed(
        self,
        provider_event_id: str,
        *,
        expected_claimed_at: datetime,
        detail: str,
    ) -> None:
        """Record failure only if this processor still owns its claim."""
        self._finalize_claim(
            provider_event_id,
            expected_claimed_at=expected_claimed_at,
            values={
                "status": "FAILED",
                "failure_detail": detail[:1000],
            },
        )

    def _finalize_claim(
        self,
        provider_event_id: str,
        *,
        expected_claimed_at: datetime,
        values: dict[str, object],
    ) -> None:
        if expected_claimed_at.tzinfo is None:
            raise ValueError("expected_claimed_at must be timezone-aware")
        result = self._session.execute(
            update(WebhookEventRecord)
            .where(
                WebhookEventRecord.provider_event_id == provider_event_id,
                WebhookEventRecord.status == "PROCESSING",
                WebhookEventRecord.claimed_at == expected_claimed_at,
            )
            .values(**values)
        )
        if result.rowcount != 1:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            raise WebhookClaimUnavailable(
                "webhook event claim is no longer owned by this processor"
            )

    def _require(self, provider_event_id: str) -> WebhookEventRecord:
        record = self.get(provider_event_id)
        if record is None:
            raise ValueError("webhook event does not exist")
        return record

    def get(self, provider_event_id: str) -> WebhookEventRecord | None:
        return self._session.get(WebhookEventRecord, provider_event_id)

    def list_recent(
        self, *, limit: int, offset: int = 0
    ) -> tuple[WebhookEventRecord, ...]:
        return tuple(
            self._session.scalars(
                select(WebhookEventRecord)
                .order_by(
                    WebhookEventRecord.received_at.desc(),
                    WebhookEventRecord.provider_event_id.asc(),
                )
                .limit(_bounded_limit(limit))
                .offset(_bounded_offset(offset))
            )
        )

    def ready_ids(
        self, *, limit: int, now: datetime, lease_timeout: timedelta
    ) -> tuple[str, ...]:
        """Return a deterministic bounded snapshot; ``claim`` remains atomic."""
        lease_expires_at = _lease_expires_at(now, lease_timeout)
        return tuple(
            self._session.scalars(
                select(WebhookEventRecord.provider_event_id)
                .where(_claimable_webhook_condition(lease_expires_at))
                .order_by(
                    WebhookEventRecord.received_at.asc(),
                    WebhookEventRecord.provider_event_id.asc(),
                )
                .limit(_bounded_limit(limit))
            )
        )


def _lease_expires_at(claimed_at: datetime, lease_timeout: timedelta) -> datetime:
    if claimed_at.tzinfo is None:
        raise ValueError("claimed_at must be timezone-aware")
    if lease_timeout <= timedelta(0):
        raise ValueError("processing lease timeout must be positive")
    return claimed_at - lease_timeout


def _claimable_webhook_condition(lease_expires_at: datetime):  # type: ignore[no-untyped-def]
    """SQL predicate shared by direct and ``SKIP LOCKED`` claim paths."""
    return or_(
        WebhookEventRecord.status.in_(("RECEIVED", "FAILED")),
        (
            (WebhookEventRecord.status == "PROCESSING")
            & (WebhookEventRecord.claimed_at < lease_expires_at)
        ),
    )


class NormalizedEventRepository:
    """Persist canonical provider facts without treating raw webhook JSON as graph data."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def persist(self, *, provider_event_id: str, event: Event) -> Event:
        payload = event.model_dump(mode="json")
        existing = self._session.get(NormalizedEventRecord, str(event.id))
        by_provider = self._session.scalar(
            select(NormalizedEventRecord).where(
                NormalizedEventRecord.provider_event_id == provider_event_id
            )
        )
        record = existing or by_provider
        if record is None:
            self._session.add(
                NormalizedEventRecord(
                    event_id=str(event.id),
                    provider_event_id=provider_event_id,
                    event_type=event.event_type.value,
                    account_id=str(event.account_id),
                    occurred_at=event.occurred_at,
                    payload=payload,
                )
            )
            return event
        if record.event_id != str(event.id) or record.payload != payload:
            raise ImmutablePersistenceConflict(
                "provider event already has a different canonical event"
            )
        return Event.model_validate(record.payload)

    def get_for_provider(self, provider_event_id: str) -> Event | None:
        record = self._session.scalar(
            select(NormalizedEventRecord).where(
                NormalizedEventRecord.provider_event_id == provider_event_id
            )
        )
        return None if record is None else Event.model_validate(record.payload)


def _bounded_limit(limit: int) -> int:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return limit


def _bounded_offset(offset: int) -> int:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    return offset
