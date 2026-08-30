"""Small session-owned repositories for immutable operational lineage."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import Select, delete, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from mayajaal.api.contracts import (
    InvestigationJob,
    InvestigationJobStatus,
    InvestigationRun,
    PersistedInvestigationReport,
    RiskCase,
)
from mayajaal.calibration import ProbabilityEstimate
from mayajaal.features import FeatureKind, FeatureSchema, FeatureVector
from mayajaal.investigation import InvestigationReport, InvestigationRequest, report_id
from mayajaal.policy import PolicyDecision
from mayajaal.schemas import Event
from mayajaal.scoring import ScoreObservation
from mayajaal.scoring.provenance import feature_vector_id

from .models import (
    FeatureVectorRecord,
    InvestigationJobRecord,
    InvestigationReportRecord,
    InvestigationRequestRecord,
    InvestigationRunRecord,
    NormalizedEventRecord,
    PolicyDecisionRecord,
    ProbabilityEstimateRecord,
    RiskCaseDecisionRecord,
    RiskCaseRecord,
    RiskEvaluationRecord,
    RiskProcessingFailureRecord,
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


def _conflict_ignored_insert(
    session: Session,
    record_type: type[object],
    values: Mapping[str, object],
    *,
    index_elements: list[str] | None = None,
    index_where: object | None = None,
) -> Any:
    """Use the database's atomic no-overwrite insert for immutable rows."""
    dialect = session.get_bind().dialect.name
    table = cast(Any, record_type.__table__)  # type: ignore[attr-defined]
    if dialect == "postgresql":
        statement: Any = postgresql_insert(table).values(**values)
        return statement.on_conflict_do_nothing(
            index_elements=index_elements, index_where=cast(Any, index_where)
        )
    if dialect == "sqlite":
        statement = sqlite_insert(table).values(**values)
        return statement.on_conflict_do_nothing(
            index_elements=index_elements, index_where=cast(Any, index_where)
        )
    raise ValueError("immutable lineage requires PostgreSQL or SQLite")


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
            candidate = self._make_record(value, payload)
            values = {
                column.name: cast(object, getattr(candidate, column.name))
                for column in candidate.__table__.columns
            }
            self._session.execute(
                _conflict_ignored_insert(self._session, self._record_type, values)
            )
            existing = self._session.get(self._record_type, key)
            if existing is None:
                raise RuntimeError("immutable insert did not yield a durable row")
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

    def get_for_decision(self, decision_id: str) -> ScoreObservation | None:
        row = self._session.get(PolicyDecisionRecord, decision_id)
        return None if row is None else self.get(row.score_id)


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
            self._session.execute(
                _conflict_ignored_insert(
                    self._session,
                    FeatureVectorRecord,
                    {
                        "feature_vector_id": vector_id,
                        "account_id": vector.account_id,
                        "scoring_cutoff": vector.cutoff,
                        "payload": payload,
                    },
                )
            )
            existing = self._session.get(FeatureVectorRecord, vector_id)
            if existing is None:
                raise RuntimeError("feature vector insert did not yield a durable row")
        elif existing.payload != payload:
            raise ImmutablePersistenceConflict(
                f"immutable feature vector {vector_id} already exists with different payload"
            )
        return vector_id

    def get(self, vector_id: str) -> FeatureVector | None:
        """Load and verify one exact persisted vector against this schema."""
        record = self._session.get(FeatureVectorRecord, vector_id)
        if record is None:
            return None
        payload = record.payload
        try:
            raw_values = payload["values"]
            if not isinstance(raw_values, dict):
                raise TypeError("feature vector values must be an object")
            raw_feature_values = cast(dict[str, float | int | str], raw_values)
            # JSONB object storage does not preserve insertion order. Rebuild
            # the authoritative FeatureVector mapping in frozen-schema order.
            values = {
                definition.name: raw_feature_values[definition.name]
                for definition in self._schema.definitions
            }
            for definition in self._schema.definitions:
                value = values.get(definition.name)
                if (
                    definition.kind is FeatureKind.NUMERIC
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    # PostgreSQL JSONB can deserialize 1.0 as the integer 1;
                    # restore the authoritative FeatureVector numeric contract.
                    values[definition.name] = float(value)
            vector = FeatureVector(
                account_id=str(payload["account_id"]),
                cutoff=datetime.fromisoformat(str(payload["cutoff"])),
                values=cast(dict[str, float | str], values),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ImmutablePersistenceConflict(
                "persisted feature vector is malformed"
            ) from error
        if (
            vector.account_id != record.account_id
            or vector.cutoff != _require_aware_datetime(record.scoring_cutoff)
        ):
            raise ImmutablePersistenceConflict(
                "persisted feature vector columns disagree"
            )
        try:
            trusted_vector_id = feature_vector_id(self._schema, vector)
        except (TypeError, ValueError) as error:
            raise ImmutablePersistenceConflict(
                "persisted feature vector does not satisfy the frozen schema"
            ) from error
        if trusted_vector_id != vector_id:
            raise ImmutablePersistenceConflict(
                "persisted feature vector ID/schema mismatch"
            )
        return vector


class RiskEvaluationRepository:
    """Replay key from one processed provider delivery to immutable decision lineage."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, provider_event_id: str) -> tuple[str, str | None] | None:
        row = self._session.get(RiskEvaluationRecord, provider_event_id)
        return None if row is None else (row.decision_id, row.case_id)

    def persist(
        self, provider_event_id: str, decision_id: str, case_id: str | None
    ) -> tuple[str, str | None, bool]:
        values = {
            "provider_event_id": provider_event_id,
            "decision_id": decision_id,
            "case_id": case_id,
        }
        statement = _conflict_ignored_insert(
            self._session, RiskEvaluationRecord, values
        )
        result = cast(Any, self._session.execute(statement))
        inserted: bool = result.rowcount == 1
        row = self._session.get(RiskEvaluationRecord, provider_event_id)
        if row is None:
            raise RuntimeError("risk evaluation insert did not yield a durable row")
        if row.decision_id != decision_id or row.case_id != case_id:
            raise ImmutablePersistenceConflict(
                "risk evaluation replay has different lineage"
            )
        return row.decision_id, row.case_id, inserted


class RiskProcessingFailureRepository:
    """Small mutable marker for bounded, non-successful Stage 12C attempts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, provider_event_id: str) -> RiskProcessingFailureRecord | None:
        return self._session.get(RiskProcessingFailureRecord, provider_event_id)

    def persist_failed(
        self, provider_event_id: str, *, attempted_at: datetime, detail: str
    ) -> None:
        values = {
            "provider_event_id": provider_event_id,
            "status": "FAILED",
            "last_attempt_at": attempted_at,
            "failure_detail": detail[:1000],
        }
        dialect = self._session.get_bind().dialect.name
        if dialect == "postgresql":
            statement = postgresql_insert(RiskProcessingFailureRecord).values(**values)
        elif dialect == "sqlite":
            statement = sqlite_insert(RiskProcessingFailureRecord).values(**values)
        else:
            raise ValueError("risk processing failures require PostgreSQL or SQLite")
        self._session.execute(
            statement.on_conflict_do_update(
                index_elements=["provider_event_id"],
                set_={
                    "status": values["status"],
                    "last_attempt_at": values["last_attempt_at"],
                    "failure_detail": values["failure_detail"],
                },
            )
        )

    def clear(self, provider_event_id: str) -> None:
        self._session.execute(
            delete(RiskProcessingFailureRecord).where(
                RiskProcessingFailureRecord.provider_event_id == provider_event_id
            )
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

    def get_for_decision(self, decision_id: str) -> ProbabilityEstimate | None:
        row = self._session.get(PolicyDecisionRecord, decision_id)
        return None if row is None else self.get(row.probability_estimate_id)


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

    def has_decision(self, case_id: str, decision_id: str) -> bool:
        return (
            self._session.get(
                RiskCaseDecisionRecord, {"case_id": case_id, "decision_id": decision_id}
            )
            is not None
        )

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

    def open_or_create(self, value: RiskCase) -> RiskCase:
        """Return the one open episode, tolerating a concurrent opener."""
        existing = self.open_for_subject(value.subject_id)
        if existing is not None:
            return existing
        payload = payload_for(value)
        values = {
            "case_id": value.case_id,
            "subject_type": value.subject_type.value,
            "subject_id": value.subject_id,
            "status": value.status.value,
            "opened_at": value.opened_at,
            "closed_at": value.closed_at,
            "opening_decision_id": value.opening_decision_id,
            "payload": payload,
        }
        statement = _conflict_ignored_insert(
            self._session,
            RiskCaseRecord,
            values,
            index_elements=["subject_type", "subject_id"],
            index_where=text("status = 'OPEN'"),
        )
        self._session.execute(statement)
        opened = self.open_for_subject(value.subject_id)
        if opened is None:
            raise RuntimeError("open case insert did not yield an open episode")
        return opened

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


class InvestigationJobUnavailable(ValueError):
    """A job is completed or currently owned by another executor."""


class InvestigationJobRepository:
    """Lease/fence mutable operational work without weakening completed lineage."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, run_id: str) -> InvestigationJob | None:
        row = self._session.get(InvestigationJobRecord, run_id)
        return None if row is None else self._from_record(row)

    def enqueue(self, value: InvestigationJob) -> tuple[InvestigationJob, bool]:
        values = {
            "run_id": value.run_id,
            "decision_id": value.decision_id,
            "case_id": value.case_id,
            "idempotency_key": value.idempotency_key,
            "status": value.status.value,
            "created_at": value.created_at,
            "last_attempt_at": value.last_attempt_at,
            "claimed_at": value.claimed_at,
            "failure_detail": value.failure_detail,
        }
        statement = _conflict_ignored_insert(
            self._session,
            InvestigationJobRecord,
            values,
            index_elements=["case_id", "decision_id", "idempotency_key"],
        )
        if self._session.get_bind().dialect.name == "postgresql":
            inserted = (
                self._session.execute(
                    statement.returning(InvestigationJobRecord.run_id)
                ).scalar_one_or_none()
                is not None
            )
        else:
            inserted = cast(Any, self._session.execute(statement)).rowcount == 1
        stored = self.get_for_idempotency(
            value.case_id, value.decision_id, value.idempotency_key
        )
        if stored is None:
            raise RuntimeError("investigation job insert did not yield a durable row")
        if inserted and stored != value:
            raise ImmutablePersistenceConflict("investigation job ID already exists")
        return stored, inserted

    def get_for_idempotency(
        self, case_id: str, decision_id: str, idempotency_key: str
    ) -> InvestigationJob | None:
        row = self._session.scalar(
            select(InvestigationJobRecord).where(
                InvestigationJobRecord.case_id == case_id,
                InvestigationJobRecord.decision_id == decision_id,
                InvestigationJobRecord.idempotency_key == idempotency_key,
            )
        )
        return None if row is None else self._from_record(row)

    def claim(
        self, run_id: str, *, claimed_at: datetime, lease_timeout: timedelta
    ) -> InvestigationJob:
        expires_at = _lease_expires_at(claimed_at, lease_timeout)
        result = self._session.execute(
            update(InvestigationJobRecord)
            .where(
                InvestigationJobRecord.run_id == run_id,
                or_(
                    InvestigationJobRecord.status.in_(("QUEUED", "FAILED")),
                    (InvestigationJobRecord.status == "RUNNING")
                    & (InvestigationJobRecord.claimed_at < expires_at),
                ),
            )
            .values(
                status="RUNNING",
                claimed_at=claimed_at,
                last_attempt_at=claimed_at,
                failure_detail=None,
            )
        )
        if result.rowcount != 1:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            raise InvestigationJobUnavailable("investigation job is not available")
        claimed = self.get(run_id)
        if claimed is None:
            raise RuntimeError("claimed investigation job disappeared")
        return claimed

    def complete(self, run_id: str, *, claimed_at: datetime) -> None:
        self._finalize(
            run_id, claimed_at, {"status": "COMPLETED", "failure_detail": None}
        )

    def fail(self, run_id: str, *, claimed_at: datetime, detail: str) -> None:
        self._finalize(
            run_id, claimed_at, {"status": "FAILED", "failure_detail": detail[:1000]}
        )

    def _finalize(
        self, run_id: str, claimed_at: datetime, values: dict[str, object]
    ) -> None:
        result = self._session.execute(
            update(InvestigationJobRecord)
            .where(
                InvestigationJobRecord.run_id == run_id,
                InvestigationJobRecord.status == "RUNNING",
                InvestigationJobRecord.claimed_at == claimed_at,
            )
            .values(**values)
        )
        if result.rowcount != 1:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            raise InvestigationJobUnavailable(
                "investigation job claim is no longer owned"
            )

    @staticmethod
    def _from_record(row: InvestigationJobRecord) -> InvestigationJob:
        return InvestigationJob(
            run_id=row.run_id,
            decision_id=row.decision_id,
            case_id=row.case_id,
            idempotency_key=row.idempotency_key,
            status=InvestigationJobStatus(row.status),
            created_at=_require_aware_datetime(row.created_at),
            last_attempt_at=_aware_datetime(row.last_attempt_at),
            claimed_at=_aware_datetime(row.claimed_at),
            failure_detail=row.failure_detail,
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

    def unscored_processed_ids(self, *, limit: int) -> tuple[str, ...]:
        """Find projected non-setup canonical events with no durable evaluation."""
        return tuple(
            self._session.scalars(
                select(WebhookEventRecord.provider_event_id)
                .join(
                    NormalizedEventRecord,
                    NormalizedEventRecord.provider_event_id
                    == WebhookEventRecord.provider_event_id,
                )
                .outerjoin(
                    RiskEvaluationRecord,
                    RiskEvaluationRecord.provider_event_id
                    == WebhookEventRecord.provider_event_id,
                )
                .outerjoin(
                    RiskProcessingFailureRecord,
                    RiskProcessingFailureRecord.provider_event_id
                    == WebhookEventRecord.provider_event_id,
                )
                .where(
                    WebhookEventRecord.status == "PROCESSED",
                    NormalizedEventRecord.event_type != "ACCOUNT_CREATED",
                    RiskEvaluationRecord.provider_event_id.is_(None),
                    RiskProcessingFailureRecord.provider_event_id.is_(None),
                )
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


def _aware_datetime(value: datetime | None) -> datetime | None:
    """SQLite test rows lose offsets; PostgreSQL keeps the original instant."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _require_aware_datetime(value: datetime) -> datetime:
    converted = _aware_datetime(value)
    assert converted is not None
    return converted


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

    def known_at(self, cutoff: datetime) -> tuple[Event, ...]:
        """Return canonical facts available to Mayajaal by the fixed cutoff."""
        return tuple(
            Event.model_validate(row.payload)
            for row in self._session.scalars(
                select(NormalizedEventRecord)
                .join(
                    WebhookEventRecord,
                    WebhookEventRecord.provider_event_id
                    == NormalizedEventRecord.provider_event_id,
                )
                .where(
                    NormalizedEventRecord.occurred_at <= cutoff,
                    WebhookEventRecord.received_at <= cutoff,
                )
                .order_by(
                    NormalizedEventRecord.occurred_at.asc(),
                    NormalizedEventRecord.event_id.asc(),
                )
            )
        )


def _bounded_limit(limit: int) -> int:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return limit


def _bounded_offset(offset: int) -> int:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    return offset
