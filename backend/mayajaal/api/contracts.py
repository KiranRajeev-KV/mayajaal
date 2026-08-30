"""Small operational contracts for case and investigation execution lifecycle."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, cast

from pydantic import Field, JsonValue, model_validator

from mayajaal.investigation import (
    InvestigationExecution,
    InvestigationStatus,
    InvestigationSubjectType,
    investigation_id,
    report_id,
)
from mayajaal.schemas.common import AwareDatetime, SchemaModel

NonEmptyId = Annotated[str, Field(min_length=1)]


class RiskCaseStatus(StrEnum):
    """The minimal operational lifecycle of a long-lived risk episode."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class InvestigationJobStatus(StrEnum):
    """Durable operational state before a completed investigation exists."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class InvestigationJob(SchemaModel):
    """Recoverable user-requested investigation work, separate from a report."""

    run_id: NonEmptyId
    decision_id: NonEmptyId
    case_id: NonEmptyId
    status: InvestigationJobStatus
    created_at: AwareDatetime
    last_attempt_at: AwareDatetime | None = None
    claimed_at: AwareDatetime | None = None
    failure_detail: str | None = Field(default=None, max_length=1000)


class RiskCase(SchemaModel):
    """Operational container for a subject's risk episode, not a model output."""

    case_id: NonEmptyId
    subject_type: InvestigationSubjectType
    subject_id: NonEmptyId
    status: RiskCaseStatus = RiskCaseStatus.OPEN
    opened_at: AwareDatetime
    opening_decision_id: NonEmptyId
    closed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "RiskCase":
        if self.status is RiskCaseStatus.OPEN and self.closed_at is not None:
            raise ValueError("open risk case cannot have closed_at")
        if self.status is RiskCaseStatus.CLOSED:
            if self.closed_at is None:
                raise ValueError("closed risk case requires closed_at")
            if self.closed_at < self.opened_at:
                raise ValueError("risk case closed_at cannot precede opened_at")
        return self


class InvestigationRun(SchemaModel):
    """One operational attempt to investigate a persisted decision/request."""

    run_id: NonEmptyId
    decision_id: NonEmptyId
    investigation_id: NonEmptyId
    agent_model_id: NonEmptyId
    status: InvestigationStatus
    started_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    case_id: NonEmptyId | None = None
    provenance: dict[str, JsonValue] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_temporal_lifecycle(self) -> "InvestigationRun":
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("investigation run completed_at cannot precede started_at")
        return self

    @classmethod
    def from_execution(
        cls,
        *,
        run_id: str,
        execution: InvestigationExecution,
        started_at: datetime,
        completed_at: datetime | None = None,
        case_id: str | None = None,
    ) -> "InvestigationRun":
        """Derive run provenance from the verified execution, never caller IDs."""
        provenance = investigation_provenance(
            request=execution.report.request,
            config=execution.config,
            agent_model_id=execution.agent_model_id,
            snapshot=execution.snapshot,
        )
        return cls(
            run_id=run_id,
            decision_id=execution.report.request.decision_id,
            investigation_id=str(provenance["investigation_id"]),
            agent_model_id=execution.agent_model_id,
            status=execution.report.status,
            started_at=started_at,
            completed_at=completed_at,
            case_id=case_id,
            provenance=cast(dict[str, JsonValue], provenance),
        )


class PersistedInvestigationReport(SchemaModel):
    """Operational report identity paired with its authoritative report payload."""

    report_id: NonEmptyId
    run_id: NonEmptyId
    investigation_id: NonEmptyId
    report: "InvestigationReport"

    @classmethod
    def from_execution(
        cls,
        *,
        run: InvestigationRun,
        execution: InvestigationExecution,
    ) -> "PersistedInvestigationReport":
        """Derive the report identity through the existing report provenance API."""
        expected_investigation_id = investigation_id(
            request=execution.report.request,
            config=execution.config,
            agent_model_id=execution.agent_model_id,
            snapshot=execution.snapshot,
        )
        if run.investigation_id != expected_investigation_id:
            raise ValueError("investigation run does not match execution provenance")
        if run.decision_id != execution.report.request.decision_id:
            raise ValueError("investigation run does not match execution request")
        return cls(
            report_id=report_id(expected_investigation_id, execution.report),
            run_id=run.run_id,
            investigation_id=expected_investigation_id,
            report=execution.report,
        )


from mayajaal.investigation import (  # noqa: E402
    InvestigationReport,
    investigation_provenance,
)
