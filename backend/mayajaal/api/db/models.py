"""SQLAlchemy persistence representations for immutable runtime lineage.

These models deliberately do not replace the authoritative domain contracts.
Each row retains a full JSON payload and only materializes fields needed for
identity, lineage, and the immediate subject/time/status lookup paths.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .base import Base

Payload = dict[str, Any]
JsonPayload = JSON().with_variant(JSONB, "postgresql")


class ScoreObservationRecord(Base):
    """Stored representation of one immutable score observation."""

    __tablename__ = "score_observations"
    __table_args__ = (
        Index(
            "ix_score_observations_subject_id_scoring_cutoff",
            "subject_id",
            "scoring_cutoff",
        ),
    )

    score_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    scoring_cutoff: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    feature_vector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_model_score: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class FeatureVectorRecord(Base):
    """Immutable decision-time feature input, keyed by trusted vector identity."""

    __tablename__ = "feature_vectors"
    feature_vector_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(255), nullable=False)
    scoring_cutoff: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class RiskEvaluationRecord(Base):
    """One durable event-to-decision linkage for replay-safe runtime scoring."""

    __tablename__ = "risk_evaluations"
    provider_event_id: Mapped[str] = mapped_column(
        ForeignKey("webhook_events.provider_event_id"), primary_key=True
    )
    decision_id: Mapped[str] = mapped_column(
        ForeignKey("policy_decisions.decision_id"), nullable=False
    )
    case_id: Mapped[str | None] = mapped_column(ForeignKey("risk_cases.case_id"))


class ProbabilityEstimateRecord(Base):
    """Stored representation of one calibrated child of a score observation."""

    __tablename__ = "probability_estimates"
    __table_args__ = (
        Index(
            "ix_probability_estimates_subject_id_scoring_cutoff",
            "subject_id",
            "scoring_cutoff",
        ),
    )

    probability_estimate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    score_id: Mapped[str] = mapped_column(
        ForeignKey("score_observations.score_id"), nullable=False
    )
    base_model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    probability_model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_vector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scoring_cutoff: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    calibrated_probability: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class PolicyDecisionRecord(Base):
    """Stored representation of an immutable cost-sensitive policy decision."""

    __tablename__ = "policy_decisions"
    __table_args__ = (
        Index(
            "ix_policy_decisions_subject_id_scoring_cutoff",
            "subject_id",
            "scoring_cutoff",
        ),
        Index("ix_policy_decisions_context_id", "context_id"),
    )

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    probability_estimate_id: Mapped[str] = mapped_column(
        ForeignKey("probability_estimates.probability_estimate_id"), nullable=False
    )
    score_id: Mapped[str] = mapped_column(
        ForeignKey("score_observations.score_id"), nullable=False
    )
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    base_model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    probability_model_id: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_vector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scoring_cutoff: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    context_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chosen_action: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class InvestigationRequestRecord(Base):
    """One immutable account investigation request for a policy decision."""

    __tablename__ = "investigation_requests"
    __table_args__ = (
        Index(
            "ix_investigation_requests_subject_id_cutoff_time",
            "subject_id",
            "cutoff_time",
        ),
    )

    decision_id: Mapped[str] = mapped_column(
        ForeignKey("policy_decisions.decision_id"), primary_key=True
    )
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    probability_estimate_id: Mapped[str] = mapped_column(
        ForeignKey("probability_estimates.probability_estimate_id"), nullable=False
    )
    score_id: Mapped[str] = mapped_column(
        ForeignKey("score_observations.score_id"), nullable=False
    )
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    cutoff_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    context_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    policy_action: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class InvestigationReportRecord(Base):
    """One trusted report for one operational investigation run."""

    __tablename__ = "investigation_reports"
    __table_args__ = (UniqueConstraint("run_id"),)

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("investigation_runs.run_id"), nullable=False
    )
    investigation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_action: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    pattern: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class RiskCaseRecord(Base):
    """Storage representation of an operational, long-lived risk case."""

    __tablename__ = "risk_cases"
    __table_args__ = (
        Index("ix_risk_cases_subject_id_opened_at", "subject_id", "opened_at"),
        Index("ix_risk_cases_status_opened_at", "status", "opened_at"),
        Index(
            "uq_risk_cases_open_subject",
            "subject_type",
            "subject_id",
            unique=True,
            postgresql_where=text("status = 'OPEN'"),
            sqlite_where=text("status = 'OPEN'"),
        ),
    )

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    opening_decision_id: Mapped[str] = mapped_column(
        ForeignKey("policy_decisions.decision_id"), nullable=False
    )
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class RiskCaseDecisionRecord(Base):
    """Association allowing a risk case to retain policy decisions over time."""

    __tablename__ = "risk_case_decisions"

    case_id: Mapped[str] = mapped_column(
        ForeignKey("risk_cases.case_id"), primary_key=True
    )
    decision_id: Mapped[str] = mapped_column(
        ForeignKey("policy_decisions.decision_id"), primary_key=True
    )


class InvestigationRunRecord(Base):
    """Operational execution attempt; provenance identity is not its primary key."""

    __tablename__ = "investigation_runs"
    __table_args__ = (
        Index(
            "ix_investigation_runs_decision_id_started_at", "decision_id", "started_at"
        ),
        Index("ix_investigation_runs_case_id_started_at", "case_id", "started_at"),
    )

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    decision_id: Mapped[str] = mapped_column(
        ForeignKey("investigation_requests.decision_id"), nullable=False
    )
    case_id: Mapped[str | None] = mapped_column(
        ForeignKey("risk_cases.case_id"), nullable=True
    )
    investigation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)


class WebhookEventRecord(Base):
    """Durable, append-only provider delivery inbox row.

    The provider's delivery ID is the database identity: it is deliberately
    not replaced by an application-generated identifier.
    """

    __tablename__ = "webhook_events"
    __table_args__ = (Index("ix_webhook_events_received_at", "received_at"),)

    provider_event_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    raw_body: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    raw_body_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_detail: Mapped[str | None] = mapped_column(String(1000))


class NormalizedEventRecord(Base):
    """Trusted canonical event derived from one durable provider delivery."""

    __tablename__ = "normalized_events"
    __table_args__ = (
        Index(
            "ix_normalized_events_account_id_occurred_at", "account_id", "occurred_at"
        ),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    provider_event_id: Mapped[str] = mapped_column(
        ForeignKey("webhook_events.provider_event_id"), unique=True, nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(36), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)
