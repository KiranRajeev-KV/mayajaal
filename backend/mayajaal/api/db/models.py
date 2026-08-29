"""SQLAlchemy persistence representations for immutable runtime lineage.

These models deliberately do not replace the authoritative domain contracts.
Each row retains a full JSON payload and only materializes fields needed for
identity, lineage, and the immediate subject/time/status lookup paths.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, String
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
    """One immutable report keyed by its already-unique request decision ID.

    The current authoritative report contract has no independent report field;
    it is therefore persisted as a one-to-one child of its request. Future case
    orchestration can introduce a report-run identifier without overloading this
    operational lineage boundary.
    """

    __tablename__ = "investigation_reports"

    decision_id: Mapped[str] = mapped_column(
        ForeignKey("investigation_requests.decision_id"), primary_key=True
    )
    policy_action: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    pattern: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[Payload] = mapped_column(JsonPayload, nullable=False)
