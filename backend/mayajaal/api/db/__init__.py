"""Synchronous SQLAlchemy foundation for Mayajaal operational storage."""

from .base import Base
from .config import DATABASE_URL_ENVIRONMENT_VARIABLE, DatabaseConfig
from .engine import DatabaseRuntime, create_database_runtime, ping_database
from .models import (
    InvestigationReportRecord,
    InvestigationRequestRecord,
    InvestigationRunRecord,
    PolicyDecisionRecord,
    ProbabilityEstimateRecord,
    RiskCaseDecisionRecord,
    RiskCaseRecord,
    ScoreObservationRecord,
)
from .repositories import (
    ImmutablePersistenceConflict,
    InvestigationReportRepository,
    InvestigationRequestRepository,
    InvestigationRunRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    RiskCaseRepository,
    ScoreObservationRepository,
)
from .session import SessionFactory, session_scope

__all__ = [
    "DATABASE_URL_ENVIRONMENT_VARIABLE",
    "Base",
    "DatabaseConfig",
    "DatabaseRuntime",
    "ImmutablePersistenceConflict",
    "InvestigationReportRecord",
    "InvestigationReportRepository",
    "InvestigationRequestRecord",
    "InvestigationRequestRepository",
    "InvestigationRunRecord",
    "InvestigationRunRepository",
    "PolicyDecisionRecord",
    "PolicyDecisionRepository",
    "ProbabilityEstimateRecord",
    "ProbabilityEstimateRepository",
    "RiskCaseDecisionRecord",
    "RiskCaseRecord",
    "RiskCaseRepository",
    "ScoreObservationRecord",
    "ScoreObservationRepository",
    "SessionFactory",
    "create_database_runtime",
    "ping_database",
    "session_scope",
]
