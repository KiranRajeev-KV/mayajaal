"""Transactional persistence coordination for already-produced runtime lineage."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from mayajaal.calibration import ProbabilityEstimate
from mayajaal.investigation import InvestigationExecution, InvestigationRequest
from mayajaal.policy import PolicyDecision
from mayajaal.scoring import ScoreObservation

from .contracts import InvestigationRun, PersistedInvestigationReport, RiskCase
from .db.repositories import (
    InvestigationReportRepository,
    InvestigationRequestRepository,
    InvestigationRunRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    RiskCaseRepository,
    ScoreObservationRepository,
)


@dataclass(frozen=True)
class PersistedRuntimeLineage:
    """References created by one caller-owned transactional persistence operation."""

    investigation_run: InvestigationRun
    investigation_report: PersistedInvestigationReport


class RuntimeLineagePersistenceService:
    """Persist trusted runtime outputs; it never scores, investigates, or commits."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def persist_execution(
        self,
        *,
        score_observation: ScoreObservation,
        probability_estimate: ProbabilityEstimate,
        policy_decision: PolicyDecision,
        investigation_request: InvestigationRequest,
        execution: InvestigationExecution,
        run_id: str,
        started_at: datetime,
        completed_at: datetime | None = None,
        risk_case: RiskCase | None = None,
        case_id: str | None = None,
    ) -> PersistedRuntimeLineage:
        """Persist one verified execution and all of its upstream immutable parents.

        The surrounding ``session_scope`` owns commit/rollback, so any error in
        this sequence leaves no partially committed operational lineage.
        """
        _validate_lineage(
            score_observation,
            probability_estimate,
            policy_decision,
            investigation_request,
            execution,
            risk_case,
        )
        ScoreObservationRepository(self._session).persist(score_observation)
        ProbabilityEstimateRepository(self._session).persist(probability_estimate)
        PolicyDecisionRepository(self._session).persist(policy_decision)
        InvestigationRequestRepository(self._session).persist(investigation_request)
        if risk_case is not None:
            case_repository = RiskCaseRepository(self._session)
            case_repository.persist(risk_case)
            case_repository.attach_decision(
                risk_case.case_id, policy_decision.decision_id
            )
        run = InvestigationRun.from_execution(
            run_id=run_id,
            execution=execution,
            started_at=started_at,
            completed_at=completed_at,
            case_id=case_id
            if case_id is not None
            else (None if risk_case is None else risk_case.case_id),
        )
        InvestigationRunRepository(self._session).persist(run)
        report = PersistedInvestigationReport.from_execution(
            run=run, execution=execution
        )
        InvestigationReportRepository(self._session).persist(report)
        return PersistedRuntimeLineage(
            investigation_run=run,
            investigation_report=report,
        )


def _validate_lineage(
    score_observation: ScoreObservation,
    probability_estimate: ProbabilityEstimate,
    policy_decision: PolicyDecision,
    investigation_request: InvestigationRequest,
    execution: InvestigationExecution,
    risk_case: RiskCase | None,
) -> None:
    """Reject mismatched parent objects before staging any rows in the session."""
    request = execution.report.request
    if investigation_request != request:
        raise ValueError("execution report request does not match supplied request")
    if investigation_request.decision_id != policy_decision.decision_id:
        raise ValueError("investigation request does not match policy decision")
    if probability_estimate.score_id != score_observation.score_id:
        raise ValueError("probability estimate does not match score observation")
    if (
        policy_decision.probability_estimate_id
        != probability_estimate.probability_estimate_id
    ):
        raise ValueError("policy decision does not match probability estimate")
    if policy_decision.score_id != score_observation.score_id:
        raise ValueError("policy decision does not match score observation")
    if risk_case is not None:
        if risk_case.subject_id != policy_decision.subject_id:
            raise ValueError("risk case subject does not match policy decision")
        if risk_case.subject_type != investigation_request.subject_type:
            raise ValueError(
                "risk case subject type does not match investigation request"
            )
