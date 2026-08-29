"""Small session-owned repositories for immutable operational lineage."""

from collections.abc import Callable
from typing import cast

from sqlalchemy.orm import Session

from mayajaal.calibration import ProbabilityEstimate
from mayajaal.investigation import InvestigationReport, InvestigationRequest
from mayajaal.policy import PolicyDecision
from mayajaal.scoring import ScoreObservation

from .models import (
    InvestigationReportRecord,
    InvestigationRequestRecord,
    PolicyDecisionRecord,
    ProbabilityEstimateRecord,
    ScoreObservationRecord,
)
from .serialization import from_payload, payload_for


class ImmutablePersistenceConflict(ValueError):
    """Raised when one authoritative ID is reused for different semantics."""


class _ImmutableRepository[
    Domain: (
        ScoreObservation,
        ProbabilityEstimate,
        PolicyDecision,
        InvestigationRequest,
        InvestigationReport,
    ),
    Record: (
        ScoreObservationRecord,
        ProbabilityEstimateRecord,
        PolicyDecisionRecord,
        InvestigationRequestRecord,
        InvestigationReportRecord,
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


class InvestigationReportRepository(
    _ImmutableRepository[InvestigationReport, InvestigationReportRecord]
):
    """Persistence boundary for one immutable report per existing request."""

    def __init__(self, session: Session) -> None:
        super().__init__(
            session,
            record_type=InvestigationReportRecord,
            domain_type=InvestigationReport,
            key_for=lambda value: value.request.decision_id,
            make_record=lambda value, payload: InvestigationReportRecord(
                decision_id=value.request.decision_id,
                policy_action=value.policy_action.value,
                status=value.status.value,
                pattern=value.pattern.value,
                payload=payload,
            ),
        )
