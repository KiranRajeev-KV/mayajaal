"""Stage 12C orchestration: processed event to persisted risk decision only."""

from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy.orm import Session

from mayajaal.calibration import ProbabilityModel, estimate_probability
from mayajaal.evaluation import FrozenFullEvaluation
from mayajaal.features import FeatureService
from mayajaal.graph import Neo4jGraphRepository
from mayajaal.investigation import InvestigationSubjectType
from mayajaal.policy import (
    DecisionContext,
    PolicyAction,
    PolicyDecision,
    PolicyModel,
    decide,
)
from mayajaal.schemas import EventType
from mayajaal.scoring.service import score_feature_vector

from .contracts import RiskCase
from .db import (
    FeatureVectorRepository,
    NormalizedEventRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    RiskCaseRepository,
    RiskEvaluationRepository,
    ScoreObservationRepository,
    WebhookEventRepository,
)
from .db.session import SessionFactory
from .webhooks import WebhookProcessingStatus


class RiskEvaluationUnavailable(ValueError):
    """A provider event is not eligible for a truthful risk evaluation."""


@dataclass(frozen=True)
class RiskEvaluationResult:
    provider_event_id: str
    decision_id: str | None
    case_id: str | None
    reused: bool


class RuntimeRiskScoringService:
    """Compose feature, frozen scoring, calibration, policy, and persistence.

    The scoring cutoff is the normalized event's ``ingested_at`` / inbox receipt
    time, not provider occurrence time. Thus an out-of-order older event is only
    visible after Mayajaal actually knew it, while feature values still retain
    their provider occurrence timestamps for the existing formulas.
    """

    def __init__(
        self,
        sessions: SessionFactory,
        graph: Neo4jGraphRepository,
        frozen: FrozenFullEvaluation,
        probability_model: ProbabilityModel,
        policy_model: PolicyModel,
    ) -> None:
        if (
            frozen.base_model_id != probability_model.base_model_id
            or policy_model.base_model_id != frozen.base_model_id
        ):
            raise ValueError("runtime artifacts have mismatched base-model lineage")
        if policy_model.probability_model_id != probability_model.probability_model_id:
            raise ValueError("runtime policy and calibration artifacts do not match")
        self._sessions, self._graph, self._frozen = sessions, graph, frozen
        self._probability_model, self._policy_model = probability_model, policy_model

    def process(self, provider_event_id: str) -> RiskEvaluationResult:
        with self._sessions.begin() as session:
            existing = RiskEvaluationRepository(session).get(provider_event_id)
            if existing is not None:
                return RiskEvaluationResult(
                    provider_event_id, existing[0], existing[1], True
                )
            delivery = WebhookEventRepository(session).get(provider_event_id)
            if (
                delivery is None
                or delivery.status != WebhookProcessingStatus.PROCESSED.value
            ):
                raise RiskEvaluationUnavailable(
                    "risk evaluation requires a PROCESSED webhook event"
                )
            event = NormalizedEventRepository(session).get_for_provider(
                provider_event_id
            )
            if event is None:
                raise RiskEvaluationUnavailable(
                    "processed webhook is missing normalized event"
                )
            if event.event_type is EventType.ACCOUNT_CREATED:
                return RiskEvaluationResult(provider_event_id, None, None, False)
            context = _decision_context(delivery.payload)
            cutoff = event.ingested_at
            projection = self._graph.feature_projection_at(cutoff)
            feature_service = FeatureService(projection)
            vector = feature_service.extract(str(event.account_id), cutoff)
            score = score_feature_vector(self._frozen, vector)
            estimate = estimate_probability(
                self._probability_model, score, scoring_context_id=context.context_id
            )
            decision = decide(
                self._policy_model, self._probability_model, score, estimate, context
            )
            FeatureVectorRepository(session, feature_service.schema).persist(vector)
            ScoreObservationRepository(session).persist(score)
            ProbabilityEstimateRepository(session).persist(estimate)
            PolicyDecisionRepository(session).persist(decision)
            case_id = self._link_case(session, decision, cutoff)
            RiskEvaluationRepository(session).persist(
                provider_event_id, decision.decision_id, case_id
            )
            return RiskEvaluationResult(
                provider_event_id, decision.decision_id, case_id, False
            )

    def _link_case(
        self, session: Session, decision: PolicyDecision, cutoff: datetime
    ) -> str | None:
        # REVIEW/BLOCK are operationally actionable; ALLOW is not. Keep this rule
        # here rather than in a generic persistence repository.
        if decision.chosen_action is PolicyAction.ALLOW:
            return None
        repository = RiskCaseRepository(session)
        existing = repository.open_for_subject(decision.subject_id)
        if existing is not None:
            repository.attach_decision(existing.case_id, decision.decision_id)
            return existing.case_id
        case = RiskCase(
            case_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"mayajaal:risk-case:{decision.subject_id}:{decision.decision_id}",
                )
            ),
            subject_type=InvestigationSubjectType.ACCOUNT,
            subject_id=decision.subject_id,
            opened_at=cutoff,
            opening_decision_id=decision.decision_id,
        )
        repository.persist(case)
        repository.attach_decision(case.case_id, decision.decision_id)
        return case.case_id


def _decision_context(payload: dict[str, object]) -> DecisionContext:
    """Read only namespaced simulator policy context; never alter Event."""
    envelope = payload.get("payload")
    envelope_values = (
        cast(dict[str, object], envelope) if isinstance(envelope, dict) else {}
    )
    fixture = envelope_values.get("mayajaal")
    if not isinstance(fixture, dict):
        raise RiskEvaluationUnavailable(
            "processed event lacks Mayajaal decision context"
        )
    fixture_values = cast(dict[str, object], fixture)
    exposure = fixture_values.get("exposure_paise")
    context_id = fixture_values.get("context_id")
    if not isinstance(exposure, int) or isinstance(exposure, bool):
        raise RiskEvaluationUnavailable("processed event lacks integer exposure_paise")
    if context_id is not None and not isinstance(context_id, str):
        raise RiskEvaluationUnavailable("invalid decision context_id")
    return DecisionContext(exposure_paise=exposure, context_id=context_id)
