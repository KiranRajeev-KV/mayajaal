"""Framework-neutral contracts for bounded, read-only investigations."""

from enum import StrEnum
from typing import Annotated

from pydantic import Field, JsonValue, model_validator

from mayajaal.calibration import (
    ProbabilityEstimate,
    ProbabilityModel,
    verify_probability_estimate,
)
from mayajaal.policy import PolicyAction, PolicyDecision
from mayajaal.schemas.common import AwareDatetime, SchemaModel
from mayajaal.scoring import ScoreObservation

from .errors import GroundingFailureCode

NonEmptyId = Annotated[str, Field(min_length=1)]


class InvestigationStatus(StrEnum):
    """Lifecycle state of a future investigation result."""

    COMPLETED = "COMPLETED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FAILED = "FAILED"


class InvestigationPattern(StrEnum):
    """Initial, deliberately broad investigation pattern taxonomy."""

    PROMO_RING = "PROMO_RING"
    REFUND_RING = "REFUND_RING"
    MIXED_ABUSE = "MIXED_ABUSE"
    BENIGN_SHARED_IDENTITY = "BENIGN_SHARED_IDENTITY"
    INCONCLUSIVE = "INCONCLUSIVE"


class InvestigationSubjectType(StrEnum):
    """Subject types eligible for the initial account-scored workflow."""

    ACCOUNT = "ACCOUNT"


class EvidenceSource(StrEnum):
    """Closed, read-only evidence sources allowed for the next slice."""

    MODEL_EXPLANATION = "MODEL_EXPLANATION"
    TEMPORAL_GRAPH = "TEMPORAL_GRAPH"
    EVENT_HISTORY = "EVENT_HISTORY"
    IDENTITY_SUMMARY = "IDENTITY_SUMMARY"
    CASE_TIMELINE = "CASE_TIMELINE"


class EvidenceType(StrEnum):
    """Closed factual evidence types allowed for the next slice."""

    RISK_DRIVER = "RISK_DRIVER"
    SHARED_DEVICE = "SHARED_DEVICE"
    SHARED_PAYMENT_IDENTITY = "SHARED_PAYMENT_IDENTITY"
    SHARED_IP = "SHARED_IP"
    SHARED_ADDRESS = "SHARED_ADDRESS"
    RELATED_ACCOUNT_ACTIVITY = "RELATED_ACCOUNT_ACTIVITY"
    IDENTITY_NEIGHBORHOOD = "IDENTITY_NEIGHBORHOOD"
    PROMOTION_ACTIVITY = "PROMOTION_ACTIVITY"
    REFUND_ACTIVITY = "REFUND_ACTIVITY"
    TIMELINE_EVENT = "TIMELINE_EVENT"


class InvestigationTriggerReason(StrEnum):
    """Deterministic explanation for an investigation trigger outcome."""

    REVIEW_ACTION = "REVIEW_ACTION"
    BLOCK_ACTION = "BLOCK_ACTION"
    UNSTABLE_ALLOW = "UNSTABLE_ALLOW"
    STABLE_ALLOW = "STABLE_ALLOW"
    DISABLED_BY_CONFIG = "DISABLED_BY_CONFIG"


class InvestigationTriggerConfig(SchemaModel):
    """Explicit action/stability cases that may open an investigation."""

    investigate_review: bool = True
    investigate_block: bool = True
    investigate_unstable_allow: bool = True


class ReasoningEffort(StrEnum):
    """Known reasoning-effort names; provider/model support remains specific."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class InvestigationConfig(SchemaModel):
    """Validated limits and non-secret model selection for investigations."""

    max_tool_calls: int = Field(default=8, ge=1)
    max_iterations: int = Field(default=4, ge=1)
    max_graph_hops: int = Field(default=2, ge=0)
    max_graph_nodes: int = Field(default=200, ge=1)
    max_graph_edges: int = Field(default=500, ge=1)
    max_related_accounts: int = Field(default=50, ge=1)
    max_events_per_tool: int = Field(default=100, ge=1)
    max_timeline_events: int = Field(default=20, ge=1)
    max_risk_drivers: int = Field(default=5, ge=1)
    model_name: str | None = Field(default=None, min_length=1)
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    triggers: InvestigationTriggerConfig = Field(
        default_factory=InvestigationTriggerConfig
    )


class InvestigationTrigger(SchemaModel):
    """Pure trigger result; it never changes the supplied policy decision."""

    should_investigate: bool
    reason: InvestigationTriggerReason

    @model_validator(mode="after")
    def validate_reason_matches_outcome(self) -> "InvestigationTrigger":
        triggered_reasons = {
            InvestigationTriggerReason.REVIEW_ACTION,
            InvestigationTriggerReason.BLOCK_ACTION,
            InvestigationTriggerReason.UNSTABLE_ALLOW,
        }
        if self.should_investigate != (self.reason in triggered_reasons):
            raise ValueError("trigger reason does not match should_investigate")
        return self


class InvestigationRequest(SchemaModel):
    """Immutable policy-decision binding supplied to a future investigator."""

    decision_id: NonEmptyId
    policy_id: NonEmptyId
    probability_estimate_id: NonEmptyId
    score_id: NonEmptyId
    feature_vector_id: NonEmptyId
    subject_type: InvestigationSubjectType
    subject_id: NonEmptyId
    cutoff_time: AwareDatetime
    context_id: str | None = Field(default=None, min_length=1)
    policy_action: PolicyAction
    decision_is_stable_across_scenarios: bool

    @classmethod
    def from_policy_decision(
        cls,
        decision: PolicyDecision,
        probability_model: ProbabilityModel,
        score_observation: ScoreObservation,
        probability_estimate: ProbabilityEstimate,
    ) -> "InvestigationRequest":
        """Bind a request to a verified score and its immutable policy decision."""
        verified_estimate = verify_probability_estimate(
            probability_estimate, probability_model, score_observation
        )
        if (
            decision.base_model_id != verified_estimate.base_model_id
            or decision.probability_model_id != verified_estimate.probability_model_id
            or decision.probability_estimate_id
            != verified_estimate.probability_estimate_id
            or decision.score_id != verified_estimate.score_id
            or decision.subject_id != verified_estimate.subject_id
            or decision.feature_vector_id != verified_estimate.feature_vector_id
            or decision.raw_model_score != verified_estimate.raw_model_score
            or decision.calibrated_fraud_probability
            != verified_estimate.calibrated_probability
            or decision.scoring_cutoff != verified_estimate.scoring_cutoff
        ):
            raise ValueError(
                "policy decision does not match verified probability estimate"
            )
        return cls(
            decision_id=decision.decision_id,
            policy_id=decision.policy_id,
            probability_estimate_id=decision.probability_estimate_id,
            score_id=verified_estimate.score_id,
            feature_vector_id=verified_estimate.feature_vector_id,
            subject_type=InvestigationSubjectType.ACCOUNT,
            subject_id=verified_estimate.subject_id,
            cutoff_time=verified_estimate.scoring_cutoff,
            context_id=decision.context.context_id,
            policy_action=decision.chosen_action,
            decision_is_stable_across_scenarios=decision.decision_is_stable_across_scenarios,
        )


class EvidenceItem(SchemaModel):
    """One factual observation known no later than the request cutoff."""

    evidence_id: NonEmptyId
    evidence_type: EvidenceType
    source: EvidenceSource
    observed_at: AwareDatetime
    cutoff_time: AwareDatetime
    subject_ids: tuple[NonEmptyId, ...] = Field(min_length=1)
    facts: dict[NonEmptyId, JsonValue] = Field(min_length=1)

    @classmethod
    def from_request(
        cls,
        request: InvestigationRequest,
        *,
        evidence_id: str,
        evidence_type: EvidenceType,
        source: EvidenceSource,
        observed_at: AwareDatetime,
        subject_ids: tuple[str, ...],
        facts: dict[str, JsonValue],
    ) -> "EvidenceItem":
        """Create cutoff-bound evidence without caller control of its cutoff."""
        return cls(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            source=source,
            observed_at=observed_at,
            cutoff_time=request.cutoff_time,
            subject_ids=subject_ids,
            facts=facts,
        )

    def verify_for_request(self, request: InvestigationRequest) -> "EvidenceItem":
        """Reject evidence whose fixed cutoff differs from its investigation."""
        if self.cutoff_time != request.cutoff_time:
            raise ValueError(
                "evidence cutoff_time does not match investigation request"
            )
        return self

    @model_validator(mode="after")
    def validate_temporal_and_label_safety(self) -> "EvidenceItem":
        if self.observed_at > self.cutoff_time:
            raise ValueError("evidence observed_at cannot be after cutoff_time")
        if _contains_evaluation_label_key(self.facts):
            raise ValueError("evidence facts cannot contain evaluation-only label keys")
        return self


class EvidenceFinding(SchemaModel):
    """A future report claim that must identify its supporting observations."""

    claim: str = Field(min_length=1)
    evidence_ids: tuple[NonEmptyId, ...] = Field(min_length=1)


class RelatedEntity(SchemaModel):
    """A related entity named by a report and bound to factual evidence."""

    entity_id: NonEmptyId
    entity_type: NonEmptyId
    evidence_ids: tuple[NonEmptyId, ...] = Field(min_length=1)


class InvestigationUsage(SchemaModel):
    """Future tool-budget consumption recorded with a structured report."""

    tool_calls: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    graph_nodes: int = Field(default=0, ge=0)
    graph_edges: int = Field(default=0, ge=0)
    related_accounts: int = Field(default=0, ge=0)
    events_retrieved: int = Field(default=0, ge=0)


class GroundingFailureDiagnostic(SchemaModel):
    """Non-authoritative debugging metadata for a rejected model candidate.

    It is deliberately separate from ``InvestigationReport``: a failed report
    carries no model claims and this diagnostic is not report provenance.
    """

    code: GroundingFailureCode
    detail: str = Field(min_length=1)
    rejected_candidate: dict[str, JsonValue] | None = None


class InvestigationReport(SchemaModel):
    """Future structured, evidence-referenced output with no action authority."""

    request: InvestigationRequest
    policy_action: PolicyAction
    status: InvestigationStatus
    pattern: InvestigationPattern = InvestigationPattern.INCONCLUSIVE
    key_findings: tuple[EvidenceFinding, ...] = ()
    counterevidence: tuple[EvidenceFinding, ...] = ()
    timeline_evidence_ids: tuple[NonEmptyId, ...] = ()
    related_entities: tuple[RelatedEntity, ...] = ()
    evidence_ids: tuple[NonEmptyId, ...] = ()
    summary: str | None = Field(default=None, min_length=1)
    limitations: tuple[str, ...] = ()
    usage: InvestigationUsage = Field(default_factory=InvestigationUsage)

    @model_validator(mode="after")
    def validate_request_action_and_evidence_references(self) -> "InvestigationReport":
        if self.policy_action is not self.request.policy_action:
            raise ValueError(
                "report policy_action must match the immutable request action"
            )
        declared_ids = set(self.evidence_ids)
        referenced_ids = {
            evidence_id
            for finding in (*self.key_findings, *self.counterevidence)
            for evidence_id in finding.evidence_ids
        } | set(self.timeline_evidence_ids)
        if not referenced_ids.issubset(declared_ids):
            raise ValueError("report findings must reference declared evidence_ids")
        if self.status in {
            InvestigationStatus.BUDGET_EXHAUSTED,
            InvestigationStatus.FAILED,
        } and (
            self.key_findings
            or self.counterevidence
            or self.timeline_evidence_ids
            or self.related_entities
            or self.evidence_ids
        ):
            raise ValueError("operational investigation reports cannot retain claims")
        return self


def _contains_evaluation_label_key(value: JsonValue) -> bool:
    """Reject known evaluation-only label fields from structured evidence facts."""
    forbidden_keys = {
        "abusetypes",
        "coordinationclusterid",
        "fraudlabel",
        "groundtruth",
        "groundtruthlabel",
        "iscoordinatedabuse",
        "isfraud",
        "syntheticlabels",
    }
    if isinstance(value, dict):
        for key, nested_value in value.items():
            normalized_key = "".join(
                character for character in key.casefold() if character.isalnum()
            )
            if normalized_key in forbidden_keys or _contains_evaluation_label_key(
                nested_value
            ):
                return True
    elif isinstance(value, list):
        return any(_contains_evaluation_label_key(item) for item in value)
    return False
