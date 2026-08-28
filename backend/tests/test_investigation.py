"""Focused tests for read-only investigation contracts and trigger rules."""

import inspect
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from math import log

import mayajaal.investigation.models as investigation_models
import mayajaal.investigation.triggers as investigation_triggers
from mayajaal.calibration import (
    CalibrationConfig,
    ProbabilityModel,
    SigmoidCalibrator,
    estimate_probability,
)
from mayajaal.investigation import (
    EvidenceFinding,
    EvidenceItem,
    EvidenceSource,
    EvidenceType,
    InvestigationConfig,
    InvestigationPattern,
    InvestigationReport,
    InvestigationRequest,
    InvestigationStatus,
    InvestigationSubjectType,
    InvestigationTriggerConfig,
    InvestigationTriggerReason,
    RelatedEntity,
    should_investigate,
)
from mayajaal.policy import (
    DecisionContext,
    PolicyAction,
    PolicyConfig,
    build_policy_model,
    decide,
)
from mayajaal.scoring import ScoreObservation, score_id, score_observation_semantics


def probability_model() -> ProbabilityModel:
    """Create a minimal verified probability model fixture."""
    return ProbabilityModel(
        base_model_id="base-model-fixture",
        probability_model_id="probability-model-fixture",
        calibration_config=CalibrationConfig(
            minimum_positive_samples=1, minimum_negative_samples=1
        ),
        calibrator=SigmoidCalibrator(coefficient=1.0, intercept=0.0),
        frozen_provenance={"base_model_id": "base-model-fixture"},
    )


def scored_decision(probability: float, *, exposure_paise: int = 250_000):
    """Create one verified policy decision without any model-training concern."""
    probability_parent = probability_model()
    raw_score = log(probability / (1.0 - probability))
    semantics = score_observation_semantics(
        score_contract_version=1,
        base_model_id=probability_parent.base_model_id,
        subject_id="account-123",
        scoring_cutoff=cutoff(),
        raw_model_score=raw_score,
        feature_vector_id="feature-vector-fixture",
    )
    score_observation = ScoreObservation(
        score_id=score_id(**semantics),
        base_model_id=probability_parent.base_model_id,
        subject_id="account-123",
        scoring_cutoff=cutoff(),
        raw_model_score=raw_score,
        feature_vector_id="feature-vector-fixture",
    )
    estimate = estimate_probability(
        probability_parent,
        score_observation,
        scoring_context_id="order-123",
    )
    policy_model = build_policy_model(probability_parent, PolicyConfig())
    policy_decision = decide(
        policy_model,
        probability_parent,
        score_observation,
        estimate,
        DecisionContext(exposure_paise=exposure_paise, context_id="order-123"),
    )
    return probability_parent, score_observation, estimate, policy_decision


def decision(probability: float, *, exposure_paise: int = 250_000):
    """Return only the policy decision for trigger-focused fixtures."""
    return scored_decision(probability, exposure_paise=exposure_paise)[3]


def cutoff() -> datetime:
    """Return one explicit cutoff for contract fixtures."""
    return datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


class InvestigationContractTests(unittest.TestCase):
    def test_default_triggers_preserve_policy_and_cover_all_actions(self) -> None:
        review = decision(0.05)
        original_review = review
        block = decision(0.90)
        unstable_allow = decision(0.007)
        stable_allow = decision(0.001)
        self.assertIs(review.chosen_action, PolicyAction.REVIEW)
        self.assertIs(block.chosen_action, PolicyAction.BLOCK)
        self.assertIs(unstable_allow.chosen_action, PolicyAction.ALLOW)
        self.assertFalse(unstable_allow.decision_is_stable_across_scenarios)
        self.assertIs(stable_allow.chosen_action, PolicyAction.ALLOW)
        self.assertTrue(stable_allow.decision_is_stable_across_scenarios)

        self.assertEqual(
            should_investigate(review, InvestigationConfig()).reason,
            InvestigationTriggerReason.REVIEW_ACTION,
        )
        self.assertEqual(
            should_investigate(block, InvestigationConfig()).reason,
            InvestigationTriggerReason.BLOCK_ACTION,
        )
        self.assertEqual(
            should_investigate(unstable_allow, InvestigationConfig()).reason,
            InvestigationTriggerReason.UNSTABLE_ALLOW,
        )
        stable_trigger = should_investigate(stable_allow, InvestigationConfig())
        self.assertFalse(stable_trigger.should_investigate)
        self.assertEqual(stable_trigger.reason, InvestigationTriggerReason.STABLE_ALLOW)
        self.assertIs(review, original_review)

    def test_trigger_configuration_overrides_are_explicit(self) -> None:
        disabled = InvestigationConfig(
            triggers=InvestigationTriggerConfig(
                investigate_review=False,
                investigate_block=False,
                investigate_unstable_allow=False,
            )
        )
        for policy_decision in (decision(0.05), decision(0.90), decision(0.01)):
            trigger = should_investigate(policy_decision, disabled)
            self.assertFalse(trigger.should_investigate)
            self.assertEqual(
                trigger.reason, InvestigationTriggerReason.DISABLED_BY_CONFIG
            )

    def test_request_binds_decision_lineage_to_verified_scoring_cutoff(self) -> None:
        (
            probability_parent,
            score_observation,
            probability_estimate,
            policy_decision,
        ) = scored_decision(0.05)
        request = InvestigationRequest.from_policy_decision(
            policy_decision,
            probability_parent,
            score_observation,
            probability_estimate,
        )
        self.assertEqual(request.decision_id, policy_decision.decision_id)
        self.assertEqual(request.policy_id, policy_decision.policy_id)
        self.assertEqual(
            request.probability_estimate_id, policy_decision.probability_estimate_id
        )
        self.assertEqual(request.score_id, score_observation.score_id)
        self.assertEqual(request.feature_vector_id, score_observation.feature_vector_id)
        self.assertIs(request.policy_action, policy_decision.chosen_action)
        self.assertEqual(request.cutoff_time, probability_estimate.scoring_cutoff)
        self.assertEqual(request.cutoff_time, policy_decision.scoring_cutoff)
        self.assertIs(request.subject_type, InvestigationSubjectType.ACCOUNT)
        self.assertEqual(request.subject_id, "account-123")
        self.assertEqual(request.context_id, "order-123")
        self.assertEqual(
            request.decision_is_stable_across_scenarios,
            policy_decision.decision_is_stable_across_scenarios,
        )
        with self.assertRaises(ValueError):
            _ = InvestigationRequest(
                decision_id="",
                policy_id="policy-1",
                probability_estimate_id="estimate-1",
                score_id="score-1",
                feature_vector_id="feature-vector-1",
                subject_type=InvestigationSubjectType.ACCOUNT,
                subject_id="account-1",
                cutoff_time=cutoff(),
                policy_action=PolicyAction.REVIEW,
                decision_is_stable_across_scenarios=True,
            )
        with self.assertRaises(ValueError):
            _ = InvestigationRequest(
                decision_id="decision-1",
                policy_id="policy-1",
                probability_estimate_id="estimate-1",
                score_id="score-1",
                feature_vector_id="feature-vector-1",
                subject_type=InvestigationSubjectType.ACCOUNT,
                subject_id="account-123",
                cutoff_time=datetime(2026, 5, 1, 12, 0),  # noqa: DTZ001
                policy_action=PolicyAction.REVIEW,
                decision_is_stable_across_scenarios=True,
            )
        changed_cutoff = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
        changed_score_semantics = score_observation_semantics(
            score_contract_version=1,
            base_model_id=probability_parent.base_model_id,
            subject_id="account-123",
            scoring_cutoff=changed_cutoff,
            raw_model_score=probability_estimate.raw_model_score,
            feature_vector_id="feature-vector-fixture",
        )
        changed_score = ScoreObservation(
            score_id=score_id(**changed_score_semantics),
            base_model_id=probability_parent.base_model_id,
            subject_id="account-123",
            scoring_cutoff=changed_cutoff,
            raw_model_score=probability_estimate.raw_model_score,
            feature_vector_id="feature-vector-fixture",
        )
        changed_cutoff_estimate = estimate_probability(
            probability_parent,
            changed_score,
            scoring_context_id=probability_estimate.scoring_context_id,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _ = InvestigationRequest.from_policy_decision(
                policy_decision,
                probability_parent,
                changed_score,
                changed_cutoff_estimate,
            )
        with self.assertRaisesRegex(ValueError, "semantics or calibrated probability"):
            _ = InvestigationRequest.from_policy_decision(
                policy_decision,
                probability_parent,
                score_observation,
                replace(
                    probability_estimate,
                    scoring_cutoff=datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
                ),
            )

    def test_investigation_budgets_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(max_tool_calls=0)
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(max_iterations=0)
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(max_graph_hops=-1)
        with self.assertRaises(ValueError):
            _ = InvestigationConfig(max_events_per_tool=0)

    def test_evidence_requires_structured_cutoff_safe_label_free_facts(self) -> None:
        evidence = EvidenceItem(
            evidence_id="evidence-1",
            evidence_type=EvidenceType.SHARED_DEVICE,
            source=EvidenceSource.TEMPORAL_GRAPH,
            observed_at=cutoff(),
            cutoff_time=cutoff(),
            subject_ids=("account-123", "device-456"),
            facts={"shared_account_count": 3, "identity_type": "device"},
        )
        self.assertEqual(evidence.facts["shared_account_count"], 3)
        (
            probability_parent,
            score_observation,
            probability_estimate,
            policy_decision,
        ) = scored_decision(0.05)
        request = InvestigationRequest.from_policy_decision(
            policy_decision,
            probability_parent,
            score_observation,
            probability_estimate,
        )
        request_bound = EvidenceItem.from_request(
            request,
            evidence_id="evidence-request-1",
            evidence_type=EvidenceType.SHARED_DEVICE,
            source=EvidenceSource.TEMPORAL_GRAPH,
            observed_at=cutoff(),
            subject_ids=("account-123", "device-456"),
            facts={"shared_account_count": 3},
        )
        self.assertEqual(request_bound.cutoff_time, request.cutoff_time)
        self.assertEqual(request_bound.verify_for_request(request), request_bound)
        with self.assertRaisesRegex(ValueError, "does not match investigation request"):
            _ = evidence.model_copy(
                update={"cutoff_time": datetime(2026, 5, 2, 12, 0, tzinfo=UTC)}
            ).verify_for_request(request)
        with self.assertRaisesRegex(ValueError, "after cutoff_time"):
            _ = EvidenceItem(
                evidence_id="evidence-1",
                evidence_type=EvidenceType.SHARED_DEVICE,
                source=EvidenceSource.TEMPORAL_GRAPH,
                observed_at=datetime(2026, 5, 1, 12, 1, tzinfo=UTC),
                cutoff_time=cutoff(),
                subject_ids=("account-123",),
                facts={"shared_account_count": 3},
            )
        with self.assertRaisesRegex(ValueError, "evaluation-only label"):
            _ = EvidenceItem(
                evidence_id="evidence-1",
                evidence_type=EvidenceType.SHARED_DEVICE,
                source=EvidenceSource.TEMPORAL_GRAPH,
                observed_at=cutoff(),
                cutoff_time=cutoff(),
                subject_ids=("account-123",),
                facts={"nested": {"synthetic_labels": True}},
            )
        with self.assertRaises(ValueError):
            _ = EvidenceItem(
                evidence_id="",
                evidence_type=EvidenceType.SHARED_DEVICE,
                source=EvidenceSource.TEMPORAL_GRAPH,
                observed_at=cutoff(),
                cutoff_time=cutoff(),
                subject_ids=(),
                facts={},
            )
        with self.assertRaises(ValueError):
            _ = EvidenceItem.model_validate(
                {
                    "evidence_id": "evidence-1",
                    "evidence_type": "ARBITRARY_TYPE",
                    "source": "ARBITRARY_SOURCE",
                    "observed_at": cutoff().isoformat(),
                    "cutoff_time": cutoff().isoformat(),
                    "subject_ids": ["account-123"],
                    "facts": {"shared_account_count": 3},
                }
            )

    def test_report_carries_input_action_and_only_evidence_referenced_claims(
        self,
    ) -> None:
        (
            probability_parent,
            score_observation,
            probability_estimate,
            policy_decision,
        ) = scored_decision(0.05)
        request = InvestigationRequest.from_policy_decision(
            policy_decision,
            probability_parent,
            score_observation,
            probability_estimate,
        )
        report = InvestigationReport(
            request=request,
            policy_action=request.policy_action,
            status=InvestigationStatus.COMPLETED,
            pattern=InvestigationPattern.PROMO_RING,
            key_findings=(
                EvidenceFinding(
                    claim="A promotion was reused by related accounts.",
                    evidence_ids=("evidence-1",),
                ),
            ),
            related_entities=(
                RelatedEntity(
                    entity_id="promotion-1",
                    entity_type="promotion",
                    evidence_ids=("evidence-1",),
                ),
            ),
            evidence_ids=("evidence-1",),
            summary="Evidence supports additional review.",
        )
        self.assertIs(report.policy_action, request.policy_action)
        alternate_action = (
            PolicyAction.ALLOW
            if request.policy_action is not PolicyAction.ALLOW
            else PolicyAction.BLOCK
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            _ = InvestigationReport(
                request=request,
                policy_action=alternate_action,
                status=InvestigationStatus.COMPLETED,
            )
        with self.assertRaises(ValueError):
            _ = InvestigationReport.model_validate(
                {
                    "request": request.model_dump(mode="json"),
                    "policy_action": request.policy_action.value,
                    "status": InvestigationStatus.COMPLETED.value,
                    "evidence_ids": ["evidence-1"],
                    "enforcement_action": PolicyAction.BLOCK.value,
                }
            )
        with self.assertRaisesRegex(ValueError, "declared evidence_ids"):
            _ = InvestigationReport(
                request=request,
                policy_action=request.policy_action,
                status=InvestigationStatus.COMPLETED,
                key_findings=(
                    EvidenceFinding(claim="Unsupported", evidence_ids=("missing",)),
                ),
                evidence_ids=("evidence-1",),
            )

    def test_investigation_package_has_no_model_or_agent_dependencies(self) -> None:
        source = "\n".join(
            (
                inspect.getsource(investigation_models),
                inspect.getsource(investigation_triggers),
            )
        ).casefold()
        for forbidden_dependency in (
            "langchain",
            "openai",
            "catboost",
            "mayajaal.synthetic",
        ):
            self.assertNotIn(forbidden_dependency, source)


if __name__ == "__main__":
    _ = unittest.main()
