"""Focused tests for read-only investigation contracts and trigger rules."""

import inspect
import unittest
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
    InvestigationConfig,
    InvestigationPattern,
    InvestigationReport,
    InvestigationRequest,
    InvestigationStatus,
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


def decision(probability: float, *, exposure_paise: int = 250_000):
    """Create one verified policy decision without any model-training concern."""
    probability_parent = probability_model()
    estimate = estimate_probability(
        probability_parent,
        log(probability / (1.0 - probability)),
        scoring_context_id="order-123",
    )
    return decide(
        build_policy_model(probability_parent, PolicyConfig()),
        probability_parent,
        estimate,
        DecisionContext(exposure_paise=exposure_paise, context_id="order-123"),
    )


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

    def test_request_binds_decision_lineage_and_requires_aware_cutoff(self) -> None:
        policy_decision = decision(0.05)
        request = InvestigationRequest.from_policy_decision(
            policy_decision, subject_id="account-123", cutoff_time=cutoff()
        )
        self.assertEqual(request.decision_id, policy_decision.decision_id)
        self.assertEqual(request.policy_id, policy_decision.policy_id)
        self.assertEqual(
            request.probability_estimate_id, policy_decision.probability_estimate_id
        )
        self.assertIs(request.policy_action, policy_decision.chosen_action)
        self.assertEqual(
            request.decision_is_stable_across_scenarios,
            policy_decision.decision_is_stable_across_scenarios,
        )
        with self.assertRaises(ValueError):
            _ = InvestigationRequest(
                decision_id="",
                policy_id="policy-1",
                probability_estimate_id="estimate-1",
                subject_id="account-1",
                cutoff_time=cutoff(),
                policy_action=PolicyAction.REVIEW,
                decision_is_stable_across_scenarios=True,
            )
        with self.assertRaises(ValueError):
            _ = InvestigationRequest.from_policy_decision(
                policy_decision,
                subject_id="account-123",
                cutoff_time=datetime(2026, 5, 1, 12, 0),  # noqa: DTZ001
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
            evidence_type="identity_reuse",
            source="temporal_graph",
            observed_at=cutoff(),
            cutoff_time=cutoff(),
            subject_ids=("account-123", "device-456"),
            facts={"shared_account_count": 3, "identity_type": "device"},
        )
        self.assertEqual(evidence.facts["shared_account_count"], 3)
        with self.assertRaisesRegex(ValueError, "after cutoff_time"):
            _ = EvidenceItem(
                evidence_id="evidence-1",
                evidence_type="identity_reuse",
                source="temporal_graph",
                observed_at=datetime(2026, 5, 1, 12, 1, tzinfo=UTC),
                cutoff_time=cutoff(),
                subject_ids=("account-123",),
                facts={"shared_account_count": 3},
            )
        with self.assertRaisesRegex(ValueError, "evaluation-only label"):
            _ = EvidenceItem(
                evidence_id="evidence-1",
                evidence_type="identity_reuse",
                source="temporal_graph",
                observed_at=cutoff(),
                cutoff_time=cutoff(),
                subject_ids=("account-123",),
                facts={"nested": {"synthetic_labels": True}},
            )
        with self.assertRaises(ValueError):
            _ = EvidenceItem(
                evidence_id="",
                evidence_type="identity_reuse",
                source="temporal_graph",
                observed_at=cutoff(),
                cutoff_time=cutoff(),
                subject_ids=(),
                facts={},
            )

    def test_report_carries_input_action_and_only_evidence_referenced_claims(
        self,
    ) -> None:
        request = InvestigationRequest.from_policy_decision(
            decision(0.05), subject_id="account-123", cutoff_time=cutoff()
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
                RelatedEntity(entity_id="promotion-1", entity_type="promotion"),
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
