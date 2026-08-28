"""Focused behavioural tests for the model-neutral expected-cost policy."""

import json
import unittest
from dataclasses import replace
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory

from mayajaal.calibration import (
    CalibrationConfig,
    ProbabilityEstimate,
    ProbabilityModel,
    SigmoidCalibrator,
    estimate_probability,
    probability_estimate_id,
    verify_probability_estimate,
)
from mayajaal.policy import (
    ActionCost,
    DecisionContext,
    PolicyAction,
    PolicyConfig,
    PolicyDecision,
    PolicyModel,
    ProbabilitySensitivityConfig,
    build_policy_model,
    decide,
    load_policy_decision,
    load_policy_model,
    odds_adjusted_probability,
    policy_provenance,
    save_policy_artifacts,
)


def probability_model() -> ProbabilityModel:
    """Return a minimal already-verified probability lineage fixture."""
    return ProbabilityModel(
        base_model_id="base-model-fixture",
        probability_model_id="probability-model-fixture",
        calibration_config=CalibrationConfig(
            minimum_positive_samples=1, minimum_negative_samples=1
        ),
        calibrator=SigmoidCalibrator(coefficient=1.0, intercept=0.0),
        frozen_provenance={"base_model_id": "base-model-fixture"},
    )


def policy(config: PolicyConfig | None = None):
    """Create one independently verified policy fixture."""
    return build_policy_model(probability_model(), config or PolicyConfig())


def estimate(probability: float, *, context_id: str | None = None):
    """Produce a verified estimate through the fixture's identity sigmoid."""
    return estimate_probability(
        probability_model(),
        log(probability / (1.0 - probability)),
        scoring_context_id=context_id,
    )


def make_decision(
    policy_model: PolicyModel,
    probability_estimate: ProbabilityEstimate,
    context: DecisionContext,
) -> PolicyDecision:
    """Exercise the production policy boundary with a verified parent model."""
    return decide(policy_model, probability_model(), probability_estimate, context)


def costs_by_action(decision: PolicyDecision) -> dict[PolicyAction, ActionCost]:
    """Make cost assertions readable without affecting the public contract."""
    return {cost.action: cost for cost in decision.expected_costs}


class CostSensitivePolicyTests(unittest.TestCase):
    def test_odds_sensitivity_is_relative_and_handles_probability_endpoints(
        self,
    ) -> None:
        self.assertEqual(odds_adjusted_probability(0.007, 1.0), 0.007)
        self.assertAlmostEqual(odds_adjusted_probability(0.007, 0.5), 0.0035123)
        self.assertAlmostEqual(odds_adjusted_probability(0.007, 2.0), 0.0139027)
        self.assertEqual(odds_adjusted_probability(0.0, 0.5), 0.0)
        self.assertEqual(odds_adjusted_probability(1.0, 2.0), 1.0)

    def test_odds_multiplier_validation_is_strict(self) -> None:
        with self.assertRaises(ValueError):
            _ = ProbabilitySensitivityConfig(optimistic_odds_multiplier=0.0)
        with self.assertRaises(ValueError):
            _ = ProbabilitySensitivityConfig(optimistic_odds_multiplier=1.01)
        with self.assertRaises(ValueError):
            _ = ProbabilitySensitivityConfig(stressed_odds_multiplier=0.99)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            _ = odds_adjusted_probability(0.1, 0.0)

    def test_probability_estimate_is_verified_and_has_deterministic_lineage(
        self,
    ) -> None:
        first = estimate_probability(
            probability_model(), 0.25, scoring_context_id="o-1"
        )
        second = estimate_probability(
            probability_model(), 0.25, scoring_context_id="o-1"
        )
        self.assertEqual(first, second)
        self.assertEqual(first.calibrated_probability, second.calibrated_probability)
        self.assertEqual(verify_probability_estimate(first, probability_model()), first)
        self.assertNotEqual(
            first.probability_estimate_id,
            estimate_probability(
                probability_model(), 0.26, scoring_context_id="o-1"
            ).probability_estimate_id,
        )
        changed_model = replace(probability_model(), probability_model_id="other-model")
        self.assertNotEqual(
            first.probability_estimate_id,
            estimate_probability(
                changed_model, 0.25, scoring_context_id="o-1"
            ).probability_estimate_id,
        )
        with self.assertRaisesRegex(ValueError, "semantics or calibrated probability"):
            _ = verify_probability_estimate(
                replace(first, calibrated_probability=0.9), probability_model()
            )
        with self.assertRaisesRegex(ValueError, "semantics or calibrated probability"):
            _ = verify_probability_estimate(
                replace(first, raw_model_score=0.9), probability_model()
            )
        with self.assertRaisesRegex(ValueError, "probability_model_id"):
            _ = verify_probability_estimate(first, changed_model)

    def test_decide_requires_a_probability_recomputed_by_verified_model(self) -> None:
        model = policy()
        probability_parent = probability_model()
        valid_estimate = estimate_probability(probability_parent, 0.25)
        context = DecisionContext(exposure_paise=10_000)
        self.assertIsInstance(
            decide(model, probability_parent, valid_estimate, context), PolicyDecision
        )

        fake_probability = 0.9
        fake_estimate = replace(
            valid_estimate,
            calibrated_probability=fake_probability,
            probability_estimate_id=probability_estimate_id(
                base_model_id=valid_estimate.base_model_id,
                probability_model_id=valid_estimate.probability_model_id,
                probability_estimate_contract_version=1,
                raw_model_score=valid_estimate.raw_model_score,
                calibrated_probability=fake_probability,
                scoring_context_id=valid_estimate.scoring_context_id,
            ),
        )
        with self.assertRaisesRegex(ValueError, "semantics or calibrated probability"):
            _ = decide(model, probability_parent, fake_estimate, context)

        tampered_raw_score = 0.9
        tampered_raw_estimate = replace(
            valid_estimate,
            raw_model_score=tampered_raw_score,
            probability_estimate_id=probability_estimate_id(
                base_model_id=valid_estimate.base_model_id,
                probability_model_id=valid_estimate.probability_model_id,
                probability_estimate_contract_version=1,
                raw_model_score=tampered_raw_score,
                calibrated_probability=valid_estimate.calibrated_probability,
                scoring_context_id=valid_estimate.scoring_context_id,
            ),
        )
        with self.assertRaisesRegex(ValueError, "semantics or calibrated probability"):
            _ = decide(model, probability_parent, tampered_raw_estimate, context)

        wrong_probability_parent = replace(
            probability_parent,
            calibrator=SigmoidCalibrator(coefficient=2.0, intercept=0.0),
        )
        with self.assertRaisesRegex(ValueError, "semantics or calibrated probability"):
            _ = decide(model, wrong_probability_parent, valid_estimate, context)
        with self.assertRaisesRegex(ValueError, "probability_model_id"):
            _ = decide(
                model,
                replace(probability_parent, probability_model_id="wrong-model"),
                valid_estimate,
                context,
            )
        with self.assertRaisesRegex(ValueError, "policy model"):
            _ = decide(
                replace(model, policy_id="wrong-policy"),
                probability_parent,
                valid_estimate,
                context,
            )

    def test_decide_binds_scoring_and_decision_contexts_when_both_are_present(
        self,
    ) -> None:
        model = policy()
        probability_parent = probability_model()
        matched_estimate = estimate_probability(
            probability_parent, 0.25, scoring_context_id="order-123"
        )
        matching_context = DecisionContext(
            exposure_paise=10_000, context_id="order-123"
        )
        matching = decide(model, probability_parent, matched_estimate, matching_context)
        self.assertEqual(matching.scoring_context_id, matching.context.context_id)
        self.assertEqual(
            matching.decision_id,
            "82e7337ee08e00cabf34f60fd16a0a56948effb14409498a10c52b63d6b6cb92",
        )

        with self.assertRaisesRegex(ValueError, "scoring_context_id.*context_id"):
            _ = decide(
                model,
                probability_parent,
                matched_estimate,
                DecisionContext(exposure_paise=10_000, context_id="order-456"),
            )

        self.assertIsInstance(
            decide(
                model,
                probability_parent,
                estimate_probability(probability_parent, 0.25),
                DecisionContext(exposure_paise=10_000),
            ),
            PolicyDecision,
        )
        self.assertIsInstance(
            decide(
                model,
                probability_parent,
                matched_estimate,
                DecisionContext(exposure_paise=10_000),
            ),
            PolicyDecision,
        )
        self.assertIsInstance(
            decide(
                model,
                probability_parent,
                estimate_probability(probability_parent, 0.25),
                DecisionContext(exposure_paise=10_000, context_id="order-123"),
            ),
            PolicyDecision,
        )

        with TemporaryDirectory() as directory:
            artifacts = save_policy_artifacts(
                Path(directory),
                model,
                probability_parent,
                matched_estimate,
                matching_context,
                matching,
            )
            self.assertEqual(
                load_policy_decision(artifacts["decision"], model, probability_parent),
                matching,
            )

    def test_low_intermediate_and_high_risk_choose_allow_review_block(self) -> None:
        model = policy()
        context = DecisionContext(exposure_paise=250_000)
        self.assertIs(
            make_decision(model, estimate(0.001), context).chosen_action,
            PolicyAction.ALLOW,
        )
        self.assertIs(
            make_decision(model, estimate(0.05), context).chosen_action,
            PolicyAction.REVIEW,
        )
        self.assertIs(
            make_decision(model, estimate(0.90), context).chosen_action,
            PolicyAction.BLOCK,
        )

    def test_exposure_amount_changes_the_economic_action(self) -> None:
        model = policy()
        self.assertIs(
            make_decision(
                model, estimate(0.06), DecisionContext(exposure_paise=1_000)
            ).chosen_action,
            PolicyAction.ALLOW,
        )
        self.assertIs(
            make_decision(
                model, estimate(0.06), DecisionContext(exposure_paise=250_000)
            ).chosen_action,
            PolicyAction.REVIEW,
        )

    def test_legitimate_blocking_cost_prevents_reckless_blocking(self) -> None:
        model = policy(
            PolicyConfig(
                block_legitimate_margin_loss_fraction=1.0,
                block_legitimate_friction_cost_paise=500_000,
            )
        )
        decision = make_decision(
            model, estimate(0.90), DecisionContext(exposure_paise=250_000)
        )
        self.assertIs(decision.chosen_action, PolicyAction.REVIEW)

    def test_expected_costs_include_residual_review_loss_exactly(self) -> None:
        model = policy()
        decision = make_decision(
            model, estimate(0.10), DecisionContext(exposure_paise=10_000)
        )
        costs = costs_by_action(decision)
        review = costs[PolicyAction.REVIEW]
        self.assertEqual(review.fraud_cost_paise, 3_500.0)
        self.assertEqual(review.legitimate_cost_paise, 2_000.0)
        self.assertEqual(review.expected_cost_paise, 2_150.0)
        self.assertAlmostEqual(review.delta_from_chosen_paise, 1_150.0)

    def test_exact_tie_uses_configured_stable_action_order(self) -> None:
        zero_costs = PolicyConfig(
            review_operational_cost_paise=0,
            review_legitimate_friction_cost_paise=0,
            review_fraud_residual_loss_fraction=0.0,
            block_operational_cost_paise=0,
            block_legitimate_margin_loss_fraction=0.0,
            block_legitimate_friction_cost_paise=0,
            block_fraud_residual_loss_fraction=0.0,
            allow_fraud_exposure_loss_fraction=0.0,
        )
        self.assertIs(
            make_decision(
                policy(zero_costs), estimate(0.5), DecisionContext(exposure_paise=99)
            ).chosen_action,
            PolicyAction.ALLOW,
        )
        block_first = zero_costs.model_copy(
            update={
                "tie_break_order": (
                    PolicyAction.BLOCK,
                    PolicyAction.REVIEW,
                    PolicyAction.ALLOW,
                )
            }
        )
        self.assertIs(
            make_decision(
                policy(block_first), estimate(0.5), DecisionContext(exposure_paise=99)
            ).chosen_action,
            PolicyAction.BLOCK,
        )

    def test_probability_and_cost_validation_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "within \\[0, 1\\]"):
            _ = odds_adjusted_probability(1.01, 1.0)
        with self.assertRaises(ValueError):
            _ = DecisionContext(exposure_paise=-1)
        with self.assertRaises(ValueError):
            _ = PolicyConfig(review_operational_cost_paise=-1)
        with self.assertRaises(ValueError):
            _ = PolicyConfig(tie_break_order=(PolicyAction.ALLOW, PolicyAction.ALLOW))

    def test_sensitivity_scenarios_report_stability_and_instability(self) -> None:
        model = policy()
        unstable = make_decision(
            model, estimate(0.01), DecisionContext(exposure_paise=250_000)
        )
        self.assertFalse(unstable.decision_is_stable_across_scenarios)
        self.assertEqual(
            tuple(item.scenario for item in unstable.scenarios),
            ("optimistic", "stressed"),
        )
        self.assertTrue(
            all(
                0.0 <= item.assumed_fraud_probability <= 1.0
                for item in unstable.scenarios
            )
        )
        stable = make_decision(
            model, estimate(0.90), DecisionContext(exposure_paise=250_000)
        )
        self.assertTrue(stable.decision_is_stable_across_scenarios)

    def test_policy_id_binds_semantics_but_not_paths_or_formatting(self) -> None:
        first = policy()
        equivalent = build_policy_model(
            probability_model(),
            PolicyConfig.model_validate(
                {
                    "block_fraud_residual_loss_fraction": 0.01,
                    "block_legitimate_friction_cost_paise": 1000,
                    "block_legitimate_margin_loss_fraction": 0.10,
                    "block_operational_cost_paise": 200,
                    "review_fraud_residual_loss_fraction": 0.20,
                    "review_legitimate_friction_cost_paise": 500,
                    "review_operational_cost_paise": 1500,
                    "allow_fraud_exposure_loss_fraction": 1.0,
                    "allow_legitimate_cost_paise": 0,
                    "allow_operational_cost_paise": 0,
                    "tie_break_order": ["ALLOW", "REVIEW", "BLOCK"],
                    "sensitivity": {
                        "stressed_odds_multiplier": 2.0,
                        "optimistic_odds_multiplier": 0.5,
                    },
                }
            ),
        )
        self.assertEqual(first.policy_id, equivalent.policy_id)
        changed_config = first.config.model_copy(
            update={"review_operational_cost_paise": 1_501}
        )
        self.assertNotEqual(first.policy_id, policy(changed_config).policy_id)
        changed_probability = replace(
            probability_model(), probability_model_id="probability-model-other"
        )
        self.assertNotEqual(
            first.policy_id,
            build_policy_model(changed_probability, first.config).policy_id,
        )

    def test_policy_artifact_rejects_tampering_and_lineage_mismatch(self) -> None:
        model = policy()
        decision = make_decision(
            model, estimate(0.10), DecisionContext(exposure_paise=10_000)
        )
        with TemporaryDirectory() as directory:
            artifacts = save_policy_artifacts(
                Path(directory),
                model,
                probability_model(),
                estimate(0.10),
                decision.context,
                decision,
            )
            loaded = load_policy_model(
                artifacts["policy_model"],
                probability_model(),
                expected_policy_id=model.policy_id,
            )
            self.assertEqual(loaded, model)
            saved_decision = json.loads(
                artifacts["decision"].read_text(encoding="utf-8")
            )
            self.assertEqual(saved_decision["policy_id"], model.policy_id)
            self.assertEqual(
                saved_decision["probability_model_id"],
                probability_model().probability_model_id,
            )
            self.assertEqual(
                load_policy_decision(artifacts["decision"], model, probability_model()),
                decision,
            )
            document = json.loads(artifacts["policy_model"].read_text(encoding="utf-8"))
            document["provenance"]["policy_config"]["review_operational_cost_paise"] = 9
            artifacts["policy_model"].write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash or semantics"):
                _ = load_policy_model(artifacts["policy_model"], probability_model())

            artifacts = save_policy_artifacts(
                Path(directory),
                model,
                probability_model(),
                estimate(0.10),
                decision.context,
                decision,
            )
            wrong_probability = replace(
                probability_model(), probability_model_id="wrong-probability-model"
            )
            with self.assertRaisesRegex(ValueError, "probability_model_id"):
                _ = load_policy_model(artifacts["policy_model"], wrong_probability)

    def test_decision_id_and_artifact_integrity_cover_semantic_fields(self) -> None:
        model = policy()
        first_estimate = estimate(0.10, context_id="order-1")
        first = make_decision(
            model, first_estimate, DecisionContext(exposure_paise=10_000)
        )
        same = make_decision(
            model, first_estimate, DecisionContext(exposure_paise=10_000)
        )
        self.assertEqual(first.decision_id, same.decision_id)
        self.assertNotEqual(
            first.decision_id,
            make_decision(
                model, first_estimate, DecisionContext(exposure_paise=10_001)
            ).decision_id,
        )
        self.assertNotEqual(
            first.decision_id,
            make_decision(
                model,
                estimate(0.11, context_id="order-1"),
                DecisionContext(exposure_paise=10_000),
            ).decision_id,
        )
        altered_policy = policy(PolicyConfig(review_operational_cost_paise=1_501))
        self.assertNotEqual(
            first.decision_id,
            make_decision(
                altered_policy, first_estimate, DecisionContext(exposure_paise=10_000)
            ).decision_id,
        )
        with TemporaryDirectory() as directory:
            artifacts = save_policy_artifacts(
                Path(directory),
                model,
                probability_model(),
                first_estimate,
                first.context,
                first,
            )
            self.assertEqual(
                load_policy_decision(artifacts["decision"], model, probability_model()),
                first,
            )
            original = json.loads(artifacts["decision"].read_text(encoding="utf-8"))
            for field, value in (
                ("chosen_action", "BLOCK"),
                ("calibrated_fraud_probability", 0.9),
                ("raw_model_score", 0.9),
                ("probability_estimate_id", "tampered-estimate"),
                ("probability_model_id", "tampered-model"),
                ("policy_id", "tampered-policy"),
                ("decision_margin_paise", 1.0),
                ("decision_is_stable_across_scenarios", False),
                ("decision_id", "tampered"),
            ):
                document = json.loads(json.dumps(original))
                document[field] = value
                artifacts["decision"].write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "decision|probability estimate"
                ):
                    _ = load_policy_decision(
                        artifacts["decision"], model, probability_model()
                    )
            document = json.loads(json.dumps(original))
            document["context"]["exposure_paise"] = 999
            artifacts["decision"].write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decision"):
                _ = load_policy_decision(
                    artifacts["decision"], model, probability_model()
                )
            document = json.loads(json.dumps(original))
            document["expected_costs"][0]["expected_cost_paise"] = 999.0
            artifacts["decision"].write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decision"):
                _ = load_policy_decision(
                    artifacts["decision"], model, probability_model()
                )
            document = json.loads(json.dumps(original))
            document["scenarios"][0]["chosen_action"] = "BLOCK"
            artifacts["decision"].write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decision"):
                _ = load_policy_decision(
                    artifacts["decision"], model, probability_model()
                )

    def test_save_rejects_a_caller_altered_decision(self) -> None:
        model = policy()
        probability_parent = probability_model()
        probability_estimate = estimate_probability(probability_parent, 0.25)
        context = DecisionContext(exposure_paise=10_000)
        decision = decide(model, probability_parent, probability_estimate, context)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "verified reconstruction"):
                _ = save_policy_artifacts(
                    Path(directory),
                    model,
                    probability_parent,
                    probability_estimate,
                    context,
                    replace(decision, chosen_action=PolicyAction.ALLOW),
                )
            artifacts = save_policy_artifacts(
                Path(directory),
                model,
                probability_parent,
                probability_estimate,
                context,
                decision,
            )
            self.assertEqual(
                load_policy_decision(artifacts["decision"], model, probability_parent),
                decision,
            )

    def test_policy_never_imports_synthetic_labels_or_catboost(self) -> None:
        package_directory = Path(__file__).parents[1] / "mayajaal" / "policy"
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in package_directory.glob("*.py")
        )
        self.assertNotIn("synthetic_labels", source)
        self.assertNotIn("catboost", source.casefold())

    def test_sensitivity_assumptions_are_semantic_policy_inputs(self) -> None:
        first = policy()
        changed = first.config.model_copy(
            update={
                "sensitivity": ProbabilitySensitivityConfig(
                    optimistic_odds_multiplier=0.25,
                    stressed_odds_multiplier=4.0,
                )
            }
        )
        self.assertNotEqual(first.policy_id, policy(changed).policy_id)
        self.assertNotEqual(
            policy_provenance(probability_model(), first.config),
            policy_provenance(probability_model(), changed),
        )


if __name__ == "__main__":
    _ = unittest.main()
