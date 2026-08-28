"""Focused behavioural tests for the model-neutral expected-cost policy."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from mayajaal.calibration import (
    CalibrationConfig,
    ProbabilityModel,
    SigmoidCalibrator,
)
from mayajaal.policy import (
    ActionCost,
    DecisionContext,
    PolicyAction,
    PolicyConfig,
    PolicyDecision,
    ProbabilitySensitivityConfig,
    build_policy_model,
    decide,
    load_policy_model,
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


def costs_by_action(decision: PolicyDecision) -> dict[PolicyAction, ActionCost]:
    """Make cost assertions readable without affecting the public contract."""
    return {cost.action: cost for cost in decision.expected_costs}


class CostSensitivePolicyTests(unittest.TestCase):
    def test_low_intermediate_and_high_risk_choose_allow_review_block(self) -> None:
        model = policy()
        context = DecisionContext(exposure_paise=250_000)
        self.assertIs(decide(model, 0.001, context).chosen_action, PolicyAction.ALLOW)
        self.assertIs(decide(model, 0.05, context).chosen_action, PolicyAction.REVIEW)
        self.assertIs(decide(model, 0.90, context).chosen_action, PolicyAction.BLOCK)

    def test_exposure_amount_changes_the_economic_action(self) -> None:
        model = policy()
        self.assertIs(
            decide(model, 0.06, DecisionContext(exposure_paise=1_000)).chosen_action,
            PolicyAction.ALLOW,
        )
        self.assertIs(
            decide(model, 0.06, DecisionContext(exposure_paise=250_000)).chosen_action,
            PolicyAction.REVIEW,
        )

    def test_legitimate_blocking_cost_prevents_reckless_blocking(self) -> None:
        model = policy(
            PolicyConfig(
                block_legitimate_margin_loss_fraction=1.0,
                block_legitimate_friction_cost_paise=500_000,
            )
        )
        decision = decide(model, 0.90, DecisionContext(exposure_paise=250_000))
        self.assertIs(decision.chosen_action, PolicyAction.REVIEW)

    def test_expected_costs_include_residual_review_loss_exactly(self) -> None:
        model = policy()
        decision = decide(model, 0.10, DecisionContext(exposure_paise=10_000))
        costs = costs_by_action(decision)
        review = costs[PolicyAction.REVIEW]
        self.assertEqual(review.fraud_cost_paise, 3_500.0)
        self.assertEqual(review.legitimate_cost_paise, 2_000.0)
        self.assertEqual(review.expected_cost_paise, 2_150.0)
        self.assertEqual(review.delta_from_chosen_paise, 1_150.0)

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
            decide(
                policy(zero_costs), 0.5, DecisionContext(exposure_paise=99)
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
            decide(
                policy(block_first), 0.5, DecisionContext(exposure_paise=99)
            ).chosen_action,
            PolicyAction.BLOCK,
        )

    def test_probability_and_cost_validation_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "within \\[0, 1\\]"):
            _ = decide(policy(), 1.01, DecisionContext(exposure_paise=1))
        with self.assertRaises(ValueError):
            _ = DecisionContext(exposure_paise=-1)
        with self.assertRaises(ValueError):
            _ = PolicyConfig(review_operational_cost_paise=-1)
        with self.assertRaises(ValueError):
            _ = PolicyConfig(tie_break_order=(PolicyAction.ALLOW, PolicyAction.ALLOW))

    def test_sensitivity_scenarios_report_stability_and_instability(self) -> None:
        model = policy()
        unstable = decide(model, 0.05, DecisionContext(exposure_paise=250_000))
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
        stable = decide(model, 0.90, DecisionContext(exposure_paise=250_000))
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
                        "stressed_probability_shift": 0.05,
                        "optimistic_probability_shift": -0.05,
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
        decision = decide(model, 0.10, DecisionContext(exposure_paise=10_000))
        with TemporaryDirectory() as directory:
            artifacts = save_policy_artifacts(Path(directory), model, decision)
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
            document = json.loads(artifacts["policy_model"].read_text(encoding="utf-8"))
            document["provenance"]["policy_config"]["review_operational_cost_paise"] = 9
            artifacts["policy_model"].write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash or semantics"):
                _ = load_policy_model(artifacts["policy_model"], probability_model())

            artifacts = save_policy_artifacts(Path(directory), model, decision)
            wrong_probability = replace(
                probability_model(), probability_model_id="wrong-probability-model"
            )
            with self.assertRaisesRegex(ValueError, "probability_model_id"):
                _ = load_policy_model(artifacts["policy_model"], wrong_probability)

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
                    optimistic_probability_shift=-0.10,
                    stressed_probability_shift=0.10,
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
