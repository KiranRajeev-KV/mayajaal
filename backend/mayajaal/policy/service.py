"""Pure deterministic expected-cost decisions over calibrated probabilities."""

from .models import (
    ActionCost,
    DecisionContext,
    PolicyAction,
    PolicyDecision,
    ScenarioDecision,
    validate_probability,
)
from .provenance import PolicyModel


def decide(
    policy_model: PolicyModel,
    calibrated_fraud_probability: float,
    context: DecisionContext,
) -> PolicyDecision:
    """Choose the minimum conditional expected-cost action deterministically.

    This is intentionally a business-policy computation, not a score threshold:
    a different exposure can change the minimum-cost action at the same
    calibrated probability.
    """
    probability = validate_probability(calibrated_fraud_probability)
    expected_costs, chosen_action, margin = _decision_at_probability(
        policy_model, probability, context
    )
    sensitivity = policy_model.config.sensitivity
    scenarios = tuple(
        _scenario_decision(
            policy_model,
            context,
            name,
            shift,
            probability,
        )
        for name, shift in (
            ("optimistic", sensitivity.optimistic_probability_shift),
            ("stressed", sensitivity.stressed_probability_shift),
        )
    )
    return PolicyDecision(
        policy_id=policy_model.policy_id,
        base_model_id=policy_model.base_model_id,
        probability_model_id=policy_model.probability_model_id,
        calibrated_fraud_probability=probability,
        context=context,
        chosen_action=chosen_action,
        expected_costs=expected_costs,
        decision_margin_paise=margin,
        scenarios=scenarios,
        decision_is_stable_across_scenarios=all(
            scenario.chosen_action is chosen_action for scenario in scenarios
        ),
    )


def _scenario_decision(
    policy_model: PolicyModel,
    context: DecisionContext,
    name: str,
    shift: float,
    base_probability: float,
) -> ScenarioDecision:
    assumed_probability = min(max(base_probability + shift, 0.0), 1.0)
    expected_costs, chosen_action, margin = _decision_at_probability(
        policy_model, assumed_probability, context
    )
    return ScenarioDecision(
        scenario=name,
        probability_shift=shift,
        assumed_fraud_probability=assumed_probability,
        chosen_action=chosen_action,
        expected_costs=expected_costs,
        decision_margin_paise=margin,
    )


def _decision_at_probability(
    policy_model: PolicyModel,
    probability: float,
    context: DecisionContext,
) -> tuple[tuple[ActionCost, ...], PolicyAction, float]:
    conditional_costs = _conditional_costs(policy_model, context)
    raw_costs = {
        action: probability * fraud_cost + (1.0 - probability) * legitimate_cost
        for action, (fraud_cost, legitimate_cost) in conditional_costs.items()
    }
    action_rank = {
        action: index
        for index, action in enumerate(policy_model.config.tie_break_order)
    }
    chosen_action = min(
        raw_costs, key=lambda action: (raw_costs[action], action_rank[action])
    )
    chosen_cost = raw_costs[chosen_action]
    ordered = tuple(
        ActionCost(
            action=action,
            fraud_cost_paise=conditional_costs[action][0],
            legitimate_cost_paise=conditional_costs[action][1],
            expected_cost_paise=raw_costs[action],
            delta_from_chosen_paise=raw_costs[action] - chosen_cost,
        )
        for action in PolicyAction
    )
    remaining = sorted(
        cost for action, cost in raw_costs.items() if action is not chosen_action
    )
    margin = remaining[0] - chosen_cost
    return ordered, chosen_action, margin


def _conditional_costs(
    policy_model: PolicyModel, context: DecisionContext
) -> dict[PolicyAction, tuple[float, float]]:
    config = policy_model.config
    exposure = float(context.exposure_paise)
    return {
        PolicyAction.ALLOW: (
            float(config.allow_operational_cost_paise)
            + exposure * config.allow_fraud_exposure_loss_fraction,
            float(
                config.allow_operational_cost_paise + config.allow_legitimate_cost_paise
            ),
        ),
        PolicyAction.REVIEW: (
            float(config.review_operational_cost_paise)
            + exposure * config.review_fraud_residual_loss_fraction,
            float(
                config.review_operational_cost_paise
                + config.review_legitimate_friction_cost_paise
            ),
        ),
        PolicyAction.BLOCK: (
            float(config.block_operational_cost_paise)
            + exposure * config.block_fraud_residual_loss_fraction,
            float(
                config.block_operational_cost_paise
                + config.block_legitimate_friction_cost_paise
            )
            + exposure * config.block_legitimate_margin_loss_fraction,
        ),
    }
