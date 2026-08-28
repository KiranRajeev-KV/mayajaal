"""Pure deterministic expected-cost decisions over verified probabilities."""

from math import isfinite

from mayajaal.calibration import (
    ProbabilityEstimate,
    ProbabilityModel,
    verify_probability_estimate,
)

from .models import (
    ActionCost,
    DecisionContext,
    PolicyAction,
    PolicyDecision,
    ScenarioDecision,
)
from .provenance import (
    PolicyModel,
    build_policy_model,
    decision_id,
    decision_semantics,
)


def decide(
    policy_model: PolicyModel,
    probability_model: ProbabilityModel,
    probability_estimate: ProbabilityEstimate,
    context: DecisionContext,
) -> PolicyDecision:
    """Choose the minimum conditional expected-cost action deterministically.

    This is intentionally a business-policy computation, not a score threshold:
    a different exposure can change the minimum-cost action at the same
    calibrated probability. The estimate is a score-derived child of the
    calibrated probability model, never an unbound caller-provided float.
    """
    _verify_probability_contract(policy_model, probability_model, probability_estimate)
    _verify_context_binding(probability_estimate, context)
    probability = probability_estimate.calibrated_probability
    expected_costs, chosen_action, margin = _decision_at_probability(
        policy_model, probability, context
    )
    sensitivity = policy_model.config.sensitivity
    scenarios = tuple(
        _scenario_decision(policy_model, context, name, odds_multiplier, probability)
        for name, odds_multiplier in (
            ("optimistic", sensitivity.optimistic_odds_multiplier),
            ("stressed", sensitivity.stressed_odds_multiplier),
        )
    )
    stable = all(scenario.chosen_action is chosen_action for scenario in scenarios)
    semantics = decision_semantics(
        base_model_id=policy_model.base_model_id,
        probability_model_id=policy_model.probability_model_id,
        probability_estimate_id=probability_estimate.probability_estimate_id,
        policy_id=policy_model.policy_id,
        calibrated_fraud_probability=probability,
        context=context,
        chosen_action=chosen_action.value,
        expected_costs=expected_costs,
        decision_margin_paise=margin,
        scenarios=scenarios,
        decision_is_stable_across_scenarios=stable,
    )
    return PolicyDecision(
        policy_id=policy_model.policy_id,
        base_model_id=policy_model.base_model_id,
        probability_model_id=policy_model.probability_model_id,
        probability_estimate_id=probability_estimate.probability_estimate_id,
        decision_id=decision_id(**semantics),
        raw_model_score=probability_estimate.raw_model_score,
        calibrated_fraud_probability=probability,
        scoring_context_id=probability_estimate.scoring_context_id,
        context=context,
        chosen_action=chosen_action,
        expected_costs=expected_costs,
        decision_margin_paise=margin,
        scenarios=scenarios,
        decision_is_stable_across_scenarios=stable,
    )


def odds_adjusted_probability(probability: float, odds_multiplier: float) -> float:
    """Apply a relative odds assumption without treating it as recalibration."""
    if not isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError(
            "calibrated fraud probability must be finite and within [0, 1]"
        )
    if not isfinite(odds_multiplier) or odds_multiplier <= 0.0:
        raise ValueError("odds multiplier must be finite and greater than zero")
    if probability == 0.0 or probability == 1.0:
        return probability
    odds = probability / (1.0 - probability)
    scenario_odds = odds * odds_multiplier
    return scenario_odds / (1.0 + scenario_odds)


def _verify_probability_contract(
    policy_model: PolicyModel,
    probability_model: ProbabilityModel,
    estimate: ProbabilityEstimate,
) -> None:
    """Verify score-derived probability and its binding to policy lineage."""
    if probability_model.base_model_id != policy_model.base_model_id:
        raise ValueError("probability model base_model_id does not match policy")
    if probability_model.probability_model_id != policy_model.probability_model_id:
        raise ValueError("probability model probability_model_id does not match policy")
    expected_policy_model = build_policy_model(probability_model, policy_model.config)
    if policy_model != expected_policy_model:
        raise ValueError("policy model does not match verified probability lineage")
    _ = verify_probability_estimate(estimate, probability_model)


def _verify_context_binding(
    estimate: ProbabilityEstimate,
    context: DecisionContext,
) -> None:
    """Prevent a context-bound score from being applied to another context."""
    if (
        estimate.scoring_context_id is not None
        and context.context_id is not None
        and estimate.scoring_context_id != context.context_id
    ):
        raise ValueError(
            "probability estimate scoring_context_id does not match decision context_id"
        )


def _scenario_decision(
    policy_model: PolicyModel,
    context: DecisionContext,
    name: str,
    odds_multiplier: float,
    base_probability: float,
) -> ScenarioDecision:
    assumed_probability = odds_adjusted_probability(base_probability, odds_multiplier)
    expected_costs, chosen_action, margin = _decision_at_probability(
        policy_model, assumed_probability, context
    )
    return ScenarioDecision(
        scenario=name,
        odds_multiplier=odds_multiplier,
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
    return ordered, chosen_action, remaining[0] - chosen_cost


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
