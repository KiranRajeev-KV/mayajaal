"""Portable artifact writers and verifiers for deterministic policy decisions."""

import json
from pathlib import Path
from typing import cast

from mayajaal.calibration import (
    ProbabilityModel,
    estimate_probability,
    verify_probability_estimate,
)

from .models import DecisionContext, PolicyDecision
from .provenance import (
    DECISION_PROVENANCE_CONTRACT_VERSION,
    PolicyModel,
    save_policy_model,
)
from .service import decide


def save_policy_artifacts(
    output_directory: Path, policy_model: PolicyModel, decision: PolicyDecision
) -> dict[str, Path]:
    """Persist a policy contract and its deterministic decision result."""
    _verify_decision_lineage(decision, policy_model)
    output_directory.mkdir(parents=True, exist_ok=True)
    policy_path = save_policy_model(
        output_directory / "policy_model.json", policy_model
    )
    decision_path = output_directory / "policy_decision.json"
    decision_path.write_text(
        json.dumps(_decision_document(decision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"policy_model": policy_path, "decision": decision_path}


def load_policy_decision(
    decision_path: Path,
    policy_model: PolicyModel,
    probability_model: ProbabilityModel,
    *,
    expected_decision_id: str | None = None,
) -> PolicyDecision:
    """Reconstruct a decision from trusted parents and reject every mismatch."""
    try:
        raw_document: object = json.loads(decision_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"missing policy decision artifact: {decision_path}"
        ) from error
    if not isinstance(raw_document, dict):
        raise ValueError("invalid policy decision artifact")
    document = cast(dict[str, object], raw_document)
    required = {
        "decision_contract_version",
        "policy_id",
        "base_model_id",
        "probability_model_id",
        "probability_estimate_id",
        "decision_id",
        "calibrated_fraud_probability",
        "context",
        "chosen_action",
        "expected_costs",
        "decision_margin_paise",
        "scenarios",
        "decision_is_stable_across_scenarios",
        "raw_model_score",
        "scoring_context_id",
    }
    if set(document) != required:
        raise ValueError("policy decision artifact has unsupported or missing fields")
    if document["decision_contract_version"] != DECISION_PROVENANCE_CONTRACT_VERSION:
        raise ValueError("unsupported policy decision provenance contract version")
    if document["base_model_id"] != probability_model.base_model_id:
        raise ValueError("decision base_model_id does not match verified lineage")
    if document["probability_model_id"] != probability_model.probability_model_id:
        raise ValueError(
            "decision probability_model_id does not match verified lineage"
        )
    if document["policy_id"] != policy_model.policy_id:
        raise ValueError("decision policy_id does not match verified policy")
    context = _context(document["context"])
    raw_score = _finite_float(document["raw_model_score"], "raw_model_score")
    scoring_context_id = document["scoring_context_id"]
    if scoring_context_id is not None and not isinstance(scoring_context_id, str):
        raise ValueError("invalid scoring_context_id in policy decision")
    estimate = estimate_probability(
        probability_model, raw_score, scoring_context_id=scoring_context_id
    )
    verify_probability_estimate(estimate, probability_model)
    if estimate.probability_estimate_id != document["probability_estimate_id"]:
        raise ValueError(
            "decision probability_estimate_id does not match verified estimate"
        )
    reconstructed = decide(policy_model, estimate, context)
    if document != _decision_document(reconstructed):
        raise ValueError("policy decision semantics or integrity identifier mismatch")
    if (
        expected_decision_id is not None
        and reconstructed.decision_id != expected_decision_id
    ):
        raise ValueError("decision_id does not match expected decision lineage")
    return reconstructed


def _verify_decision_lineage(
    decision: PolicyDecision, policy_model: PolicyModel
) -> None:
    if decision.policy_id != policy_model.policy_id:
        raise ValueError("decision policy_id does not match policy model")
    if decision.base_model_id != policy_model.base_model_id:
        raise ValueError("decision base_model_id does not match policy model")
    if decision.probability_model_id != policy_model.probability_model_id:
        raise ValueError("decision probability_model_id does not match policy model")
    if not decision.probability_estimate_id or not decision.decision_id:
        raise ValueError(
            "decision must include estimate and decision integrity identifiers"
        )


def _context(value: object) -> DecisionContext:
    try:
        return DecisionContext.model_validate(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid decision context in policy decision") from error


def _finite_float(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"invalid {name} in policy decision")
    return float(value)


def _decision_document(decision: PolicyDecision) -> dict[str, object]:
    """Serialize exactly the semantics needed to reconstruct and verify a decision."""
    return {
        "decision_contract_version": DECISION_PROVENANCE_CONTRACT_VERSION,
        "policy_id": decision.policy_id,
        "base_model_id": decision.base_model_id,
        "probability_model_id": decision.probability_model_id,
        "probability_estimate_id": decision.probability_estimate_id,
        "decision_id": decision.decision_id,
        "raw_model_score": decision.raw_model_score,
        "calibrated_fraud_probability": decision.calibrated_fraud_probability,
        "scoring_context_id": decision.scoring_context_id,
        "context": decision.context.model_dump(mode="json"),
        "chosen_action": decision.chosen_action.value,
        "expected_costs": [_cost_document(cost) for cost in decision.expected_costs],
        "decision_margin_paise": decision.decision_margin_paise,
        "scenarios": [_scenario_document(scenario) for scenario in decision.scenarios],
        "decision_is_stable_across_scenarios": decision.decision_is_stable_across_scenarios,
    }


def _cost_document(cost: object) -> dict[str, object]:
    from .models import ActionCost

    if not isinstance(cost, ActionCost):
        raise TypeError("invalid action cost")
    return {
        "action": cost.action.value,
        "fraud_cost_paise": cost.fraud_cost_paise,
        "legitimate_cost_paise": cost.legitimate_cost_paise,
        "expected_cost_paise": cost.expected_cost_paise,
        "delta_from_chosen_paise": cost.delta_from_chosen_paise,
    }


def _scenario_document(scenario: object) -> dict[str, object]:
    from .models import ScenarioDecision

    if not isinstance(scenario, ScenarioDecision):
        raise TypeError("invalid scenario decision")
    return {
        "scenario": scenario.scenario,
        "odds_multiplier": scenario.odds_multiplier,
        "assumed_fraud_probability": scenario.assumed_fraud_probability,
        "chosen_action": scenario.chosen_action.value,
        "expected_costs": [_cost_document(cost) for cost in scenario.expected_costs],
        "decision_margin_paise": scenario.decision_margin_paise,
    }
