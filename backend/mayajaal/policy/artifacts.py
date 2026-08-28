"""Portable artifact writers for deterministic policy decisions."""

import json
from dataclasses import asdict
from pathlib import Path

from .models import PolicyDecision
from .provenance import PolicyModel, save_policy_model


def save_policy_artifacts(
    output_directory: Path, policy_model: PolicyModel, decision: PolicyDecision
) -> dict[str, Path]:
    """Persist the policy contract and one presentation-independent decision."""
    if decision.policy_id != policy_model.policy_id:
        raise ValueError("decision policy_id does not match policy model")
    if decision.probability_model_id != policy_model.probability_model_id:
        raise ValueError("decision probability_model_id does not match policy model")
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


def _decision_document(decision: PolicyDecision) -> dict[str, object]:
    """Convert the one Pydantic context nested in the dataclass decision."""
    document = asdict(decision)
    document["context"] = decision.context.model_dump(mode="json")
    return document
