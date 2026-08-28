"""Deterministic lineage contracts for a calibrated probability policy."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mayajaal.calibration import ProbabilityModel

from .models import PolicyConfig

POLICY_PROVENANCE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class PolicyModel:
    """A verified merchant policy bound to one calibrated probability model."""

    base_model_id: str
    probability_model_id: str
    policy_id: str
    config: PolicyConfig


def canonical_hash(value: object) -> str:
    """Return a path- and presentation-independent SHA-256 JSON hash."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def policy_semantics(
    *,
    probability_model_id: str,
    policy_contract_version: int,
    policy_config: object,
) -> dict[str, object]:
    """Return precisely the inputs that define the policy's semantics."""
    return {
        "policy_contract_version": policy_contract_version,
        "probability_model_id": probability_model_id,
        "policy_config": policy_config,
    }


def policy_id(
    *,
    probability_model_id: str,
    policy_contract_version: int,
    policy_config: object,
) -> str:
    """Hash one complete probability-bound policy configuration."""
    return canonical_hash(
        policy_semantics(
            probability_model_id=probability_model_id,
            policy_contract_version=policy_contract_version,
            policy_config=policy_config,
        )
    )


def policy_provenance(
    probability_model: ProbabilityModel, config: PolicyConfig
) -> dict[str, object]:
    """Build the persisted lineage contract for one policy model."""
    semantics = policy_semantics(
        probability_model_id=probability_model.probability_model_id,
        policy_contract_version=POLICY_PROVENANCE_CONTRACT_VERSION,
        policy_config=config.model_dump(mode="json"),
    )
    return {
        **semantics,
        "policy_id": policy_id(
            probability_model_id=probability_model.probability_model_id,
            policy_contract_version=POLICY_PROVENANCE_CONTRACT_VERSION,
            policy_config=config.model_dump(mode="json"),
        ),
        "base_model_id": probability_model.base_model_id,
    }


def build_policy_model(
    probability_model: ProbabilityModel, config: PolicyConfig
) -> PolicyModel:
    """Bind validated economics to an already verified probability-model lineage."""
    provenance = policy_provenance(probability_model, config)
    return PolicyModel(
        base_model_id=probability_model.base_model_id,
        probability_model_id=probability_model.probability_model_id,
        policy_id=str(provenance["policy_id"]),
        config=config,
    )


def save_policy_model(output_path: Path, policy_model: PolicyModel) -> Path:
    """Persist a portable policy contract without paths or runtime decisions."""
    document = {
        "status": "VALID",
        "provenance": {
            "policy_contract_version": POLICY_PROVENANCE_CONTRACT_VERSION,
            "base_model_id": policy_model.base_model_id,
            "probability_model_id": policy_model.probability_model_id,
            "policy_config": policy_model.config.model_dump(mode="json"),
            "policy_id": policy_model.policy_id,
        },
    }
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def load_policy_model(
    policy_path: Path,
    probability_model: ProbabilityModel,
    *,
    expected_policy_id: str | None = None,
) -> PolicyModel:
    """Load a policy only when it exactly binds the supplied verified lineage."""
    try:
        untrusted_document: object = json.loads(policy_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing policy model artifact: {policy_path}") from error
    if not isinstance(untrusted_document, dict):
        raise ValueError("invalid policy model artifact")
    document = cast(dict[str, object], untrusted_document)
    if document.get("status") != "VALID":
        raise ValueError("invalid policy model artifact")
    raw_provenance = document.get("provenance")
    if not isinstance(raw_provenance, dict):
        raise ValueError("policy model artifact is missing provenance")
    provenance = cast(dict[str, object], raw_provenance)
    required = {
        "policy_contract_version",
        "base_model_id",
        "probability_model_id",
        "policy_config",
        "policy_id",
    }
    if not required.issubset(provenance):
        raise ValueError("policy provenance is missing required fields")
    if provenance["policy_contract_version"] != POLICY_PROVENANCE_CONTRACT_VERSION:
        raise ValueError("unsupported policy provenance contract version")
    if provenance["base_model_id"] != probability_model.base_model_id:
        raise ValueError(
            "policy base_model_id does not match probability-model lineage"
        )
    if provenance["probability_model_id"] != probability_model.probability_model_id:
        raise ValueError("policy probability_model_id does not match verified lineage")
    try:
        config = PolicyConfig.model_validate(provenance["policy_config"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid policy configuration in policy provenance") from error
    expected = policy_provenance(probability_model, config)
    if provenance != expected:
        raise ValueError("policy provenance hash or semantics mismatch")
    policy = PolicyModel(
        base_model_id=probability_model.base_model_id,
        probability_model_id=probability_model.probability_model_id,
        policy_id=str(provenance["policy_id"]),
        config=config,
    )
    if expected_policy_id is not None and policy.policy_id != expected_policy_id:
        raise ValueError("policy_id does not match expected policy lineage")
    return policy
