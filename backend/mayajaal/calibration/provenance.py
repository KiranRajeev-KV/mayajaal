"""Stable, model-neutral lineage contracts for calibrated probabilities."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from .models import (
    CalibrationConfig,
    CalibrationMethod,
    ProbabilityEstimate,
    SigmoidCalibrator,
)

CALIBRATION_PROVENANCE_CONTRACT_VERSION = 1
PROBABILITY_ESTIMATE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ProbabilityModel:
    """A verified calibrated probability mapping with explicit base lineage.

    Future policy artifacts should bind to ``probability_model_id`` rather than
    independently rebuilding assumptions about the CatBoost model or calibrator.
    """

    base_model_id: str
    probability_model_id: str
    calibration_config: CalibrationConfig
    calibrator: SigmoidCalibrator
    frozen_provenance: dict[str, object] | None


def canonical_hash(value: object) -> str:
    """Return SHA-256 of canonical JSON independent of presentation details."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def probability_model_provenance(
    *,
    base_model_id: str,
    calibration_config: CalibrationConfig,
    calibrator: SigmoidCalibrator,
    frozen_provenance: dict[str, object] | None,
) -> dict[str, object]:
    """Build the deterministic probability-model identity and lineage payload."""
    if not base_model_id:
        raise ValueError("calibrated probability provenance requires a base_model_id")
    if calibrator.method is not calibration_config.method:
        raise ValueError("calibrator method must match calibration config method")
    semantics = probability_model_semantics(
        base_model_id=base_model_id,
        calibration_contract_version=CALIBRATION_PROVENANCE_CONTRACT_VERSION,
        calibration_method=calibrator.method.value,
        calibration_config=calibration_config.model_dump(mode="json"),
        calibrator_parameters=asdict(calibrator),
    )
    return {
        **semantics,
        "probability_model_id": probability_model_id(
            base_model_id=base_model_id,
            calibration_contract_version=CALIBRATION_PROVENANCE_CONTRACT_VERSION,
            calibration_method=calibrator.method.value,
            calibration_config=calibration_config.model_dump(mode="json"),
            calibrator_parameters=asdict(calibrator),
        ),
        "frozen_provenance": frozen_provenance,
    }


def probability_model_semantics(
    *,
    base_model_id: str,
    calibration_contract_version: int,
    calibration_method: str,
    calibration_config: object,
    calibrator_parameters: object,
) -> dict[str, object]:
    """Return canonical identity inputs usable by future calibration methods."""
    return {
        "calibration_contract_version": calibration_contract_version,
        "base_model_id": base_model_id,
        "calibration_method": calibration_method,
        "calibration_config": calibration_config,
        "calibrator_parameters": calibrator_parameters,
    }


def probability_model_id(
    *,
    base_model_id: str,
    calibration_contract_version: int,
    calibration_method: str,
    calibration_config: object,
    calibrator_parameters: object,
) -> str:
    """Hash model-neutral calibrated-probability semantics into one ID."""
    return canonical_hash(
        probability_model_semantics(
            base_model_id=base_model_id,
            calibration_contract_version=calibration_contract_version,
            calibration_method=calibration_method,
            calibration_config=calibration_config,
            calibrator_parameters=calibrator_parameters,
        )
    )


def probability_estimate_semantics(
    *,
    base_model_id: str,
    probability_model_id: str,
    probability_estimate_contract_version: int,
    raw_model_score: float,
    calibrated_probability: float,
    scoring_context_id: str | None,
) -> dict[str, object]:
    """Return canonical semantic inputs for one score-derived probability."""
    return {
        "probability_estimate_contract_version": probability_estimate_contract_version,
        "base_model_id": base_model_id,
        "probability_model_id": probability_model_id,
        "raw_model_score": raw_model_score,
        "calibrated_probability": calibrated_probability,
        "scoring_context_id": scoring_context_id,
    }


def probability_estimate_id(
    *,
    base_model_id: str,
    probability_model_id: str,
    probability_estimate_contract_version: int,
    raw_model_score: float,
    calibrated_probability: float,
    scoring_context_id: str | None,
) -> str:
    """Hash the semantic score-to-probability result, not its storage location."""
    return canonical_hash(
        probability_estimate_semantics(
            base_model_id=base_model_id,
            probability_model_id=probability_model_id,
            probability_estimate_contract_version=probability_estimate_contract_version,
            raw_model_score=raw_model_score,
            calibrated_probability=calibrated_probability,
            scoring_context_id=scoring_context_id,
        )
    )


def estimate_probability(
    probability_model: ProbabilityModel,
    raw_model_score: float,
    *,
    scoring_context_id: str | None = None,
) -> ProbabilityEstimate:
    """Derive one probability only through a previously verified mapping."""
    from .service import predict_probability

    calibrated_probability = predict_probability(
        probability_model.calibrator, (raw_model_score,)
    )[0]
    return ProbabilityEstimate(
        base_model_id=probability_model.base_model_id,
        probability_model_id=probability_model.probability_model_id,
        probability_estimate_id=probability_estimate_id(
            base_model_id=probability_model.base_model_id,
            probability_model_id=probability_model.probability_model_id,
            probability_estimate_contract_version=PROBABILITY_ESTIMATE_CONTRACT_VERSION,
            raw_model_score=raw_model_score,
            calibrated_probability=calibrated_probability,
            scoring_context_id=scoring_context_id,
        ),
        raw_model_score=raw_model_score,
        calibrated_probability=calibrated_probability,
        scoring_context_id=scoring_context_id,
    )


def verify_probability_estimate(
    estimate: ProbabilityEstimate, probability_model: ProbabilityModel
) -> ProbabilityEstimate:
    """Recompute an estimate through trusted calibration and reject tampering."""
    if estimate.base_model_id != probability_model.base_model_id:
        raise ValueError(
            "probability estimate base_model_id does not match verified lineage"
        )
    if estimate.probability_model_id != probability_model.probability_model_id:
        raise ValueError(
            "probability estimate probability_model_id does not match verified lineage"
        )
    expected = estimate_probability(
        probability_model,
        estimate.raw_model_score,
        scoring_context_id=estimate.scoring_context_id,
    )
    if estimate != expected:
        raise ValueError(
            "probability estimate semantics or calibrated probability mismatch"
        )
    return expected


def load_probability_model(
    calibrator_path: Path,
    *,
    expected_base_model_id: str | None = None,
    expected_probability_model_id: str | None = None,
) -> ProbabilityModel:
    """Load and verify a persisted calibrated probability-model contract."""
    try:
        document = _document(json.loads(calibrator_path.read_text(encoding="utf-8")))
    except FileNotFoundError as error:
        raise ValueError(
            f"missing sigmoid calibrator artifact: {calibrator_path}"
        ) from error
    if document["status"] != "VALID":
        raise ValueError("cannot load an invalid calibrated probability model")
    provenance = _provenance(document["provenance"])
    try:
        config = CalibrationConfig.model_validate(provenance["calibration_config"])
        parameters = _required_mapping(
            provenance["calibrator_parameters"], "calibrator parameters"
        )
        calibrator = SigmoidCalibrator(
            coefficient=float(str(parameters["coefficient"])),
            intercept=float(str(parameters["intercept"])),
            method=CalibrationMethod(str(parameters["method"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid calibrated probability-model parameters") from error
    if calibrator.method is not config.method:
        raise ValueError("calibration method mismatch in probability-model provenance")
    root_parameters = document["parameters"]
    if root_parameters != asdict(calibrator):
        raise ValueError("calibrator parameters mismatch probability-model provenance")
    expected_provenance = probability_model_provenance(
        base_model_id=str(provenance["base_model_id"]),
        calibration_config=config,
        calibrator=calibrator,
        frozen_provenance=_optional_mapping(provenance.get("frozen_provenance")),
    )
    if provenance != expected_provenance:
        raise ValueError("probability-model provenance hash or semantics mismatch")
    frozen_provenance = _optional_mapping(provenance.get("frozen_provenance"))
    if (
        frozen_provenance is not None
        and frozen_provenance.get("base_model_id") != provenance["base_model_id"]
    ):
        raise ValueError("frozen provenance base_model_id mismatch")
    model = ProbabilityModel(
        base_model_id=str(provenance["base_model_id"]),
        probability_model_id=str(provenance["probability_model_id"]),
        calibration_config=config,
        calibrator=calibrator,
        frozen_provenance=frozen_provenance,
    )
    if (
        expected_base_model_id is not None
        and model.base_model_id != expected_base_model_id
    ):
        raise ValueError(
            "probability model base_model_id does not match expected lineage"
        )
    if (
        expected_probability_model_id is not None
        and model.probability_model_id != expected_probability_model_id
    ):
        raise ValueError(
            "probability_model_id does not match expected calibrated-model lineage"
        )
    return model


def _document(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("invalid sigmoid calibrator artifact")
    document = cast(dict[str, object], value)
    required = {"status", "parameters", "provenance"}
    if not required.issubset(document):
        raise ValueError("sigmoid calibrator artifact is missing required fields")
    return document


def _provenance(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("invalid probability-model provenance")
    provenance = cast(dict[str, object], value)
    required = {
        "calibration_contract_version",
        "base_model_id",
        "probability_model_id",
        "calibration_method",
        "calibration_config",
        "calibrator_parameters",
        "frozen_provenance",
    }
    if not required.issubset(provenance):
        raise ValueError("probability-model provenance is missing required fields")
    if (
        provenance["calibration_contract_version"]
        != CALIBRATION_PROVENANCE_CONTRACT_VERSION
    ):
        raise ValueError("unsupported calibration provenance contract version")
    if provenance["calibration_method"] != "sigmoid":
        raise ValueError(
            "unsupported calibration method in probability-model provenance"
        )
    if not isinstance(provenance["base_model_id"], str) or not isinstance(
        provenance["probability_model_id"], str
    ):
        raise ValueError("invalid probability-model lineage identifiers")
    return provenance


def _optional_mapping(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid frozen provenance in probability-model contract")
    return cast(dict[str, object], value)


def _required_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid {name} in probability-model contract")
    return cast(dict[str, object], value)
