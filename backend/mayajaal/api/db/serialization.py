"""Pydantic-backed JSON payload conversion for immutable domain contracts."""

from typing import Any, cast

from pydantic import TypeAdapter

from mayajaal.api.contracts import InvestigationRun, RiskCase
from mayajaal.calibration import ProbabilityEstimate
from mayajaal.investigation import InvestigationReport, InvestigationRequest
from mayajaal.policy import PolicyDecision
from mayajaal.scoring import ScoreObservation

DomainObject = (
    ScoreObservation
    | ProbabilityEstimate
    | PolicyDecision
    | InvestigationRequest
    | InvestigationReport
    | InvestigationRun
    | RiskCase
)

_SCORE_ADAPTER = TypeAdapter(ScoreObservation)
_ESTIMATE_ADAPTER = TypeAdapter(ProbabilityEstimate)
_DECISION_ADAPTER = TypeAdapter(PolicyDecision)
_REQUEST_ADAPTER = TypeAdapter(InvestigationRequest)
_REPORT_ADAPTER = TypeAdapter(InvestigationReport)
_RUN_ADAPTER = TypeAdapter(InvestigationRun)
_CASE_ADAPTER = TypeAdapter(RiskCase)


def payload_for(value: DomainObject) -> dict[str, object]:
    """Produce a JSONB-safe canonical Pydantic serialization of a domain object."""
    payload = _adapter_for_type(type(value)).dump_python(
        value, mode="json", round_trip=True
    )
    if not isinstance(payload, dict):
        raise TypeError("domain serialization must produce an object payload")
    return cast(dict[str, object], payload)


def from_payload(model: type[DomainObject], payload: object) -> DomainObject:
    """Fail closed when a persisted payload no longer validates its contract."""
    try:
        return cast(DomainObject, _adapter_for_type(model).validate_python(payload))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "stored domain payload failed authoritative validation"
        ) from error


def _adapter_for_type(model: type[DomainObject]) -> TypeAdapter[Any]:
    # Pydantic's invariant generic adapter cannot represent this heterogeneous
    # dispatch map more narrowly; callers preserve the concrete model boundary.
    adapters: dict[type[DomainObject], TypeAdapter[Any]] = {
        ScoreObservation: _SCORE_ADAPTER,
        ProbabilityEstimate: _ESTIMATE_ADAPTER,
        PolicyDecision: _DECISION_ADAPTER,
        InvestigationRequest: _REQUEST_ADAPTER,
        InvestigationReport: _REPORT_ADAPTER,
        InvestigationRun: _RUN_ADAPTER,
        RiskCase: _CASE_ADAPTER,
    }
    try:
        return adapters[model]
    except KeyError as error:
        raise TypeError(f"unsupported persisted domain type: {model!r}") from error
