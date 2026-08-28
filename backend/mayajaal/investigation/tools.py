"""Fixed LangChain wrappers around the bounded investigation evidence service.

The wrappers deliberately expose no model-controlled arguments.  A later
orchestrator may choose among these five tools, but the request, score, and
all limits remain in :class:`InvestigationToolContext` created by trusted
application code.
"""

from dataclasses import dataclass, field
from threading import Lock

from langchain_core.tools import BaseTool, tool
from pydantic import JsonValue

from mayajaal.scoring import ScoreObservation

from .allowlist import INVESTIGATION_TOOL_NAMES as _INVESTIGATION_TOOL_NAMES
from .ledger import EvidenceLedger
from .models import InvestigationConfig, InvestigationRequest
from .service import EvidenceService

INVESTIGATION_TOOL_NAMES = _INVESTIGATION_TOOL_NAMES


class InvestigationToolBudgetExhausted(ValueError):
    """Raised before a tool can access evidence after the global cap is used."""


@dataclass
class InvestigationToolBudget:
    """Mutable, deterministic per-investigation counter owned by trusted code."""

    max_tool_calls: int
    used_tool_calls: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject invalid direct construction as well as invalid configuration."""
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least one")
        if not 0 <= self.used_tool_calls <= self.max_tool_calls:
            raise ValueError("used_tool_calls must be between zero and max_tool_calls")

    @property
    def remaining_tool_calls(self) -> int:
        """Return how many evidence calls remain in this investigation."""
        return self.max_tool_calls - self.used_tool_calls

    def consume(self) -> None:
        """Reserve one call, rejecting the request before any retrieval occurs."""
        with self._lock:
            if self.used_tool_calls >= self.max_tool_calls:
                raise InvestigationToolBudgetExhausted(
                    "investigation tool-call budget exhausted"
                )
            self.used_tool_calls += 1


@dataclass(frozen=True)
class InvestigationToolContext:
    """Trusted run-scoped dependencies for the fixed investigation tool allowlist."""

    request: InvestigationRequest
    evidence_service: EvidenceService
    score_observation: ScoreObservation
    config: InvestigationConfig
    budget: InvestigationToolBudget = field(repr=False)
    ledger: EvidenceLedger = field(repr=False)

    @classmethod
    def create(
        cls,
        *,
        request: InvestigationRequest,
        evidence_service: EvidenceService,
        score_observation: ScoreObservation,
        config: InvestigationConfig,
    ) -> "InvestigationToolContext":
        """Create a context whose global budget comes from validated config only."""
        if (
            score_observation.score_id != request.score_id
            or score_observation.subject_id != request.subject_id
            or score_observation.scoring_cutoff != request.cutoff_time
            or score_observation.feature_vector_id != request.feature_vector_id
        ):
            raise ValueError("score observation does not match investigation request")
        return cls(
            request=request,
            evidence_service=evidence_service,
            score_observation=score_observation,
            config=config,
            budget=InvestigationToolBudget(max_tool_calls=config.max_tool_calls),
            ledger=EvidenceLedger(request),
        )


def build_investigation_tools(
    context: InvestigationToolContext,
) -> tuple[BaseTool, ...]:
    """Build the complete, fixed, zero-argument read-only investigation allowlist.

    The decorated functions close over the trusted runtime context.  LangChain
    therefore cannot expose subject IDs, cutoff timestamps, traversal budgets,
    event limits, or arbitrary graph queries in their JSON schemas.
    """

    @tool
    def risk_explanation() -> list[dict[str, JsonValue]]:
        """Return bounded TreeSHAP raw-score drivers for the fixed investigation."""
        return _invoke(context, "risk_explanation")

    @tool
    def identity_neighborhood() -> list[dict[str, JsonValue]]:
        """Return the bounded account and identity neighbourhood at the fixed cutoff."""
        return _invoke(context, "identity_neighborhood")

    @tool
    def shared_identity_summary() -> list[dict[str, JsonValue]]:
        """Return cutoff-safe summaries of shared devices, payments, IPs, and addresses."""
        return _invoke(context, "shared_identity_summary")

    @tool
    def related_activity() -> list[dict[str, JsonValue]]:
        """Return bounded historical activity for the fixed subject and ranked peers."""
        return _invoke(context, "related_activity")

    @tool
    def case_timeline() -> list[dict[str, JsonValue]]:
        """Return the bounded chronological case timeline at the fixed cutoff."""
        return _invoke(context, "case_timeline")

    return (
        risk_explanation,
        identity_neighborhood,
        shared_identity_summary,
        related_activity,
        case_timeline,
    )


def _invoke(
    context: InvestigationToolContext,
    capability: str,
) -> list[dict[str, JsonValue]]:
    """Consume the global budget and serialize one matching evidence capability."""
    context.budget.consume()
    service = context.evidence_service
    if capability == "risk_explanation":
        items = service.get_risk_explanation(context.request, context.score_observation)
    elif capability == "identity_neighborhood":
        items = service.get_identity_neighborhood(context.request)
    elif capability == "shared_identity_summary":
        items = service.get_shared_identity_summary(context.request)
    elif capability == "related_activity":
        items = service.get_related_activity(context.request)
    elif capability == "case_timeline":
        items = service.get_case_timeline(context.request)
    else:
        raise ValueError(f"unsupported investigation capability: {capability}")
    context.ledger.record(capability, items)
    return [item.model_dump(mode="json") for item in items]
