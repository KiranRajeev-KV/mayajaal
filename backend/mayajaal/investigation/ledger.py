"""Run-scoped records of the factual evidence actually returned to an agent."""

from dataclasses import dataclass, field

from pydantic import Field, field_validator

from mayajaal.schemas.common import SchemaModel

from .allowlist import INVESTIGATION_TOOL_NAMES
from .models import (
    EvidenceItem,
    InvestigationConfig,
    InvestigationReport,
    InvestigationRequest,
)
from .provenance import evidence_id


class InvestigationToolTrace(SchemaModel):
    """One approved evidence-tool invocation and its returned observations."""

    call_index: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    returned_evidence_ids: tuple[str, ...]

    @field_validator("tool_name")
    @classmethod
    def validate_fixed_allowlist(cls, value: str) -> str:
        """Reject traces that did not invoke one of the five approved tools."""
        if value not in INVESTIGATION_TOOL_NAMES:
            raise ValueError("tool trace name is not in the fixed allowlist")
        return value


@dataclass(frozen=True)
class EvidenceLedgerSnapshot:
    """Immutable ledger state used for validation, provenance, and persistence."""

    evidence: tuple[EvidenceItem, ...]
    tool_trace: tuple[InvestigationToolTrace, ...]


@dataclass(frozen=True)
class InvestigationExecution:
    """One bounded agent outcome with its application-owned evidence ledger."""

    report: InvestigationReport
    snapshot: EvidenceLedgerSnapshot
    agent_model_id: str
    config: InvestigationConfig


@dataclass
class EvidenceLedger:
    """Application-owned ledger that only admits request-bound factual evidence."""

    request: InvestigationRequest
    _items: dict[str, EvidenceItem] = field(default_factory=dict, init=False)
    _trace: list[InvestigationToolTrace] = field(default_factory=list, init=False)

    def record(self, tool_name: str, items: tuple[EvidenceItem, ...]) -> None:
        """Verify and record exactly one approved tool result in call order."""
        if tool_name not in INVESTIGATION_TOOL_NAMES:
            raise ValueError("evidence tool name is not in the fixed allowlist")
        returned_ids: list[str] = []
        for item in items:
            _verify_evidence(item, self.request)
            existing = self._items.get(item.evidence_id)
            if existing is not None and existing != item:
                raise ValueError("conflicting duplicate evidence_id returned by tools")
            self._items.setdefault(item.evidence_id, item)
            returned_ids.append(item.evidence_id)
        self._trace.append(
            InvestigationToolTrace(
                call_index=len(self._trace) + 1,
                tool_name=tool_name,
                returned_evidence_ids=tuple(returned_ids),
            )
        )

    def snapshot(self) -> EvidenceLedgerSnapshot:
        """Return deterministic evidence and trace records for one completed run."""
        return EvidenceLedgerSnapshot(
            evidence=tuple(self._items.values()), tool_trace=tuple(self._trace)
        )

    @classmethod
    def from_snapshot(
        cls, request: InvestigationRequest, snapshot: EvidenceLedgerSnapshot
    ) -> "EvidenceLedger":
        """Rebuild a ledger only if persisted evidence and trace remain valid."""
        ledger = cls(request)
        items_by_id = {item.evidence_id: item for item in snapshot.evidence}
        if len(items_by_id) != len(snapshot.evidence):
            raise ValueError("persisted evidence contains duplicate evidence_id")
        expected_call_index = 1
        for trace in snapshot.tool_trace:
            if trace.call_index != expected_call_index:
                raise ValueError("tool trace call indexes must be consecutive")
            if trace.tool_name not in INVESTIGATION_TOOL_NAMES:
                raise ValueError("tool trace name is not in the fixed allowlist")
            expected_call_index += 1
            trace_items: list[EvidenceItem] = []
            for item_id in trace.returned_evidence_ids:
                item = items_by_id.get(item_id)
                if item is None:
                    raise ValueError("tool trace references missing evidence")
                trace_items.append(item)
            ledger.record(trace.tool_name, tuple(trace_items))
        if tuple(ledger.snapshot().evidence) != snapshot.evidence:
            raise ValueError("persisted evidence is not represented by its tool trace")
        return ledger


def _verify_evidence(item: EvidenceItem, request: InvestigationRequest) -> None:
    """Verify request/cutoff binding and the evidence's deterministic identity."""
    item.verify_for_request(request)
    expected_id = evidence_id(
        request,
        evidence_type=item.evidence_type,
        source=item.source,
        observed_at=item.observed_at,
        subject_ids=item.subject_ids,
        facts=item.facts,
    )
    if item.evidence_id != expected_id:
        raise ValueError("evidence_id does not match deterministic evidence semantics")
