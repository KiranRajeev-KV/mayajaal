"""Run-scoped records of the factual evidence actually returned to an agent."""

import re
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
    _aliases: dict[str, str] = field(default_factory=dict, init=False)
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
            if existing is None:
                self._items[item.evidence_id] = item
                self._aliases[item.evidence_id] = _evidence_alias(len(self._items))
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

    def alias_for(self, canonical_evidence_id: str) -> str:
        """Return the deterministic model-facing reference for admitted evidence."""
        try:
            return self._aliases[canonical_evidence_id]
        except KeyError as error:
            raise ValueError(
                "evidence_id has not been admitted to this ledger"
            ) from error

    def resolve_alias(self, alias: str) -> str:
        """Resolve one strict, run-local evidence alias to its canonical ID."""
        if not _EVIDENCE_ALIAS_PATTERN.fullmatch(alias):
            raise ValueError("evidence reference must use the E001 alias format")
        for canonical_id, admitted_alias in self._aliases.items():
            if admitted_alias == alias:
                return canonical_id
        raise ValueError("evidence reference is not admitted to this investigation")

    def resolve_aliases(self, aliases: tuple[str, ...]) -> tuple[str, ...]:
        """Resolve ordered model references without any fuzzy matching."""
        return tuple(self.resolve_alias(alias) for alias in aliases)

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


_EVIDENCE_ALIAS_PATTERN = re.compile(r"E[0-9]{3,}")


def _evidence_alias(admission_index: int) -> str:
    """Return a stable, readable reference based only on ledger admission order."""
    return f"E{admission_index:03d}"
