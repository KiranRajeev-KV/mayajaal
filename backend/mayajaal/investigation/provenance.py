"""Deterministic semantic identities for read-only evidence observations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from mayajaal.scoring import canonical_hash

from .allowlist import INVESTIGATION_TOOL_NAMES
from .models import (
    EvidenceSource,
    EvidenceType,
    GroundingFailureDiagnostic,
    InvestigationConfig,
    InvestigationReport,
    InvestigationRequest,
)

if TYPE_CHECKING:
    from .ledger import EvidenceLedgerSnapshot, InvestigationToolTrace

# Related-activity facts now distinguish full-history aggregates from bounded
# detailed retrieval metadata. This changes canonical evidence semantics.
EVIDENCE_CONTRACT_VERSION = 2
INVESTIGATION_PROVENANCE_CONTRACT_VERSION = 2
REPORT_PROVENANCE_CONTRACT_VERSION = 1
# Diagnostics are debug-only and deliberately excluded from investigation and
# report identities, but persisted diagnostics still need independent integrity.
DIAGNOSTIC_PROVENANCE_CONTRACT_VERSION = 1
AGENT_PROMPT_CONTRACT_VERSION = 3


def evidence_id(
    request: InvestigationRequest,
    *,
    evidence_type: EvidenceType,
    source: EvidenceSource,
    observed_at: datetime,
    subject_ids: tuple[str, ...],
    facts: Mapping[str, object],
) -> str:
    """Hash factual evidence semantics, excluding retrieval-time presentation."""
    return canonical_hash(
        {
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "decision_id": request.decision_id,
            "policy_id": request.policy_id,
            "probability_estimate_id": request.probability_estimate_id,
            "score_id": request.score_id,
            "subject_id": request.subject_id,
            "cutoff_time": request.cutoff_time.isoformat(),
            "evidence_type": evidence_type.value,
            "source": source.value,
            "observed_at": observed_at.isoformat(),
            "subject_ids": list(subject_ids),
            "facts": facts,
        }
    )


def investigation_provenance(
    *,
    request: InvestigationRequest,
    config: InvestigationConfig,
    agent_model_id: str,
    snapshot: EvidenceLedgerSnapshot,
) -> dict[str, object]:
    """Return deterministic identity inputs for one bounded investigation run."""
    if not agent_model_id:
        raise ValueError("investigation agent model identity must be non-empty")
    _validate_trace_allowlist(snapshot)
    return {
        "investigation_provenance_contract_version": INVESTIGATION_PROVENANCE_CONTRACT_VERSION,
        "agent_prompt_contract_version": AGENT_PROMPT_CONTRACT_VERSION,
        "decision_id": request.decision_id,
        "investigation_request": request.model_dump(mode="json"),
        "investigation_config": config.model_dump(mode="json"),
        "agent_model_id": agent_model_id,
        "tool_allowlist": list(INVESTIGATION_TOOL_NAMES),
        "tool_trace": [_trace_semantics(trace) for trace in snapshot.tool_trace],
        "evidence_ids": [item.evidence_id for item in snapshot.evidence],
        "investigation_id": investigation_id(
            request=request,
            config=config,
            agent_model_id=agent_model_id,
            snapshot=snapshot,
        ),
    }


def investigation_id(
    *,
    request: InvestigationRequest,
    config: InvestigationConfig,
    agent_model_id: str,
    snapshot: EvidenceLedgerSnapshot,
) -> str:
    """Hash run semantics, deliberately excluding nondeterministic report prose."""
    if not agent_model_id:
        raise ValueError("investigation agent model identity must be non-empty")
    _validate_trace_allowlist(snapshot)
    return canonical_hash(
        {
            "investigation_provenance_contract_version": INVESTIGATION_PROVENANCE_CONTRACT_VERSION,
            "agent_prompt_contract_version": AGENT_PROMPT_CONTRACT_VERSION,
            "decision_id": request.decision_id,
            "investigation_request": request.model_dump(mode="json"),
            "investigation_config": config.model_dump(mode="json"),
            "agent_model_id": agent_model_id,
            "tool_allowlist": list(INVESTIGATION_TOOL_NAMES),
            "tool_trace": [_trace_semantics(trace) for trace in snapshot.tool_trace],
            "evidence_ids": [item.evidence_id for item in snapshot.evidence],
        }
    )


def report_id(investigation_id: str, report: InvestigationReport) -> str:
    """Hash the complete grounded report separately from run provenance."""
    if not investigation_id:
        raise ValueError("investigation_id must be non-empty")
    return canonical_hash(
        {
            "report_provenance_contract_version": REPORT_PROVENANCE_CONTRACT_VERSION,
            "investigation_id": investigation_id,
            "investigation_report": report.model_dump(mode="json"),
        }
    )


def diagnostic_id(investigation_id: str, diagnostic: GroundingFailureDiagnostic) -> str:
    """Hash debug diagnostic content without changing trusted report identity."""
    if not investigation_id:
        raise ValueError("investigation_id must be non-empty")
    return canonical_hash(
        {
            "diagnostic_provenance_contract_version": DIAGNOSTIC_PROVENANCE_CONTRACT_VERSION,
            "investigation_id": investigation_id,
            "grounding_failure": diagnostic.model_dump(mode="json"),
        }
    )


def _trace_semantics(trace: InvestigationToolTrace) -> dict[str, object]:
    return {
        "call_index": trace.call_index,
        "tool_name": trace.tool_name,
        "returned_evidence_ids": list(trace.returned_evidence_ids),
    }


def _validate_trace_allowlist(snapshot: EvidenceLedgerSnapshot) -> None:
    """Keep hash inputs bound to the same fixed tools used at runtime."""
    if any(
        trace.tool_name not in INVESTIGATION_TOOL_NAMES for trace in snapshot.tool_trace
    ):
        raise ValueError("tool trace name is not in the fixed allowlist")
