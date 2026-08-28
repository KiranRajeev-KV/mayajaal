"""Deterministic semantic identities for read-only evidence observations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from mayajaal.scoring import canonical_hash

from .models import (
    EvidenceSource,
    EvidenceType,
    InvestigationConfig,
    InvestigationReport,
    InvestigationRequest,
)

if TYPE_CHECKING:
    from .ledger import EvidenceLedgerSnapshot, InvestigationToolTrace

EVIDENCE_CONTRACT_VERSION = 1
INVESTIGATION_PROVENANCE_CONTRACT_VERSION = 1
REPORT_PROVENANCE_CONTRACT_VERSION = 1
AGENT_PROMPT_CONTRACT_VERSION = 1


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
    tool_allowlist: tuple[str, ...],
    snapshot: EvidenceLedgerSnapshot,
) -> dict[str, object]:
    """Return deterministic identity inputs for one bounded investigation run."""
    if not agent_model_id:
        raise ValueError("investigation agent model identity must be non-empty")
    return {
        "investigation_provenance_contract_version": INVESTIGATION_PROVENANCE_CONTRACT_VERSION,
        "agent_prompt_contract_version": AGENT_PROMPT_CONTRACT_VERSION,
        "decision_id": request.decision_id,
        "investigation_request": request.model_dump(mode="json"),
        "investigation_config": config.model_dump(mode="json"),
        "agent_model_id": agent_model_id,
        "tool_allowlist": list(tool_allowlist),
        "tool_trace": [_trace_semantics(trace) for trace in snapshot.tool_trace],
        "evidence_ids": [item.evidence_id for item in snapshot.evidence],
        "investigation_id": investigation_id(
            request=request,
            config=config,
            agent_model_id=agent_model_id,
            tool_allowlist=tool_allowlist,
            snapshot=snapshot,
        ),
    }


def investigation_id(
    *,
    request: InvestigationRequest,
    config: InvestigationConfig,
    agent_model_id: str,
    tool_allowlist: tuple[str, ...],
    snapshot: EvidenceLedgerSnapshot,
) -> str:
    """Hash run semantics, deliberately excluding nondeterministic report prose."""
    if not agent_model_id:
        raise ValueError("investigation agent model identity must be non-empty")
    return canonical_hash(
        {
            "investigation_provenance_contract_version": INVESTIGATION_PROVENANCE_CONTRACT_VERSION,
            "agent_prompt_contract_version": AGENT_PROMPT_CONTRACT_VERSION,
            "decision_id": request.decision_id,
            "investigation_request": request.model_dump(mode="json"),
            "investigation_config": config.model_dump(mode="json"),
            "agent_model_id": agent_model_id,
            "tool_allowlist": list(tool_allowlist),
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


def _trace_semantics(trace: InvestigationToolTrace) -> dict[str, object]:
    return {
        "call_index": trace.call_index,
        "tool_name": trace.tool_name,
        "returned_evidence_ids": list(trace.returned_evidence_ids),
    }
