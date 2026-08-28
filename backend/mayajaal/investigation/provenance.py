"""Deterministic semantic identities for read-only evidence observations."""

from collections.abc import Mapping
from datetime import datetime

from mayajaal.scoring import canonical_hash

from .models import EvidenceSource, EvidenceType, InvestigationRequest

EVIDENCE_CONTRACT_VERSION = 1


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
