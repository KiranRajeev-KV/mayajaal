"""Deterministic referential grounding checks for structured investigation reports."""

from .errors import GroundingFailureCode, InvestigationGroundingError
from .ledger import EvidenceLedgerSnapshot
from .models import (
    EvidenceItem,
    EvidenceType,
    InvestigationReport,
    InvestigationRequest,
)


def validate_report_grounding(
    report: InvestigationReport,
    request: InvestigationRequest,
    snapshot: EvidenceLedgerSnapshot,
) -> InvestigationReport:
    """Require every structured factual reference to exist in this run's ledger.

    This intentionally validates references, not prose meaning.  The ledger is
    application-owned and every entry is independently request/cutoff bound.
    """
    if report.request != request:
        raise InvestigationGroundingError(
            GroundingFailureCode.GROUNDING_VALIDATION_FAILED,
            "report request does not match investigation",
        )
    evidence_by_id = {item.evidence_id: item for item in snapshot.evidence}
    if len(evidence_by_id) != len(snapshot.evidence):
        raise InvestigationGroundingError(
            GroundingFailureCode.GROUNDING_VALIDATION_FAILED,
            "ledger contains duplicate evidence IDs",
        )
    for item in snapshot.evidence:
        try:
            item.verify_for_request(request)
        except ValueError as error:
            raise InvestigationGroundingError(
                GroundingFailureCode.GROUNDING_VALIDATION_FAILED,
                "ledger evidence does not match investigation cutoff",
            ) from error
    _verify_report_ids(report.evidence_ids, evidence_by_id, "report")
    for finding in (*report.key_findings, *report.counterevidence):
        _verify_report_ids(finding.evidence_ids, evidence_by_id, "finding")
    timeline_items = _verify_report_ids(
        report.timeline_evidence_ids, evidence_by_id, "timeline"
    )
    if any(
        item.evidence_type is not EvidenceType.TIMELINE_EVENT for item in timeline_items
    ):
        raise InvestigationGroundingError(
            GroundingFailureCode.WRONG_TIMELINE_EVIDENCE_TYPE,
            "timeline references must cite TIMELINE_EVENT evidence",
        )
    for related in report.related_entities:
        _verify_report_ids(
            related.evidence_ids,
            evidence_by_id,
            "related entity",
            failure_code=GroundingFailureCode.UNGROUNDED_RELATED_ENTITY,
        )
    return report


def _verify_report_ids(
    ids: tuple[str, ...],
    evidence_by_id: dict[str, EvidenceItem],
    label: str,
    *,
    failure_code: GroundingFailureCode = GroundingFailureCode.UNKNOWN_EVIDENCE_REFERENCE,
) -> tuple[EvidenceItem, ...]:
    result: list[EvidenceItem] = []
    for item_id in ids:
        item = evidence_by_id.get(item_id)
        if item is None:
            raise InvestigationGroundingError(
                failure_code,
                f"{label} references evidence not returned to this agent",
            )
        result.append(item)
    return tuple(result)
