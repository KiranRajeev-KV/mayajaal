"""Verified persistence for grounded, bounded investigation runs."""

import json
from pathlib import Path
from typing import cast

from mayajaal.schemas.common import SchemaModel

from .grounding import validate_report_grounding
from .ledger import (
    EvidenceLedger,
    EvidenceLedgerSnapshot,
    InvestigationExecution,
    InvestigationToolTrace,
)
from .models import (
    EvidenceItem,
    InvestigationConfig,
    InvestigationReport,
    InvestigationRequest,
)
from .provenance import (
    INVESTIGATION_PROVENANCE_CONTRACT_VERSION,
    REPORT_PROVENANCE_CONTRACT_VERSION,
    investigation_provenance,
    report_id,
)


def save_investigation_artifacts(
    output_directory: Path,
    execution: InvestigationExecution,
) -> dict[str, Path]:
    """Verify a run from trusted contents before writing any report artifacts."""
    request = execution.report.request
    snapshot = _verified_snapshot(request, execution.snapshot)
    validate_report_grounding(execution.report, request, snapshot)
    provenance = investigation_provenance(
        request=request,
        config=execution.config,
        agent_model_id=execution.agent_model_id,
        snapshot=snapshot,
    )
    investigation_id = str(provenance["investigation_id"])
    report_identity = report_id(investigation_id, execution.report)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "provenance": output_directory / "investigation_provenance.json",
        "evidence": output_directory / "evidence.json",
        "report": output_directory / "investigation_report.json",
    }
    paths["provenance"].write_text(
        _document({"status": "VALID", "provenance": provenance}), encoding="utf-8"
    )
    paths["evidence"].write_text(
        _document(
            {
                "investigation_provenance_contract_version": INVESTIGATION_PROVENANCE_CONTRACT_VERSION,
                "investigation_id": investigation_id,
                "evidence": [
                    item.model_dump(mode="json") for item in snapshot.evidence
                ],
                "tool_trace": [
                    item.model_dump(mode="json") for item in snapshot.tool_trace
                ],
            }
        ),
        encoding="utf-8",
    )
    paths["report"].write_text(
        _document(
            {
                "report_provenance_contract_version": REPORT_PROVENANCE_CONTRACT_VERSION,
                "investigation_id": investigation_id,
                "report_id": report_identity,
                "report": execution.report.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    return paths


def load_investigation_artifacts(
    output_directory: Path,
    request: InvestigationRequest,
    config: InvestigationConfig,
    *,
    agent_model_id: str,
) -> InvestigationExecution:
    """Load only an intact report that reconstructs from trusted parent inputs."""
    provenance_document = _read_document(
        output_directory / "investigation_provenance.json"
    )
    evidence_document = _read_document(output_directory / "evidence.json")
    report_document = _read_document(output_directory / "investigation_report.json")
    provenance = _mapping(provenance_document.get("provenance"), "provenance")
    if provenance_document.get("status") != "VALID":
        raise ValueError("invalid investigation provenance artifact")
    report = cast(
        InvestigationReport,
        _model(InvestigationReport, report_document.get("report"), "report"),
    )
    if report.request != request:
        raise ValueError("investigation report request does not match trusted request")
    persisted_config = cast(
        InvestigationConfig,
        _model(
            InvestigationConfig,
            provenance.get("investigation_config"),
            "investigation configuration",
        ),
    )
    if persisted_config != config:
        raise ValueError(
            "investigation artifact configuration does not match trusted expected configuration"
        )
    snapshot = EvidenceLedgerSnapshot(
        evidence=tuple(
            cast(EvidenceItem, _model(EvidenceItem, value, "evidence"))
            for value in _sequence(evidence_document.get("evidence"), "evidence")
        ),
        tool_trace=tuple(
            cast(
                InvestigationToolTrace,
                _model(InvestigationToolTrace, value, "tool trace"),
            )
            for value in _sequence(evidence_document.get("tool_trace"), "tool trace")
        ),
    )
    snapshot = _verified_snapshot(request, snapshot)
    validate_report_grounding(report, request, snapshot)
    expected_provenance = investigation_provenance(
        request=request,
        config=config,
        agent_model_id=agent_model_id,
        snapshot=snapshot,
    )
    if provenance != expected_provenance:
        raise ValueError("investigation provenance semantics or identifier mismatch")
    investigation_id = str(expected_provenance["investigation_id"])
    if (
        evidence_document.get("investigation_provenance_contract_version")
        != INVESTIGATION_PROVENANCE_CONTRACT_VERSION
        or evidence_document.get("investigation_id") != investigation_id
    ):
        raise ValueError("evidence artifact investigation lineage mismatch")
    expected_report_id = report_id(investigation_id, report)
    if (
        report_document.get("report_provenance_contract_version")
        != REPORT_PROVENANCE_CONTRACT_VERSION
        or report_document.get("investigation_id") != investigation_id
        or report_document.get("report_id") != expected_report_id
    ):
        raise ValueError("investigation report provenance or content mismatch")
    return InvestigationExecution(
        report=report,
        snapshot=snapshot,
        agent_model_id=agent_model_id,
        config=persisted_config,
    )


def _verified_snapshot(
    request: InvestigationRequest, snapshot: EvidenceLedgerSnapshot
) -> EvidenceLedgerSnapshot:
    """Reconstruct ledger state to verify evidence IDs, bindings, and traces."""
    return EvidenceLedger.from_snapshot(request, snapshot).snapshot()


def _document(value: dict[str, object]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _read_document(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing investigation artifact: {path}") from error
    return _mapping(raw, "investigation artifact")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid {name}")
    return cast(dict[str, object], value)


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"invalid {name}")
    return tuple(cast(list[object], value))


def _model(model: type[SchemaModel], value: object, name: str) -> SchemaModel:
    try:
        return model.model_validate(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name} artifact") from error
