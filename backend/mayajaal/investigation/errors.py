"""Typed, deterministic reasons why investigation grounding was rejected."""

from enum import StrEnum


class GroundingFailureCode(StrEnum):
    """Closed diagnostics for application-owned referential validation failures."""

    MALFORMED_EVIDENCE_REFERENCE = "MALFORMED_EVIDENCE_REFERENCE"
    UNKNOWN_EVIDENCE_REFERENCE = "UNKNOWN_EVIDENCE_REFERENCE"
    UNDECLARED_EVIDENCE_REFERENCE = "UNDECLARED_EVIDENCE_REFERENCE"
    WRONG_TIMELINE_EVIDENCE_TYPE = "WRONG_TIMELINE_EVIDENCE_TYPE"
    UNGROUNDED_RELATED_ENTITY = "UNGROUNDED_RELATED_ENTITY"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    GROUNDING_VALIDATION_FAILED = "GROUNDING_VALIDATION_FAILED"


class InvestigationGroundingError(ValueError):
    """A safe, typed reason an untrusted candidate cannot be grounded."""

    def __init__(self, code: GroundingFailureCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")
