"""Storage-independent models emitted by deterministic entity resolution."""

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from mayajaal.schemas.common import SchemaModel


class ResolutionEntityType(StrEnum):
    """Kinds of raw identifiers resolved before graph construction."""

    ADDRESS = "address"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    PAYMENT_IDENTITY = "payment_identity"
    DEVICE = "device"


class ResolutionMethod(StrEnum):
    """The deterministic policy path that produced a resolution."""

    EXACT = "EXACT"
    NORMALIZED = "NORMALIZED"
    FUZZY = "FUZZY"


class ResolutionResult(SchemaModel):
    """One raw entity's link to a canonical entity in its identifier domain."""

    entity_type: ResolutionEntityType
    raw_entity_id: UUID
    canonical_entity_id: UUID
    method: ResolutionMethod
    score: float = Field(ge=0, le=100)
    evidence: str = Field(min_length=1)


class ResolutionBundle(SchemaModel):
    """All results, sorted into a reproducible cross-domain order."""

    results: tuple[ResolutionResult, ...]
