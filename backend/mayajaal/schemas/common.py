"""Shared schema primitives and validation helpers."""

from datetime import datetime
from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict


def _require_timezone(value: datetime) -> datetime:
    """Reject naive timestamps so temporal graph edges are unambiguous."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return value


AwareDatetime = Annotated[datetime, AfterValidator(_require_timezone)]


class SchemaModel(BaseModel):
    """Base model that rejects unknown fields and validates later assignments."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", validate_assignment=True
    )
