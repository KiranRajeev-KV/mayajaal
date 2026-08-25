"""Deterministic identity resolution."""

from .models import (
    ResolutionBundle,
    ResolutionEntityType,
    ResolutionMethod,
    ResolutionResult,
)
from .policy import ADDRESS_FUZZY_THRESHOLD, resolve_addresses, resolve_all

__all__ = [
    "ADDRESS_FUZZY_THRESHOLD",
    "ResolutionBundle",
    "ResolutionEntityType",
    "ResolutionMethod",
    "ResolutionResult",
    "resolve_addresses",
    "resolve_all",
]
