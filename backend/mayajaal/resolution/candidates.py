"""Bounded address candidate generation using deterministic address partitions."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from mayajaal.schemas import Address

from .normalizers import (
    normalize_address_component,
    normalize_city,
    normalize_postal_code,
    normalize_text,
)


@dataclass(frozen=True)
class NormalizedAddress:
    """Address fields used by matching; recipient names are deliberately excluded."""

    entity_id: UUID
    line1: str
    line2: str
    city: str
    region: str
    postal_code: str
    country_code: str

    @property
    def exact_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.line1,
            self.line2,
            self.city,
            self.region,
            self.postal_code,
            self.country_code,
        )

    @property
    def candidate_key(self) -> tuple[str, str, str]:
        """Strict locality partition that prevents unrelated global comparisons."""
        return (self.country_code, self.city, self.postal_code)

    @property
    def comparison_text(self) -> str:
        return " ".join(part for part in (self.line1, self.line2, self.region) if part)


def normalize_address(address: Address) -> NormalizedAddress:
    """Extract a stable matching representation from an address entity."""
    return NormalizedAddress(
        entity_id=address.id,
        line1=normalize_address_component(address.line1),
        line2=normalize_address_component(address.line2),
        city=normalize_city(address.city),
        region=normalize_text(address.region or ""),
        postal_code=normalize_postal_code(address.postal_code),
        country_code=normalize_text(address.country_code).upper(),
    )


def address_candidate_buckets(
    addresses: Iterable[NormalizedAddress],
) -> dict[tuple[str, str, str], tuple[NormalizedAddress, ...]]:
    """Group addresses into bounded fuzzy-comparison sets in stable ID order."""
    buckets: defaultdict[tuple[str, str, str], list[NormalizedAddress]] = defaultdict(
        list
    )
    for address in addresses:
        buckets[address.candidate_key].append(address)
    return {
        key: tuple(sorted(values, key=lambda value: str(value.entity_id)))
        for key, values in buckets.items()
    }
