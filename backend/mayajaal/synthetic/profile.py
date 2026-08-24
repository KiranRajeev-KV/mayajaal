"""Configuration for a deterministic synthetic fraud-world run."""

from datetime import UTC, datetime, timedelta
from typing import Self

from pydantic import Field, model_validator

from mayajaal.schemas.common import AwareDatetime, SchemaModel


class GenerationProfile(SchemaModel):
    """Small, explicit control surface for world size, topology, and time range."""

    seed: int = Field(ge=0)
    normal_account_count: int = Field(default=20, ge=0)
    shared_household_count: int = Field(default=3, ge=0)
    accounts_per_shared_household: int = Field(default=3, gt=0)
    promo_ring_count: int = Field(default=1, ge=0)
    refund_ring_count: int = Field(default=1, ge=0)
    mixed_ring_count: int = Field(default=1, ge=0)
    accounts_per_ring: int = Field(default=4, gt=1)
    start_at: AwareDatetime = datetime(2026, 1, 1, tzinfo=UTC)
    end_at: AwareDatetime = datetime(2026, 4, 1, tzinfo=UTC)

    @model_validator(mode="after")
    def validate_time_window(self) -> Self:
        if self.end_at - self.start_at < timedelta(days=14):
            raise ValueError("generation window must be at least 14 days")
        return self
