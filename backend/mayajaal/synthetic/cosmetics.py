"""Local Faker-backed cosmetic values, isolated from fraud topology."""

from dataclasses import dataclass

from faker import Faker


@dataclass(frozen=True)
class CosmeticIdentity:
    """Human-readable account and shipping-address details."""

    name: str
    email: str
    line1: str
    city: str
    region: str
    postal_code: str


class CosmeticFactory:
    """Produces deterministic, non-behavioral details through one local Faker instance."""

    def __init__(self, seed: int) -> None:
        self._faker = Faker("en_IN")
        self._faker.seed_instance(seed)
        self._serial = 0

    def identity(self) -> CosmeticIdentity:
        """Return one realistic-looking identity without influencing topology."""
        self._serial += 1
        username = self._faker.user_name().replace(".", "_")
        return CosmeticIdentity(
            name=self._faker.name(),
            email=f"{username}{self._serial}@example.test",
            line1=self._faker.street_address(),
            city=self._faker.city(),
            region=self._faker.state(),
            postal_code=self._faker.postcode(),
        )
