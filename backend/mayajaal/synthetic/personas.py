"""Hidden ordinary-customer behaviour parameters used by the generator."""

from dataclasses import dataclass
from enum import StrEnum

from numpy.random import Generator


class PersonaName(StrEnum):
    """Ordinary commerce behaviour archetypes; none imply a label."""

    OCCASIONAL = "occasional"
    REPEAT = "repeat"
    PROMO_SENSITIVE = "promo_sensitive"
    HIGH_VALUE = "high_value"
    COMMUTER = "commuter"
    DIGITAL_FIRST = "digital_first"


@dataclass(frozen=True)
class PersonaSpec:
    """Correlated, hidden parameters for one ordinary customer archetype."""

    name: PersonaName
    order_rate: float
    promotion_probability: float
    refund_multiplier: float
    value_multiplier: float
    preferred_hour: int
    mobile_probability: float
    travel_multiplier: float


PERSONAS: dict[PersonaName, PersonaSpec] = {
    PersonaName.OCCASIONAL: PersonaSpec(
        PersonaName.OCCASIONAL, 1.0, 0.12, 0.7, 0.8, 20, 0.65, 0.7
    ),
    PersonaName.REPEAT: PersonaSpec(
        PersonaName.REPEAT, 4.5, 0.20, 1.0, 1.0, 19, 0.60, 0.9
    ),
    PersonaName.PROMO_SENSITIVE: PersonaSpec(
        PersonaName.PROMO_SENSITIVE, 2.7, 0.62, 1.1, 0.72, 21, 0.72, 0.8
    ),
    PersonaName.HIGH_VALUE: PersonaSpec(
        PersonaName.HIGH_VALUE, 2.2, 0.08, 0.65, 2.7, 13, 0.40, 1.1
    ),
    PersonaName.COMMUTER: PersonaSpec(
        PersonaName.COMMUTER, 2.4, 0.18, 0.9, 0.95, 8, 0.88, 2.2
    ),
    PersonaName.DIGITAL_FIRST: PersonaSpec(
        PersonaName.DIGITAL_FIRST, 3.0, 0.28, 1.0, 1.15, 22, 0.82, 1.15
    ),
}


def choose_persona(weights: dict[str, float], rng: Generator) -> PersonaSpec:
    """Select a validated persona from configurable relative weights."""
    names = tuple(PersonaName(name) for name in weights)
    values = tuple(weights[name.value] for name in names)
    index = int(rng.choice(len(names), p=[value / sum(values) for value in values]))
    return PERSONAS[names[index]]
