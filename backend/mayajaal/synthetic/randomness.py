"""Stable, scoped random streams for deterministic world generation."""

from hashlib import blake2b

from numpy.random import Generator, default_rng


def generator_for(seed: int, scope: str) -> Generator:
    """Return a reproducible stream isolated from other generation scopes."""
    digest = blake2b(f"mayajaal:{seed}:{scope}".encode(), digest_size=16).digest()
    return default_rng(int.from_bytes(digest, byteorder="big", signed=False))
