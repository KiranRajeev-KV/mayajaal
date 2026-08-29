"""Shared SQLAlchemy declarative metadata for future operational ORM models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Single metadata root; Stage 11A intentionally declares no tables."""
