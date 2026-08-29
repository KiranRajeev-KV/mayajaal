"""Shared SQLAlchemy declarative metadata for operational ORM models."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Names are generated before Alembic sees metadata, keeping upgrade/downgrade
# operations stable across PostgreSQL installations and autogeneration runs.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Single metadata root for persistence-only SQLAlchemy models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
