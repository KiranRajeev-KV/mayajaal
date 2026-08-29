"""Alembic environment wired to Mayajaal's one operational metadata root."""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from mayajaal.api.db import Base, DatabaseConfig
from mayajaal.api.db import models as _operational_models

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Importing this module registers all current operational tables on Base before
# Alembic autogeneration compares metadata with PostgreSQL.
_ = _operational_models


def _database_url() -> str:
    """Read the same secret-bearing URL boundary used by the application."""
    return DatabaseConfig.from_environment().database_url.get_secret_value()


def run_migrations_offline() -> None:
    """Configure Alembic without creating an application connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations through a short-lived Alembic-owned connection."""
    section = config.get_section(config.config_ini_section) or {}
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=_database_url(),
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, compare_type=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
