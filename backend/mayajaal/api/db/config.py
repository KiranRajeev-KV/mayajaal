"""Validated secret-bearing PostgreSQL configuration boundary."""

import os
from collections.abc import Mapping

from pydantic import SecretStr, field_validator
from sqlalchemy import make_url
from sqlalchemy.exc import ArgumentError

from mayajaal.schemas.common import SchemaModel

DATABASE_URL_ENVIRONMENT_VARIABLE = "MAYAJAAL_DATABASE_URL"
_POSTGRES_DRIVER_NAME = "postgresql+psycopg"


class DatabaseConfig(SchemaModel):
    """Database connection settings sourced only from the process environment."""

    database_url: SecretStr

    @field_validator("database_url")
    @classmethod
    def _require_sync_psycopg_postgres(cls, value: SecretStr) -> SecretStr:
        """Reject malformed URLs and drivers outside the supported sync stack."""
        try:
            url = make_url(value.get_secret_value())
        except (ArgumentError, ValueError) as error:
            message = (
                f"{DATABASE_URL_ENVIRONMENT_VARIABLE} must be a valid "
                "SQLAlchemy PostgreSQL URL"
            )
            raise ValueError(message) from error

        if url.drivername != _POSTGRES_DRIVER_NAME:
            message = (
                f"{DATABASE_URL_ENVIRONMENT_VARIABLE} must use {_POSTGRES_DRIVER_NAME}"
            )
            raise ValueError(message)
        if not url.host or not url.database or not url.username or url.password is None:
            message = (
                f"{DATABASE_URL_ENVIRONMENT_VARIABLE} must include host, database, "
                "username, and password"
            )
            raise ValueError(message)
        return value

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "DatabaseConfig":
        """Read the one required secret-bearing setting without logging it."""
        source = os.environ if environment is None else environment
        value = source.get(DATABASE_URL_ENVIRONMENT_VARIABLE)
        if value is None or not value.strip():
            message = f"{DATABASE_URL_ENVIRONMENT_VARIABLE} must be set"
            raise ValueError(message)
        return cls.model_validate({"database_url": value})
