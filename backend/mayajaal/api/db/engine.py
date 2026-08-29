"""Application-owned synchronous SQLAlchemy engine and health boundary."""

from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text

from .config import DatabaseConfig
from .session import SessionFactory, create_session_factory


@dataclass(frozen=True)
class DatabaseRuntime:
    """One application-owned engine and session factory for one database URL."""

    config: DatabaseConfig
    engine: Engine
    sessions: SessionFactory

    def dispose(self) -> None:
        """Close pooled connections during application shutdown."""
        self.engine.dispose()


def create_database_runtime(config: DatabaseConfig) -> DatabaseRuntime:
    """Create the single synchronous database runtime for an application process."""
    engine = create_engine(
        config.database_url.get_secret_value(),
        pool_pre_ping=True,
    )
    return DatabaseRuntime(
        config=config,
        engine=engine,
        sessions=create_session_factory(engine),
    )


def ping_database(engine: Engine) -> bool:
    """Return whether a real connection can execute a minimal PostgreSQL query."""
    with engine.connect() as connection:
        return connection.execute(text("SELECT 1")).scalar_one() == 1
