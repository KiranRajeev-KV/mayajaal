"""High-value unit tests for the operational database foundation."""

import unittest

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mayajaal.api.db import Base, DatabaseConfig, session_scope


class DatabaseFoundationTests(unittest.TestCase):
    """Protect the secret boundary and transactional lifecycle before Stage 11B."""

    def test_database_url_is_environment_only_and_requires_sync_psycopg(self) -> None:
        with self.assertRaisesRegex(ValueError, "MAYAJAAL_DATABASE_URL must be set"):
            DatabaseConfig.from_environment({})

        with self.assertRaisesRegex(ValueError, "postgresql\\+psycopg"):
            DatabaseConfig.from_environment(
                {"MAYAJAAL_DATABASE_URL": "postgresql://user:password@localhost/db"}
            )

        config = DatabaseConfig.from_environment(
            {
                "MAYAJAAL_DATABASE_URL": (
                    "postgresql+psycopg://user:password@localhost:5433/mayajaal"
                )
            }
        )
        self.assertNotIn("password", repr(config))

    def test_session_scope_rolls_back_on_failure_and_metadata_starts_empty(
        self,
    ) -> None:
        engine = create_engine("sqlite://")
        sessions = sessionmaker(bind=engine)
        session = None
        try:
            with (
                self.assertRaisesRegex(RuntimeError, "rollback"),
                session_scope(sessions) as session,
            ):
                session.execute(text("SELECT 1"))
                raise RuntimeError("rollback")

            self.assertIsNotNone(session)
            self.assertFalse(session.in_transaction())
            self.assertEqual(Base.metadata.tables, {})
        finally:
            engine.dispose()
