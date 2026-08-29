"""Explicit transactional scope for future operational repositories."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

SessionFactory = sessionmaker[Session]


def create_session_factory(engine: Engine) -> SessionFactory:
    """Bind one reusable, non-expiring session factory to the application engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(sessions: SessionFactory) -> Generator[Session, None, None]:
    """Commit on success; roll back and close a session on every failure path."""
    session = sessions()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
