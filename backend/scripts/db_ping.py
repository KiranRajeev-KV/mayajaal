"""Verify that the configured operational PostgreSQL database is reachable."""

from mayajaal.api.db import DatabaseConfig, create_database_runtime, ping_database


def main() -> int:
    """Connect once, execute SELECT 1, and close the application runtime."""
    runtime = create_database_runtime(DatabaseConfig.from_environment())
    try:
        if not ping_database(runtime.engine):
            return 1
    finally:
        runtime.dispose()
    print("PostgreSQL ping succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
