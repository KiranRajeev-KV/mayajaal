"""Clear the dedicated local Neo4j derived graph before changing datasets."""

import argparse
import os

from mayajaal.api.env import load_environment
from mayajaal.graph import Neo4jGraphRepository

load_environment()


def parse_arguments() -> argparse.Namespace:
    """Parse local Neo4j connection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uri", default=os.environ.get("MAYAJAAL_NEO4J_URI", "bolt://localhost:7687")
    )
    parser.add_argument(
        "--username", default=os.environ.get("MAYAJAAL_NEO4J_USERNAME", "neo4j")
    )
    parser.add_argument(
        "--password", default=os.environ.get("MAYAJAAL_NEO4J_PASSWORD", "mayajaal")
    )
    return parser.parse_args()


def main() -> int:
    """Delete all data from the local database used exclusively for this graph."""
    arguments = parse_arguments()
    repository = Neo4jGraphRepository(
        arguments.uri, (arguments.username, arguments.password)
    )
    try:
        repository.clear()
    finally:
        repository.close()
    print("Cleared the derived Neo4j graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
