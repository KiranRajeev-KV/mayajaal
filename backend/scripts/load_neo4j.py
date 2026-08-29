"""Generate, resolve, and load a deterministic world into local Neo4j."""

import argparse
import os
from pathlib import Path

from mayajaal.api.env import load_environment
from mayajaal.graph import Neo4jGraphRepository, build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import generate_world
from mayajaal.synthetic.config import load_generation_config

load_environment()


def parse_arguments() -> argparse.Namespace:
    """Parse a generation profile and local Neo4j connection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
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
    """Load the generated world while keeping Neo4j a derived projection."""
    arguments = parse_arguments()
    config = load_generation_config(arguments.config.resolve())
    world = generate_world(config.synthetic_world)
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    repository = Neo4jGraphRepository(
        arguments.uri, (arguments.username, arguments.password)
    )
    try:
        report = repository.load(build_graph_projection(world, resolution))
    finally:
        repository.close()
    print(
        "Loaded derived graph: "
        f"{report.node_count} nodes, {report.relationship_count} event relationships."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
