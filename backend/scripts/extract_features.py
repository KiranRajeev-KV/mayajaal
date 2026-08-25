"""Generate resolved cutoff-safe account feature vectors as a Parquet table."""

import argparse
from datetime import datetime
from pathlib import Path

from mayajaal.features import FeatureService, feature_frame
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import generate_world
from mayajaal.synthetic.config import load_generation_config


def parse_arguments() -> argparse.Namespace:
    """Parse the generation config, cutoff, and Parquet output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--cutoff",
        help="ISO-8601 cutoff; defaults to synthetic_world.end_at",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Parquet path; defaults to <output.directory>/account_features.parquet",
    )
    return parser.parse_args()


def main() -> int:
    """Resolve a generated world and export only cutoff-safe feature vectors."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    cutoff = (
        datetime.fromisoformat(arguments.cutoff)
        if arguments.cutoff is not None
        else config.synthetic_world.end_at
    )
    if cutoff.tzinfo is None:
        raise ValueError("--cutoff must include a timezone offset")
    output = (
        arguments.output or Path(config.output.directory) / "account_features.parquet"
    )
    if not output.is_absolute():
        output = config_path.parent / output

    world = generate_world(config.synthetic_world)
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    service = FeatureService(build_graph_projection(world, resolution))
    vectors = service.extract_many(
        (str(account.id) for account in world.accounts), cutoff
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_frame(vectors, service.schema).write_parquet(output)
    print(
        f"Wrote {len(vectors)} account feature vectors at {cutoff.isoformat()}: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
