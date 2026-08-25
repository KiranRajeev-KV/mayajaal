"""Train and persist the deterministic CatBoost graph-feature baseline."""

import argparse
from datetime import datetime
from pathlib import Path

from mayajaal.baseline import label_vectors, save_baseline, train_baseline
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import generate_world
from mayajaal.synthetic.config import load_generation_config


def parse_arguments() -> argparse.Namespace:
    """Parse the generation config, cutoff, and artifact output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--cutoff",
        help="ISO-8601 cutoff; defaults to synthetic_world.end_at",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; defaults to <output.directory>/catboost-baseline",
    )
    return parser.parse_args()


def main() -> int:
    """Train class-balanced CatBoost and save its model, schema, and SHAP plot."""
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
    output_directory = (
        arguments.output_dir or Path(config.output.directory) / "catboost-baseline"
    )
    if not output_directory.is_absolute():
        output_directory = config_path.parent / output_directory

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
    examples = label_vectors(vectors, world, cutoff)
    baseline = train_baseline(examples, service.schema)
    artifacts = save_baseline(baseline, vectors, output_directory)
    print(f"Trained on {len(examples)} account samples at {cutoff.isoformat()}.")
    print(f"Model: {artifacts.model_path}")
    print(f"Metadata: {artifacts.metadata_path}")
    print(f"SHAP summary: {artifacts.shap_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
