"""Generate Mayajaal's deterministic synthetic Parquet dataset."""

import argparse
from pathlib import Path

from mayajaal.synthetic import (
    diagnose_world,
    export_parquet,
    generate_world,
    guardrail_failures,
    write_diagnostics,
)
from mayajaal.synthetic.config import load_generation_config


def parse_arguments() -> argparse.Namespace:
    """Parse the optional config path and output-directory override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML configuration path (default: config.toml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output.directory value from the TOML file",
    )
    return parser.parse_args()


def main() -> int:
    """Generate and export all entity/event tables, then print their paths."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    output_directory = arguments.output_dir or Path(config.output.directory)
    if not output_directory.is_absolute():
        output_directory = config_path.parent / output_directory

    world = generate_world(config.synthetic_world)
    paths = export_parquet(world, output_directory)
    diagnostics = diagnose_world(world)
    diagnostics_path = write_diagnostics(
        diagnostics, output_directory / "diagnostics.json"
    )
    guardrails = guardrail_failures(diagnostics, config.synthetic_world)
    print(f"Generated {len(world.events)} events for {len(world.accounts)} accounts.")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"diagnostics: {diagnostics_path}")
    if guardrails:
        print(f"diagnostic warnings: {', '.join(guardrails)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
