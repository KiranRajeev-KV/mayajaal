"""Fit validation-only sigmoid calibration for the frozen full CatBoost baseline."""

import argparse
from dataclasses import asdict
from pathlib import Path

from mayajaal.calibration import calibrate_records, save_calibration_artifacts
from mayajaal.evaluation import (
    build_split_manifest,
    fit_full_catboost_scores,
)
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import generate_world, profile_for_total_accounts
from mayajaal.synthetic.config import load_generation_config


def parse_arguments() -> argparse.Namespace:
    """Parse the canonical config and optional calibration artifact location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <output.directory>/calibration-evaluation",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Derive the configured validation.full_account_count benchmark population.",
    )
    return parser.parse_args()


def main() -> int:
    """Rebuild a deterministic benchmark and calibrate only its full model."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    output_directory = (
        arguments.output_dir or Path(config.output.directory) / "calibration-evaluation"
    )
    if not output_directory.is_absolute():
        output_directory = config_path.parent / output_directory
    profile = (
        profile_for_total_accounts(
            config.synthetic_world,
            config.synthetic_world.validation.full_account_count,
        )
        if arguments.full
        else config.synthetic_world
    )
    world = generate_world(profile)
    resolution = resolve_all(
        accounts=world.accounts,
        addresses=world.addresses,
        ip_addresses=world.ip_addresses,
        payment_identities=world.payment_identities,
        devices=world.devices,
    )
    service = FeatureService(build_graph_projection(world, resolution))
    manifest = build_split_manifest(
        world,
        config.evaluation,
        start_at=profile.start_at,
        end_at=profile.end_at,
    )
    full_records, raw_scores, schema, _ = fit_full_catboost_scores(service, manifest)
    predictions, report = calibrate_records(
        full_records, raw_scores, config.calibration
    )
    artifacts = save_calibration_artifacts(
        output_directory,
        predictions,
        report,
        metadata={
            "name": "validation-only sigmoid calibration",
            "model_variant": "full",
            "seed": profile.seed,
            "calibration_config": config.calibration.model_dump(mode="json"),
            "evaluation_config": config.evaluation.model_dump(mode="json"),
            "cutoffs": {
                "train": manifest.train_cutoff.isoformat(),
                "validation": manifest.validation_cutoff.isoformat(),
                "test": manifest.test_cutoff.isoformat(),
            },
            "feature_schema": [asdict(item) for item in schema.definitions],
            "leakage_policy": "CatBoost is fit on train; the frozen raw-score sigmoid is fit only on validation; test is evaluated once and never used for fitting, threshold selection, or method selection.",
        },
    )
    print(f"calibration: {report.fit.status.value}")
    for reason in report.fit.reasons:
        print(f"  {reason}")
    print(f"calibration evaluation: {artifacts['evaluation']}")
    print(f"calibration predictions: {artifacts['predictions']}")
    test = report.test_calibrated
    if test is not None:
        print(
            "test calibrated: "
            f"Brier={test.brier_score:.6f}, log_loss={test.log_loss:.6f}, "
            f"AP={test.average_precision!r}, ROC-AUC={test.roc_auc!r}"
        )
    else:
        print("test calibrated: unavailable because validation calibration is invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
