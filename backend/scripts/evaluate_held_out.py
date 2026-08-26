"""Run the deterministic chronological held-out CatBoost evaluation."""

import argparse
from pathlib import Path

from mayajaal.baseline import predict_raw_score
from mayajaal.evaluation import (
    EvaluationSplit,
    FrozenFullArtifactInput,
    build_split_manifest,
    evaluate_catboost,
    held_out_validity,
    save_catboost_evaluation_models,
    vectors_for_manifest,
    write_evaluation_artifacts,
)
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.resolution import resolve_all
from mayajaal.synthetic import generate_world, profile_for_total_accounts
from mayajaal.synthetic.config import load_generation_config


def parse_arguments() -> argparse.Namespace:
    """Parse the canonical config and optional held-out artifact location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <output.directory>/held-out-evaluation",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Derive the configured validation.full_account_count benchmark population.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate, resolve, score, and persist one chronology-respecting benchmark."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    output_directory = (
        arguments.output_dir or Path(config.output.directory) / "held-out-evaluation"
    )
    if not output_directory.is_absolute():
        output_directory = config_path.parent / output_directory

    profile = (
        profile_for_total_accounts(
            config.synthetic_world, config.synthetic_world.validation.full_account_count
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
    records, thresholds, reports, schemas, models = evaluate_catboost(
        service, manifest, config.evaluation
    )
    model_artifacts = save_catboost_evaluation_models(
        models,
        service,
        manifest,
        output_directory / "models",
        shap_sample_count=profile.validation.shap_sample_count,
    )
    vectors = vectors_for_manifest(service, manifest)
    full_raw_scores = {
        sample.sample_id: predict_raw_score(models["full"], vectors[sample.sample_id])
        for sample in manifest.samples
    }
    artifacts = write_evaluation_artifacts(
        output_directory,
        manifest,
        records,
        thresholds,
        reports,
        schemas,
        config.evaluation,
        seed=profile.seed,
        frozen_full=FrozenFullArtifactInput(
            records=records["full"],
            raw_scores=full_raw_scores,
            schema=schemas["full"],
            model_artifacts=model_artifacts["full"],
            training_config=models["full"].config,
            generation_profile=profile,
        ),
    )
    validity = held_out_validity(thresholds, reports)
    print(f"held-out benchmark: {validity.status.value}")
    for reason in validity.reasons:
        print(f"  {reason}")
    print(f"evaluation: {artifacts['evaluation']}")
    print(f"predictions: {artifacts['predictions']}")
    print(f"frozen full provenance: {artifacts['full_provenance']}")
    for name, report in sorted(reports.items()):
        test = report[EvaluationSplit.TEST]
        print(
            f"{name} test: AP={test.average_precision!r}, ROC-AUC={test.roc_auc!r}, "
            f"F1={test.f1!r}, threshold={thresholds[name].threshold!r}"
        )
        print(f"{name} model: {model_artifacts[name].model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
