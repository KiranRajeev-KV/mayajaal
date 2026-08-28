"""Make one auditable expected-cost decision from a verified calibrated model."""

import argparse
from pathlib import Path

from mayajaal.calibration import estimate_probability, load_probability_model
from mayajaal.evaluation import load_frozen_full_evaluation, vectors_for_manifest
from mayajaal.features import FeatureService
from mayajaal.graph import build_graph_projection
from mayajaal.policy import (
    DecisionContext,
    build_policy_model,
    decide,
    save_policy_artifacts,
)
from mayajaal.resolution import resolve_all
from mayajaal.scoring.service import score_feature_vector
from mayajaal.synthetic import generate_world, profile_for_total_accounts
from mayajaal.synthetic.config import load_generation_config


def parse_arguments() -> argparse.Namespace:
    """Parse one decision-time context without reading labels or model scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Defaults to <output.directory>/calibration-evaluation",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        required=True,
        help="Frozen evaluation sample whose cutoff-safe account vector is scored.",
    )
    parser.add_argument("--exposure-paise", type=int, required=True)
    parser.add_argument("--context-id", type=str)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Defaults to <output.directory>/held-out-evaluation.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the configured 10k population when its frozen evaluation is selected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <output.directory>/policy-decision",
    )
    return parser.parse_args()


def _resolve_path(path: Path, config_directory: Path) -> Path:
    """Resolve user-relative artifacts consistently with the other local CLIs."""
    return path if path.is_absolute() else config_directory / path


def main() -> int:
    """Score one frozen-model feature vector, then persist one policy decision."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    calibration_directory = _resolve_path(
        arguments.calibration_dir
        or Path(config.output.directory) / "calibration-evaluation",
        config_path.parent,
    )
    evaluation_directory = _resolve_path(
        arguments.evaluation_dir
        or Path(config.output.directory) / "held-out-evaluation",
        config_path.parent,
    )
    output_directory = _resolve_path(
        arguments.output_dir or Path(config.output.directory) / "policy-decision",
        config_path.parent,
    )
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
    frozen = load_frozen_full_evaluation(
        evaluation_directory,
        expected_profile=profile,
        expected_evaluation_config=config.evaluation,
    )
    vectors = vectors_for_manifest(
        FeatureService(build_graph_projection(world, resolution)), frozen.manifest
    )
    try:
        vector = vectors[arguments.sample_id]
    except KeyError as error:
        raise ValueError(
            "sample_id is not present in the frozen split manifest"
        ) from error
    score_observation = score_feature_vector(frozen, vector)
    probability_model = load_probability_model(
        calibration_directory / "sigmoid_calibrator.json",
        expected_base_model_id=frozen.base_model_id,
    )
    policy_model = build_policy_model(probability_model, config.policy)
    probability_estimate = estimate_probability(
        probability_model,
        score_observation,
        scoring_context_id=arguments.context_id,
    )
    decision = decide(
        policy_model,
        probability_model,
        score_observation,
        probability_estimate,
        DecisionContext(
            exposure_paise=arguments.exposure_paise,
            context_id=arguments.context_id,
        ),
    )
    artifacts = save_policy_artifacts(
        output_directory,
        policy_model,
        probability_model,
        score_observation,
        probability_estimate,
        decision.context,
        decision,
    )
    print(f"policy_id: {decision.policy_id}")
    print(f"probability_model_id: {decision.probability_model_id}")
    print(f"score_id: {decision.score_id}")
    print(f"probability_estimate_id: {decision.probability_estimate_id}")
    print(f"decision_id: {decision.decision_id}")
    print(f"action: {decision.chosen_action.value}")
    print(f"decision margin (paise): {decision.decision_margin_paise:.6f}")
    print(f"policy model: {artifacts['policy_model']}")
    print(f"policy decision: {artifacts['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
