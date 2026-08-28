"""Make one auditable expected-cost decision from a verified calibrated model."""

import argparse
from pathlib import Path

from mayajaal.calibration import estimate_probability, load_probability_model
from mayajaal.policy import (
    DecisionContext,
    build_policy_model,
    decide,
    save_policy_artifacts,
)
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
        "--raw-model-score",
        type=float,
        required=True,
        help="Transform this margin through the verified sigmoid artifact.",
    )
    parser.add_argument("--exposure-paise", type=int, required=True)
    parser.add_argument("--context-id", type=str)
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
    """Verify calibrated lineage, apply merchant costs, and persist one decision."""
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    calibration_directory = _resolve_path(
        arguments.calibration_dir
        or Path(config.output.directory) / "calibration-evaluation",
        config_path.parent,
    )
    output_directory = _resolve_path(
        arguments.output_dir or Path(config.output.directory) / "policy-decision",
        config_path.parent,
    )
    probability_model = load_probability_model(
        calibration_directory / "sigmoid_calibrator.json"
    )
    policy_model = build_policy_model(probability_model, config.policy)
    probability_estimate = estimate_probability(
        probability_model,
        arguments.raw_model_score,
        scoring_context_id=arguments.context_id,
    )
    decision = decide(
        policy_model,
        probability_estimate,
        DecisionContext(
            exposure_paise=arguments.exposure_paise,
            context_id=arguments.context_id,
        ),
    )
    artifacts = save_policy_artifacts(output_directory, policy_model, decision)
    print(f"policy_id: {decision.policy_id}")
    print(f"probability_model_id: {decision.probability_model_id}")
    print(f"probability_estimate_id: {decision.probability_estimate_id}")
    print(f"decision_id: {decision.decision_id}")
    print(f"action: {decision.chosen_action.value}")
    print(f"decision margin (paise): {decision.decision_margin_paise:.6f}")
    print(f"policy model: {artifacts['policy_model']}")
    print(f"policy decision: {artifacts['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
