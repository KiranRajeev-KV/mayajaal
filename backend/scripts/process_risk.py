"""Score one already-PROCESSED webhook event through frozen Stage 12C artifacts."""

import argparse
from pathlib import Path

from mayajaal.api.db import DatabaseConfig, create_database_runtime
from mayajaal.api.env import load_environment
from mayajaal.api.event_processing import Neo4jRuntimeConfig
from mayajaal.api.risk_scoring import RuntimeRiskScoringService
from mayajaal.calibration import load_probability_model
from mayajaal.evaluation import load_frozen_full_evaluation
from mayajaal.graph import Neo4jGraphRepository
from mayajaal.policy import load_policy_model
from mayajaal.synthetic import profile_for_total_accounts
from mayajaal.synthetic.config import load_generation_config


def _resolve(path: Path, config_directory: Path) -> Path:
    return path if path.is_absolute() else config_directory / path


def main() -> int:
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("artifacts/held-out-standard-10k-final"),
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("artifacts/calibration-standard-10k-final"),
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path("artifacts/policy-standard-10k-final/policy_model.json"),
    )
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = load_generation_config(config_path)
    profile = profile_for_total_accounts(
        config.synthetic_world, config.synthetic_world.validation.full_account_count
    )
    frozen = load_frozen_full_evaluation(
        _resolve(arguments.evaluation_dir, config_path.parent),
        expected_profile=profile,
        expected_evaluation_config=config.evaluation,
    )
    probability = load_probability_model(
        _resolve(arguments.calibration_dir, config_path.parent)
        / "sigmoid_calibrator.json",
        expected_base_model_id=frozen.base_model_id,
    )
    policy = load_policy_model(
        _resolve(arguments.policy_path, config_path.parent), probability
    )
    database = create_database_runtime(DatabaseConfig.from_environment())
    graph_config = Neo4jRuntimeConfig.from_environment()
    graph = Neo4jGraphRepository(
        graph_config.uri, (graph_config.username, graph_config.password)
    )
    try:
        result = RuntimeRiskScoringService(
            database.sessions, graph, frozen, probability, policy
        ).process(arguments.event_id)
        print(
            f"{result.provider_event_id}: decision={result.decision_id} case={result.case_id} reused={result.reused}"
        )
    finally:
        graph.close()
        database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
