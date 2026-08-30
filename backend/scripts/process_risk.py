"""Score one already-PROCESSED webhook event through frozen Stage 12C artifacts."""

import argparse
from pathlib import Path

from mayajaal.api.env import load_environment
from mayajaal.api.runtime import (
    RuntimeArtifactConfig,
    create_realtime_application_runtime,
)


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
        default=Path("artifacts/policy-decision-verified/policy_model.json"),
    )
    arguments = parser.parse_args()
    runtime = create_realtime_application_runtime(
        artifacts=RuntimeArtifactConfig(
            config_path=arguments.config,
            evaluation_dir=arguments.evaluation_dir,
            calibration_dir=arguments.calibration_dir,
            policy_path=arguments.policy_path,
        )
    )
    try:
        result = runtime.risk_scoring.process(arguments.event_id)
        print(
            f"{result.provider_event_id}: decision={result.decision_id} case={result.case_id} reused={result.reused}"
        )
    finally:
        runtime.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
