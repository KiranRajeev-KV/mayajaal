"""Long-lived Stage 12D runtime resources shared by API and recovery CLIs."""

from dataclasses import dataclass
from pathlib import Path

from mayajaal.calibration import ProbabilityModel, load_probability_model
from mayajaal.evaluation import FrozenFullEvaluation, load_frozen_full_evaluation
from mayajaal.graph import Neo4jGraphRepository
from mayajaal.policy import PolicyModel, load_policy_model
from mayajaal.synthetic import profile_for_total_accounts
from mayajaal.synthetic.config import load_generation_config

from .db import DatabaseConfig, DatabaseRuntime, create_database_runtime
from .event_processing import Neo4jRuntimeConfig, WebhookEventProcessor
from .realtime_pipeline import RealtimeRiskPipelineService
from .risk_scoring import RuntimeRiskScoringService


@dataclass(frozen=True)
class RuntimeArtifactConfig:
    """Verified artifact locations, relative to the generation config by default."""

    config_path: Path = Path("config.toml")
    evaluation_dir: Path = Path("artifacts/held-out-standard-10k-final")
    calibration_dir: Path = Path("artifacts/calibration-standard-10k-final")
    policy_path: Path = Path("artifacts/policy-decision-verified/policy_model.json")


@dataclass
class RealtimeApplicationRuntime:
    """One process-owned database, graph driver, verified artifacts, and pipeline."""

    database: DatabaseRuntime
    graph: Neo4jGraphRepository
    frozen: FrozenFullEvaluation
    probability_model: ProbabilityModel
    policy_model: PolicyModel
    risk_scoring: RuntimeRiskScoringService
    pipeline: RealtimeRiskPipelineService
    owns_database: bool = True

    def dispose(self) -> None:
        self.graph.close()
        if self.owns_database:
            self.database.dispose()


def create_realtime_application_runtime(
    *,
    database: DatabaseRuntime | None = None,
    artifacts: RuntimeArtifactConfig | None = None,
) -> RealtimeApplicationRuntime:
    """Load and cross-check frozen artifacts exactly once for a runtime lifecycle."""
    artifact_config = artifacts or RuntimeArtifactConfig()
    config_path = artifact_config.config_path.resolve()
    config = load_generation_config(config_path)
    profile = profile_for_total_accounts(
        config.synthetic_world, config.synthetic_world.validation.full_account_count
    )
    frozen = load_frozen_full_evaluation(
        _resolve(artifact_config.evaluation_dir, config_path.parent),
        expected_profile=profile,
        expected_evaluation_config=config.evaluation,
    )
    probability = load_probability_model(
        _resolve(artifact_config.calibration_dir, config_path.parent)
        / "sigmoid_calibrator.json",
        expected_base_model_id=frozen.base_model_id,
    )
    policy = load_policy_model(
        _resolve(artifact_config.policy_path, config_path.parent), probability
    )
    runtime = database or create_database_runtime(DatabaseConfig.from_environment())
    graph_config = Neo4jRuntimeConfig.from_environment()
    graph = Neo4jGraphRepository(
        graph_config.uri, (graph_config.username, graph_config.password)
    )
    webhook_processor = WebhookEventProcessor(runtime.sessions, graph)
    scoring = RuntimeRiskScoringService(
        runtime.sessions, graph, frozen, probability, policy
    )
    return RealtimeApplicationRuntime(
        database=runtime,
        graph=graph,
        frozen=frozen,
        probability_model=probability,
        policy_model=policy,
        risk_scoring=scoring,
        pipeline=RealtimeRiskPipelineService(
            runtime.sessions, webhook_processor, scoring
        ),
        owns_database=database is None,
    )


def _resolve(path: Path, config_directory: Path) -> Path:
    return path if path.is_absolute() else config_directory / path
