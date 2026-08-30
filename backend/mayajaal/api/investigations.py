"""Stage 12E user-triggered investigation execution around existing contracts."""

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from mayajaal.calibration import ProbabilityModel
from mayajaal.evaluation import FrozenFullEvaluation
from mayajaal.features import FeatureService
from mayajaal.graph import Neo4jGraphRepository
from mayajaal.investigation import (
    EvidenceService,
    InvestigationAgentService,
    InvestigationConfig,
)

from .contracts import InvestigationJobStatus
from .db import (
    InvestigationJobRepository,
    InvestigationJobUnavailable,
    InvestigationRequestRepository,
    NormalizedEventRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    ScoreObservationRepository,
)
from .db.session import SessionFactory
from .orchestration import RuntimeLineagePersistenceService

DEFAULT_INVESTIGATION_LEASE_TIMEOUT = timedelta(minutes=10)


@dataclass(frozen=True)
class InvestigationProcessResult:
    """Small operational outcome for an explicit/background attempt."""

    run_id: str
    status: InvestigationJobStatus
    reused: bool


class InvestigationExecutionService:
    """Run one durable user-requested job without changing agent semantics."""

    def __init__(
        self,
        sessions: SessionFactory,
        graph: Neo4jGraphRepository,
        frozen: FrozenFullEvaluation,
        probability_model: ProbabilityModel,
        config: InvestigationConfig,
        *,
        agent_factory: Callable[[InvestigationConfig], InvestigationAgentService]
        | None = None,
        lease_timeout: timedelta = DEFAULT_INVESTIGATION_LEASE_TIMEOUT,
    ) -> None:
        if lease_timeout <= timedelta(0):
            raise ValueError("investigation lease timeout must be positive")
        self._sessions = sessions
        self._graph = graph
        self._frozen = frozen
        self._probability_model = probability_model
        self._config = config.model_copy(deep=True)
        # Agent counters are intentionally per service instance/run, not shared.
        self._agent_factory: Callable[
            [InvestigationConfig], InvestigationAgentService
        ] = agent_factory or _new_agent
        self._lease_timeout = lease_timeout

    @property
    def config(self) -> InvestigationConfig:
        return self._config.model_copy(deep=True)

    def process(self, run_id: str) -> InvestigationProcessResult:
        """Claim, execute, and atomically persist one job's trusted outcome."""
        now = datetime.now(tz=UTC)
        with self._sessions.begin() as session:
            jobs = InvestigationJobRepository(session)
            existing = jobs.get(run_id)
            if existing is None:
                raise ValueError("investigation job does not exist")
            if existing.status is InvestigationJobStatus.COMPLETED:
                return InvestigationProcessResult(run_id, existing.status, True)
            try:
                job = jobs.claim(
                    run_id, claimed_at=now, lease_timeout=self._lease_timeout
                )
            except InvestigationJobUnavailable:
                current = jobs.get(run_id)
                if current is not None:
                    return InvestigationProcessResult(run_id, current.status, True)
                raise

        try:
            with self._sessions() as session:
                request = InvestigationRequestRepository(session).get(job.decision_id)
                score = ScoreObservationRepository(session).get_for_decision(
                    job.decision_id
                )
                probability = ProbabilityEstimateRepository(session).get_for_decision(
                    job.decision_id
                )
                decision = PolicyDecisionRepository(session).get(job.decision_id)
                if (
                    request is None
                    or score is None
                    or probability is None
                    or decision is None
                ):
                    raise ValueError(
                        "investigation job has incomplete decision lineage"
                    )
                events = NormalizedEventRepository(session).known_at(
                    request.cutoff_time
                )

            projection = self._graph.feature_projection_at(request.cutoff_time)
            evidence = EvidenceService(
                projection=projection,
                events=events,
                feature_service=FeatureService(projection),
                frozen_evaluation=self._frozen,
                config=self._config,
            )
            execution = self._agent_factory(self._config).run_execution(
                request=request, evidence_service=evidence, score_observation=score
            )
            completed = datetime.now(tz=UTC)
            with self._sessions.begin() as session:
                RuntimeLineagePersistenceService(session).persist_execution(
                    score_observation=score,
                    probability_estimate=probability,
                    policy_decision=decision,
                    investigation_request=request,
                    execution=execution,
                    run_id=run_id,
                    started_at=job.last_attempt_at or now,
                    completed_at=completed,
                    case_id=job.case_id,
                )
                InvestigationJobRepository(session).complete(run_id, claimed_at=now)
            return InvestigationProcessResult(
                run_id, InvestigationJobStatus.COMPLETED, False
            )
        except Exception as error:
            with (
                self._sessions.begin() as session,
                suppress(InvestigationJobUnavailable),
            ):
                InvestigationJobRepository(session).fail(
                    run_id, claimed_at=now, detail=_failure_detail(error)
                )
            return InvestigationProcessResult(
                run_id, InvestigationJobStatus.FAILED, False
            )


def _failure_detail(error: Exception) -> str:
    """Keep provider/runtime diagnostics bounded and free of tracebacks."""
    return (f"{type(error).__name__}: {error}".strip() or "investigation failed")[:1000]


def _new_agent(config: InvestigationConfig) -> InvestigationAgentService:
    return InvestigationAgentService(config=config)
