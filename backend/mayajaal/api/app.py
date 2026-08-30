"""Operational FastAPI application with post-commit realtime orchestration."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, NoReturn, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import Field
from sqlalchemy.orm import Session

from mayajaal.investigation import (
    InvestigationPattern,
    InvestigationRequest,
    InvestigationStatus,
)
from mayajaal.policy import PolicyAction
from mayajaal.schemas import Event, EventType
from mayajaal.schemas.common import SchemaModel

from .contracts import (
    InvestigationJob,
    InvestigationJobStatus,
    InvestigationRun,
    RiskCase,
    RiskCaseStatus,
)
from .db import (
    DatabaseRuntime,
    InvestigationJobRepository,
    InvestigationReportRepository,
    InvestigationRequestRepository,
    InvestigationRunRepository,
    NormalizedEventRepository,
    PolicyDecisionRepository,
    ProbabilityEstimateRepository,
    RiskCaseRepository,
    RiskEvaluationRepository,
    RiskProcessingFailureRepository,
    ScoreObservationRepository,
    SessionFactory,
    WebhookEventRepository,
    WebhookPayloadConflict,
    ping_database,
)
from .env import load_environment
from .realtime_pipeline import RealtimePipelineState, RealtimeRiskPipelineService
from .runtime import RealtimeApplicationRuntime, create_realtime_application_runtime
from .webhooks import (
    RazorpayWebhookEnvelope,
    WebhookConfig,
    WebhookInboxService,
    WebhookIngestResult,
    WebhookProcessingStatus,
    verify_razorpay_signature,
    webhook_record_status,
)

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 100
LOGGER = logging.getLogger(__name__)


class HealthResponse(SchemaModel):
    """Small readiness response without connection or secret details."""

    status: str


class CaseResponse(SchemaModel):
    """Stable public projection of an operational risk case."""

    case_id: str
    subject_type: str
    subject_id: str
    status: RiskCaseStatus
    opened_at: datetime
    closed_at: datetime | None
    opening_decision_id: str

    @classmethod
    def from_case(cls, value: RiskCase) -> "CaseResponse":
        return cls(
            case_id=value.case_id,
            subject_type=value.subject_type.value,
            subject_id=value.subject_id,
            status=value.status,
            opened_at=value.opened_at,
            closed_at=value.closed_at,
            opening_decision_id=value.opening_decision_id,
        )


class CaseListResponse(SchemaModel):
    """Bounded deterministic page of cases."""

    items: tuple[CaseResponse, ...]
    limit: int = Field(ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(ge=0)


class DecisionResponse(SchemaModel):
    """Bounded frontend projection of immutable decision lineage."""

    decision_id: str
    subject_id: str
    scoring_cutoff: datetime
    raw_model_score: float
    calibrated_probability: float
    policy_action: PolicyAction
    decision_margin_paise: float
    decision_is_stable_across_scenarios: bool
    context_id: str | None

    @classmethod
    def from_decision(cls, value: object) -> "DecisionResponse":
        from mayajaal.policy import PolicyDecision

        if not isinstance(value, PolicyDecision):
            raise TypeError("decision response requires a policy decision")
        return cls(
            decision_id=value.decision_id,
            subject_id=value.subject_id,
            scoring_cutoff=value.scoring_cutoff,
            raw_model_score=value.raw_model_score,
            calibrated_probability=value.calibrated_fraud_probability,
            policy_action=value.chosen_action,
            decision_margin_paise=value.decision_margin_paise,
            decision_is_stable_across_scenarios=value.decision_is_stable_across_scenarios,
            context_id=value.context.context_id,
        )


class InvestigationRunResponse(SchemaModel):
    """Stable public projection of one operational execution attempt."""

    run_id: str
    decision_id: str
    case_id: str | None
    investigation_id: str
    agent_model_id: str
    status: InvestigationStatus
    started_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_run(cls, value: InvestigationRun) -> "InvestigationRunResponse":
        return cls(
            run_id=value.run_id,
            decision_id=value.decision_id,
            case_id=value.case_id,
            investigation_id=value.investigation_id,
            agent_model_id=value.agent_model_id,
            status=value.status,
            started_at=value.started_at,
            completed_at=value.completed_at,
        )


class InvestigationRunListResponse(SchemaModel):
    """Bounded deterministic page of investigation runs."""

    items: tuple[InvestigationRunResponse, ...]
    limit: int = Field(ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(ge=0)


class InvestigationTriggerRequest(SchemaModel):
    """The frontend selects immutable decision lineage, never a free-form prompt."""

    decision_id: str = Field(min_length=1, max_length=64)


class InvestigationTriggerResponse(SchemaModel):
    run_id: str
    case_id: str
    decision_id: str
    status: InvestigationJobStatus


class InvestigationJobResponse(SchemaModel):
    """Polling shape that remains truthful before a report exists."""

    run_id: str
    case_id: str
    decision_id: str
    status: InvestigationJobStatus
    created_at: datetime
    last_attempt_at: datetime | None
    failure_detail: str | None
    report_status: InvestigationStatus | None = None

    @classmethod
    def from_job(
        cls, value: InvestigationJob, completed: InvestigationRun | None
    ) -> "InvestigationJobResponse":
        return cls(
            run_id=value.run_id,
            case_id=value.case_id,
            decision_id=value.decision_id,
            status=value.status,
            created_at=value.created_at,
            last_attempt_at=value.last_attempt_at,
            failure_detail=value.failure_detail,
            report_status=None if completed is None else completed.status,
        )


class InvestigationReportResponse(SchemaModel):
    """Read-only public projection of a verified report, excluding raw JSONB."""

    report_id: str
    run_id: str
    investigation_id: str
    decision_id: str
    policy_action: PolicyAction
    status: InvestigationStatus
    pattern: InvestigationPattern
    evidence_ids: tuple[str, ...]
    summary: str | None
    limitations: tuple[str, ...]


class WebhookEventResponse(SchemaModel):
    """Safe read-only operational projection; it deliberately omits raw bytes."""

    provider_event_id: str
    provider: str
    event_type: str
    provider_created_at: datetime
    received_at: datetime
    raw_body_sha256: str
    status: WebhookProcessingStatus

    @classmethod
    def from_record(cls, value: object) -> "WebhookEventResponse":
        # A narrow structural boundary avoids exposing persistence payloads in
        # the public API while keeping this response decoupled from ORM internals.
        from .db import WebhookEventRecord

        if not isinstance(value, WebhookEventRecord):
            raise TypeError("webhook response requires a webhook event record")
        return cls(
            provider_event_id=value.provider_event_id,
            provider=value.provider,
            event_type=value.event_type,
            provider_created_at=value.provider_created_at,
            received_at=value.received_at,
            raw_body_sha256=value.raw_body_sha256,
            status=webhook_record_status(value),
        )


class WebhookEventListResponse(SchemaModel):
    """Bounded deterministic page of durable webhook inbox records."""

    items: tuple[WebhookEventResponse, ...]
    limit: int = Field(ge=1, le=MAX_PAGE_LIMIT)
    offset: int = Field(ge=0)


class WebhookPipelineResultResponse(SchemaModel):
    """Trusted event outcome for the frontend; no raw webhook payload leaks."""

    provider_event_id: str
    processing_status: WebhookProcessingStatus
    pipeline_state: RealtimePipelineState
    canonical_event_type: str | None
    decision_id: str | None
    case_id: str | None
    policy_action: PolicyAction | None
    calibrated_probability: float | None


def create_app(
    database_runtime: DatabaseRuntime | None = None,
    webhook_config: WebhookConfig | None = None,
    realtime_runtime: RealtimeApplicationRuntime | None = None,
) -> FastAPI:
    """Create the operational read API with one shared runtime per lifespan."""
    load_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        operational = realtime_runtime
        if operational is None:
            operational = create_realtime_application_runtime(database=database_runtime)
        app.state.realtime_runtime = operational
        app.state.database_runtime = operational.database
        app.state.realtime_tasks = set()
        app.state.investigation_tasks = set()
        app.state.webhook_config = (
            webhook_config
            if webhook_config is not None
            else WebhookConfig.from_environment()
        )
        try:
            yield
        finally:
            tasks = cast(set[asyncio.Task[None]], app.state.realtime_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            investigation_tasks = cast(
                set[asyncio.Task[None]], app.state.investigation_tasks
            )
            if investigation_tasks:
                await asyncio.gather(*investigation_tasks, return_exceptions=True)
            if realtime_runtime is None:
                operational.dispose()

    app = FastAPI(title="Mayajaal operational API", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        try:
            ping_database(_runtime_from_request(request).engine)
            _realtime_runtime_from_request(request).graph.verify_connectivity()
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="required dependency is not ready",
            ) from error
        return HealthResponse(status="ready")

    @app.get("/cases", response_model=CaseListResponse)
    def list_cases(
        session: Annotated[Session, Depends(get_session)],
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> CaseListResponse:
        cases = RiskCaseRepository(session).list_recent(limit=limit, offset=offset)
        return CaseListResponse(
            items=tuple(CaseResponse.from_case(value) for value in cases),
            limit=limit,
            offset=offset,
        )

    @app.get("/cases/{case_id}", response_model=CaseResponse)
    def get_case(
        case_id: str, session: Annotated[Session, Depends(get_session)]
    ) -> CaseResponse:
        case = RiskCaseRepository(session).get(case_id)
        if case is None:
            _not_found("case")
        assert case is not None
        return CaseResponse.from_case(case)

    @app.get("/decisions/{decision_id}", response_model=DecisionResponse)
    def get_decision(
        decision_id: str, session: Annotated[Session, Depends(get_session)]
    ) -> DecisionResponse:
        decision = PolicyDecisionRepository(session).get(decision_id)
        if decision is None:
            _not_found("decision")
        assert decision is not None
        return DecisionResponse.from_decision(decision)

    @app.get(
        "/cases/{case_id}/investigations",
        response_model=InvestigationRunListResponse,
    )
    def list_case_investigations(
        case_id: str,
        session: Annotated[Session, Depends(get_session)],
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> InvestigationRunListResponse:
        if RiskCaseRepository(session).get(case_id) is None:
            _not_found("case")
        runs = InvestigationRunRepository(session).list_for_case(
            case_id, limit=limit, offset=offset
        )
        return InvestigationRunListResponse(
            items=tuple(InvestigationRunResponse.from_run(value) for value in runs),
            limit=limit,
            offset=offset,
        )

    @app.get("/investigations/{run_id}", response_model=InvestigationJobResponse)
    def get_investigation(
        run_id: str, session: Annotated[Session, Depends(get_session)]
    ) -> InvestigationJobResponse:
        job = InvestigationJobRepository(session).get(run_id)
        if job is None:
            _not_found("investigation")
        assert job is not None
        return InvestigationJobResponse.from_job(
            job, InvestigationRunRepository(session).get(run_id)
        )

    @app.post(
        "/cases/{case_id}/investigations",
        response_model=InvestigationTriggerResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def trigger_investigation(
        case_id: str,
        body: InvestigationTriggerRequest,
        request: Request,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> InvestigationTriggerResponse:
        """Durably enqueue a manual investigation before any model work begins."""
        key = _validated_idempotency_key(idempotency_key)
        run_id = str(uuid4())
        try:
            job, created = await run_in_threadpool(
                _enqueue_investigation_sync,
                _realtime_runtime_from_request(request),
                run_id,
                case_id,
                body.decision_id,
                key,
            )
        except LookupError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            ) from error
        if created:
            task = asyncio.create_task(
                _run_investigation(
                    _realtime_runtime_from_request(request).investigations, job.run_id
                )
            )
            tasks = cast(set[asyncio.Task[None]], request.app.state.investigation_tasks)
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        return InvestigationTriggerResponse(
            run_id=job.run_id,
            case_id=job.case_id,
            decision_id=job.decision_id,
            status=job.status,
        )

    @app.get(
        "/investigations/{run_id}/report", response_model=InvestigationReportResponse
    )
    def get_investigation_report(
        run_id: str, session: Annotated[Session, Depends(get_session)]
    ) -> InvestigationReportResponse:
        report = InvestigationReportRepository(session).get_for_run(run_id)
        if report is None:
            _not_found("investigation report")
        assert report is not None
        value = report.report
        return InvestigationReportResponse(
            report_id=report.report_id,
            run_id=report.run_id,
            investigation_id=report.investigation_id,
            decision_id=value.request.decision_id,
            policy_action=value.policy_action,
            status=value.status,
            pattern=value.pattern,
            evidence_ids=value.evidence_ids,
            summary=value.summary,
            limitations=value.limitations,
        )

    @app.post("/webhooks/razorpay", response_model=WebhookIngestResult)
    async def receive_razorpay_webhook(
        request: Request,
    ) -> WebhookIngestResult:
        """Durably accept one signature-verified Razorpay-shaped delivery only."""
        raw_body = await request.body()
        signature = request.headers.get("X-Razorpay-Signature")
        provider_event_id = request.headers.get("x-razorpay-event-id")
        if not _is_valid_provider_event_id(provider_event_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="missing event ID"
            )
        assert provider_event_id is not None
        secret = _webhook_config_from_request(
            request
        ).razorpay_webhook_secret.get_secret_value()
        if not verify_razorpay_signature(
            raw_body=raw_body, signature=signature, secret=secret
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature"
            )
        try:
            envelope = RazorpayWebhookEnvelope.model_validate_json(raw_body)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid webhook envelope",
            ) from error
        try:
            result = await run_in_threadpool(
                _accept_webhook_sync,
                _runtime_from_request(request).sessions,
                provider_event_id,
                envelope,
                raw_body,
            )
        except WebhookPayloadConflict as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="conflicting duplicate delivery",
            ) from error
        pipeline = _pipeline_from_request(request, required=False)
        if pipeline is not None:
            # The inbox transaction above has returned, so this task can never
            # make an accepted delivery disappear. It owns fresh sessions.
            task = asyncio.create_task(_run_pipeline(pipeline, provider_event_id))
            tasks = cast(set[asyncio.Task[None]], request.app.state.realtime_tasks)
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        return result

    @app.get("/webhooks/events", response_model=WebhookEventListResponse)
    def list_webhook_events(
        session: Annotated[Session, Depends(get_session)],
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> WebhookEventListResponse:
        records = WebhookEventRepository(session).list_recent(
            limit=limit, offset=offset
        )
        return WebhookEventListResponse(
            items=tuple(WebhookEventResponse.from_record(record) for record in records),
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/webhooks/events/{provider_event_id}", response_model=WebhookEventResponse
    )
    def get_webhook_event(
        provider_event_id: str,
        session: Annotated[Session, Depends(get_session)],
    ) -> WebhookEventResponse:
        record = WebhookEventRepository(session).get(provider_event_id)
        if record is None:
            _not_found("webhook event")
        assert record is not None
        return WebhookEventResponse.from_record(record)

    @app.get(
        "/webhooks/events/{provider_event_id}/result",
        response_model=WebhookPipelineResultResponse,
    )
    def get_webhook_pipeline_result(
        provider_event_id: str,
        session: Annotated[Session, Depends(get_session)],
    ) -> WebhookPipelineResultResponse:
        record = WebhookEventRepository(session).get(provider_event_id)
        if record is None:
            _not_found("webhook event")
        assert record is not None
        event = NormalizedEventRepository(session).get_for_provider(provider_event_id)
        evaluation = RiskEvaluationRepository(session).get(provider_event_id)
        failure = RiskProcessingFailureRepository(session).get(provider_event_id)
        decision = (
            None
            if evaluation is None
            else PolicyDecisionRepository(session).get(evaluation[0])
        )
        return WebhookPipelineResultResponse(
            provider_event_id=provider_event_id,
            processing_status=webhook_record_status(record),
            pipeline_state=_pipeline_state(record.status, event, evaluation, failure),
            canonical_event_type=None if event is None else event.event_type.value,
            decision_id=None if evaluation is None else evaluation[0],
            case_id=None if evaluation is None else evaluation[1],
            policy_action=None if decision is None else decision.chosen_action,
            calibrated_probability=(
                None if decision is None else decision.calibrated_fraud_probability
            ),
        )

    _ = (
        health,
        list_cases,
        get_case,
        get_decision,
        list_case_investigations,
        get_investigation,
        trigger_investigation,
        get_investigation_report,
        receive_razorpay_webhook,
        list_webhook_events,
        get_webhook_event,
        get_webhook_pipeline_result,
    )
    return app


def get_session(request: Request) -> Generator[Session, None, None]:
    """Yield one short-lived, non-committing SQLAlchemy session per request."""
    session = _runtime_from_request(request).sessions()
    try:
        yield session
    finally:
        session.close()


def _runtime_from_request(request: Request) -> DatabaseRuntime:
    try:
        return cast(DatabaseRuntime, request.app.state.database_runtime)
    except AttributeError as error:
        raise RuntimeError("database runtime is not initialized") from error


def _realtime_runtime_from_request(request: Request) -> RealtimeApplicationRuntime:
    try:
        return cast(RealtimeApplicationRuntime, request.app.state.realtime_runtime)
    except AttributeError as error:
        raise RuntimeError("realtime runtime is not initialized") from error


def _webhook_config_from_request(request: Request) -> WebhookConfig:
    try:
        return cast(WebhookConfig, request.app.state.webhook_config)
    except AttributeError as error:
        raise RuntimeError("webhook configuration is not initialized") from error


def _pipeline_from_request(
    request: Request, *, required: bool = True
) -> RealtimeRiskPipelineService | None:
    try:
        return cast(
            RealtimeApplicationRuntime, request.app.state.realtime_runtime
        ).pipeline
    except AttributeError:
        if required:
            raise RuntimeError("realtime pipeline is not initialized") from None
    return None


def _enqueue_investigation_sync(
    runtime: RealtimeApplicationRuntime,
    run_id: str,
    case_id: str,
    decision_id: str,
    idempotency_key: str,
) -> tuple[InvestigationJob, bool]:
    """Validate immutable lineage and commit the operational job in one scope."""
    with runtime.database.sessions.begin() as session:
        case_repository = RiskCaseRepository(session)
        if case_repository.get(case_id) is None:
            raise LookupError("case not found")
        if not case_repository.has_decision(case_id, decision_id):
            raise ValueError("decision does not belong to case")
        decision = PolicyDecisionRepository(session).get(decision_id)
        score = ScoreObservationRepository(session).get_for_decision(decision_id)
        estimate = ProbabilityEstimateRepository(session).get_for_decision(decision_id)
        if decision is None or score is None or estimate is None:
            raise ValueError("decision has incomplete trusted lineage")
        investigation_request = InvestigationRequest.from_policy_decision(
            decision, runtime.probability_model, score, estimate
        )
        InvestigationRequestRepository(session).persist(investigation_request)
        return InvestigationJobRepository(session).enqueue(
            InvestigationJob(
                run_id=run_id,
                decision_id=decision_id,
                case_id=case_id,
                idempotency_key=idempotency_key,
                status=InvestigationJobStatus.QUEUED,
                created_at=datetime.now(tz=UTC),
            )
        )


def _validated_idempotency_key(value: str | None) -> str:
    if value is None or not value.strip() or value != value.strip() or len(value) > 255:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Idempotency-Key must be a non-empty value up to 255 characters",
        )
    return value


def _is_valid_provider_event_id(value: str | None) -> bool:
    """Accept a provider ID only when it is nonempty, bounded, and unambiguous."""
    return (
        value is not None
        and bool(value.strip())
        and value == value.strip()
        and len(value) <= 255
    )


def _accept_webhook_sync(
    sessions: SessionFactory,
    provider_event_id: str,
    envelope: RazorpayWebhookEnvelope,
    raw_body: bytes,
) -> WebhookIngestResult:
    """Run synchronous SQLAlchemy/psycopg work in FastAPI's worker threadpool."""
    with sessions.begin() as session:
        return WebhookInboxService(session).accept(
            provider_event_id=provider_event_id,
            envelope=envelope,
            raw_body=raw_body,
            received_at=datetime.now(tz=UTC),
        )


async def _run_pipeline(
    pipeline: RealtimeRiskPipelineService, provider_event_id: str
) -> None:
    """Isolate best-effort post-commit work from the accepted webhook response."""
    try:
        await run_in_threadpool(pipeline.process, provider_event_id)
    except Exception:
        LOGGER.exception("realtime webhook pipeline failed for %s", provider_event_id)


async def _run_investigation(service: object, run_id: str) -> None:
    """Best-effort post-commit manual work; durable jobs support recovery."""
    from .investigations import InvestigationExecutionService

    if not isinstance(service, InvestigationExecutionService):
        raise TypeError("investigation runtime service is invalid")
    try:
        await run_in_threadpool(service.process, run_id)
    except Exception:
        LOGGER.exception("investigation job failed for %s", run_id)


def _not_found(resource: str) -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"{resource} not found"
    )


def _pipeline_state(
    webhook_status: str,
    event: Event | None,
    evaluation: tuple[str, str | None] | None,
    failure: object | None,
) -> RealtimePipelineState:
    if evaluation is not None:
        return RealtimePipelineState.SCORED
    if failure is not None:
        return RealtimePipelineState.SCORING_FAILED
    if event is not None and event.event_type is EventType.ACCOUNT_CREATED:
        return RealtimePipelineState.SETUP
    if webhook_status == WebhookProcessingStatus.FAILED.value:
        return RealtimePipelineState.WEBHOOK_FAILED
    return RealtimePipelineState.PROCESSING
