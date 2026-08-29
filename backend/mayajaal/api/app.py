"""Thin, read-only FastAPI application for operational cases and investigations."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, NoReturn, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from mayajaal.investigation import InvestigationPattern, InvestigationStatus
from mayajaal.policy import PolicyAction
from mayajaal.schemas.common import SchemaModel

from .contracts import InvestigationRun, RiskCase, RiskCaseStatus
from .db import (
    DatabaseConfig,
    DatabaseRuntime,
    InvestigationReportRepository,
    InvestigationRunRepository,
    RiskCaseRepository,
    WebhookEventRepository,
    WebhookPayloadConflict,
    create_database_runtime,
    ping_database,
)
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


def create_app(
    database_runtime: DatabaseRuntime | None = None,
    webhook_config: WebhookConfig | None = None,
) -> FastAPI:
    """Create the operational read API with one shared runtime per lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        runtime = database_runtime
        owns_runtime = runtime is None
        if runtime is None:
            runtime = create_database_runtime(DatabaseConfig.from_environment())
        app.state.database_runtime = runtime
        app.state.webhook_config = (
            webhook_config
            if webhook_config is not None
            else WebhookConfig.from_environment()
        )
        try:
            yield
        finally:
            if owns_runtime:
                runtime.dispose()

    app = FastAPI(title="Mayajaal operational API", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        try:
            ping_database(_runtime_from_request(request).engine)
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database is not ready",
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

    @app.get("/investigations/{run_id}", response_model=InvestigationRunResponse)
    def get_investigation(
        run_id: str, session: Annotated[Session, Depends(get_session)]
    ) -> InvestigationRunResponse:
        run = InvestigationRunRepository(session).get(run_id)
        if run is None:
            _not_found("investigation")
        assert run is not None
        return InvestigationRunResponse.from_run(run)

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
        session: Annotated[Session, Depends(get_session)],
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
            with session.begin():
                result = WebhookInboxService(session).accept(
                    provider_event_id=provider_event_id,
                    envelope=envelope,
                    raw_body=raw_body,
                    received_at=datetime.now(tz=UTC),
                )
        except WebhookPayloadConflict as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="conflicting duplicate delivery",
            ) from error
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

    _ = (
        health,
        list_cases,
        get_case,
        list_case_investigations,
        get_investigation,
        get_investigation_report,
        receive_razorpay_webhook,
        list_webhook_events,
        get_webhook_event,
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


def _webhook_config_from_request(request: Request) -> WebhookConfig:
    try:
        return cast(WebhookConfig, request.app.state.webhook_config)
    except AttributeError as error:
        raise RuntimeError("webhook configuration is not initialized") from error


def _is_valid_provider_event_id(value: str | None) -> bool:
    """Accept a provider ID only when it is nonempty, bounded, and unambiguous."""
    return (
        value is not None
        and bool(value.strip())
        and value == value.strip()
        and len(value) <= 255
    )


def _not_found(resource: str) -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"{resource} not found"
    )
