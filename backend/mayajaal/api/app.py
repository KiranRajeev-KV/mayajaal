"""Thin, read-only FastAPI application for operational cases and investigations."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import datetime
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
    create_database_runtime,
    ping_database,
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


def create_app(database_runtime: DatabaseRuntime | None = None) -> FastAPI:
    """Create the operational read API with one shared runtime per lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        runtime = database_runtime
        owns_runtime = runtime is None
        if runtime is None:
            runtime = create_database_runtime(DatabaseConfig.from_environment())
        app.state.database_runtime = runtime
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

    _ = (
        health,
        list_cases,
        get_case,
        list_case_investigations,
        get_investigation,
        get_investigation_report,
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


def _not_found(resource: str) -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"{resource} not found"
    )
