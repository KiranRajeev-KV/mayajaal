"""add operational case investigation runs

Revision ID: 57142160cad9
Revises: c263e0cacd3d
Create Date: 2026-08-29 13:12:32.706826
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "57142160cad9"
down_revision: str | Sequence[str] | None = "c263e0cacd3d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add cases/runs and replace the disposable Stage 11B report shape."""
    op.create_table(
        "risk_cases",
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("subject_type", sa.String(length=32), nullable=False),
        sa.Column("subject_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("opening_decision_id", sa.String(length=64), nullable=False),
        sa.Column("payload", _json_payload(), nullable=False),
        sa.ForeignKeyConstraint(
            ["opening_decision_id"],
            ["policy_decisions.decision_id"],
            name=op.f("fk_risk_cases_opening_decision_id_policy_decisions"),
        ),
        sa.PrimaryKeyConstraint("case_id", name=op.f("pk_risk_cases")),
    )
    op.create_index(
        "ix_risk_cases_status_opened_at",
        "risk_cases",
        ["status", "opened_at"],
        unique=False,
    )
    op.create_index(
        "ix_risk_cases_subject_id_opened_at",
        "risk_cases",
        ["subject_id", "opened_at"],
        unique=False,
    )
    op.create_table(
        "risk_case_decisions",
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("decision_id", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["risk_cases.case_id"],
            name=op.f("fk_risk_case_decisions_case_id_risk_cases"),
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["policy_decisions.decision_id"],
            name=op.f("fk_risk_case_decisions_decision_id_policy_decisions"),
        ),
        sa.PrimaryKeyConstraint(
            "case_id", "decision_id", name=op.f("pk_risk_case_decisions")
        ),
    )
    op.create_table(
        "investigation_runs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("decision_id", sa.String(length=64), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=True),
        sa.Column("investigation_id", sa.String(length=64), nullable=False),
        sa.Column("agent_model_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", _json_payload(), nullable=False),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["risk_cases.case_id"],
            name=op.f("fk_investigation_runs_case_id_risk_cases"),
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["investigation_requests.decision_id"],
            name=op.f("fk_investigation_runs_decision_id_investigation_requests"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_investigation_runs")),
    )
    op.create_index(
        "ix_investigation_runs_case_id_started_at",
        "investigation_runs",
        ["case_id", "started_at"],
        unique=False,
    )
    op.create_index(
        "ix_investigation_runs_decision_id_started_at",
        "investigation_runs",
        ["decision_id", "started_at"],
        unique=False,
    )

    # Stage 11B's report table is disposable hackathon development state. A
    # report cannot be migrated safely because it has no run/provenance ID;
    # re-create it rather than inventing a fake historical execution attempt.
    op.drop_table("investigation_reports")
    op.create_table(
        "investigation_reports",
        sa.Column("report_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("investigation_id", sa.String(length=64), nullable=False),
        sa.Column("policy_action", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("pattern", sa.String(length=64), nullable=False),
        sa.Column("payload", _json_payload(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["investigation_runs.run_id"],
            name=op.f("fk_investigation_reports_run_id_investigation_runs"),
        ),
        sa.PrimaryKeyConstraint("report_id", name=op.f("pk_investigation_reports")),
        sa.UniqueConstraint("run_id", name=op.f("uq_investigation_reports_run_id")),
    )


def downgrade() -> None:
    """Restore the Stage 11B schema; Stage 11C runtime rows are disposable."""
    op.drop_table("investigation_reports")
    op.create_table(
        "investigation_reports",
        sa.Column("decision_id", sa.String(length=64), nullable=False),
        sa.Column("policy_action", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("pattern", sa.String(length=64), nullable=False),
        sa.Column("payload", _json_payload(), nullable=False),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["investigation_requests.decision_id"],
            name=op.f("fk_investigation_reports_decision_id_investigation_requests"),
        ),
        sa.PrimaryKeyConstraint("decision_id", name=op.f("pk_investigation_reports")),
    )
    op.drop_index(
        "ix_investigation_runs_decision_id_started_at",
        table_name="investigation_runs",
    )
    op.drop_index(
        "ix_investigation_runs_case_id_started_at",
        table_name="investigation_runs",
    )
    op.drop_table("investigation_runs")
    op.drop_table("risk_case_decisions")
    op.drop_index("ix_risk_cases_subject_id_opened_at", table_name="risk_cases")
    op.drop_index("ix_risk_cases_status_opened_at", table_name="risk_cases")
    op.drop_table("risk_cases")


def _json_payload() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
