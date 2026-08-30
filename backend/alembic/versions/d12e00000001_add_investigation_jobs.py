"""add durable Stage 12E investigation jobs

Revision ID: d12e00000001
Revises: d12d00000001
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d12e00000001"
down_revision: str | Sequence[str] | None = "d12d00000001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "investigation_jobs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("decision_id", sa.String(length=64), nullable=False),
        sa.Column("case_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_detail", sa.String(length=1000), nullable=True),
        sa.ForeignKeyConstraint(["case_id"], ["risk_cases.case_id"]),
        sa.ForeignKeyConstraint(
            ["decision_id"], ["investigation_requests.decision_id"]
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(
        "ix_investigation_jobs_status_created_at",
        "investigation_jobs",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_investigation_jobs_status_created_at", table_name="investigation_jobs"
    )
    op.drop_table("investigation_jobs")
