"""add idempotency keys to durable investigation jobs

Revision ID: d12e00000002
Revises: d12e00000001
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d12e00000002"
down_revision: str | Sequence[str] | None = "d12e00000001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "investigation_jobs",
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
    )
    # Existing pre-idempotency jobs remain distinct, completed operational work.
    op.execute(
        "UPDATE investigation_jobs SET idempotency_key = run_id "
        "WHERE idempotency_key IS NULL"
    )
    op.alter_column("investigation_jobs", "idempotency_key", nullable=False)
    op.create_unique_constraint(
        op.f("uq_investigation_jobs_case_id"),
        "investigation_jobs",
        ["case_id", "decision_id", "idempotency_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("uq_investigation_jobs_case_id"),
        "investigation_jobs",
        type_="unique",
    )
    op.drop_column("investigation_jobs", "idempotency_key")
