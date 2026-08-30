"""add Stage 12D durable scoring failures

Revision ID: d12d00000001
Revises: d12c00000002
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d12d00000001"
down_revision: str | Sequence[str] | None = "d12c00000002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "risk_processing_failures",
        sa.Column(
            "provider_event_id",
            sa.String(length=255),
            sa.ForeignKey("webhook_events.provider_event_id"),
            primary_key=True,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failure_detail", sa.String(length=1000), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("risk_processing_failures")
