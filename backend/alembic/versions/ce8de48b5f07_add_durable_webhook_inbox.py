"""add durable webhook inbox

Revision ID: ce8de48b5f07
Revises: 57142160cad9
Create Date: 2026-08-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "ce8de48b5f07"
down_revision: str | Sequence[str] | None = "57142160cad9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create only the Stage 12A append-only provider delivery inbox."""
    op.create_table(
        "webhook_events",
        sa.Column("provider_event_id", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=255), nullable=False),
        sa.Column("provider_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_body", sa.LargeBinary(), nullable=False),
        sa.Column("raw_body_sha256", sa.String(length=64), nullable=False),
        sa.Column("payload", _json_payload(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_detail", sa.String(length=1000), nullable=True),
        sa.PrimaryKeyConstraint("provider_event_id", name=op.f("pk_webhook_events")),
    )
    op.create_index(
        "ix_webhook_events_received_at",
        "webhook_events",
        ["received_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the disposable Stage 12A inbox schema."""
    op.drop_index("ix_webhook_events_received_at", table_name="webhook_events")
    op.drop_table("webhook_events")


def _json_payload() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
