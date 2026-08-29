"""add normalized webhook events

Revision ID: a5b7d6e8f901
Revises: ce8de48b5f07
Create Date: 2026-08-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a5b7d6e8f901"
down_revision: str | Sequence[str] | None = "ce8de48b5f07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add only canonical webhook-event persistence and a short claim marker."""
    op.add_column(
        "webhook_events",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "normalized_events",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("provider_event_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("account_id", sa.String(length=36), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", _json_payload(), nullable=False),
        sa.ForeignKeyConstraint(
            ["provider_event_id"],
            ["webhook_events.provider_event_id"],
            name=op.f("fk_normalized_events_provider_event_id_webhook_events"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_normalized_events")),
        sa.UniqueConstraint(
            "provider_event_id", name=op.f("uq_normalized_events_provider_event_id")
        ),
    )
    op.create_index(
        "ix_normalized_events_account_id_occurred_at",
        "normalized_events",
        ["account_id", "occurred_at"],
        unique=False,
    )


def downgrade() -> None:
    """Return to the Stage 12A inbox-only schema."""
    op.drop_index(
        "ix_normalized_events_account_id_occurred_at", table_name="normalized_events"
    )
    op.drop_table("normalized_events")
    op.drop_column("webhook_events", "claimed_at")


def _json_payload() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
