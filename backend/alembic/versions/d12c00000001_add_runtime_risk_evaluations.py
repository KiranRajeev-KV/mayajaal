"""add Stage 12C decision-time inputs and event lineage

Revision ID: d12c00000001
Revises: a5b7d6e8f901
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "d12c00000001"
down_revision: str | Sequence[str] | None = "a5b7d6e8f901"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.create_table(
        "feature_vectors",
        sa.Column("feature_vector_id", sa.String(length=64), primary_key=True),
        sa.Column("account_id", sa.String(length=255), nullable=False),
        sa.Column("scoring_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", json_type, nullable=False),
    )
    op.create_table(
        "risk_evaluations",
        sa.Column(
            "provider_event_id",
            sa.String(length=255),
            sa.ForeignKey("webhook_events.provider_event_id"),
            primary_key=True,
        ),
        sa.Column(
            "decision_id",
            sa.String(length=64),
            sa.ForeignKey("policy_decisions.decision_id"),
            nullable=False,
        ),
        sa.Column(
            "case_id",
            sa.String(length=64),
            sa.ForeignKey("risk_cases.case_id"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("risk_evaluations")
    op.drop_table("feature_vectors")
