"""enforce one open risk episode per subject

Revision ID: d12c00000002
Revises: d12c00000001
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d12c00000002"
down_revision: str | Sequence[str] | None = "d12c00000001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_risk_cases_open_subject",
        "risk_cases",
        ["subject_type", "subject_id"],
        unique=True,
        postgresql_where=sa.text("status = 'OPEN'"),
    )


def downgrade() -> None:
    op.drop_index("uq_risk_cases_open_subject", table_name="risk_cases")
