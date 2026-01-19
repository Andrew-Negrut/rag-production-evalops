"""add chunk meta column

Revision ID: 8e4c2b3f9a21
Revises: 651bb092a953
Create Date: 2026-01-19 19:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "8e4c2b3f9a21"
down_revision: Union[str, None] = "651bb092a953"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add meta column with default so existing rows are populated.
    op.add_column(
        "chunks",
        sa.Column(
            "meta",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("chunks", "meta")
