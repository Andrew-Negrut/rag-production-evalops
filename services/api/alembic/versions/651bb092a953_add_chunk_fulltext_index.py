"""add chunk fulltext index

Revision ID: 651bb092a953
Revises: 2bd7adb7c007
Create Date: 2026-01-03 23:13:32.314457

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '651bb092a953'
down_revision: Union[str, None] = '2bd7adb7c007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Generated column for FTS
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS search_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    """)

    # GIN index for fast full-text search
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_chunks_search_tsv
        ON chunks USING GIN (search_tsv)
    """)

def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_search_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search_tsv")