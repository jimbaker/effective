"""ledger: record WHO wrote each row (writer_task, writer_placement)

Two nullable columns so the store can tell a placed-writer collision — two ops the substrate
placed distinctly writing one `event_id`, where the second row is dropped by
`ON CONFLICT DO NOTHING` and the run reports success — from the two conflicts that are
CORRECT and must stay silent: a crash-window re-append, and the same message triaged in two
generations.

Additive and metadata-only, like `0002`. No DEFAULT and no backfill: an existing row genuinely
has no known writer, and NULL is that statement. `refuse_placed_writer_collision` reads unknown
as allow, so no historical row becomes a landmine and no table rewrite is needed.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ledger", sa.Column("writer_task", sa.String(), nullable=True))
    op.add_column("ledger", sa.Column("writer_placement", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("ledger", "writer_placement")
    op.drop_column("ledger", "writer_task")
