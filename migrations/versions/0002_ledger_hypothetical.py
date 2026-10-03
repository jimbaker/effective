"""ledger.hypothetical — fence off counterfactual (D) lineages

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-23

A boolean marker so a counterfactual sibling lineage lives in the same ledger as the canonical
record but is filtered out of every default projection (`WHERE NOT hypothetical`), rather than
separated into another table: a denormalized column rather than a derive-by-join.

Safe under the append-only trigger: `ADD COLUMN … DEFAULT false` is DDL with a constant default,
a metadata-only change on Postgres 11+ (no per-row rewrite), so the `BEFORE UPDATE OR DELETE`
row-level trigger `ledger_no_mutate` does not fire. Existing rows read as `false`, the canonical
lineage, with no backfill.
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ledger",
        sa.Column(
            "hypothetical",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("ledger", "hypothetical")
