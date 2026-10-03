"""Append-only decision ledger — a deliberately light SQLModel table + writer.

The "two bookkeepers" split: Absurd checkpoints are disposable
execution state; this is the canonical, append-only semantic record. The table
is write-mostly (no Read/Create variants); its payload is JSONB, typed in Python
on read via the consuming domain's event union.

Appends are idempotent (``ON CONFLICT (event_id) DO NOTHING``); a trigger blocks
UPDATE/DELETE (see the Alembic migration). Imports SQLAlchemy/SQLModel, so it is
pulled only by the production/worker path — never by the recording/replay core.
"""

from datetime import datetime
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import BigInteger, Boolean, Column, DateTime, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import Field, SQLModel, create_engine

import effective.pgkeys  # noqa: F401  — registers the `Key` psycopg dumper
from effective.ops import LedgerRow, Writer, refuse_placed_writer_collision


def to_sqlalchemy_url(dsn: str) -> str:
    """``postgresql://`` (psql / absurd-sdk) -> ``postgresql+psycopg://`` (SQLAlchemy/psycopg3)."""
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://") :]
    return dsn


class LedgerEntry(SQLModel, table=True):
    __tablename__ = "ledger"

    seq: int | None = Field(
        default=None, sa_column=Column(BigInteger, primary_key=True, autoincrement=True)
    )
    event_id: str = Field(index=True, unique=True)
    kind: str
    workflow_run_id: str | None = Field(default=None, index=True)
    payload: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    recorded_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=text("now()"), nullable=False),
    )
    # A counterfactual (D) lineage is fenced off, not separated: every row of a hypothetical run
    # carries this flag (denormalized from the `forked` genesis, `effective.counterfactual`), so
    # the canonical audit view filters with one predicate — `WHERE NOT hypothetical` — and no join
    # (fork-note §4 / §10, the column over derive-by-join). The default projection is
    # canonical-only; a counterfactual view opts *in*. The flag is itself append-only (the
    # UPDATE/DELETE trigger covers hypothetical rows too), so a fork cannot be laundered into the
    # record by flipping it.
    hypothetical: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default=text("false")),
    )
    # WHO wrote the row (`effective.ops.Writer`). Both nullable, and the nullability is the
    # contract rather than a migration convenience: a direct writer append (a fork's genesis or
    # seal, which bypass `ctx.step`) genuinely has no placement, and every row written before
    # these columns existed has none either. The collision check reads unknown as ALLOW, so
    # neither becomes a landmine.
    writer_task: str | None = Field(default=None)
    writer_placement: str | None = Field(default=None)


class PostgresLedger:
    """Append-only writer over the ``ledger`` table. Satisfies the handler's
    ``LedgerWriter`` protocol (a single ``append(row)`` method).

    ``hypothetical`` marks the whole lineage: a counterfactual (D) run constructs its writer with
    ``hypothetical=True`` and every row it appends is fenced off from the canonical view. It is a
    property of the *writer*, not the row, because a fork is a whole sibling lineage — the flag is
    set once at construction and denormalized to each append (fork-note §4)."""

    def __init__(
        self, dsn: str, workflow_run_id: str | None = None, *, hypothetical: bool = False
    ) -> None:
        self._engine = create_engine(to_sqlalchemy_url(dsn))
        self.workflow_run_id = workflow_run_id
        self.hypothetical = hypothetical

    def append(self, row: LedgerRow, *, writer: Writer | None = None) -> None:
        table = LedgerEntry.__table__  # ty: ignore[unresolved-attribute]
        stmt = (
            pg_insert(table)
            .values(
                event_id=row.event_id,
                kind=row.kind,
                workflow_run_id=self.workflow_run_id,
                payload=to_jsonable_python(row),
                hypothetical=self.hypothetical,
                writer_task=None if writer is None else writer.task,
                writer_placement=None if writer is None else writer.placement.stored(),
            )
            .on_conflict_do_nothing(index_elements=["event_id"])
            .returning(table.c.event_id)
        )
        # One transaction, so the read-back cannot see a different world than the insert lost to
        # — the same rule `SqliteApp.spawn` states for its lock, spelled in the idiom this engine
        # has. A concurrent inserter blocks on the unique index until it commits, and the SELECT
        # then reads the committed winner.
        with self._engine.begin() as conn:
            if conn.execute(stmt).first() is not None:
                return
            held = conn.execute(
                select(table.c.writer_task, table.c.writer_placement).where(
                    table.c.event_id == str(row.event_id.stored())
                )
            ).first()
        refuse_placed_writer_collision(row.event_id, writer, None if held is None else tuple(held))

    def close(self) -> None:
        self._engine.dispose()
