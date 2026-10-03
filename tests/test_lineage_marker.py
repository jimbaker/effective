"""The `hypothetical` lineage marker + the `Forked` genesis.

The marker fences a counterfactual off from the canonical record: a fork lives in the SAME
ledger as the audit trail but every row carries `hypothetical=True`, so the default projection is
`WHERE NOT hypothetical`: one predicate, no join. These pins cover the marker
on both engines, the genesis event's provenance, and the two properties that make the fence sound:
a hypothetical row is itself append-only (a fork cannot be laundered into the record by flipping
the flag), and a projection rebuilt over a ledger containing hypotheticals is unchanged.

The Postgres tests need the pinned test container (`just pgt-up` + `alembic upgrade head`); they
skip without it. The genesis-event and SQLite tests are infra-free.
"""

import os

import psycopg
import pytest

from effective.counterfactual import FORKED_KIND, Forked, genesis_row
from effective.keys import Key
from effective.ops import LedgerRow

DSN = os.environ.get("DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective")


def _pg() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2) as c:
            c.execute("SELECT hypothetical FROM ledger LIMIT 0")  # column present == migrated
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not _pg(), reason="no migrated test Postgres (pgt-up + alembic)")


# --- the genesis event (infra-free) --------------------------------------------------------


def test_the_genesis_records_its_provenance():
    row = genesis_row(
        "r-fork-0",
        forked_from="r-base",
        forked_at_event="reviewed:m1",
        delta={"decision": "reject"},
    )
    assert row.kind == FORKED_KIND
    assert (
        row.event_id.stored() == "forked:r-fork-0"
    )  # deterministic, so a crash-replay is idempotent
    assert row.get("forked_from") == "r-base"
    assert row.get("forked_at_event") == "reviewed:m1"  # the event address, not an op index
    assert row.get("delta") == {"decision": "reject"}
    assert row.get("at_op_index") is None  # the trace SEEK is the driver's job


def test_the_genesis_carries_the_resolved_all_ops_seek_when_given_one():
    """`at_op_index` is an all-ops index — the convention for the fork's `at`."""
    row = genesis_row("r-fork-0", forked_from="r-base", forked_at_event="e1", at_op_index=4)
    assert row.get("at_op_index") == 4
    assert Forked.model_validate_json(row.model_dump_json()).at_op_index == 4


def test_the_genesis_id_is_deterministic_per_child_run():
    a = genesis_row("r-fork-7", forked_from="r-base", forked_at_event="e1")
    b = genesis_row("r-fork-7", forked_from="r-base", forked_at_event="e1")
    assert a.event_id == b.event_id  # idempotent append by event_id


# --- the marker on Postgres ----------------------------------------------------------------


@pg
def test_a_canonical_writer_marks_rows_not_hypothetical():
    from uuid import uuid4

    from effective.ledger import PostgresLedger

    rid = f"canon-{uuid4().hex[:8]}"
    ledger = PostgresLedger(DSN, workflow_run_id=rid)
    try:
        ledger.append(LedgerRow(event_id=Key.parse(f"{rid}:e1"), kind="request_processed"))
    finally:
        ledger.close()
    with psycopg.connect(DSN) as c:
        rows = c.execute(
            "SELECT hypothetical FROM ledger WHERE workflow_run_id = %s", (rid,)
        ).fetchall()
    assert rows == [(False,)]


@pg
def test_a_hypothetical_writer_marks_every_row_and_writes_a_genesis():
    from uuid import uuid4

    from effective.ledger import PostgresLedger

    base, fork = f"base-{uuid4().hex[:8]}", None
    fork = f"fork-{uuid4().hex[:8]}"
    ledger = PostgresLedger(DSN, workflow_run_id=fork, hypothetical=True)
    try:
        ledger.append(genesis_row(fork, forked_from=base, forked_at_event=f"{base}:e1"))
        ledger.append(LedgerRow(event_id=Key.parse(f"{fork}:tail"), kind="request_processed"))
    finally:
        ledger.close()
    with psycopg.connect(DSN) as c:
        rows = c.execute(
            "SELECT kind, hypothetical FROM ledger WHERE workflow_run_id = %s ORDER BY seq",
            (fork,),
        ).fetchall()
    assert rows == [(FORKED_KIND, True), ("request_processed", True)]  # whole lineage marked


@pg
def test_a_hypothetical_row_is_ITSELF_append_only():
    """The fence is sound only if a fork cannot be laundered into the record by flipping the flag.
    The append-only trigger covers hypothetical rows too — an UPDATE is blocked."""
    from uuid import uuid4

    from effective.ledger import PostgresLedger

    rid = f"launder-{uuid4().hex[:8]}"
    ledger = PostgresLedger(DSN, workflow_run_id=rid, hypothetical=True)
    try:
        ledger.append(LedgerRow(event_id=Key.parse(f"{rid}:e1"), kind="request_processed"))
    finally:
        ledger.close()
    with psycopg.connect(DSN) as c, pytest.raises(psycopg.errors.RaiseException):
        c.execute("UPDATE ledger SET hypothetical = false WHERE workflow_run_id = %s", (rid,))


@pg
def test_a_projection_rebuild_ignores_hypothetical_rows():
    """A projection rebuilt over a ledger that CONTAINS a fork is unchanged: the canonical view
    never sees the counterfactual. Two runs of the same shape, one canonical, one hypothetical;
    the canonical one rebuilds into a total, the fork rebuilds into nothing (its rows are all
    filtered, so `rebuild_total` sees an empty lineage and returns None)."""
    from decimal import Decimal
    from uuid import uuid4

    from _approval_domain import processed_row, rebuild_total
    from sqlmodel import Session, create_engine

    from effective.ledger import PostgresLedger, to_sqlalchemy_url

    canon = f"c-{uuid4().hex[:8]}"
    fork = f"f-{uuid4().hex[:8]}"

    real = PostgresLedger(DSN, workflow_run_id=canon)
    hyp = PostgresLedger(DSN, workflow_run_id=fork, hypothetical=True)
    try:
        real.append(processed_row(canon))
        hyp.append(genesis_row(fork, forked_from=canon, forked_at_event=f"processed:{canon}"))
        hyp.append(processed_row(fork))
    finally:
        real.close()
        hyp.close()

    engine = create_engine(to_sqlalchemy_url(DSN))
    try:
        with Session(engine) as session:
            assert rebuild_total(session, canon) == Decimal("4.50")  # canonical run rebuilt
            assert rebuild_total(session, fork) is None  # fork fenced off: empty lineage
    finally:
        engine.dispose()
