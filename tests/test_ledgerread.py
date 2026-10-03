"""A run's ledger rows read back canonical: a fork's rows under the same run id stay fenced off."""

import uuid
from contextlib import closing

import psycopg
import pytest
from _durable import DSN, pg_ready

from effective.keys import Segment, compose_key
from effective.ledger import PostgresLedger
from effective.ledgerread import pg_payloads, sqlite_payloads
from effective.ops import LedgerRow
from effective.sqlite import SqliteLedger


def _row(kind: str, run_id: str) -> LedgerRow:
    return LedgerRow(event_id=compose_key(t"entry:{Segment(kind)},{Segment(run_id)}"), kind=kind)


def test_sqlite_reads_the_canonical_rows_in_append_order(sqlite_app):
    app = sqlite_app()
    run_id = "r1"
    for kind, hypothetical in (("first", False), ("forked", True), ("second", False)):
        ledger = SqliteLedger(app.conn, run_id, app.write_lock, hypothetical=hypothetical)
        ledger.append(_row(kind, run_id))
    assert [row["kind"] for row in sqlite_payloads(app.conn, run_id)] == ["first", "second"]


def test_sqlite_reads_no_rows_from_a_store_with_no_ledger(tmp_path):
    import sqlite3

    with closing(sqlite3.connect(tmp_path / "empty.db")) as conn:
        assert sqlite_payloads(conn, "r1") == ()


@pytest.mark.skipif(not pg_ready(), reason="needs the test Postgres (just pgt-up)")
def test_postgres_reads_the_canonical_rows_in_append_order():
    run_id = str(uuid.uuid4())
    for kind, hypothetical in (("first", False), ("forked", True), ("second", False)):
        ledger = PostgresLedger(DSN, workflow_run_id=run_id, hypothetical=hypothetical)
        ledger.append(_row(kind, run_id))
        ledger.close()
    with psycopg.connect(DSN) as conn:
        assert [row["kind"] for row in pg_payloads(conn, run_id)] == ["first", "second"]
