"""A run's canonical ledger rows, as stored, on either engine.

Canonical means `NOT hypothetical`: a fork's rows are fenced off from every reader of the record.
Rows come in append order (`seq`), payloads decoded to dicts, so a reader such as
`runview.run_view` sees the same value whichever engine answered.
"""

import json
import sqlite3
from typing import Any

import psycopg

from effective.sql import bind


def sqlite_payloads(conn: sqlite3.Connection, run_id: str) -> tuple[dict[str, Any], ...]:
    """The rows of `run_id`; a store with no ledger table has none."""
    try:
        rows = conn.execute(
            *bind(
                t"SELECT payload FROM ledger WHERE workflow_run_id={run_id} AND NOT hypothetical "
                t"ORDER BY seq"
            )
        ).fetchall()
    except sqlite3.OperationalError:
        return ()
    return tuple(json.loads(row[0]) for row in rows)


def pg_payloads(conn: psycopg.Connection[Any], run_id: str) -> tuple[dict[str, Any], ...]:
    """The rows of `run_id`; `jsonb` arrives decoded."""
    rows = conn.execute(
        t"SELECT payload FROM ledger WHERE workflow_run_id={run_id} AND NOT hypothetical "
        t"ORDER BY seq"
    ).fetchall()
    return tuple(row[0] for row in rows)
