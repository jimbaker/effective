"""Integration tests for the DurableHandler against a real local Postgres + worker.

Gated on a reachable Postgres with the Absurd schema + ledger table (skipped
otherwise), so the infra-free suite stays green on any laptop. The domain
interpreter is a fake returning canned typed values — no real LLM. What we are
proving is the *durable execution*: checkpointed steps, JSON round-trip into
typed objects, durable HITL suspend/resume, and a dedicated append-only ledger.
"""

import os
from uuid import uuid4

import psycopg
import pytest
from _approval_domain import CannedDomain, process_refund, review_name
from psycopg.types.json import Json

from effective.handlers.durable import DurableHandler
from effective.ledger import PostgresLedger

DSN = os.environ.get("DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective")


def _pg_ready() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2) as conn:
            # require both the Absurd schema and the ledger table
            conn.execute("SELECT 1 FROM ledger LIMIT 0")
            conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='absurd'")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pg_ready(), reason="no local Postgres with Absurd + ledger")


def _absurd():
    from effective.absurd_worker import absurd_worker

    return absurd_worker(DSN)


def _run_until_result(app, task_id, max_batches: int = 12):
    for _ in range(max_batches):
        snap = app.fetch_task_result(task_id)
        if snap and snap.state in ("completed", "failed", "cancelled"):
            return snap
        app.work_batch()
    return app.fetch_task_result(task_id)


def _ledger_kinds(run_id: str) -> list[str]:
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            "SELECT kind FROM ledger WHERE workflow_run_id = %s ORDER BY seq", (run_id,)
        ).fetchall()
    return [r[0] for r in rows]


def test_auto_path_commits_and_writes_ledger():
    app = _absurd()
    domain = CannedDomain(amount="42.00")  # under threshold -> auto -> commit
    mid = f"auto-{uuid4().hex[:8]}"

    @app.register_task("it-refund-auto", default_max_attempts=3)
    def task(params, ctx):
        ledger = PostgresLedger(DSN, workflow_run_id=params["request_id"])
        try:
            return DurableHandler(ctx, domain, ledger=ledger).run(
                lambda: process_refund(params["request_id"])
            )
        finally:
            ledger.close()

    spawned = app.spawn("it-refund-auto", {"request_id": mid})
    snap = _run_until_result(app, spawned["task_id"])

    assert snap is not None
    assert snap.state == "completed"
    assert snap.result["status"] == "committed"
    assert domain.calls == ["fetch_request", "assess_request"]
    # the dedicated append-only ledger captured the decision trace
    assert _ledger_kinds(mid) == ["assessment", "commitment"]
    app.close()


def test_hitl_suspends_then_resumes_and_ledger_records_review():
    app = _absurd()
    domain = CannedDomain(amount="500.00")  # over threshold -> review -> awaits event
    mid = f"hitl-{uuid4().hex[:8]}"

    @app.register_task("it-refund-hitl", default_max_attempts=3)
    def task(params, ctx):
        ledger = PostgresLedger(DSN, workflow_run_id=params["request_id"])
        try:
            return DurableHandler(ctx, domain, ledger=ledger).run(
                lambda: process_refund(params["request_id"])
            )
        finally:
            ledger.close()

    spawned = app.spawn("it-refund-hitl", {"request_id": mid})

    app.work_batch()  # runs to the review await and suspends
    mid_snap = app.fetch_task_result(spawned["task_id"])
    if mid_snap is not None:
        assert mid_snap.state != "completed"  # parked at the event, not done yet
    assert _ledger_kinds(mid) == ["assessment"]  # only the pre-suspension event so far

    app.emit_event(review_name(mid).stored(), {"decision": "approve", "actor": "approver"})
    snap = _run_until_result(app, spawned["task_id"])

    assert snap is not None
    assert snap.state == "completed"
    assert snap.result["status"] == "committed"
    assert _ledger_kinds(mid) == ["assessment", "review", "commitment"]
    app.close()


def test_ledger_is_append_only():
    eid = f"trig-{uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO ledger (event_id, kind, payload) VALUES (%s, %s, %s)",
            (eid, "test", Json({"x": 1})),
        )
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute(t"UPDATE ledger SET kind = 'mutated' WHERE event_id = {eid}")
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute(t"DELETE FROM ledger WHERE event_id = {eid}")
