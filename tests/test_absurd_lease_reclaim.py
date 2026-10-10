"""Expired-lease reclaim on the deployed engine: the recovery path, tested.

A worker that dies mid-task leaves its row `running` with a `claim_expires_at` in the past.
Nothing re-queues it explicitly: Absurd's `claim_task` sweeps expired leases as part of the next
claim, which is the whole recovery story for a crashed worker. That is the property the durable
substrate rests on when a pod is evicted, and it was asserted nowhere — the crash tests kill the
worker between ops and drive a FRESH one, which exercises replay rather than reclaim.
"""

import uuid

import psycopg
import pytest
from _durable import DSN, absurd, pg_ready

from effective.api import step
from effective.domain import CallTool
from effective.engines.absurd import ConcurrentAbsurdCtx
from effective.handlers.durable import DurableHandler

pytestmark = pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")


class _Domain:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op):
        self.calls.append(op.name)
        return "ok"


def _wf():
    yield from step("tool:a", CallTool(name="a", result_schema=str))
    yield from step("tool:b", CallTool(name="b", result_schema=str))
    return "done"


@pytest.fixture
def conn():
    c = psycopg.connect(DSN, autocommit=True)
    yield c
    c.close()


def test_an_expired_lease_is_reclaimed_and_the_run_completes(conn):
    """Strand the task with a dead worker's lease, then prove the next claim takes it."""
    app = absurd()
    domain = _Domain()
    name = f"lease-{uuid.uuid4().hex[:8]}"

    @app.register_task(name, default_max_attempts=5)
    def task(params, ctx):
        return DurableHandler(ConcurrentAbsurdCtx(ctx), domain).run(_wf)

    task_id = app.spawn(name, {"run_id": "r1"})

    # A worker that claimed and then died: `claim_tasks` takes the lease and we never execute,
    # which is the real shape (`work_batch` would claim AND run it to completion). The lease
    # lives on the RUN row — `t_default` carries only `last_attempt_run`, the pointer to it.
    claimed = app.app.claim_tasks(batch_size=1, worker_id="dead-worker")
    assert claimed, "the dead worker must actually hold a lease for this to test anything"
    conn.execute(
        t"UPDATE absurd.r_default SET claim_expires_at = now() - interval '1 hour' "
        t"WHERE task_id = {task_id}::uuid AND state = 'running'"
    )
    stranded = conn.execute(
        t"SELECT state, claim_expires_at < now() FROM absurd.r_default "
        t"WHERE task_id = {task_id}::uuid ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    assert stranded == ("running", True), stranded  # running, lease elapsed, nobody coming

    for _ in range(24):
        snap = app.fetch_task_result(task_id)
        if snap is not None and snap.state in ("completed", "failed"):
            break
        app.work_batch()

    snap = app.fetch_task_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap  # reclaimed, not stranded
    assert snap.result == "done"
    assert domain.calls == ["a", "b"], "the reclaiming worker ran the whole workflow"
