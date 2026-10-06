"""An attempt is one execution of a task, counted from 1 against the task's limit, on both engines.

Absurd counts this way, and the embedded engine conforms. A task body reads its attempt from its
ctx, and a spawn carries the limit its child runs under, so a spawned child's budget does not
depend on which engine enqueued it.
"""

import json
import os
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from _conformance import IMMEDIATE_RETRY, private
from _durable import DSN, pg_ready

from effective.api import call_tool
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, SpawnResult
from effective.handlers.absurd import DurableHandler, spawn_done_name
from effective.interpreters.tools import make_tool_runner, spawn_tool
from effective.keys import Key
from effective.ops import DONE_EVENT_PARAM, Writer


def crashing(times: int, seen: list[tuple[int, int | None, bool]]):
    """A body that records the attempt it runs as, and crashes on its first `times` executions."""

    def body(params, ctx):
        attempt = ctx.attempt
        seen.append((attempt.number, attempt.limit, attempt.final))
        if len(seen) <= times:
            raise ValueError("a crash")
        return "ok"

    return body


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_each_execution_is_numbered_from_one_against_its_limit(backend, failures):
    """Reddens if an engine counts failures where Absurd counts executions: the task that finishes
    after `failures` crashes ran `failures + 1` times, and each execution saw its own number."""
    name = private("counted")
    seen: list[tuple[int, int | None, bool]] = []
    backend.register_body(name, crashing(failures, seen))

    task = backend.spawn(name, str(uuid4()), max_attempts=3)
    snapshot = backend.run_until_result(task)

    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert seen == [(n, 3, n == 3) for n in range(1, failures + 2)]
    assert backend.task_attempts(task) == failures + 1


def test_a_task_that_crashes_to_death_ends_on_its_last_attempt(backend):
    """Reddens if the last execution is not the one numbered by the limit: the body must be able to
    tell, while it runs, that no retry follows."""
    name = private("dying")
    seen: list[tuple[int, int | None, bool]] = []
    backend.register_body(name, crashing(9, seen))

    task = backend.spawn(name, str(uuid4()), max_attempts=2)
    snapshot = backend.run_until_result(task)

    assert snapshot is not None
    assert snapshot.state == "failed", snapshot
    assert seen == [(1, 2, False), (2, 2, True)]
    assert backend.task_attempts(task) == 2


def test_a_spawn_carries_its_attempt_budget_to_its_child(backend):
    """Reddens if a spawned child's limit comes from the engine that enqueued it: without the
    spawn's budget the embedded engine gives 3 and a separate Absurd spawner app gives 5."""
    parent, child = private("parent"), private("child")
    seen: list[tuple[int, int | None, bool]] = []
    backend.register_body(child, crashing(9, seen))
    domain = MeteredInterpreter(
        llm=lambda _op: ("", Usage()),
        tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(backend.spawner)}),
    )

    def spawns():
        # The args as a model's tool request writes them.
        request = {"task_name": child, "params": {}, "max_attempts": 2}
        spawned = yield from call_tool(SPAWN_TOOL, request, SpawnResult)
        return spawned.task_id

    backend.register_body(
        parent, lambda params, ctx: DurableHandler(ctx, domain, params=params).run(spawns)
    )
    snapshot = backend.run_until_result(backend.spawn(parent, str(uuid4())))
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    task = uuid.UUID(snapshot.result)
    ended = backend.run_until_result(task)

    assert ended is not None
    assert ended.state == "failed", ended
    assert backend.task_attempts(task) == 2


# A worker killed inside a body: nothing reaches the body's `except`, and the engine counts the
# execution when the lease expires. Run in a subprocess so the kill takes only the worker.
WORKER = textwrap.dedent(
    """
    import os, signal, sys
    engine, task, where = sys.argv[1], sys.argv[2], sys.argv[3]
    def body(params, ctx):
        os.kill(os.getpid(), signal.SIGKILL)
    if engine == "sqlite":
        import effective.sqlite as sqlite
        sqlite.CLAIM_LEASE_SECONDS = 0.0  # the lease is spent the moment it is taken
        app = sqlite.SqliteApp(where)
        app.register_task(task)(body)
        app.work_batch()
    else:
        from absurd_sdk import Absurd
        app = Absurd(os.environ["DATABASE_URL"], queue_name=where)
        app.register_task(task)(body)
        app.work_batch(claim_timeout=1)
    """
)


@dataclass(frozen=True)
class Died:
    """How a task whose worker was killed ended, and every event its queue holds."""

    ended: tuple[str, int]
    answers: dict[str, Any]


def die_in(engine: str, task: str, where: str, tmp_path) -> None:
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    died = subprocess.run(
        [sys.executable, str(script), engine, task, where],
        env={**os.environ, "DATABASE_URL": DSN},
        capture_output=True,
        timeout=60,
    )
    assert died.returncode == -9, died.stderr


def sqlite_death(tmp_path, limit: int, params: dict[str, Any] | None = None) -> Died:
    from effective.sqlite import SqliteApp

    where, task = str(tmp_path / "death.db"), private("dies")
    app = SqliteApp(where)
    try:
        app.register_task(task)(lambda params, ctx: "ok")
        task_id = app.spawn(task, params or {}, max_attempts=limit)
        die_in("sqlite", task, where, tmp_path)
        app.work_batch()
        ended = app.conn.execute(
            "SELECT state, attempt FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        answers = app.conn.execute("SELECT name, payload FROM events").fetchall()
        return Died(tuple(ended), {name: json.loads(payload) for name, payload in answers})
    finally:
        app.close()


def absurd_death(tmp_path, limit: int, params: dict[str, Any] | None = None) -> Died:
    from effective.absurd_worker import absurd_worker

    queue, task = f"q{uuid.uuid4().hex[:8]}", private("dies")
    tasks, events = f"t_{queue}", f"e_{queue}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("SELECT absurd.create_queue(%s)", (queue,))
    app: Any = absurd_worker(DSN, queue_name=queue)
    try:
        app.register_task(task)(lambda params, ctx: "ok")
        spawned = app.spawn(task, params or {}, max_attempts=limit, retry_strategy=IMMEDIATE_RETRY)
        die_in("postgres", task, queue, tmp_path)
        # The sweeping worker reads a clock past the killed run's lease, as Absurd's claim asks it.
        past_the_lease = datetime.now(UTC) + timedelta(minutes=5)
        app._conn.execute(
            "SELECT set_config('absurd.fake_now', %s, false)", (past_the_lease.isoformat(),)
        )
        row = None
        for _ in range(5):  # the sweep that fails the dead run need not claim its successor
            app.work_batch(claim_timeout=30)
            with psycopg.connect(DSN) as conn:
                row = conn.execute(
                    t"SELECT state, attempts FROM absurd.{tasks:i} "
                    t"WHERE task_id = {spawned['task_id']}"
                ).fetchone()
            if row is not None and row[0] in ("completed", "failed"):
                break
        with psycopg.connect(DSN) as conn:
            answers = conn.execute(
                t"SELECT event_name, payload FROM absurd.{events:i} WHERE payload IS NOT NULL"
            ).fetchall()
        assert row is not None
        return Died(tuple(row), dict(answers))
    finally:
        app.close()
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("SELECT absurd.drop_queue(%s)", (queue,))


ENGINES = [
    pytest.param(sqlite_death, id="sqlite"),
    pytest.param(
        absurd_death,
        id="postgres",
        marks=pytest.mark.skipif(not pg_ready(), reason="no Postgres with Absurd (just pgt-up)"),
    ),
]


@pytest.mark.parametrize("death", ENGINES)
@pytest.mark.parametrize(("limit", "ended"), [(2, ("completed", 2)), (1, ("failed", 1))])
def test_a_worker_death_counts_the_execution_it_killed(tmp_path, death, limit, ended):
    """Reddens if a killed execution goes uncounted: after one death a task with a limit of 2 runs
    once more and ends on attempt 2, and a task with a limit of 1 fails on attempt 1."""
    assert death(tmp_path, limit).ended == ended


@pytest.mark.xfail(
    strict=True,
    reason="the engine fails a task whose worker died where it claims, and no body runs there to "
    "answer the waiting parent",
)
@pytest.mark.parametrize("death", ENGINES)
def test_a_worker_death_on_a_childs_last_attempt_answers_its_parent(tmp_path, death):
    """Reddens while a parent waits forever on a child whose worker died on its last attempt."""
    placement = Writer(task=str(uuid.uuid4()), placement=Key.parse("step;tool:spawn,c"))
    done = spawn_done_name(placement).stored()

    died = death(tmp_path, 1, {DONE_EVENT_PARAM: done})

    assert died.ended == ("failed", 1)
    assert died.answers.get(done, {}).get("answer", {}).get("kind") == "failed"
