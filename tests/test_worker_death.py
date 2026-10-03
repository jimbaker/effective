"""A worker killed at a designed cut under a gather, recovered by lease reclaim, on both engines.

`_death` holds the cut. The contrast is the in-process fault injector at the same cut: an exception
ends one branch's turn and its sibling commits after it, which a death does not allow.
The predictions were written before these ran.
"""

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from _conformance import IMMEDIATE_RETRY, private
from _death import DIED, ORDER, SCENARIOS, cut, program
from _durable import DSN, pg_ready
from _schedules import Turnstile, ledger_path

from effective.checkpoints import keys, read_sqlite_conn

TESTS = Path(__file__).parent


@dataclass(frozen=True)
class Death:
    """The record a killed worker left, and the record after its recovery."""

    rows: list[str]
    appended: list[str]
    ended: tuple[str, int]
    recovered_rows: list[str]
    recovered_appended: list[str]
    result: Any


def die(engine: str, where: str, task: str, scenario: str) -> None:
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, _death; _death.work(*sys.argv[1:])",
            engine,
            where,
            task,
            DSN,
            scenario,
        ],
        cwd=TESTS,
        env={**os.environ, "DATABASE_URL": DSN},
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == DIED, child.stderr.decode()


def appended(names: Any) -> list[str]:
    """The paths whose append checkpoint exists."""
    return sorted(str(name).rsplit(",", 1)[-1] for name in names if ";ledger;" in str(name))


def sqlite_death(tmp_path: Path, scenario: str) -> Death:
    from effective.sqlite import SqliteApp, SqliteLedger

    program, handler = SCENARIOS[scenario]
    where, task, run_id = str(tmp_path / "death.db"), private("dies"), str(uuid4())
    app = SqliteApp(where)
    try:

        @app.register_task(task)
        def recovers(params: Any, ctx: Any) -> Any:
            ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock)
            return handler(ctx, ledger, dying=False).run(lambda: program(params["run_id"]))

        task_id = app.spawn(task, {"run_id": run_id}, max_attempts=3)
        die("sqlite", where, task, scenario)

        def rows() -> list[str]:
            payloads = app.conn.execute(
                "SELECT payload FROM ledger WHERE workflow_run_id=?", (run_id,)
            ).fetchall()
            return sorted(json.loads(payload)["path"] for (payload,) in payloads)

        before = rows(), appended(keys(read_sqlite_conn(app.conn, task_id)))
        snap = app.run_until_result(task_id)
        assert snap is not None
        attempts = app.conn.execute(
            "SELECT attempt FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        return Death(
            *before,
            (snap.state, attempts),
            rows(),
            appended(keys(read_sqlite_conn(app.conn, task_id))),
            snap.result,
        )
    finally:
        app.close()


def absurd_death(_tmp_path: Path, scenario: str) -> Death:
    from effective.absurd_worker import absurd_worker
    from effective.handlers.absurd import ConcurrentAbsurdCtx
    from effective.ledger import PostgresLedger

    program, handler = SCENARIOS[scenario]
    queue, task, run_id = f"q{uuid.uuid4().hex[:8]}", private("dies"), str(uuid4())
    tasks, checkpoints = f"t_{queue}", f"c_{queue}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("SELECT absurd.create_queue(%s)", (queue,))
    app: Any = absurd_worker(DSN, queue_name=queue)
    try:

        @app.register_task(task)
        def recovers(params: Any, ctx: Any) -> Any:
            ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
            live = handler(ConcurrentAbsurdCtx(ctx), ledger, dying=False)
            return live.run(lambda: program(params["run_id"]))

        spawned = app.spawn(
            task, {"run_id": run_id}, max_attempts=3, retry_strategy=IMMEDIATE_RETRY
        )
        task_id = spawned["task_id"]
        die("postgres", queue, task, scenario)

        def record() -> tuple[list[str], list[str]]:
            with psycopg.connect(DSN) as conn:
                payloads = conn.execute(
                    "SELECT payload FROM ledger WHERE workflow_run_id=%s", (run_id,)
                ).fetchall()
                # A set, where `read_absurd_task` wants an order: under `fake_now` the recovery's
                # checkpoints share one `updated_at`.
                names = conn.execute(
                    t"SELECT checkpoint_name FROM absurd.{checkpoints:i} "
                    t"WHERE task_id = {task_id}::uuid AND status = 'committed'"
                ).fetchall()
            return sorted(p["path"] for (p,) in payloads), appended(n for (n,) in names)

        before = record()
        # The sweeping worker reads a clock past the killed run's lease, as Absurd's claim asks it.
        past_the_lease = datetime.now(UTC) + timedelta(minutes=5)
        app._conn.execute(
            "SELECT set_config('absurd.fake_now', %s, false)", (past_the_lease.isoformat(),)
        )
        ended = None
        for _ in range(5):  # the sweep that fails the dead run need not claim its successor
            app.work_batch(claim_timeout=30)
            with psycopg.connect(DSN) as conn:
                ended = conn.execute(
                    t"SELECT state, attempts FROM absurd.{tasks:i} WHERE task_id = {task_id}"
                ).fetchone()
            if ended is not None and ended[0] in ("completed", "failed"):
                break
        assert ended is not None
        snapshot = app.fetch_task_result(task_id)
        return Death(*before, tuple(ended), *record(), snapshot.result)
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
def test_a_worker_killed_under_a_gather_leaves_the_cut_and_recovers(tmp_path, death):
    died = death(tmp_path, "gather")

    assert died.rows == ["a0", "b0", "b1"], "D2: b1's row landed, and a1 never ran"
    assert died.appended == ["a0", "b0"], "D3: b1 died in its window, and a1 never ran"
    assert died.ended == ("completed", 2), "D4"
    assert died.recovered_rows == sorted(ORDER), "D4: every row once, b1's re-written in place"
    assert died.recovered_appended == sorted(ORDER), "D4"


@pytest.mark.parametrize("death", ENGINES)
def test_a_losers_op_the_death_interrupted_after_the_choice_never_reruns(tmp_path, death):
    """D5 of the race build's predictions. The loser's first append
    was admitted before the flag and interrupted by the death after the choice was saved; its row
    stays, its checkpoint never commits, and recovery stops the loser there."""
    died = death(tmp_path, "race")

    assert died.rows == ["l0", "w0"], "l0's row landed before the death"
    assert died.appended == ["w0"], "l0 died in its window, and l1 never ran"
    assert died.ended == ("completed", 2)
    assert died.recovered_rows == ["l0", "w0"], "l1 never ran"
    assert died.recovered_appended == ["w0"], "l0 was never run again after the choice"
    assert died.result == [["Won", 0], ["Stopped", 1]]


def test_an_injected_fault_at_the_cut_lets_a_sibling_commit(backend):
    """N1: the injector is a failure, not a death. `b1`'s raise ends its turn, and `a1` commits
    after it."""
    name, run_id = private("fails"), str(uuid4())
    turnstile = Turnstile(ORDER, ledger_path)
    backend.register(name, program, None, cut(), [turnstile.layer()])
    task = backend.spawn(name, run_id, max_attempts=1)
    snap = backend.run_until_result(task)

    assert snap.state == "failed", snap
    assert sorted(row["path"] for row in backend.ledger_payloads(run_id)) == sorted(ORDER)
    assert turnstile.outcomes[-2:] == [("b1", "FaultInjected"), ("a1", "committed")]
