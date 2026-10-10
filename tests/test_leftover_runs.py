"""A run one test leaves non-terminal, and what the session does about it (`tests/conftest.py`)."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import pytest
from _durable import DSN, absurd, cancel_leftover_runs, is_disposable, pg_ready
from conftest import _LOCK_KEY

from effective.engines import TaskState
from effective.engines.absurd import AbsurdEngine


def _marked() -> bool:
    with psycopg.connect(DSN, connect_timeout=2) as conn:
        return is_disposable(conn)


pytestmark = pytest.mark.skipif(
    not (pg_ready() and _marked()), reason="no disposable Postgres with Absurd + ledger"
)

ROOT = Path(__file__).parent.parent


def serial_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    """A serial pytest session on this process's database, loading this suite's conftest."""
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_XDIST_WORKER"}
    env |= {"DATABASE_URL": DSN, "PYTHONPATH": str(ROOT / "tests")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "-o", "addopts=", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )


def state(engine: AbsurdEngine, task_id: UUID) -> TaskState:
    snapshot = engine.fetch_task_result(task_id)
    assert snapshot is not None
    return snapshot.state


def test_a_leftover_run_spends_a_batch_until_it_is_cancelled():
    app = absurd()
    try:
        orphan = app.spawn(f"orphan-{uuid4().hex[:8]}", {})
        name = f"own-{uuid4().hex[:8]}"
        app.register_task(name)(lambda params, ctx: "done")

        first = app.spawn(name, {})
        app.work_batch()
        assert state(app, first) == "pending"

        with psycopg.connect(DSN, autocommit=True) as conn:
            cancel_leftover_runs(conn)
        for spawned in (orphan, first):
            assert state(app, spawned) == "cancelled"

        second = app.spawn(name, {})
        app.work_batch()
        assert state(app, second) == "completed"
    finally:
        app.close()


def test_an_undeclared_database_is_refused_the_cancel():
    """The helper checks the marker itself, so no caller can cancel runs on a database that did
    not declare itself disposable."""

    class Unmarked:
        def execute(self, query, params=None):
            assert "_pgtest_disposable" in str(query), f"wrote to an undeclared database: {query}"
            return type("R", (), {"fetchone": staticmethod(lambda: (None,))})()

    with pytest.raises(PermissionError, match="not marked disposable"):
        cancel_leftover_runs(Unmarked())


PROBE = """
from uuid import uuid4

from _durable import absurd


def test_a_leaves_a_run_nobody_registers():
    app = absurd()
    app.spawn(f"orphan-{uuid4().hex[:8]}", {})
    app.close()


def test_b_drives_its_own_run_with_one_batch():
    app = absurd()
    try:
        name = f"own-{uuid4().hex[:8]}"
        app.register_task(name)(lambda params, ctx: "done")
        spawned = app.spawn(name, {})
        app.work_batch()
        assert app.fetch_task_result(spawned).state == "completed"
    finally:
        app.close()
"""


@pytest.mark.skipif(
    "PYTEST_XDIST_WORKER" not in os.environ,
    reason="a serial session holds its database, so a second session on it is refused",
)
def test_the_teardown_cancels_what_a_test_left_for_the_next(tmp_path):
    """Two tests in a session of their own on this worker's database, which that session resets
    first: the first leaves a run no app registers, and the second's single batch runs its own."""
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE)
    run = serial_pytest("-p", "conftest", "-c", str(ROOT / "pyproject.toml"), str(probe))
    assert run.returncode == 0, run.stdout + run.stderr


def test_a_second_serial_run_on_one_database_is_refused():
    """Refused before its reset, so a refused run has cancelled nothing. Under a serial parent the
    parent's own session holds the lock and the holder's attempt fails; the subprocess is refused
    either way."""
    with psycopg.connect(DSN, autocommit=True) as holder:
        holder.execute(t"SELECT pg_try_advisory_lock({_LOCK_KEY}, hashtext(current_database()))")
        run = serial_pytest("--collect-only", __file__)
    assert run.returncode == pytest.ExitCode.USAGE_ERROR, run.stdout + run.stderr
    assert "another serial test run" in run.stdout + run.stderr
