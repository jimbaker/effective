"""`effective.runs` — written because no surface could tell a dead run from an empty one.

The property under test is a DISCRIMINATION, so the cases are the ones it must separate: completed,
failed, and parked all leave different task rows and two of them leave the same tape.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from effective.api import await_event, call_tool, compose_key
from effective.cost import MeteredInterpreter, Usage
from effective.engines.sqlite import SqliteApp
from effective.handlers.durable import DurableHandler
from effective.keys import Segment
from effective.runs import RunState, read_sqlite_runs, read_sqlite_runs_conn


def _ok(_run_id: str):
    yield from call_tool("work", {}, str)
    return "done"


def _boom(_run_id: str):
    raise ValueError("the payload was not what the park awaited")
    yield  # pragma: no cover  -- a workflow is a generator


def _parks(run_id: str):
    return (yield from await_event(compose_key(t"review:{Segment(run_id)}"), dict))


WORKFLOWS = {"ok": _ok, "boom": _boom, "parks": _parks}


@pytest.fixture
def store(tmp_path: Path) -> Path:
    db = tmp_path / "runs.db"
    app = SqliteApp(str(db))
    for name, workflow in WORKFLOWS.items():

        @app.register_task(name)
        def task(params, ctx, _wf=workflow):
            return DurableHandler(
                ctx, MeteredInterpreter(llm=lambda _op: ({}, Usage()), tools=lambda _op: "ran")
            ).run(lambda: _wf(params["run_id"]))

    for name in WORKFLOWS:
        app.spawn(name, {"run_id": f"r-{name}"})
    for _ in range(12):
        if not app.work_batch():
            break
    app.close()
    return db


def test_the_three_endings_are_three_different_states(store):
    by_name = {r.task_name: r for r in read_sqlite_runs(store)}
    assert by_name["ok"].state is RunState.COMPLETED
    assert by_name["boom"].state is RunState.FAILED
    assert by_name["parks"].state is RunState.PARKED


def test_a_failed_run_carries_its_reason(store):
    """The field the type exists for. "It failed" without the message sends a reader to a
    database, which is the drop-out that produced this module."""
    boom = next(r for r in read_sqlite_runs(store) if r.task_name == "boom")
    assert boom.failure is not None
    assert "not what the park awaited" in boom.failure
    assert next(r for r in read_sqlite_runs(store) if r.task_name == "ok").failure is None


def test_a_failed_run_and_a_parked_one_are_both_UNFINISHED_but_not_the_same_obligation(store):
    """`live` is the property a surface actually branches on, and the split that matters inside it
    is parked-waiting-on-me against ready-waiting-on-a-drain."""
    by_name = {r.task_name: r for r in read_sqlite_runs(store)}
    assert by_name["parks"].state.live
    assert not by_name["boom"].state.live
    assert not by_name["ok"].state.live


def test_the_engine_word_is_kept_beside_the_normalized_one(store):
    """SQLite says `waiting` and Absurd says `sleeping` for the same situation. A consumer reads
    `state`; someone debugging the engine reads `raw_state`."""
    parked = next(r for r in read_sqlite_runs(store) if r.task_name == "parks")
    assert parked.state is RunState.PARKED
    assert parked.raw_state == "waiting"


def test_the_ledger_run_id_is_not_the_task_id(store):
    """A task and a run are different things: `run_id` is a caller's convention in `params`."""
    ok = next(r for r in read_sqlite_runs(store) if r.task_name == "ok")
    assert ok.run_id == "r-ok"
    assert str(ok.task_id) != "r-ok"


def test_an_unmapped_engine_state_is_refused_rather_than_defaulted(store):
    """A `.get(raw, READY)` would turn a word nobody mapped into a confident wrong answer — and
    the wrong answer would be the reassuring one."""
    with closing(sqlite3.connect(store)) as conn:
        conn.execute("UPDATE tasks SET state='quiesced' WHERE name='ok'")
        conn.commit()
    with pytest.raises(ValueError, match="unmapped engine state"):
        read_sqlite_runs(store)


def test_the_conn_form_reads_an_in_memory_store():
    """An `:memory:` store has no path to reopen; `checkpoints` and `parked` ship the pair."""
    app = SqliteApp(":memory:")

    @app.register_task("ok")
    def task(params, ctx):
        return DurableHandler(
            ctx, MeteredInterpreter(llm=lambda _op: ({}, Usage()), tools=lambda _op: "ran")
        ).run(lambda: _ok(params["run_id"]))

    app.run_until_result(app.spawn("ok", {"run_id": "r-mem"}))
    try:
        (run,) = read_sqlite_runs_conn(app.conn)
        assert run.state is RunState.COMPLETED
        assert run.run_id == "r-mem"
    finally:
        app.close()
