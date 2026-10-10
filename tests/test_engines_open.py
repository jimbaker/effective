"""An engine opened by its URL runs, drives and reports a task the same way on both engines."""

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4, uuid7

import pytest
from _durable import DSN
from pydantic import BaseModel

from effective.api import await_event, call_tool, sleep_until
from effective.cancel import OpCancelled
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL
from effective.engines import Drives, Engine, NoDriver, TaskState, open
from effective.engines import sqlite as sqlite_engine
from effective.engines.absurd import AbsurdEngine, sdk_failure
from effective.engines.sqlite import SqliteApp
from effective.handlers.durable import DurableHandler
from effective.interpreters.tools import make_tool_runner, spawn_tool
from effective.ops import DONE_EVENT_PARAM, Unretryable
from effective.spawning import join_answer, spawn_child


class NoDomain:
    def run(self, op: Any) -> Any:
        raise AssertionError(f"no domain op expected, got {op!r}")


class Tools:
    def run(self, op: Any) -> Any:
        return {"tool": op.name}


@pytest.mark.parametrize(
    ("url", "kind"),
    [("sqlite://", SqliteApp), ("sqlite:///:memory:", SqliteApp)],
    ids=["bare", "named memory"],
)
def test_a_sqlite_url_opens_the_embedded_engine(url, kind):
    opened = open(url)
    assert isinstance(opened, kind)
    opened.close()


def test_every_engine_open_returns_runs_and_drives_tasks():
    """Checked by the type checker: each engine `open` can return conforms to both protocols."""
    opened = open("sqlite://")
    runs: Engine = opened
    drives: Drives = opened
    assert runs is drives
    opened.close()


@pytest.mark.parametrize(
    "url", ["sqlite://somehost/x.db", "sqlite:///x.db?mode=ro", "sqlite:///x.db#part"]
)
def test_a_sqlite_url_with_parts_no_store_reads_is_refused(url, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="file path and nothing else"):
        open(url)
    assert list(tmp_path.iterdir()) == []


def test_a_spawn_of_fewer_than_one_attempt_is_refused(engine):
    with pytest.raises(ValueError, match="max_attempts was 0"):
        engine.spawn("noop", {}, max_attempts=0)


def test_a_url_with_no_driver_is_refused_by_name():
    with pytest.raises(NoDriver, match="temporal"):
        open("temporal://localhost:7233")


def test_an_empty_store_has_no_batch_to_work(engine):
    assert engine.work_batch() is False


def test_a_spawned_task_completes_with_its_result(engine):
    @engine.register_task("double")
    def double(params, ctx):
        return params["n"] * 2

    snapshot = engine.run_until_result(engine.spawn("double", {"n": 21}))
    assert snapshot is not None
    assert snapshot.state is TaskState.COMPLETED
    assert snapshot.result == 42


def test_a_task_spawns_before_this_process_registers_it(engine):
    """Another process may be the one that runs it, so the spawn names no registration."""
    task_id = engine.spawn("elsewhere", {"n": 1})
    engine.register_task("elsewhere")(lambda params, ctx: params["n"])
    done = engine.run_until_result(task_id)
    assert done is not None
    assert done.state is TaskState.COMPLETED


@pytest.mark.parametrize("second_name", ["noop", "other"], ids=["same name", "another name"])
def test_a_repeated_idempotency_key_returns_the_task_already_spawned(engine, second_name):
    """The key alone names the task, so a repeat under another name is still the first task."""
    engine.register_task("noop")(lambda params, ctx: None)
    first = engine.spawn("noop", {}, idempotency_key="the-one")
    assert engine.spawn(second_name, {"n": 2}, idempotency_key="the-one") == first


class Answer(BaseModel):
    ok: bool


def test_a_model_is_sent_as_its_plain_json_in_an_event_and_in_spawn_params(engine):
    def waiting():
        return (yield from await_event("go:model", dict))

    @engine.register_task("waits")
    def waits(params, ctx):
        return [params["answer"], DurableHandler(ctx, NoDomain()).run(waiting)]

    task_id = engine.spawn("waits", {"answer": Answer(ok=False)})
    engine.run_until_result(task_id)
    engine.emit_event("go:model", Answer(ok=True))
    done = engine.run_until_result(task_id)
    assert done is not None
    assert done.result == [{"ok": False}, {"ok": True}]


def test_a_task_sleeping_until_a_time_is_sleeping(engine):
    def sleeps():
        return (yield from sleep_until(datetime.now(UTC) + timedelta(hours=1)))

    engine.register_task("sleeps")(lambda params, ctx: DurableHandler(ctx, NoDomain()).run(sleeps))
    task_id = engine.spawn("sleeps", {})
    snapshot = engine.run_until_result(task_id)
    assert snapshot is not None
    assert snapshot.state is TaskState.SLEEPING
    engine.cancel(task_id)


def test_a_failed_task_reports_its_errors_type_name(engine):
    @engine.register_task("boom")
    def boom(params, ctx):
        raise ValueError("kaboom")

    snapshot = engine.run_until_result(engine.spawn("boom", {}, max_attempts=1))
    assert snapshot is not None
    assert snapshot.state is TaskState.FAILED
    assert snapshot.failure is not None
    assert snapshot.failure.kind == "ValueError"
    assert "ValueError" in str(snapshot.failure)
    assert "kaboom" in str(snapshot.failure)


def test_a_failed_tasks_text_carries_the_notes_beside_its_error(engine):
    @engine.register_task("noted")
    def noted(params, ctx):
        error = ValueError("kaboom")
        error.add_note("a crash beside it")
        raise error

    snapshot = engine.run_until_result(engine.spawn("noted", {}, max_attempts=1))
    assert snapshot is not None
    assert "a crash beside it" in str(snapshot.failure)


def test_a_task_awaiting_an_event_is_parked_until_the_event_arrives(engine):
    def waiting(run_id: str):
        return (yield from await_event("go:" + run_id, dict))

    @engine.register_task("waits")
    def waits(params, ctx):
        return DurableHandler(ctx, NoDomain()).run(lambda: waiting(params["run_id"]))

    task_id = engine.spawn("waits", {"run_id": "r1"})
    parked = engine.run_until_result(task_id)
    assert parked is not None
    assert parked.state is TaskState.WAITING
    assert [p.task_id for p in engine.parked()] == [task_id]

    engine.emit_event("go:r1", {"ok": True})
    done = engine.run_until_result(task_id)
    assert done is not None
    assert done.state is TaskState.COMPLETED
    assert done.result == {"ok": True}
    assert engine.parked() == ()


def test_cancelling_a_parked_child_ends_it_and_raises_the_cancel_in_its_parent(engine):
    """The child has no run to send its ending from, so the cancel sends it."""
    children = []

    def spawner(task_name, params, idempotency_key, queue, *, max_attempts=None):
        children.append(
            engine.spawn(task_name, params, max_attempts, idempotency_key=idempotency_key)
        )
        return str(children[-1])

    def waiting(run_id: str):
        return (yield from await_event("never:" + run_id, dict))

    def supervises():
        spawned = yield from spawn_child("waits", "c", {})
        try:
            return ["returned", (yield from join_answer(spawned))]
        except OpCancelled as cancelled:
            return ["cancelled", cancelled.partial]

    domain = MeteredInterpreter(
        llm=lambda _op: ("", Usage()),
        tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)}),
    )
    engine.register_task("waits")(
        lambda params, ctx: DurableHandler(ctx, NoDomain(), params=params).run(
            lambda: waiting("child")
        )
    )
    engine.register_task("parent")(
        lambda params, ctx: DurableHandler(ctx, domain, params=params).run(supervises)
    )

    parent = engine.spawn("parent", {"run_id": "p"})
    engine.run_until_result(parent)
    [child] = children
    engine.run_until_result(child)
    engine.cancel(child)

    ended = engine.fetch_task_result(child)
    assert ended is not None
    assert ended.state is TaskState.CANCELLED
    assert ended.failure is None
    assert child not in [p.task_id for p in engine.parked()]
    heard = engine.run_until_result(parent)
    assert heard is not None
    assert heard.state is TaskState.COMPLETED
    assert heard.result == ["cancelled", ""]


def test_a_cancel_interrupted_before_its_delivery_delivers_on_its_retry(engine, monkeypatch):
    def waiting():
        return (yield from await_event("child-done", dict))

    engine.register_task("parent")(
        lambda params, ctx: DurableHandler(ctx, NoDomain()).run(waiting)
    )
    parent = engine.spawn("parent", {})
    engine.run_until_result(parent)
    child = engine.spawn("child", {DONE_EVENT_PARAM: "child-done"})

    def interrupted(*args):
        raise OSError("interrupted before the delivery")

    with monkeypatch.context() as patched:
        if isinstance(engine, AbsurdEngine):
            patched.setattr(engine.app, "emit_event", interrupted)
        else:
            patched.setattr(sqlite_engine, "_deliver", interrupted)
        with pytest.raises(OSError, match="interrupted before the delivery"):
            engine.cancel(child)
    engine.cancel(child)

    heard = engine.run_until_result(parent)
    assert heard is not None
    assert heard.state is TaskState.COMPLETED
    assert heard.result == {"answer": {"kind": "cancelled", "partial": ""}}


def test_a_task_returning_a_model_completes_with_its_plain_json(engine):
    engine.register_task("model")(lambda params, ctx: Answer(ok=True))
    done = engine.run_until_result(engine.spawn("model", {}))
    assert done is not None
    assert done.state is TaskState.COMPLETED
    assert done.result == {"ok": True}


def test_a_task_returning_what_no_json_holds_fails_its_attempt(engine):
    engine.register_task("opaque")(lambda params, ctx: object())
    done = engine.run_until_result(engine.spawn("opaque", {}, max_attempts=1))
    assert done is not None
    assert done.state is TaskState.FAILED


def test_a_closed_engine_opens_nothing(engine):
    """A fresh engine of the fixture's kind, closed before it read anything."""
    closed = open("sqlite://") if isinstance(engine, SqliteApp) else open(DSN)
    closed.close()
    with pytest.raises((RuntimeError, sqlite3.ProgrammingError)):
        closed.parked()


def test_a_failure_the_engine_wrote_itself_names_no_type():
    failure = sdk_failure({"name": "$ClaimTimeout", "message": "the lease ran out"})
    assert failure is not None
    assert failure.kind is None


def test_a_parent_that_leaves_its_childs_cancel_uncaught_fails_on_the_attempt_it_heard_it(engine):
    """A retry would replay the recorded cancel and raise it again, so it fails at once."""
    children, runs = [], []

    def spawner(task_name, params, idempotency_key, queue, *, max_attempts=None):
        children.append(
            engine.spawn(task_name, params, max_attempts, idempotency_key=idempotency_key)
        )
        return str(children[-1])

    def supervises():
        spawned = yield from spawn_child("waits", "c", {})
        return (yield from join_answer(spawned))

    def parent(params, ctx):
        runs.append(1)
        domain = MeteredInterpreter(
            llm=lambda _op: ("", Usage()),
            tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)}),
        )
        return DurableHandler(ctx, domain, params=params).run(supervises)

    def never():
        return (yield from await_event("never", dict))

    engine.register_task("waits")(lambda params, ctx: DurableHandler(ctx, NoDomain()).run(never))
    engine.register_task("parent")(parent)
    parent_id = engine.spawn("parent", {}, max_attempts=5)
    engine.run_until_result(parent_id)
    [child] = children
    engine.run_until_result(child)
    engine.cancel(child)
    heard = engine.run_until_result(parent_id)

    assert heard is not None
    assert heard.state is TaskState.FAILED
    assert heard.failure is not None
    assert heard.failure.kind == "OpCancelled"
    assert len(runs) == 2


def test_a_store_holding_a_task_per_name_under_one_key_returns_each_its_own():
    """A store written while the key was scoped by name; the second row is inserted as such a
    store holds it."""
    store = SqliteApp()
    first = store.spawn("a", {}, idempotency_key="k")
    store.conn.execute(
        "INSERT INTO tasks (task_id, name, params, state, idempotency_key) "
        "VALUES (?, 'b', '{}', 'ready', 'k')",
        (uuid7(),),
    )
    [(second,)] = store.conn.execute("SELECT task_id FROM tasks WHERE name = 'b'").fetchall()
    assert store.spawn("a", {}, idempotency_key="k") == first
    assert store.spawn("b", {}, idempotency_key="k") == second
    [plan] = store.conn.execute(
        "EXPLAIN QUERY PLAN SELECT task_id FROM tasks WHERE idempotency_key = 'k'"
    ).fetchall()
    assert "USING INDEX" in plan[3]
    store.close()


class Boom(Unretryable):
    pass


CONTINUES = {
    "it completes": "complete",
    "it parks on an event": "await",
    "it raises an unretryable error": "unretryable",
    "it raises a retryable error": "retryable",
}


@pytest.mark.parametrize("continues", CONTINUES.values(), ids=CONTINUES.keys())
def test_a_task_cancelled_while_its_attempt_runs_stays_cancelled_however_the_attempt_ends(
    engine, continues
):
    started, release = threading.Event(), threading.Event()
    executions = []

    def child():
        yield from call_tool("first", {}, dict)
        started.set()
        release.wait(10)
        match continues:
            case "complete":
                return "done"
            case "await":
                return (yield from await_event("later", dict))
            case "unretryable":
                raise Boom("boom")
            case "retryable":
                raise ValueError("flaky")
            case unknown:
                raise AssertionError(f"no such continuation: {unknown}")

    def body(params, ctx):
        executions.append(1)
        return DurableHandler(ctx, Tools()).run(child)

    def waits():
        return (yield from await_event("child-done", dict))

    engine.register_task("child")(body)
    engine.register_task("parent")(lambda params, ctx: DurableHandler(ctx, NoDomain()).run(waits))
    parent = engine.spawn("parent", {})
    engine.run_until_result(parent)
    task_id = engine.spawn("child", {DONE_EVENT_PARAM: "child-done"}, max_attempts=3)

    attempt = threading.Thread(target=engine.work_batch)
    attempt.start()
    assert started.wait(10)
    engine.cancel(task_id)
    release.set()
    attempt.join(20)
    while engine.work_batch():
        pass
    engine.emit_event("later", {"woke": True})
    while engine.work_batch():
        pass

    ended = engine.fetch_task_result(task_id)
    assert ended is not None
    assert ended.state is TaskState.CANCELLED
    assert len(executions) == 1
    heard = engine.run_until_result(parent)
    assert heard is not None
    assert heard.result == {"answer": {"kind": "cancelled", "partial": ""}}


def test_cancelling_a_finished_task_leaves_it_as_it_ended(engine):
    engine.register_task("noop")(lambda params, ctx: None)
    task_id = engine.spawn("noop", {})
    engine.run_until_result(task_id)
    engine.cancel(task_id)
    ended = engine.fetch_task_result(task_id)
    assert ended is not None
    assert ended.state is TaskState.COMPLETED


def test_cancelling_an_unknown_task_raises(engine):
    with pytest.raises(LookupError):
        engine.cancel(uuid4())
