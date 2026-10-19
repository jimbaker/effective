"""A run's waits read its engine's clock, so a test holds that clock and places each wake on
either engine by moving it, with no real wait."""

import ast
import importlib
import inspect
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from _clock import HeldClock
from _durable import DSN, pg_ready

import effective
from effective.api import await_until, gather, sleep_until
from effective.bridge_sqlite import park_name
from effective.engines import REPARK_MARGIN, TaskState, open
from effective.engines.absurd import AbsurdEngine, ConcurrentAbsurdCtx, SdkCtx
from effective.engines.sqlite import SqliteApp, SqliteTaskContext, _Suspend
from effective.handlers.durable import (
    _WRAPPERS,
    DurableHandler,
    RenamedAwaitCtx,
    SeedingCtx,
    _PrefixedCtx,
    _StoreCalls,
)
from effective.keys import Key
from effective.ops import DONE_EVENT_PARAM
from effective.steering import SteeringCtx
from effective.viewing import ReachedThePark, ViewingCtx

A_SECOND = timedelta(seconds=1)
A_MILLISECOND = timedelta(milliseconds=1)
AN_HOUR = timedelta(hours=1)

SRC = Path(inspect.getfile(effective)).parent.parent


class _NoDomain:
    def run(self, op: Any) -> Any:
        raise AssertionError(f"no domain op belongs in this workflow: {op}")


@pytest.fixture(params=["sqlite", "postgres"])
def held(request) -> Iterator[tuple[Any, HeldClock]]:
    """An engine on a store of its own, with one clock held for it a minute past real time."""
    clock = HeldClock(datetime.now(UTC) + timedelta(minutes=1))
    if request.param == "sqlite":
        engine = open("sqlite://", clock=clock)
        yield engine, clock
        engine.close()
        return
    if not pg_ready():
        pytest.skip("needs Postgres/Absurd (just pgt-up)")
    engine = open(DSN, queue="clock_" + uuid4().hex[:12])
    assert isinstance(engine, AbsurdEngine)
    engine.app.create_queue()
    try:
        with clock.on_absurd(engine):
            yield engine, clock
    finally:
        engine.app.drop_queue()
        engine.close()


def _state(engine: Any, task: Any) -> TaskState | None:
    snap = engine.fetch_task_result(task)
    return None if snap is None else snap.state


@pytest.mark.parametrize("where", ["top level", "in a gather"])
def test_a_parked_sleep_wakes_when_the_held_clock_reaches_it(held, where):
    """A gather's branch compares its sleep to the clock itself and parks as a value; the top
    level hands its sleep to the engine. Both wake on the held clock."""
    engine, clock = held
    wake = clock.at + AN_HOUR

    def sleeps():
        yield from sleep_until(wake)
        return "awake"

    def nap():
        if where == "top level":
            return (yield from sleeps())
        [awake] = yield from gather([sleeps])
        return awake

    engine.register_task("nap")(lambda params, ctx: DurableHandler(ctx, _NoDomain()).run(nap))
    task = engine.spawn("nap", {})
    engine.work_batch()
    clock.advance_to(wake - A_SECOND)
    engine.work_batch()
    assert _state(engine, task) is not TaskState.COMPLETED, "woke a second before its time"

    clock.advance_to(wake)
    snap = engine.run_until_result(task)
    assert snap is not None
    assert (snap.state, snap.result) == (TaskState.COMPLETED, "awake")


def test_a_repark_wakes_when_the_held_clock_reaches_its_margin(held):
    """Each engine parks a repark `REPARK_MARGIN` past its clock, so it sleeps until the clock
    moves."""
    engine, clock = held
    reparked = Key.parse("reparked")

    def body(params: Any, ctx: Any) -> str:
        if isinstance(engine, AbsurdEngine):
            ctx = ConcurrentAbsurdCtx(ctx)
        if not ctx.peek_step(reparked)[0]:
            ctx.step(reparked, lambda: True)
            ctx.repark("repark")
        return "awake"

    engine.register_task("repark")(body)
    task = engine.spawn("repark", {})
    engine.work_batch()
    engine.work_batch()
    assert _state(engine, task) is TaskState.SLEEPING

    clock.advance(timedelta(seconds=REPARK_MARGIN) - A_MILLISECOND)
    engine.work_batch()
    assert _state(engine, task) is TaskState.SLEEPING, "woke short of its margin"

    clock.advance(A_MILLISECOND)
    snap = engine.run_until_result(task)
    assert snap is not None
    assert (snap.state, snap.result) == (TaskState.COMPLETED, "awake")


def _contexts() -> list[type]:
    """Each class in the source tree that defines `sleep_until`."""
    found = []
    for path in sorted((SRC / "effective").rglob("*.py")):
        tree = ast.parse(path.read_text())
        defining = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(member, ast.FunctionDef) and member.name == "sleep_until"
                for member in node.body
            )
        ]
        if defining:
            module = importlib.import_module(".".join(path.relative_to(SRC).with_suffix("").parts))
            found += [getattr(module, name) for name in defining]
    return found


def test_every_context_that_sleeps_reads_its_engines_clock():
    """A class answers `clock` itself, since one left to `__getattr__` answers whatever it wraps
    or nothing at all."""
    contexts = _contexts()
    assert {SqliteTaskContext, SdkCtx, ViewingCtx} <= set(contexts)
    assert [cls for cls in contexts if not callable(getattr(cls, "clock", None))] == []


class _Inner:
    def clock(self) -> float:
        return 12.5


WRAPPED: dict[type, Any] = {
    _StoreCalls: lambda ctx: _StoreCalls(ctx, lambda failure: None),
    _PrefixedCtx: lambda ctx: _PrefixedCtx(ctx, "p;"),
    RenamedAwaitCtx: lambda ctx: RenamedAwaitCtx(ctx, "child"),
    SeedingCtx: lambda ctx: SeedingCtx(ctx, {}, fork_point=Key.parse("f")),
    SteeringCtx: lambda ctx: SteeringCtx(ctx, {}),
    ViewingCtx: lambda ctx: ViewingCtx(ctx, frozenset()),
}
"""Each wrapper the handler stacks over an engine's ctx, built over one whose clock reads 12.5."""


def test_every_wrapper_answers_the_clock_of_the_ctx_it_wraps():
    assert set(WRAPPED) == set(_WRAPPERS)
    assert {cls: wrap(_Inner()).clock() for cls, wrap in WRAPPED.items()} == dict.fromkeys(
        WRAPPED, 12.5
    )


def test_absurd_takes_no_clock_when_opened():
    """Absurd reads its server's clock, so a clock handed to it would be dropped."""
    with pytest.raises(ValueError, match="server's clock"):
        open(DSN, clock=HeldClock(datetime.now(UTC)))


# A held SQLite clock may stand behind real time as well as ahead of it; a read of real time where
# the app's clock belongs lands on the wrong side of a deadline in one direction or the other.


def _app(sqlite_app: Any, offset: timedelta) -> tuple[SqliteApp, HeldClock]:
    clock = HeldClock(datetime.now(UTC) + offset)
    return sqlite_app(clock=clock), clock


def _waiter(app: SqliteApp, deadline: datetime) -> Any:
    def waits():
        outcome = yield from await_until("ev:x", dict, deadline=deadline)
        return type(outcome).__name__

    app.register_task("waits")(lambda params, ctx: DurableHandler(ctx, _NoDomain()).run(waits))
    return app.spawn("waits", {})


def test_a_wait_whose_deadline_the_apps_clock_passed_expires_at_once(sqlite_app):
    app, clock = _app(sqlite_app, AN_HOUR)
    task = _waiter(app, clock.at - timedelta(minutes=1))
    snap = app.run_until_result(task, max_batches=4)
    assert snap is not None
    assert (snap.state, snap.result) == (TaskState.COMPLETED, "Expired")


def test_an_event_after_the_apps_deadline_wakes_no_wait_and_names_no_park(sqlite_app):
    app, clock = _app(sqlite_app, AN_HOUR)
    task = _waiter(app, clock.at + timedelta(minutes=10))
    app.work_batch()
    clock.advance(timedelta(minutes=20))
    assert (app.parked(), park_name(app.conn, task, clock=app.clock)) == ((), None)
    app.emit_event("ev:x", {})
    snap = app.run_until_result(task)
    assert snap is not None
    assert snap.result == "Expired"


def test_an_app_behind_real_time_claims_what_it_spawned_and_woke(sqlite_app):
    app, clock = _app(sqlite_app, -AN_HOUR)
    task = _waiter(app, clock.at + timedelta(minutes=10))
    assert app.work_batch() is True
    app.emit_event("ev:x", {})
    snap = app.run_until_result(task)
    assert snap is not None
    assert (snap.state, snap.result) == (TaskState.COMPLETED, "Arrived")


def test_a_viewed_sleep_reads_the_apps_clock(sqlite_app):
    app, clock = _app(sqlite_app, AN_HOUR)
    viewer = ViewingCtx(app.context(app.spawn("viewed", {})), frozenset())
    viewer.sleep_until(clock.at - timedelta(minutes=1), name=Key.parse("due"))
    with pytest.raises(ReachedThePark):
        viewer.sleep_until(clock.at + timedelta(minutes=1), name=Key.parse("undue"))


@pytest.mark.parametrize("ends", ["repark", "retry"])
def test_an_app_behind_real_time_requeues_at_its_own_clock(sqlite_app, ends):
    app, clock = _app(sqlite_app, -AN_HOUR)
    runs: list[None] = []

    def body(params: Any, ctx: Any) -> str:
        runs.append(None)
        if len(runs) == 1:
            if ends == "repark":
                ctx.repark("repark")
            raise RuntimeError("the first attempt fails")
        return "awake"

    app.register_task("requeued", default_max_attempts=3)(body)
    task = app.spawn("requeued", {})
    app.work_batch()
    clock.advance(timedelta(seconds=REPARK_MARGIN) + A_SECOND)
    snap = app.run_until_result(task, max_batches=4)
    assert snap is not None
    assert (snap.state, snap.result) == (TaskState.COMPLETED, "awake")


def _awaiting(app: SqliteApp, name: str) -> Any:
    """A task parked on the event `name`, which returns that event's payload."""
    app.register_task("awaits")(lambda params, ctx: ctx.await_event(Key.parse(name)))
    task = app.spawn("awaits", {})
    app.work_batch()
    assert _state(app, task) is TaskState.WAITING
    return task


def test_an_emit_from_inside_a_task_wakes_its_waiter_at_the_apps_clock(sqlite_app):
    app, _ = _app(sqlite_app, -AN_HOUR)
    waiter = _awaiting(app, "ev:x")
    app.register_task("emits")(lambda params, ctx: ctx.emit_event("ev:x", {"n": 1}))
    app.spawn("emits", {})
    snap = app.run_until_result(waiter)
    assert snap is not None
    assert snap.state is TaskState.COMPLETED


def test_an_event_stored_before_the_park_wakes_it_at_the_apps_clock(sqlite_app):
    """The event lands between the task's read and its park, so the park finds it stored."""
    app, _ = _app(sqlite_app, -AN_HOUR)
    event = Key.parse("ev:x")

    def body(params: Any, ctx: Any) -> str:
        found, _ = ctx.peek_event(event)
        if found:
            return "awake"
        app.emit_event("ev:x", {"n": 1})
        raise _Suspend(event=event)

    app.register_task("parks")(body)
    task = app.spawn("parks", {})
    snap = app.run_until_result(task)
    assert snap is not None
    assert (snap.state, snap.result) == (TaskState.COMPLETED, "awake")


@pytest.mark.parametrize("ends", ["cancelled", "failed"])
def test_a_child_that_ends_wakes_its_parent_at_the_apps_clock(sqlite_app, ends):
    app, _ = _app(sqlite_app, -AN_HOUR)
    parent = _awaiting(app, "done:child")

    def fails(params: Any, ctx: Any) -> None:
        raise RuntimeError("the child fails for good")

    app.register_task("child", default_max_attempts=1)(fails)
    child = app.spawn("child", {DONE_EVENT_PARAM: "done:child"})
    if ends == "cancelled":
        app.cancel(child)
    snap = app.run_until_result(parent)
    assert snap is not None
    assert snap.state is TaskState.COMPLETED
