"""A task fails on an error a retry would raise again, and keeps its retries otherwise.

| failure                                                   | fails its task                     |
|-----------------------------------------------------------|------------------------------------|
| a recorded value its schema rejects                       | on the attempt that loads it       |
| any other error, raised by an attempt that ran nothing    | on that attempt                    |
| fresh and read nothing the record does not hold           |                                    |
| an error an effect raised fresh                           | on the last attempt                |
| a refusal beside a crash                                  | never, once the crash clears and   |
|                                                           | the workflow catches the refusal   |

An attempt that ran something fresh changed the record, so the next attempt replays a code error
it raised and fails there.
"""

import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import psycopg
import pytest
from _conformance import private
from _durable import DSN
from _sweep import InOrder
from pydantic import BaseModel, Field, field_validator

from effective.api import (
    await_event,
    call_tool,
    direct_tool_key,
    gather,
    quorum,
    race,
    scoped,
    step,
)
from effective.bridge_absurd import read_absurd_task
from effective.checkpoints import read_sqlite_conn
from effective.combinators import Again, Chain, Done, Turn, respawn
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, CallTool
from effective.engines import TaskState
from effective.engines.absurd import (
    AbsurdEngine,
    ConcurrentAbsurdCtx,
    queue_table,
    retry_delay,
    retry_waits,
)
from effective.engines.sqlite import SqliteApp
from effective.govern import Refused
from effective.handlers import base
from effective.handlers.admission import RaceState
from effective.handlers.base import Attempt, failing_leaf
from effective.handlers.durable import DurableHandler, SeedingCtx
from effective.interpreters.tools import make_tool_runner, spawn_tool
from effective.keys import Key, compose_key
from effective.layers import op_layer
from effective.ops import RecordedValueRejected, mark_rederived


class Approval(BaseModel):
    approve: bool


class Tools:
    """Answers every tool with a small dict, except the ones a case scripts."""

    def __init__(self, scripted: dict[str, Any] | None = None) -> None:
        self.scripted = scripted or {}
        self.calls: list[str] = []

    def run(self, op: Any) -> Any:
        self.calls.append(op.name)
        match self.scripted.get(op.name):
            case None:
                return {"n": 1}
            case script:
                return script(op)


def executions(
    engine,
    workflow,
    tools: Tools,
    *,
    emit: tuple[str, Any] | None = None,
    layers: tuple[Any, ...] = (),
    name: str = "case",
):
    """Run `workflow` as one task of five attempts, a race's branches in index order; its snapshot
    and how many times it ran."""
    ran = []

    def body(params, ctx):
        ran.append(1)
        in_order: Any = InOrder(ctx)
        return DurableHandler(in_order, tools, op_layers=layers).run(workflow)

    engine.register_task(name)(body)
    task_id = engine.spawn(name, {}, max_attempts=5)
    engine.run_until_result(task_id)
    if emit is not None:
        engine.emit_event(*emit)
        engine.run_until_result(task_id)
    snapshot = engine.fetch_task_result(task_id)
    assert snapshot is not None
    return snapshot, len(ran)


def test_an_event_payload_its_schema_rejects_fails_its_task_at_once(engine):
    def awaits():
        return (yield from await_event("decision", Approval))

    snapshot, ran = executions(engine, awaits, Tools(), emit=("decision", {"approve": "maybe"}))
    assert snapshot.state is TaskState.FAILED
    assert snapshot.failure is not None
    assert snapshot.failure.kind == "RecordedValueRejected"
    assert ran == 2  # the park, then the attempt that loads the payload


def test_a_recorded_tool_result_its_schema_rejects_fails_its_task_at_once(engine):
    def reads():
        return (yield from call_tool("t", {}, Approval))

    snapshot, ran = executions(engine, reads, Tools({"t": lambda op: {"approve": "maybe"}}))
    assert snapshot.state is TaskState.FAILED
    assert snapshot.failure is not None
    assert snapshot.failure.kind == "RecordedValueRejected"
    assert ran == 1


def test_workflow_code_raising_on_a_recorded_value_fails_on_the_attempt_that_repeats_it(engine):
    def reads():
        got = yield from call_tool("t", {}, dict)
        return got["missing"]

    snapshot, ran = executions(engine, reads, Tools())
    assert snapshot.state is TaskState.FAILED
    assert snapshot.failure is not None
    assert snapshot.failure.kind == "KeyError"
    assert ran == 2


def test_workflow_code_raising_after_it_caught_a_refusal_fails_on_that_attempt(engine):
    def refuse(op):
        raise Refused(op, "no")

    def catches():
        try:
            yield from call_tool("r", {}, dict)
        except Refused as refused:
            raise ValueError("handled badly") from refused

    snapshot, ran = executions(engine, catches, Tools({"r": refuse}))
    assert snapshot.state is TaskState.FAILED
    assert snapshot.failure is not None
    assert snapshot.failure.kind == "ValueError"
    assert ran == 1


def test_a_tool_raising_fresh_keeps_every_retry(engine):
    def fail(op):
        raise TimeoutError("the tool timed out")

    def calls():
        return (yield from call_tool("t", {}, dict))

    snapshot, ran = executions(engine, calls, Tools({"t": fail}))
    assert snapshot.state is TaskState.FAILED
    assert ran == 5


def test_a_tool_raising_fresh_is_a_fresh_effect_and_no_store_read(engine):
    def fail(op):
        raise TimeoutError("the tool timed out")

    def calls():
        return (yield from call_tool("t", {}, dict))

    counts = []

    def body(params, ctx):
        handler = DurableHandler(ctx, Tools({"t": fail}))
        try:
            return handler.run(calls)
        finally:
            counts.append((handler._unreplayed.ran, handler._unreplayed.read))

    engine.register_task("case")(body)
    engine.run_until_result(engine.spawn("case", {}, max_attempts=1))
    assert counts == [(1, 0)]


def test_a_refusal_beside_a_crash_is_retried_and_caught_once_the_crash_clears(engine):
    """The retry clears the crash; the refusal alone is then delivered into the workflow, which
    catches it and completes."""
    crashes = []

    def refuse(op):
        raise Refused(op, "no")

    def crash_once(op):
        if not crashes:
            crashes.append(1)
            raise RuntimeError("the worker's tool crashed once")
        return {"n": 2}

    def catches():
        try:
            return (
                yield from gather(
                    [lambda: call_tool("r", {}, dict), lambda: call_tool("c", {}, dict)]
                )
            )
        except ExceptionGroup as refused:
            return [type(leaf).__name__ for leaf in refused.exceptions]

    snapshot, ran = executions(engine, catches, Tools({"r": refuse, "c": crash_once}))
    assert snapshot.state is TaskState.COMPLETED
    assert snapshot.result == ["Refused"]
    assert ran == 2


def test_a_loser_that_raised_after_the_choice_is_reported_once_on_a_warning_span():
    """The race returns its winner, so the loser's error reaches its reader through telemetry,
    once: the retry that replays the race reads its endings from the store."""
    started, settled = threading.Event(), threading.Event()
    spans = []

    def held(op):
        started.set()
        settled.wait(10)
        return "recorded"

    def loser():
        value = yield from call_tool("b0", {}, str)
        raise KeyError("the loser's code raised on " + value)

    def program():
        yield from race([lambda: call_tool("a", {}, dict), loser])
        return (yield from call_tool("after", {}, dict))

    class Settles:
        """Releases the held loser once the race's choice is saved."""

        def __init__(self, ctx):
            self._ctx = ctx

        def settle(self, name, value):
            saved = self._ctx.settle(name, value)
            settled.set()
            return saved

        def __getattr__(self, name):
            return getattr(self._ctx, name)

    app = SqliteApp()
    crashed = []

    def crash_once(op):
        """A fresh crash after the race, so a retry replays it."""
        if not crashed:
            crashed.append(1)
            raise RuntimeError("after the race")
        return {"n": 2}

    tools = Tools({"a": lambda op: started.wait(10) and {"n": 1}, "b0": held, "after": crash_once})

    def body(params, ctx):
        settles: Any = Settles(ctx)  # duck-typed: the ctx it wraps answers the rest
        return DurableHandler(settles, tools, sink=spans.append).run(program)

    app.register_task("race")(body)
    done = app.run_until_result(app.spawn("race", {}))
    assert done is not None
    assert done.state is TaskState.COMPLETED
    [warning] = [span for span in spans if span.severity == "warning"]
    assert warning.fields | {"message": None} == {
        "race": 0,
        "branch": 1,
        "raised": "KeyError",
        "message": None,
    }
    app.close()


def code_error() -> KeyError:
    """What the walk raises out of a workflow's own code on a recorded value."""
    error = KeyError("on a recorded value")
    mark_rederived(error)
    return error


@pytest.mark.parametrize(
    ("error", "delayed", "fails_now"),
    [
        (code_error(), False, True),
        (code_error(), True, False),
        (RecordedValueRejected("a recorded payload"), True, True),
    ],
    ids=["a code error, no wait", "a code error, a retry that waits", "a rejected record, waits"],
)
def test_a_retry_that_waits_keeps_a_code_errors_retries_and_no_other(error, delayed, fails_now):
    """A wait leaves time to deploy a fix to the code that raised; no deploy changes a record."""
    leaf = failing_leaf(error, Attempt(number=1, limit=5, delayed=delayed))
    assert (leaf is error) is fails_now


@pytest.mark.parametrize(
    ("strategy", "waits"),
    [
        (None, False),
        ({"kind": "none"}, False),
        ({"kind": "fixed", "base_seconds": 0}, False),
        ({"kind": "fixed"}, True),
        ({"kind": "exponential"}, True),
        ({"kind": "exponential", "base_seconds": 2}, True),
        ({"kind": "fixed", "base_seconds": None}, True),
        ({"kind": "fixed", "base_seconds": "60"}, True),
        ({"kind": "exponential", "base_seconds": "0"}, False),
    ],
)
def test_an_absurd_retry_waits_as_its_strategy_computes(strategy, waits):
    assert retry_waits(strategy) is waits


EXPONENTIAL = {"kind": "exponential", "base_seconds": 10}


@pytest.mark.parametrize(
    ("strategy", "attempt", "seconds"),
    [
        ({"kind": "fixed", "base_seconds": 5}, 3, 5.0),
        (EXPONENTIAL, 1, 10.0),
        (EXPONENTIAL, 3, 40.0),
        ({**EXPONENTIAL, "factor": 3}, 3, 90.0),
        ({**EXPONENTIAL, "max_seconds": 25}, 3, 25.0),
        ({**EXPONENTIAL, "max_seconds": 0}, 1, 0.0),
        ({**EXPONENTIAL, "factor": 0}, 1, 10.0),
        ({**EXPONENTIAL, "factor": 0}, 2, 0.0),
        ({**EXPONENTIAL, "factor": 10}, 400, 86400.0),
        ({**EXPONENTIAL, "factor": 0.5}, 2000, 0.0),
        ({"kind": "exponential"}, 2, 60.0),
        ({"kind": "fixed", "base_seconds": "soon"}, 1, 0.0),
        ({"kind": "linear"}, 1, 0.0),
    ],
)
def test_an_absurd_retry_delay_is_its_sqls(strategy, attempt, seconds):
    """`absurd.retry_delay_seconds`, case by case: the defaults, the factor, the cap, a factor
    that zeroes a later attempt, an overflow either way, and a strategy the SQL refuses."""
    assert retry_delay(strategy, attempt) == seconds


def test_a_store_error_raising_an_instance_a_thunk_raised_before_is_a_store_read(engine):
    """An attempt that ran its domain marks the error it raised; a later attempt's store raising
    that same instance is a store error all the same."""
    error = sqlite3.OperationalError("one instance")
    domain_calls: list[str] = []
    store_calls: list[str] = []
    counts: list[tuple[int, int]] = []

    def domain(op):
        domain_calls.append(op.name)
        if len(domain_calls) == 1:
            raise error
        return {"ok": True}

    class Busy:
        """A store whose second step raises the instance the domain raised first."""

        def __init__(self, ctx: Any) -> None:
            self._ctx = ctx

        def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
            store_calls.append(name.stored())
            if len(store_calls) == 2:
                raise error
            return self._ctx.step(name, thunk)

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._ctx, attr)

    def calls():
        return (yield from call_tool("t", {}, dict))

    def body(params, ctx):
        busy: Any = Busy(ctx)
        handler = DurableHandler(busy, Tools({"t": domain}))
        try:
            return handler.run(calls)
        finally:
            counts.append((handler._unreplayed.ran, handler._unreplayed.read))

    engine.register_task("case")(body)
    snapshot = engine.run_until_result(engine.spawn("case", {}, max_attempts=5))
    assert snapshot is not None
    assert (snapshot.state, counts) == (TaskState.COMPLETED, [(1, 0), (0, 1), (1, 0)])


def test_the_handler_leaves_a_refusals_attributes_as_it_found_them(engine):
    """A refusal is the error a workflow can catch, and a race witnesses one by its attributes, so
    counting it adds none."""
    caught: list[BaseException] = []

    def calls():
        try:
            return (yield from call_tool("t", {}, dict))
        except Refused as raised:
            caught.append(raised)
            return "caught"

    snapshot, _ = executions(engine, calls, Tools({"t": refused}))
    assert snapshot.state is TaskState.COMPLETED
    assert [sorted(vars(raised)) for raised in caught] == [["op", "reason"]]


@op_layer
def winner_fallback(op):
    """Answers the `winner` tool's store error with a value, as a deterministic fallback does."""
    try:
        return (yield op)
    except sqlite3.OperationalError:
        if getattr(getattr(op, "op", None), "name", None) == "winner":
            return {"ok": True}
        raise


class LoserFailsAfterTheChoice:
    """A store whose first step of the task calling `tool` waits for the race's choice and then
    raises `shared`; `failed` outlives the attempt."""

    def __init__(
        self,
        ctx: Any,
        shared: Exception,
        reading: threading.Event,
        published: threading.Event,
        failed: list[int],
        tool: str = "loser",
    ) -> None:
        self._ctx = ctx
        self._tool = tool
        self._shared = shared
        self._reading = reading
        self._published = published
        self._failed = failed

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        if direct_tool_key(self._tool).stored() in name.stored() and not self._failed:
            self._reading.set()
            assert self._published.wait(5)
            self._failed.append(1)
            raise self._shared
        return self._ctx.step(name, thunk)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


def test_a_losers_store_error_reusing_the_winners_instance_is_a_store_read(engine, monkeypatch):
    """The winner's domain raises an instance its layer answers; after the choice, the loser's
    store raises that same instance. The loser read the store, so the attempt is retried."""
    shared = sqlite3.OperationalError("raised by the domain and by the store")
    loser_reading, published = threading.Event(), threading.Event()
    winner_calls: list[int] = []
    counts: list[tuple[int, int]] = []
    failed: list[int] = []
    publish = RaceState.publish

    def publish_then_release(self, proposal, settle):
        chosen = publish(self, proposal, settle)
        published.set()
        return chosen

    monkeypatch.setattr(RaceState, "publish", publish_then_release)

    def winner(op):
        winner_calls.append(1)
        if len(winner_calls) == 1:
            assert loser_reading.wait(5)
            raise shared
        return {"ok": True}

    def workflow():
        answer = yield from race(
            [lambda: call_tool("winner", {}, dict), lambda: call_tool("loser", {}, dict)]
        )
        if any(type(ending).__name__ == "Raised" for ending in answer.endings):
            raise ValueError("the workflow rejects a loser that raised")
        return "recovered"

    def body(params, ctx):
        base_ctx = ConcurrentAbsurdCtx(ctx) if isinstance(engine, AbsurdEngine) else ctx
        inner: Any = LoserFailsAfterTheChoice(base_ctx, shared, loser_reading, published, failed)
        handler = DurableHandler(inner, Tools({"winner": winner}), op_layers=(winner_fallback,))
        try:
            return handler.run(workflow)
        finally:
            counts.append((handler._unreplayed.ran, handler._unreplayed.read))

    engine.register_task("case")(body)
    snapshot = engine.run_until_result(engine.spawn("case", {}, max_attempts=5))
    assert snapshot is not None
    assert (snapshot.state, snapshot.result, counts) == (
        TaskState.COMPLETED,
        "recovered",
        [(1, 1), (1, 0)],
    )


def test_a_losers_verdict_is_its_own_where_a_sibling_raises_its_instance(engine, monkeypatch):
    """A loser whose store failed after the choice and a sibling whose code raises the same
    instance each keep their own verdict: the store loser's error fails the attempt."""
    shared = sqlite3.OperationalError("one instance, two branches")
    reading, raising, published = threading.Event(), threading.Event(), threading.Event()
    counts: list[tuple[int, int]] = []
    failed: list[int] = []
    publish = RaceState.publish

    def publish_then_release(self, proposal, settle):
        chosen = publish(self, proposal, settle)
        published.set()
        return chosen

    monkeypatch.setattr(RaceState, "publish", publish_then_release)

    def winner(op):
        assert reading.wait(5)
        assert raising.wait(5)
        return {"ok": True}

    def code_loser():
        yield from ()
        raising.set()
        raise shared

    def workflow():
        answer = yield from race(
            [
                lambda: call_tool("store_loser", {}, dict),
                lambda: call_tool("winner", {}, dict),
                code_loser,
            ]
        )
        if type(answer.endings[0]).__name__ == "Raised":
            raise ValueError("the workflow rejects a store loser that raised")
        return "recovered"

    def body(params, ctx):
        base_ctx = ConcurrentAbsurdCtx(ctx) if isinstance(engine, AbsurdEngine) else ctx
        store: Any = LoserFailsAfterTheChoice(
            base_ctx, shared, reading, published, failed, tool="store_loser"
        )
        handler = DurableHandler(store, Tools({"winner": winner}))
        try:
            return handler.run(workflow)
        finally:
            counts.append((handler._unreplayed.ran, handler._unreplayed.read))

    engine.register_task("case")(body)
    snapshot = engine.run_until_result(engine.spawn("case", {}, max_attempts=5))
    assert snapshot is not None
    assert (snapshot.state, snapshot.result, counts) == (
        TaskState.COMPLETED,
        "recovered",
        [(1, 1), (0, 0)],
    )


def wrapped_executions(engine, workflow, wrap: Callable[[Any], Any], *, name: str = "case"):
    """`executions` over a ctx `wrap` builds, a race's branches in index order."""
    ran = []

    def body(params, ctx):
        ran.append(1)
        return DurableHandler(wrap(InOrder(ctx)), Tools()).run(workflow)

    engine.register_task(name)(body)
    task_id = engine.spawn(name, {}, max_attempts=5)
    snapshot = engine.run_until_result(task_id)
    assert snapshot is not None
    return snapshot, len(ran)


class ReparkReturns:
    """A ctx whose branch peeks miss each event once, as an event arriving mid-round does, and
    whose repark returns without parking, as the SDK's stale-checkpoint edge does."""

    def __init__(self, ctx: Any, missed: set[str]) -> None:
        self._ctx = ctx
        self._missed = missed

    def peek_event(self, name: Key, /) -> tuple[bool, Any]:
        if (stored := name.stored()) not in self._missed:
            self._missed.add(stored)
            return False, None
        return self._ctx.peek_event(name)

    def repark(self, name: Key, /) -> None:
        return None

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


class AttemptBusy:
    """A ctx whose first read of its attempt fails as a busy SQLite database does."""

    def __init__(self, ctx: Any, busy: list[bool]) -> None:
        self._ctx = ctx
        self._busy = busy

    @property
    def attempt(self) -> Any:
        if not self._busy:
            self._busy.append(True)
            raise sqlite3.OperationalError("database is locked")
        return self._ctx.attempt

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


def test_a_gather_whose_events_arrive_mid_round_keeps_its_retries(engine):
    """Every branch's event arrived between its peek and the re-arm, and the repark returned, so
    the gather raises its wake race; the retry resolves the branches from the record."""
    missed: set[str] = set()

    def waits():
        return (yield from await_event("ev", dict))

    def workflow():
        return (yield from gather([waits]))

    match engine:
        case AbsurdEngine():
            wrap = lambda ctx: ReparkReturns(ConcurrentAbsurdCtx(ctx), missed)  # noqa: E731
        case _:
            wrap = lambda ctx: ReparkReturns(ctx, missed)  # noqa: E731
    engine.emit_event("gather:0,0;ev", {"ok": True})
    snapshot, ran = wrapped_executions(engine, workflow, wrap)
    assert (snapshot.state, ran) == (TaskState.COMPLETED, 2)


def test_a_race_whose_attempt_read_fails_keeps_its_retries(engine):
    """Reading the attempt is a store call, and a busy store is not what a retry finds."""
    busy: list[bool] = []

    def workflow():
        answer = yield from race([lambda: call_tool("a", {}, dict)])
        return type(answer).__name__

    snapshot, ran = wrapped_executions(engine, workflow, lambda ctx: AttemptBusy(ctx, busy))
    assert (snapshot.state, ran) == (TaskState.COMPLETED, 2)


def test_a_failed_attempt_leaves_only_its_ops_on_the_record(engine):
    """What decides a retry is counted while the attempt runs, so the store holds only the ops."""
    workflow, tools, layers = a_layer_answers_a_fresh_failure(pytest.MonkeyPatch())

    def body(params, ctx):
        return DurableHandler(ctx, tools, op_layers=layers).run(workflow)

    engine.register_task("case")(body)
    task_id = engine.spawn("case", {}, max_attempts=5)
    snapshot = engine.run_until_result(task_id)
    assert snapshot is not None
    assert snapshot.state is TaskState.COMPLETED
    match engine:
        case AbsurdEngine():
            with psycopg.connect(DSN) as conn:
                every = read_absurd_task(conn, task_id, queue=engine.queue, exclude=())
        case _:
            every = read_sqlite_conn(engine.conn, task_id, exclude=())
    assert [c.key.display() for c in every] == ["step;tool:flaky"]


@pytest.mark.parametrize(
    ("strategy", "ran", "state"),
    [
        ({"kind": "fixed", "base_seconds": 0}, 2, TaskState.FAILED),
        ({"kind": "fixed", "base_seconds": None}, 1, TaskState.SLEEPING),
    ],
    ids=["a retry at once", "a retry that waits the default"],
)
def test_an_absurd_code_error_fails_when_it_repeats_unless_its_retry_waits(
    engine, strategy, ran, state
):
    """A strategy the SQL accepts, a null base among them, is read without failing the edge."""
    if not isinstance(engine, AbsurdEngine):
        pytest.skip("a retry strategy is Absurd's")
    attempts = []

    def reads():
        got = yield from call_tool("t", {}, dict)
        return got["missing"]

    def body(params, ctx):
        attempts.append(1)
        return DurableHandler(ctx, Tools()).run(reads)

    engine.register_task("case")(body)
    spawned = engine.app.spawn(
        "case", {}, max_attempts=5, retry_strategy=strategy, queue=engine.queue
    )
    snapshot = engine.run_until_result(UUID(str(spawned["task_id"])))
    assert snapshot is not None
    assert (len(attempts), snapshot.state) == (ran, state)
    runs = queue_table("r", engine.queue)
    with psycopg.connect(DSN) as conn:
        failed_of = conn.execute(
            t"SELECT failure_reason->>'name' FROM absurd.{runs:i} "
            t"WHERE task_id = {spawned['task_id']} AND failure_reason IS NOT NULL"
        ).fetchall()
    assert {name for (name,) in failed_of} == {"KeyError"}  # not an error reading its claim


class Flaky:
    """A tool that fails fresh on its first call and returns on every later one."""

    def __init__(self) -> None:
        self.failed = False

    def __call__(self, op: Any) -> Any:
        if not self.failed:
            self.failed = True
            raise TimeoutError(op.name)
        return {"ok": 7}


class Required(BaseModel):
    missing: int


def reads_missing():
    got = yield from call_tool("recorded", {}, dict)
    return got["missing"]


def reads_required():
    return (yield from call_tool("recorded", {}, Required))


def refused_composition():
    yield from call_tool("recorded", {}, dict)
    return (yield from quorum(2, [flaky_branch]))


def flaky_branch():
    return (yield from call_tool("flaky", {}, dict))


def fresh_fail_branch():
    return (yield from call_tool("fresh_fail", {}, dict))


def ok_branch():
    return (yield from call_tool("ok", {}, dict))


def raced(*branches):
    def workflow():
        answer = yield from race(list(branches))
        return type(answer).__name__

    return workflow


def fail(op):
    raise TimeoutError(op.name)


type Case = Callable[[pytest.MonkeyPatch], tuple[Any, Tools, tuple[Any, ...]]]


def a_layer_answers_a_fresh_failure(monkeypatch):
    @op_layer
    def fallback(op):
        try:
            return (yield op)
        except TimeoutError:
            return {}

    def reads():
        got = yield from call_tool("flaky", {}, dict)
        return got["ok"]

    return reads, Tools({"flaky": Flaky()}), (fallback,)


def a_schema_default_fills_what_the_record_lacked(monkeypatch):
    defaults = iter([0, 1, 1, 1, 1])

    class Defaulted(BaseModel):
        n: int = Field(default_factory=lambda: next(defaults))

    def reads():
        got = yield from call_tool("t", {}, Defaulted)
        if got.n == 0:
            raise ValueError("the default was zero")
        return got.n

    return reads, Tools({"t": lambda op: {}}), ()


def a_race_deadline_passes_between_attempts(monkeypatch):
    starts = []
    monkeypatch.setattr(base, "race_time", lambda: 0.0 if len(starts) == 1 else 20.0)

    def timed():
        starts.append(1)
        answer = yield from race([reads_missing], deadline=datetime.fromtimestamp(10, UTC))
        return type(answer).__name__

    return timed, Tools(), ()


def a_flaky_branch_beside_a_code_error(monkeypatch):
    return raced(flaky_branch, reads_missing), Tools({"flaky": Flaky()}), ()


def raced_with_flaky(bad: Callable[[], Any]) -> Case:
    return lambda monkeypatch: (raced(flaky_branch, bad), Tools({"flaky": Flaky()}), ())


def raced_before(first: Callable[[], Any], tools: dict[str, Any]) -> Case:
    return lambda monkeypatch: (raced(first, ok_branch), Tools(tools), ())


def outcomes(engine, monkeypatch, cases: dict[str, tuple[Case, Any, int]]):
    """Each case's end state, result and attempts, one task per case on one engine."""
    got = {}
    for i, (name, (case, _, _)) in enumerate(cases.items()):
        workflow, tools, layers = case(monkeypatch)
        snapshot, ran = executions(engine, workflow, tools, layers=layers, name=f"case{i}")
        got[name] = (snapshot.state, snapshot.result, ran)
    return got


RECOVERS: dict[str, tuple[Case, Any, int]] = {
    "a layer answers a fresh failure with a value it never records": (
        a_layer_answers_a_fresh_failure,
        7,
        2,
    ),
    "a schema default fills a field the record lacked": (
        a_schema_default_fills_what_the_record_lacked,
        1,
        2,
    ),
    "a race's deadline passes between attempts": (
        a_race_deadline_passes_between_attempts,
        "TimedOut",
        2,
    ),
    "a flaky branch beside a code error before the choice": (
        a_flaky_branch_beside_a_code_error,
        "Chosen",
        2,
    ),
}


def test_a_code_error_a_retry_changes_completes_on_the_next_attempt(engine, monkeypatch):
    """Each case raises in workflow code on attempt 1, and attempt 2 reaches another outcome: a
    fresh effect, a default, or a clock past a deadline."""
    assert outcomes(engine, monkeypatch, RECOVERS) == {
        name: (TaskState.COMPLETED, value, ran) for name, (_, value, ran) in RECOVERS.items()
    }


def a_race_deadline_passing_before_the_third_attempt(monkeypatch):
    starts = []
    monkeypatch.setattr(base, "race_time", lambda: 0.0 if len(starts) < 3 else 20.0)

    def timed():
        starts.append(1)
        answer = yield from race([reads_missing], deadline=datetime.fromtimestamp(10, UTC))
        return type(answer).__name__

    return timed, Tools(), ()


def the_same_race_inside_a_gather(monkeypatch):
    timed, tools, layers = a_race_deadline_passing_before_the_third_attempt(monkeypatch)
    return lambda: gather([timed]), tools, layers


LATER: dict[str, tuple[Case, Any, int]] = {
    "a race's deadline passing before the third attempt": (
        a_race_deadline_passing_before_the_third_attempt,
        "TimedOut",
        3,
    ),
    "the same race inside a gather": (the_same_race_inside_a_gather, ["TimedOut"], 3),
}


def test_a_race_failing_before_its_deadline_keeps_its_retries(engine, monkeypatch):
    """Each case raises the same error on attempts 1 and 2 with nothing run fresh, from a race
    that read the clock before its deadline, and completes on attempt 3."""
    assert outcomes(engine, monkeypatch, LATER) == {
        name: (TaskState.COMPLETED, value, ran) for name, (_, value, ran) in LATER.items()
    }


RACES: dict[str, tuple[Case, Any, int]] = {
    "a typed refusal to compose beside a flaky branch": (
        raced_with_flaky(refused_composition),
        "Chosen",
        2,
    ),
    "a rejected record beside a flaky branch": (raced_with_flaky(reads_required), "Chosen", 2),
    "a fresh failure written before a branch that succeeds": (
        raced_before(fresh_fail_branch, {"fresh_fail": fail}),
        "Chosen",
        1,
    ),
    "a rejected record written before a branch that succeeds": (
        raced_before(reads_required, {}),
        "Chosen",
        1,
    ),
}


def test_a_branch_error_before_the_choice_waits_while_a_sibling_can_win(engine, monkeypatch):
    """A branch that errs before the choice ends as that branch's slot while a sibling can still
    win, so the order the branches are written in decides nothing."""
    assert outcomes(engine, monkeypatch, RACES) == {
        name: (TaskState.COMPLETED, value, ran) for name, (_, value, ran) in RACES.items()
    }


def refused(op):
    raise Refused(op, "no")


def refuse_branch():
    return (yield from call_tool("refuse", {}, dict))


def a_fresh_failure_as_the_deadline_passes(monkeypatch):
    """The clock passes the deadline once the branch has failed, so attempt 1 ends with no branch
    left to win, and attempt 2 finds the deadline behind it before any branch runs."""
    tools = Tools({"fresh_fail": fail})
    monkeypatch.setattr(base, "race_time", lambda: 20.0 if tools.calls else 0.0)

    def timed():
        answer = yield from race([fresh_fail_branch], deadline=datetime.fromtimestamp(10, UTC))
        return type(answer).__name__

    return timed, tools, ()


UNDECIDED: dict[str, tuple[Case, tuple[TaskState, Any, int]]] = {
    "a fresh failure beside a refusal": (
        lambda monkeypatch: (
            raced(fresh_fail_branch, refuse_branch),
            Tools({"fresh_fail": fail, "refuse": refused}),
            (),
        ),
        (TaskState.FAILED, "TimeoutError", 5),
    ),
    "a code error beside a refusal": (
        lambda monkeypatch: (raced(reads_missing, refuse_branch), Tools({"refuse": refused}), ()),
        (TaskState.FAILED, "KeyError", 2),
    ),
    "a fresh failure as the deadline passes": (
        a_fresh_failure_as_the_deadline_passes,
        (TaskState.COMPLETED, "TimedOut", 2),
    ),
}


def test_a_race_no_branch_can_win_fails_with_its_errors(engine, monkeypatch):
    """No sibling is left to win, so the race fails naming the error, never answering
    `Impossible` or `TimedOut` on that attempt, and the task retries as that error would."""
    got = {}
    for i, (name, (case, _)) in enumerate(UNDECIDED.items()):
        workflow, tools, layers = case(monkeypatch)
        snapshot, ran = executions(engine, workflow, tools, layers=layers, name=f"case{i}")
        kind = snapshot.failure.kind if snapshot.failure is not None else snapshot.result
        got[name] = (snapshot.state, kind, ran)
    assert got == {name: expected for name, (_, expected) in UNDECIDED.items()}


def a_layer_raises_on_a_recorded_result(monkeypatch):
    @op_layer
    def broken(op):
        value = yield op
        return value["missing"]

    return lambda: call_tool("t", {}, dict), Tools(), (broken,)


def a_step_name_that_is_no_atom(monkeypatch):
    return lambda: step("a/b", CallTool(name="t", result_schema=dict)), Tools(), ()


def a_scope_whose_body_factory_raises(monkeypatch):
    def factory():
        raise ValueError("the scope's body failed to build")

    return lambda: scoped(Key.parse("scope"), factory), Tools(), ()


def a_validator_with_a_code_error(monkeypatch):
    class Broken(BaseModel):
        n: int

        @field_validator("n")
        @classmethod
        def broken(cls, value: int) -> int:
            raise TypeError("the validator's code raised")

    return lambda: call_tool("t", {}, Broken), Tools(), ()


def a_return_that_does_not_serialize(monkeypatch):
    def returns():
        yield from call_tool("t", {}, dict)
        return object()

    return returns, Tools(), ()


REPEATS: dict[str, tuple[Case, Any, int]] = {
    "a layer raises on a recorded result": (a_layer_raises_on_a_recorded_result, None, 2),
    "a step name that is no atom": (a_step_name_that_is_no_atom, None, 1),
    "a scope whose body factory raises": (a_scope_whose_body_factory_raises, None, 1),
    "a validator with a code error": (a_validator_with_a_code_error, None, 2),
    "a return that does not serialize": (a_return_that_does_not_serialize, None, 2),
}


def test_an_attempt_that_replays_its_record_fails_its_task(engine, monkeypatch):
    """Each case raises outside the workflow's own frame, the same way on every attempt: on the
    first attempt that runs nothing fresh."""
    assert outcomes(engine, monkeypatch, REPEATS) == {
        name: (TaskState.FAILED, None, ran) for name, (_, _, ran) in REPEATS.items()
    }


def test_a_code_error_whose_message_changes_fails_on_the_attempt_that_replays(engine):
    """The error's text holds an address that differs on every attempt, and the attempt that
    replays its record raises it all the same."""
    kept = []

    def reads():
        yield from call_tool("t", {}, dict)
        key = object()
        kept.append(key)
        return {}[key]

    snapshot, ran = executions(engine, reads, Tools())
    assert (snapshot.state, ran) == (TaskState.FAILED, 2)


SHARED = KeyError("one instance, raised by more than one task")


def test_an_error_instance_marked_in_one_task_keeps_its_retries_when_raised_fresh_in_another(
    engine,
):
    """The mark says what this raise repeats, so an instance another task's code raised twice
    is retried when a tool raises it fresh."""

    def repeats():
        yield from call_tool("t", {}, dict)
        raise SHARED

    def raise_shared(op):
        raise SHARED

    def calls():
        return (yield from call_tool("s", {}, dict))

    _, first = executions(engine, repeats, Tools(), name="first")
    _, second = executions(engine, calls, Tools({"s": raise_shared}), name="second")
    assert (first, second) == (2, 5)


def test_a_respawn_whose_enqueue_fails_fresh_keeps_its_retries(backend):
    """The successor's enqueue is a domain call, so an attempt it failed in ran something fresh."""
    chain_task, run, failures, spawns, ran = private("chain"), str(uuid4()), 2, [], []

    def flaky_spawner(task_name, params, idempotency_key, queue, *, max_attempts=None):
        spawns.append(1)
        if len(spawns) <= failures:
            raise TimeoutError("the queue was unreachable")
        return backend.spawner(
            task_name, params, idempotency_key, queue, max_attempts=max_attempts
        )

    domain = MeteredInterpreter(
        llm=lambda op: ("", Usage()),
        tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(flaky_spawner)}),
    )

    def step(state: int, turn: Turn):
        yield from ()
        return Done("finished") if turn.generation >= 1 else Again(state + 1)

    def body(params, ctx):
        ran.append(1)
        carried = Chain.from_params(params, task=chain_task, schema=int, initial=0, run_id=run)
        chain = Chain(
            task=chain_task, state=carried.state, run_id=run, generation=carried.generation
        )
        return DurableHandler(ctx, domain, params=params).run(lambda: respawn(step, chain))

    backend.register_body(chain_task, body)
    snapshot = backend.run_until_result(backend.spawn(chain_task, "r-chain", max_attempts=5))
    assert (snapshot.state, len(ran), len(backend.enqueued(chain_task)) - 1) == (
        "completed",
        3,
        1,
    )


def test_a_fork_childs_code_error_fails_on_the_attempt_that_repeats_it(engine):
    """The record of what an attempt raised is neither seeded nor steered, so a fork child
    keeps it as a task does."""
    fork_point = compose_key(t"fork-point")

    def reads_after_the_fork_point():
        yield from await_event(fork_point, dict)
        got = yield from call_tool("recorded", {}, dict)
        return got["missing"]

    ran = []

    def body(params, ctx):
        ran.append(1)
        seeded: Any = SeedingCtx(ctx, {}, fork_point=fork_point)
        return DurableHandler(seeded, Tools()).run(reads_after_the_fork_point)

    engine.register_task("child")(body)
    task_id = engine.spawn("child", {}, max_attempts=5)
    engine.emit_event(fork_point.stored(), {"ok": True})
    snapshot = engine.run_until_result(task_id)
    assert snapshot is not None
    assert (snapshot.state, len(ran)) == (TaskState.FAILED, 2)


def test_a_group_of_refusals_is_no_fresh_effect(engine):
    """A domain that refuses in a group is refused all the same, so a workflow raising after it
    caught the group fails on that attempt, as after a bare refusal."""

    def refuse(op):
        raise ExceptionGroup("denied", [Refused(op, "no")])

    def catches():
        try:
            yield from call_tool("r", {}, dict)
        except ExceptionGroup:
            raise ValueError("handled badly") from None

    snapshot, ran = executions(engine, catches, Tools({"r": refuse}))
    assert (snapshot.state, ran) == (TaskState.FAILED, 1)
