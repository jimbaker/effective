"""Unit pins for the V1 await-in-gather park machinery
that the cross-backend conformance suite can't reach deterministically: the
GatherWakeRace fallback, the preserved wall on a peek-less ctx, park-vs-failure
classification, and the poison-stub replay proof on a resumed gather."""

import pytest
from _conformance import CountingDomain, gather_await_wf

from effective.api import await_event, call_tool, gather
from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.handlers.base import Attempt, failing_leaf
from effective.handlers.durable import DurableHandler, GatherWakeRace
from effective.keys import Key
from effective.ops import leaves


class _StubCtx:
    """A sequential ctx whose peek never sees the event but whose re-arm await
    resolves instantly — the mid-round-emission race, made deterministic."""

    concurrent_safe = False

    def step(self, name, thunk, /):
        return thunk()

    def await_event(self, name, /):
        return {"late": True}  # satisfied by the time the re-arm runs

    def peek_event(self, name, /):
        return False, None  # not yet visible during the round

    def sleep_until(self, when, /, *, name: Key | None = None):
        return None


class _EnginePark(Exception):
    """Stands in for the engine's own park signal (_Suspend / SuspendTask):
    a real repark raises it and the task re-queues without burning an attempt."""


class _ReparkCtx(_StubCtx):
    """_StubCtx plus the optional repark capability — records the step name it
    was asked to park under, then parks like a real engine would."""

    def __init__(self):
        self.reparked: list[str] = []

    def repark(self, name, /):
        self.reparked.append(name)
        raise _EnginePark(name)


class _ReparkReturnsCtx(_StubCtx):
    """A repark that RETURNS without parking — the SDK's stale-checkpoint edge
    (sleep_until finds a past wake time and falls through). The caller must
    still raise GatherWakeRace rather than proceed."""

    def __init__(self):
        self.reparked: list[str] = []

    def repark(self, name, /):
        self.reparked.append(name)


class _NoPeekCtx:
    """A minimal ctx WITHOUT the optional peek capability — the wall must stay
    up here, legibly (a raw SDK ctx not wrapped in ConcurrentAbsurdCtx)."""

    def step(self, name, thunk, /):
        return thunk()

    def await_event(self, name, /):
        return {}

    def sleep_until(self, when, /, *, name: Key | None = None):
        return None


def _await_branch_wf():
    def branch():
        def thunk():
            return (yield from await_event("ev", dict))

        return thunk

    return (yield from gather([branch()]))


def test_race_without_repark_capability_still_raises_gather_wake_race():
    """F2: a parked branch can never re-run in-process, so when every parked
    branch's wake condition is already satisfied at the barrier there is
    nothing to park on. Without the optional repark capability the documented
    GatherWakeRace fallback re-queues the task via the engine's retry."""
    with pytest.raises(GatherWakeRace, match="already satisfied"):
        DurableHandler(_StubCtx(), CountingDomain()).run(_await_branch_wf)


def test_race_with_repark_capability_parks_on_the_wake_race_step():
    """The no-burn contract: a ctx bearing repark re-queues the raced task via
    the engine's park path — no GatherWakeRace, no attempt burn. The step name
    is UNFORGEABLE (`wake-race` in the second segment — an author step in a
    branch always has an integer there via the gather prefix, and a top-level
    `gather:` name is rejected by op_key) and carries the lowest raced branch's
    wake CONDITION, so a later race of the same gather on a different await
    mints a fresh name (no stale-checkpoint burn)."""
    ctx = _ReparkCtx()
    with pytest.raises(_EnginePark):
        DurableHandler(ctx, CountingDomain()).run(_await_branch_wf)
    assert ctx.reparked == ["gather:0;wake-race:0;ev"]


def test_repark_that_returns_falls_through_to_gather_wake_race():
    """Repark-if-possible, raise-if-not: when repark returns without parking
    (the stale-checkpoint edge) the loud fallback must still fire — the task
    must never proceed to re-run parked branches in-process (F2)."""
    ctx = _ReparkReturnsCtx()
    with pytest.raises(GatherWakeRace, match="already satisfied"):
        DurableHandler(ctx, CountingDomain()).run(_await_branch_wf)
    assert ctx.reparked == ["gather:0;wake-race:0;ev"]


def test_a_wake_race_keeps_its_retries():
    """The re-arm found what the round's peeks did not, so a retry resolves the branches from the
    record: the fallback's error is no repeat."""
    with pytest.raises(GatherWakeRace) as raised:
        DurableHandler(_ReparkReturnsCtx(), CountingDomain()).run(_await_branch_wf)
    assert failing_leaf(raised.value, Attempt(1, 3)) is None


def test_peekless_ctx_keeps_the_legible_wall():
    """A ctx without peek_event cannot support a branch park; the failure names
    the capability rather than dying as a mislabeled crash."""
    with pytest.raises((NotImplementedError, ExceptionGroup)) as raised:
        DurableHandler(_NoPeekCtx(), CountingDomain()).run(_await_branch_wf)
    (leaf,) = leaves(raised.value)
    assert isinstance(leaf, NotImplementedError)
    assert "peek_event capability" in str(leaf)


@pytest.fixture
def app():
    """A fresh in-memory durable engine, closed on teardown (no leaked sqlite3 connection)."""
    a = SqliteApp(":memory:")
    yield a
    a.close()


def test_branch_failure_is_still_a_failure_not_a_park(app):
    """Park-vs-failure classification: an ordinary branch exception fails the
    task exactly as before — the park machinery must not absorb it."""
    domain = CountingDomain()

    def boom_wf(run_id: str):
        def ok_branch():
            def thunk():
                return (yield from call_tool("a", {}, int))

            return thunk

        def boom_branch():
            def thunk():
                raise ValueError("branch boom")
                yield  # pragma: no cover — generator marker

            return thunk

        return (yield from gather([ok_branch(), boom_branch()]))

    @app.register_task("boom")
    def _task(params, ctx):
        return DurableHandler(ctx, domain).run(lambda: boom_wf("r1"))

    snap = app.run_until_result(app.spawn("boom", {}, max_attempts=1))
    assert snap is not None
    assert snap.state == "failed"
    assert "branch boom" in str(snap.failure)


def test_completed_parked_gather_replays_under_a_poison_domain(app):
    """The poison-stub proof on the resumed prefix: after a parked gather
    completes, re-driving the same workflow over the same store with a domain
    that RAISES if touched re-binds every op — committed steps and the
    delivered event alike — to the identical result, executing nothing."""
    domain = CountingDomain()

    @app.register_task("t")
    def _task(params, ctx):
        return DurableHandler(ctx, domain).run(lambda: gather_await_wf("r1"))

    task_id = app.spawn("t", {})
    snap = app.run_until_result(task_id)
    assert snap is not None
    assert snap.state not in ("completed", "failed")  # parked
    app.emit_event("gather:0,1;ev:r1", {"ok": True})
    snap = app.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed"
    assert domain.calls == ["a"]

    class PoisonDomain:
        def run(self, op):
            raise AssertionError("domain executed during replay")

    ctx = SqliteTaskContext(app.conn, task_id, app.write_lock)
    replayed = DurableHandler(ctx, PoisonDomain()).run(lambda: gather_await_wf("r1"))
    assert replayed == {"results": [10, {"ok": True}]}
