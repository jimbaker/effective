"""A wait that names a deadline: what each interpreter answers, and what it refuses.

`await_until` was an engine capability with no op above it, so no workflow could reach it.
These are the pins for the op, and
they sit beside the cross-engine cases in `tests/test_conformance.py`, which run the same
shapes against both engines.
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, get_args, get_origin, get_type_hints
from uuid import uuid4

import pytest
from _conformance import review_name
from _shapes import run

from effective.api import Effect, await_until, call_tool, gather
from effective.api import sleep_until as api_sleep_until
from effective.domain import DomainOp
from effective.engines.sqlite import SqliteApp
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Key
from effective.ops import (
    Addressing,
    Arrived,
    AwaitEvent,
    CompositionRefused,
    Expired,
    Race,
    SleepUntil,
    WorkflowOp,
)

pytestmark = pytest.mark.adversarial

NAME = Key.parse("ev:x")
AHEAD = datetime.now(UTC) + timedelta(hours=1)


def _waits_until(deadline: datetime = AHEAD):
    def wf():
        outcome = yield from await_until(NAME, dict, deadline=deadline)
        return outcome

    return wf


class _Tool:
    def run(self, op: DomainOp) -> Any:
        return 1


NAIVE = AHEAD.replace(tzinfo=None)

BOUNDED: dict[tuple[type, str], Callable[[Any], object]] = {
    (AwaitEvent, "deadline"): lambda at: AwaitEvent(NAME, dict, deadline=at),
    (SleepUntil, "when"): lambda at: SleepUntil(when=at),
    (Race, "deadline"): lambda at: Race(want=1, branches=(lambda: iter(()),), deadline=at),
}
"""Every op field that takes an instant as a bound, built around one."""

WHAT = {AwaitEvent: "a wait's deadline", SleepUntil: "a sleep's end", Race: "a race's deadline"}


def test_the_table_holds_every_instant_an_op_takes() -> None:
    """Read off the op union's field types, so an op that gains an instant joins the table."""
    fields = {
        (op, name)
        for arm in get_args(WorkflowOp.__value__)
        for op in [get_origin(arm) or arm]
        for name, hint in get_type_hints(op).items()
        if datetime in (hint, *get_args(hint))
    }
    assert fields == set(BOUNDED)


@pytest.mark.parametrize("instant", ["naive", "a string"])
@pytest.mark.parametrize("bound", list(BOUNDED), ids=lambda b: WHAT[b[0]])
def test_a_bound_is_an_absolute_instant(bound, instant) -> None:
    """A naive datetime reads as the local time of whichever host converts it, so an attempt that
    retries in another zone would wait on a different instant. Each op takes an aware one."""
    BOUNDED[bound](AHEAD)
    wrong = {"naive": NAIVE, "a string": AHEAD.isoformat()}[instant]
    with pytest.raises(CompositionRefused, match=f"{WHAT[bound[0]]} is an absolute instant"):
        BOUNDED[bound](wrong)


PAST = datetime(2026, 1, 1, tzinfo=UTC)
"""An instant already past, so a sleep on it ends without parking. A fixed date."""


class _Clock:
    """Answers `clock` with `PAST`, aware or with its zone dropped."""

    def __init__(self, aware: bool) -> None:
        self.aware = aware

    def run(self, op: DomainOp) -> Any:
        return PAST if self.aware else PAST.replace(tzinfo=None)


def _sleeps_on_the_clock() -> Effect[str]:
    when = yield from call_tool("clock", {}, datetime)
    yield from api_sleep_until(when)
    return "woke"


INSIDE: dict[str, Callable[[], Effect[Any]]] = {
    "the workflow": _sleeps_on_the_clock,
    "a gather branch": lambda: gather([_sleeps_on_the_clock]),
}


@pytest.mark.parametrize("where", list(INSIDE))
def test_an_instant_read_through_a_step_keeps_its_zone_on_an_engine(backend, where) -> None:
    """The instant a workflow reads through a step comes back from the checkpoint aware, and a
    naive one fails its task on the first attempt, inside a branch as outside one. A race branch
    is not a row: it refuses a sleep by its kind, before any instant is read."""
    program = lambda _run_id: INSIDE[where]()  # noqa: E731
    aware = run(backend, program, _Clock(aware=True), max_attempts=3)
    assert aware.snap.state == "completed", aware.snap
    naive = run(backend, program, _Clock(aware=False), max_attempts=3)
    assert naive.snap.state == "failed", naive.snap
    assert backend.failure_kind(naive.snap) == "CompositionRefused"
    assert backend.task_attempts(naive.task) == 1


FAR = datetime(2100, 1, 1, tzinfo=UTC)
"""Past the seconds an `int4` holds, counted from now."""


def _waits_for_an_absolute_arrival(name: Key) -> Effect[str]:
    outcome = yield from await_until(name, dict, deadline=FAR, addressing=Addressing.ABSOLUTE)
    return type(outcome).__name__


def test_a_wait_whose_deadline_is_decades_out_parks_and_takes_its_arrival(backend) -> None:
    """The name is the run's own, since an Absurd event outlives the test that emitted it."""
    name = review_name(str(uuid4()))
    outcome = run(backend, lambda _run_id: _waits_for_an_absolute_arrival(name), _Tool())
    assert outcome.snap.state == "waiting", outcome.snap
    backend.emit_event(outcome.task, name.stored(), {"ok": True})
    assert backend.run_until_result(outcome.task).result == "Arrived"


def test_the_op_reaches_the_recorder_as_an_arrival() -> None:
    """The reachability pin: a workflow yields the wait and gets a `WaitOutcome` back.

    A bounded wait is answered in its own kind, so the fixture names the arm: `responses` for a
    deadline-free await carries the payload, and here it carries the outcome.
    """
    handler = RecordingHandler(responses={"ev:x": Arrived({"n": 1})})
    assert handler.run(_waits_until()) == Arrived({"n": 1})


def test_a_canned_expiry_reaches_the_workflow_as_one() -> None:
    """The other arm, which a payload cannot spell: the fixture says the deadline won."""
    handler = RecordingHandler(responses={"ev:x": Expired()})
    assert handler.run(_waits_until()) == Expired()


def test_a_bounded_wait_replays_the_outcome_it_recorded() -> None:
    """Replay serves the tape, so the arm the workflow took the first time it takes again."""
    recorded = RecordingHandler(responses={"ev:x": Expired()})
    assert recorded.run(_waits_until()) == Expired()
    assert ReplayHandler(recorded.trace).run(_waits_until()) == Expired()


def test_a_resume_answers_a_parked_bounded_wait_in_its_own_kind() -> None:
    """A park carries its deadline, so the resume is held to the wait's own kind.

    Without the carried deadline the recorder sends a bare payload into a workflow that is
    matching on `Arrived`/`Expired`, where no arm catches it and the run returns `None`.
    """
    parked = RecordingHandler().run(_waits_until())
    assert parked.resume(Arrived({"n": 2})) == Arrived({"n": 2})

    stray = RecordingHandler().run(_waits_until())
    with pytest.raises(TypeError, match="named a deadline"):
        stray.resume({"n": 2})


def test_a_resume_may_expire_the_park_it_answers() -> None:
    """The deadline is an answer a caller can deliver, not only one a clock reaches."""
    parked = RecordingHandler().run(_waits_until())
    assert parked.resume(Expired()) == Expired()


def test_a_deadline_free_wait_still_answers_with_its_payload() -> None:
    """The field defaults to no deadline, so every await written before this reads unchanged."""
    from effective.api import await_event

    def wf():
        return (yield from await_event(NAME, dict))

    assert RecordingHandler(responses={"ev:x": {"n": 3}}).run(wf) == {"n": 3}


def test_a_bounded_wait_inside_a_gather_branch_is_refused() -> None:
    """A branch resolves its wait by peeking, and a peek reads the event alone.

    Decided by the op's kind, so the same program is refused whether or not the event has
    arrived: the canned answer below would satisfy a deadline-free await.
    """

    def branch():
        return (yield from await_until(NAME, dict, deadline=AHEAD))

    def wf():
        return (yield from gather([branch]))

    handler = RecordingHandler(responses={"ev:x": Arrived({"n": 1})})
    # A branch's refusal arrives inside an `ExceptionGroup` (the L5 shape), and the CLASS is
    # what makes it an answer a caller can relay rather than a crash.
    with pytest.raises(BaseExceptionGroup) as raised:
        handler.run(wf)
    leaves = [leaf for leaf in raised.value.exceptions]
    assert [type(leaf) for leaf in leaves] == [CompositionRefused], leaves
    assert "cannot run inside the gather branch" in str(leaves[0])


def test_a_ctx_that_waits_on_an_event_alone_refuses_by_name() -> None:
    """A capability the engine lacks is named, since dropping the deadline parks for good."""

    class _EventOnly:
        """A minimal ctx: the two required members, and no clock."""

        def step(self, name: Key, thunk: Any, /) -> Any:
            return thunk()

        def await_event(self, name: Key, /) -> Any:
            return {"n": 1}

        def sleep_until(self, when: datetime, /, *, name: Key) -> None:
            return None

    handler = DurableHandler(_EventOnly(), _Tool())
    with pytest.raises(NotImplementedError, match="waits on an event alone"):
        handler.run(_waits_until())


def test_the_op_runs_end_to_end_on_the_embedded_engine(tmp_path) -> None:
    """The reachability pin on a real engine: spawn, emit, and read the arm off the result."""
    app = SqliteApp(str(tmp_path / "bounded.db"))
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                outcome = yield from await_until(NAME, dict, deadline=AHEAD)
                match outcome:
                    case Arrived(payload=payload):
                        return {"arrived": payload}
                    case Expired():
                        return {"arrived": None}

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        app.work_batch()
        app.emit_event("ev:x", {"n": 7})
        snap = app.run_until_result(task)
        assert snap is not None
        assert snap.state == "completed", snap
        assert snap.result == {"arrived": {"n": 7}}
    finally:
        app.close()


def test_an_absolute_bounded_wait_still_answers_at_its_deadline(tmp_path) -> None:
    """`addressing` chooses who completes the name; it does not choose whether the clock counts.

    The absolute arm reroutes the wait to the root ctx, which is a naming decision. A deadline
    that survives `RELATIVE` and vanishes under `ABSOLUTE` parks a run that asked to be released,
    which is the failure `AwaitEvent.deadline`'s own docstring names.
    """

    app = SqliteApp(str(tmp_path / "absolute.db"))
    past = datetime.now(UTC) - timedelta(seconds=5)
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                outcome = yield from await_until(
                    NAME, dict, deadline=past, addressing=Addressing.ABSOLUTE
                )
                match outcome:
                    case Arrived(payload=payload):
                        return {"arm": "arrived", "payload": payload}
                    case Expired():
                        return {"arm": "expired"}

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        snap = app.run_until_result(task)
        assert snap is not None
        parked_past_it = f"an absolute bounded wait parked past its deadline: {snap}"
        assert snap.state == "completed", parked_past_it
        assert snap.result == {"arm": "expired"}
    finally:
        app.close()


def test_the_capability_probe_reads_past_every_ctx_wrapper() -> None:
    """A wrapper always defines the method, so probing the outermost ctx reports what it wraps.

    `_supports_peek`'s docstring calls this the recurring bug rather than an instance of one, and
    a probe that answers from the wrapper sends the call to an `AttributeError` several frames
    down. Asserted in BOTH directions: a wrapper over a clock-less ctx says no, and the same
    wrapper over an engine that answers says yes.
    """
    from effective.handlers.durable import _PrefixedCtx, _supports_await_until

    class _EventOnly:
        """The two required members and no clock — `TaskContext`'s floor."""

        def step(self, name: Key, thunk: Any, /) -> Any:
            return thunk()

        def await_event(self, name: Key, /) -> Any:
            return None

        def sleep_until(self, when: datetime, /, *, name: Key) -> None:
            return None

    class _Clocked(_EventOnly):
        def await_until(self, name: Key, deadline: float, decided: Key, /) -> Any:
            return Expired()

    assert _supports_await_until(_PrefixedCtx(_EventOnly(), "gather:0,0;")) is False
    assert _supports_await_until(_PrefixedCtx(_Clocked(), "gather:0,0;")) is True


def test_a_fork_childs_bounded_wait_parks_in_its_own_event_world() -> None:
    """A child re-running the base workflow waits on ITS name, or it absorbs the base's answer.

    `RenamedAwaitCtx` rescopes every awaited name to `fork:{child};{name}`; a capability added
    without that arm keeps the bare name, which is the same wrapper defect one class over from
    `_PrefixedCtx`'s frame.
    """
    from effective.handlers.durable import RenamedAwaitCtx

    asked: list[Key] = []

    class _Recording:
        def step(self, name: Key, thunk: Any, /) -> Any:
            return thunk()

        def await_event(self, name: Key, /) -> Any:
            return None

        def sleep_until(self, when: datetime, /, *, name: Key) -> None:
            return None

        def await_until(self, name: Key, deadline: float, decided: Key, /) -> Any:
            asked.append(name)
            return Expired()

    child = RenamedAwaitCtx(_Recording(), "r-child")
    child.await_until(NAME, 0.0, Key.parse("event;ev:x"))
    assert [k.stored() for k in asked] == ["fork:r-child;ev:x"], asked


def test_a_layer_injected_wait_settles_clear_of_the_op_it_wraps(tmp_path) -> None:
    """A wait's outcome belongs in the WAIT's slot, and a layer may yield one inside another op.

    `layers.LAYERED_OPS_DURABLE_ONLY` names `AwaitEvent` as an op a layer may inject, and the
    walk publishes the placement of the op it is DRIVING — so a slot read from the ambient
    placement is the wrapped op's own checkpoint. Settling there overwrites it and serves the
    wait's record back as that op's result, in the durable store, past every replay.

    Two waits, because they must also answer for themselves: the same aliasing one level up.
    """
    from effective.api import step as step_op
    from effective.domain import CallTool
    from effective.layers import op_layer
    from effective.ops import Step

    seen: list[Any] = []

    @op_layer
    def wait_first(op: Any) -> Any:
        if isinstance(op, Step):
            seen.append((yield AwaitEvent(NAME, dict, deadline=AHEAD)))
            seen.append((yield AwaitEvent(NAME, dict, deadline=AHEAD)))
        return (yield op)

    app = SqliteApp(str(tmp_path / "layer.db"))
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                return (yield from step_op("s1", CallTool(name="t", args={}, result_schema=int)))

            return DurableHandler(ctx, _Tool(), op_layers=(wait_first,)).run(wf)

        app.spawn("waiter", {})
        app.emit_event("ev:x", {"n": 1})
        app.work_batch()
        stored = dict(app.conn.execute("SELECT name, state FROM checkpoints").fetchall())
        assert stored.get("step:s1") == "1", f"the step's checkpoint holds {stored!r}"
        assert "event;ev:x" in stored, f"the wait settled nowhere of its own: {stored!r}"
    finally:
        app.close()


def test_a_counterfactual_refuses_a_wait_that_names_a_deadline() -> None:
    """A fork explores a decision, not a clock, and a deadline asks how much time passed.

    Driven rather than called, so the refusal is pinned where a fork actually meets it: the
    driver restates the op under its placed name on the way to `inspect_only`, and a restatement
    that drops the deadline puts the wait back in the permissive arm.
    """
    from effective.counterfactual import ForkedDeadline
    from effective.fork import live_drive

    gen = (x for x in [AwaitEvent(NAME, dict, deadline=AHEAD)])
    with pytest.raises(ForkedDeadline, match="explores a decision, not a clock"):
        live_drive(gen, None, _NoDomain(), {NAME: {"n": 1}})


class _NoDomain:
    """A domain the refusal fires ahead of, so nothing here answers anything."""

    def run_metered(self, op: Any) -> Any:
        raise AssertionError("the refusal should fire before any domain call")


def test_a_viewer_reads_a_settled_wait_and_reports_an_open_one() -> None:
    """A viewer READS, and both of a bounded wait's endings are writes.

    So a settled wait renders what it settled and an open one is a park, which is the same
    answer `await_event` gives for an undelivered name.
    """
    from effective.viewing import ReachedThePark, ViewingCtx

    class _Store:
        """A ctx holding one settled wait, plus `TaskContext`'s required members."""

        def __init__(self, settled: Any) -> None:
            self._settled = settled

        def step(self, name: Key, thunk: Any, /) -> Any:
            return thunk()

        def await_event(self, name: Key, /) -> Any:
            return None

        def sleep_until(self, when: datetime, /, *, name: Key) -> None:
            return None

        def peek_step(self, name: Key, /) -> tuple[bool, Any]:
            return (True, self._settled) if self._settled is not None else (False, None)

    slot = Key.parse("event;ev:x")
    assert ViewingCtx(_Store(["expired"]), ()).await_until(NAME, 0.0, slot) == Expired()
    assert ViewingCtx(_Store(["arrived", 7]), ()).await_until(NAME, 0.0, slot) == Arrived(7)
    with pytest.raises(ReachedThePark):
        ViewingCtx(_Store(None), ()).await_until(NAME, 0.0, slot)


def test_a_wake_is_spent_by_the_wait_it_answers(tmp_path) -> None:
    """One wake answers one wait, or a clock-woken task expires every later ask of the name.

    The reference spends its own the same way — the SDK clears `wake_event` on the task it holds
    as it raises — so a second ask on the same claim reads the store. Both halves of the
    bookkeeping are here: forgetting to record the wake as spent, and forgetting to consult that
    record, give the same wrong answer.
    """
    app = SqliteApp(str(tmp_path / "spent.db"))
    try:
        answers: list[str] = []

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                first = yield from await_until(NAME, dict, deadline=AHEAD)
                second = yield from await_until(NAME, dict, deadline=AHEAD)
                answers.extend(_arm(o) for o in (first, second))
                return "done"

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        app.work_batch()
        assert app.conn.execute("SELECT state FROM tasks").fetchone()[0] == "waiting"

        # the deadline arrives, then the event does — too late to wake anyone, and in the store
        app.conn.execute(
            "UPDATE tasks SET available_at=? WHERE task_id=?", (time.time() - 1.0, str(task))
        )
        app.emit_event("ev:x", {"n": 1})
        app.run_until_result(task)
        assert answers == ["expired", "arrived"], f"the wake answered twice: {answers}"
    finally:
        app.close()


def test_an_authored_wait_settles_under_its_own_name(tmp_path) -> None:
    """The first ask of a name settles at `event;{name}`, with no occurrence.

    `Key.occurrence` is byte-preserving at n <= 1, so this is what the walk already assigns and
    what `parked.pending_key` mints for the graph view. A slot placed a second time reads as the
    SECOND ask of the name, which is a different question.
    """
    app = SqliteApp(str(tmp_path / "authored.db"))
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                return (yield from await_until(NAME, dict, deadline=AHEAD))

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        app.emit_event("ev:x", {"n": 1})
        app.run_until_result(task)
        names = [r[0] for r in app.conn.execute("SELECT name FROM checkpoints").fetchall()]
        assert names == ["event;ev:x"], f"the first ask settled at {names}"
    finally:
        app.close()


def test_a_durable_forks_tail_refuses_a_wait_that_names_a_deadline() -> None:
    """The refusal reaches the durable driver, not only the two in-process ones.

    `ForkedSleep`'s docstring records that exact gap being found by an earlier review — enforced
    at two in-process sites while the durable driver slept for real. The phase is what decides:
    a replayed PREFIX happened in reality, and only the tail is the dream.
    """
    from effective.counterfactual import ForkedDeadline
    from effective.handlers.durable import Live, SeedingCtx

    class _Engine:
        def step(self, name: Key, thunk: Any, /) -> Any:
            return thunk()

        def await_event(self, name: Key, /) -> Any:
            return None

        def sleep_until(self, when: datetime, /, *, name: Key) -> None:
            return None

        def await_until(self, name: Key, deadline: float, decided: Key, /) -> Any:
            return Expired()

    child = SeedingCtx(_Engine(), {}, fork_point=NAME)
    assert child.await_until(NAME, 0.0, Key.parse("event;ev:x")) == Expired()  # the prefix

    child._phase = Live()
    with pytest.raises(ForkedDeadline, match="explores a decision, not a clock"):
        child.await_until(NAME, 0.0, Key.parse("event;ev:x"))


def _arm(outcome: Any) -> str:
    match outcome:
        case Arrived():
            return "arrived"
        case Expired():
            return "expired"
        case other:
            raise TypeError(f"not a wait outcome: {other!r}")


def test_a_reclaim_reads_the_store_rather_than_a_dead_claims_wake(tmp_path) -> None:
    """A reclaim is a new attempt, and a new attempt has no wake.

    The reference settles it: `absurd.claim_task` takes only `pending` and `sleeping` runs, an
    expired lease goes through `absurd.fail_run`, and that inserts the next attempt's run with
    `wake_event` NULL. So the attempt that follows reads the store, and the event waiting there
    answers it — whatever the clock had already told the attempt that died.

    Driven through `work_batch`, which is the whole point: the sibling pin builds ctxs by hand and
    never reaches the claim, so the claim's own clause is invisible to it.
    """
    app = SqliteApp(str(tmp_path / "reclaim.db"))
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                outcome = yield from await_until(NAME, dict, deadline=AHEAD)
                match outcome:
                    case Arrived(payload=payload):
                        return {"arm": "arrived", "payload": payload}
                    case Expired():
                        return {"arm": "expired"}

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        app.work_batch()
        assert _row(app, task) == ("waiting", "ev:x")

        # the deadline arrives, and the emit that follows leaves a waiter the clock has passed
        app.conn.execute(
            "UPDATE tasks SET available_at=? WHERE task_id=?", (time.time() - 1.0, str(task))
        )
        app.emit_event("ev:x", {"n": 1})
        assert _row(app, task) == ("waiting", "ev:x")

        # a worker claims it and dies holding the wake
        app.conn.execute(
            "UPDATE tasks SET state='running', claimed_by='dead', claim_expires_at=? "
            "WHERE task_id=?",
            (time.time() - 1.0, str(task)),
        )
        snap = app.run_until_result(task)
        assert snap is not None
        assert snap.state == "completed", snap
        assert snap.result == {"arm": "arrived", "payload": {"n": 1}}, (
            "the reclaim carried a dead claim's wake into a fresh attempt"
        )
    finally:
        app.close()


def _row(app: SqliteApp, task: Any) -> tuple[str, str | None]:
    return app.conn.execute(
        "SELECT state, waiting_event FROM tasks WHERE task_id=?", (str(task),)
    ).fetchone()


def test_a_reclaim_does_not_hand_the_next_attempt_a_spent_wake(tmp_path) -> None:
    """A wake belongs to the attempt that was woken, and a reclaim starts a new one.

    The reference settles this in `absurd.sql`: a claim takes only `pending` or `sleeping` runs,
    an expired lease goes through `fail_run`, and `fail_run` inserts the next attempt's run with
    `wake_event` NULL. So the attempt that follows a lost claim has no wake to read.

    Here the second ask has neither ending available to it — nothing was emitted and its deadline
    is an hour out — so it parks. Answering it at all means a wake the first ask already spent was
    handed across the claim, where the in-memory record of spending it does not reach.
    """
    app = SqliteApp(str(tmp_path / "spent.db"))
    died: list[bool] = []
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                first = yield from await_until(NAME, dict, deadline=AHEAD)
                if not died:
                    died.append(True)
                    raise BaseException("the worker died mid-body")
                second = yield from await_until(NAME, dict, deadline=AHEAD)
                return {"first": type(first).__name__, "second": type(second).__name__}

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        app.work_batch()
        assert _row(app, task) == ("waiting", "ev:x")

        app.conn.execute(  # the deadline arrives, so the claim that follows is the clock's
            "UPDATE tasks SET available_at=? WHERE task_id=?", (time.time() - 1.0, str(task))
        )
        with pytest.raises(BaseException, match="the worker died"):
            app.work_batch()
        app.conn.execute(  # and that claim's lease expires, holding the wake it spent
            "UPDATE tasks SET claim_expires_at=? WHERE task_id=?", (time.time() - 1.0, str(task))
        )
        for _ in range(5):
            app.work_batch()

        state, _ = _row(app, task)
        assert state == "waiting", f"the second ask was answered by the first ask's wake: {state}"
    finally:
        app.close()


def test_a_park_ended_by_a_sleep_leaves_no_wake_for_the_next_ask(tmp_path) -> None:
    """The claim clears a registration it did not take out of `waiting`. Here is where that bites.

    A sleep ends the first park without touching the column, so the claim that follows is the only
    thing standing between a stale registration and the second ask. Cross-engine coverage of this
    lives in `test_a_wake_does_not_outlive_the_park_it_ended_on_both_engines`; the mutant that
    deletes the claim's clause survives THIS file without it, and this file is where the wake's
    lifetime is documented.
    """
    app = SqliteApp(str(tmp_path / "slept.db"))
    ends_at = datetime.now(UTC) + timedelta(seconds=0.3)
    # FIXED before the run, not read inside it: a workflow that computes its own wake time reads
    # the clock between yields, and every replay pushes the sleep further out.
    wakes_at = datetime.now(UTC) + timedelta(seconds=0.6)
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                yield from await_until(NAME, dict, deadline=ends_at)
                yield from api_sleep_until(wakes_at)
                second = yield from await_until(NAME, dict, deadline=AHEAD)
                return {"second": type(second).__name__}

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        app.work_batch()
        assert _row(app, task) == ("waiting", "ev:x")

        while time.time() < ends_at.timestamp() + 0.05:
            time.sleep(0.02)
        app.work_batch()  # the clock ends the first wait, and the sleep begins
        assert _row(app, task)[0] == "sleeping"

        while time.time() < wakes_at.timestamp() + 0.05:
            time.sleep(0.02)
        app.work_batch()
        state, registered = _row(app, task)
        assert (state, registered) == ("waiting", "ev:x"), (
            f"the second ask was answered by the first park's wake: {state}"
        )
    finally:
        app.close()
