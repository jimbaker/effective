"""What instant the reference actually parks a bounded wait on.

`absurd.await_event` takes whole seconds, relative to the claim, so the park it registers ends
at `claim + ceil(deadline - claim)`. Every instant the engine stores is a microsecond
`timestamptz`, so the rounding is one parameter rather than a resolution: the two columns that
decide are written over with the deadline the workflow named (`sdk_pin_the_park`).

The behavioral half is `tests/test_conformance.py`'s bounded-wait table, which asks what an emit
inside the old rounding answers. This asks the mechanism directly, and it needs no clock to
reach a deadline, so a loaded host cannot turn it into a different measurement.
"""

import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta

import psycopg
import pytest
from _clock import HeldClock
from _durable import DSN, absurd, pg_ready
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from effective.api import await_until
from effective.engines import absurd as absurd_engine
from effective.engines.absurd import ConcurrentAbsurdCtx
from effective.handlers.durable import DurableHandler
from effective.keys import Segment, compose_key

# The park is held by patching `sdk_pin_the_park` in the module that DEFINES it, since its caller
# looks it up there; a patch on any module that merely imports it would hold nothing.

pytestmark = pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")

A_DEADLINE_THE_ROUNDING_WOULD_MOVE = 3600.5
"""Seconds out, chosen with a fractional part: `ceil` moves it by half a second."""

HALF_A_SECOND = timedelta(seconds=0.5)
"""How far past the claim a deadline sits, so `ceil` moves the park half a second past it."""

SAFETY_NET = 20.0
"""Seconds a poll waits for a condition before failing the test; no verdict turns on it."""

TOLERANCE = timedelta(milliseconds=1)
"""What a round trip through `to_timestamp` and back may cost. The defect measures 0.5 s."""


class _Ack(BaseModel):
    ok: bool = True


class _NoDomain:
    """This workflow yields one op and it is not a domain call."""

    def run(self, op):
        raise AssertionError(f"no domain op belongs in this workflow: {op}")


def _until(condition: Callable[[], bool]) -> None:
    """Poll a condition the server holds, which no event announces."""
    ends = time.monotonic() + SAFETY_NET
    while not condition():
        assert time.monotonic() < ends, "the condition never held"
        time.sleep(0.005)


def _backend(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT pg_backend_pid()").fetchone()
    assert row is not None
    return row[0]


def _waiting_on_a_lock(conn: psycopg.Connection, pid: int) -> bool:
    row = conn.execute(
        t"SELECT wait_event_type FROM pg_stat_activity WHERE pid = {pid}"
    ).fetchone()
    return row is not None and row[0] == "Lock"


@pytest.fixture
def conn():
    c = psycopg.connect(DSN, autocommit=True)
    yield c
    c.close()


DEADLINES = {
    "a rounding away": lambda: (
        datetime.now().astimezone() + timedelta(seconds=A_DEADLINE_THE_ROUNDING_WOULD_MOVE)
    ),
    "past the seconds an integer holds": lambda: datetime(2100, 1, 1).astimezone(),
}


@pytest.mark.parametrize("deadline_of", list(DEADLINES.values()), ids=list(DEADLINES))
def test_a_parked_bounded_wait_registers_the_deadline_it_was_given(conn, deadline_of):
    """The wait row and the run row both name the workflow's deadline, not a rounded one."""
    app = absurd()
    name = f"deadline-{uuid.uuid4().hex[:8]}"
    event = compose_key(t"shared:{Segment(name)}")
    deadline = deadline_of()

    def wf():
        outcome = yield from await_until(event, _Ack, deadline=deadline)
        return {"answer": type(outcome).__name__}

    @app.register_task(name, default_max_attempts=2)
    def task(params, ctx):
        return DurableHandler(ConcurrentAbsurdCtx(ctx), _NoDomain()).run(wf)

    task_id = app.spawn(name, {"run_id": "r1"})
    app.work_batch()

    parked = conn.execute(
        t"SELECT w.timeout_at, r.available_at FROM absurd.w_default w "
        t"JOIN absurd.r_default r ON r.run_id = w.run_id "
        t"WHERE w.task_id = {task_id}::uuid"
    ).fetchall()
    assert len(parked) == 1, f"the wait did not park, or parked more than once: {parked}"
    timeout_at, available_at = parked[0]
    assert abs(timeout_at - deadline) < TOLERANCE, f"the wait ends at {timeout_at}, not {deadline}"
    assert abs(available_at - deadline) < TOLERANCE, (
        f"the run wakes at {available_at}, not {deadline}"
    )


def test_the_fake_clock_reaches_a_bounded_wait_and_expires_it(conn):
    """A deadline is reachable by `absurd.fake_now`, which is what makes it testable at all.

    A held clock holds both clocks a park consults, and a park registered RELATIVE to the claim
    lands at `fake_now + ceil(...)`: past the deadline by however far the fake clock was moved,
    so advancing the clock TO the deadline cannot reach it. That is the hazard
    `HeldClock.on_absurd` names for a sleep, and an absolute park is what answers it here.

    So a boundary this pins costs no wall-clock time, which is what a rule about the instant
    itself needs: a completion at exactly the deadline is a timeout.
    """
    app = absurd()
    name = f"faked-{uuid.uuid4().hex[:8]}"
    event = compose_key(t"shared:{Segment(name)}")
    base = datetime.now().astimezone()
    deadline = base + timedelta(hours=2)

    def wf():
        outcome = yield from await_until(event, _Ack, deadline=deadline)
        return {"answer": type(outcome).__name__}

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx):
        return DurableHandler(ConcurrentAbsurdCtx(ctx), _NoDomain()).run(wf)

    task_id = app.spawn(name, {"run_id": "r1"})
    with HeldClock(base + timedelta(hours=1)).on_absurd(app) as clock:  # an hour short of due
        app.work_batch()
        parked = conn.execute(
            t"SELECT timeout_at FROM absurd.w_default WHERE task_id = {task_id}::uuid"
        ).fetchall()
        assert len(parked) == 1, f"the wait did not park under a held clock: {parked}"
        assert abs(parked[0][0] - deadline) < TOLERANCE, (
            f"the held clock reached the park: {parked[0][0]} against {deadline}"
        )

        clock.advance_to(deadline + timedelta(seconds=1))
        for _ in range(8):
            snap = app.fetch_task_result(task_id)
            if snap is not None and snap.state in ("completed", "failed"):
                break
            app.work_batch()
    snap = app.fetch_task_result(task_id)
    assert snap is not None, "the task produced no snapshot"
    assert snap.state == "completed", f"the advanced clock never reached the park: {snap}"
    assert snap.result == {"answer": "Expired"}, snap.result


def test_an_emit_racing_the_park_cannot_take_a_wait_past_its_deadline(conn, monkeypatch):
    """The park and the deadline it ends at are one transaction, so nothing reads the rounding.

    The SDK connects with autocommit, so a park and a deadline written after it would be two
    commits with a worker's PROGRESS between them, not a clock tick. A wait live in that interval
    ends at `claim + ceil(...)`, and `absurd.emit_event` takes such a wait, writes the SDK's
    arrival checkpoint and deletes the row, so no later claim asks again and the workflow is told
    an event beat a deadline it did not.

    So the emit runs on its own connection, its clock held between the deadline and the rounding,
    while the deadline write is held open. With both writes in one transaction the emit waits on
    the run's lock and reaches a wait whose deadline has passed; in two transactions it takes the
    wait.
    """
    app = absurd()
    name = f"racing-{uuid.uuid4().hex[:8]}"
    event = compose_key(t"shared:{Segment(name)}")
    clock = HeldClock(datetime.now().astimezone() + timedelta(minutes=1))
    deadline = clock.at + HALF_A_SECOND

    def wf():
        outcome = yield from await_until(event, _Ack, deadline=deadline)
        return {"answer": type(outcome).__name__}

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx):
        return DurableHandler(ConcurrentAbsurdCtx(ctx), _NoDomain()).run(wf)

    settled = absurd_engine.sdk_pin_the_park
    emitter = psycopg.connect(DSN, autocommit=True)
    HeldClock(deadline + HALF_A_SECOND / 2).hold(emitter)
    emitting = _backend(emitter)

    def emit() -> None:
        emitter.execute(
            t"SELECT absurd.emit_event('default', {event.stored()}, {Jsonb({'ok': True})})"
        )

    racer = threading.Thread(target=emit)
    rounded: list[datetime] = []  # the instant the SDK registered, read before it is written over
    blocked: list[bool] = []

    def held(sdk, key, at):
        rounded.append(
            sdk._conn.execute(
                t"SELECT timeout_at FROM absurd.w_default WHERE event_name = {event.stored()}"
            ).fetchone()[0]
        )
        racer.start()
        _until(lambda: _waiting_on_a_lock(conn, emitting) or not racer.is_alive())
        blocked.append(racer.is_alive())
        settled(sdk, key, at)
        # JOINED BY THE CALLER: one transaction and the emit is waiting on this one to commit,
        # which cannot happen until this returns.

    task_id = app.spawn(name, {"run_id": "r1"})
    try:
        with clock.on_absurd(app), monkeypatch.context() as patched:
            patched.setattr(absurd_engine, "sdk_pin_the_park", held)
            app.work_batch()
            racer.join(timeout=SAFETY_NET)

            # The schedule this measures, asserted rather than assumed: an emit that never
            # lands, or one past the rounding, passes whether or not the two writes are one
            # transaction.
            landed = conn.execute(
                t"SELECT emitted_at FROM absurd.e_default "
                t"WHERE event_name = {event.stored()} AND payload IS NOT NULL"
            ).fetchone()
            emitted_at = landed[0] if landed else None
            assert emitted_at is not None, "the emit never landed, so nothing raced the park"
            assert deadline < emitted_at < rounded[0], (
                f"the emit landed at {emitted_at}, outside the rounding it had to race: "
                f"{deadline} to {rounded[0]}"
            )
            assert blocked == [True], "the emit reached the wait before its deadline was written"

            clock.advance_to(deadline + timedelta(seconds=1))
            snap = app.run_until_result(task_id)
    finally:
        emitter.close()
    assert snap is not None, "the task produced no snapshot"
    assert snap.state == "completed", snap
    assert snap.result == {"answer": "Expired"}, (
        f"an emit inside the rounding answered a wait its deadline had ended: {snap.result}"
    )


def test_two_waits_on_one_event_each_keep_their_own_deadline(conn):
    """Two asks of one name park in turn on ONE run, and the second keeps its own deadline.

    `run_id` names the run and the wait table is keyed by run and STEP, so two asks of one name
    are two questions (`wiki/concepts/bounded-wait.md`, the slot) and two rows in turn.

    Sequential, so it measures the deadline and not the transaction: it passes with the park and
    the correction split apart. The concurrent case is the pin below it.
    """
    app = absurd()
    name = f"twice-{uuid.uuid4().hex[:8]}"
    event = compose_key(t"shared:{Segment(name)}")
    base = datetime.now().astimezone()
    first, later = base + timedelta(minutes=30), base + timedelta(hours=4)

    def wf():
        one = yield from await_until(event, _Ack, deadline=first)
        two = yield from await_until(event, _Ack, deadline=later)
        return {"first": type(one).__name__, "second": type(two).__name__}

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx):
        return DurableHandler(ConcurrentAbsurdCtx(ctx), _NoDomain()).run(wf)

    task_id = app.spawn(name, {"run_id": "r1"})
    app.work_batch()  # the first ask parks, on the real clock, half an hour out
    with HeldClock(first + timedelta(seconds=1)).on_absurd(app):
        app.work_batch()  # the clock ends it, and the second ask parks

    parked = conn.execute(
        t"SELECT timeout_at FROM absurd.w_default WHERE task_id = {task_id}::uuid"
    ).fetchall()
    assert len(parked) == 1, f"the second ask did not park alone: {parked}"
    assert abs(parked[0][0] - later) < TOLERANCE, (
        f"the second wait ends at {parked[0][0]}, not its own deadline {later}"
    )


def test_a_claim_between_the_park_and_its_deadline_cannot_take_the_next_park(conn, monkeypatch):
    """The stale write, which is the other interleaving the one transaction closes.

    A RUN outlives the claim that parked it. Split the two writes and a second worker claims the
    same run in between, settles the first wait as expired and parks the second on the same event
    an hour later; the first claim's write then lands on that second wait and ends it at the first
    deadline. Held together, the second worker cannot claim at all until the park is whole.

    Kept as its own pin because the racing emit above reaches the same transaction from an emit,
    where this one reaches it from a claim.
    """
    app, other = absurd(), absurd()
    other.app._conn.execute("SET statement_timeout = '20s'")
    claiming = _backend(other.app._conn)
    name = f"stale-{uuid.uuid4().hex[:8]}"
    event = compose_key(t"shared:{Segment(name)}")
    clock = HeldClock(datetime.now().astimezone() + timedelta(minutes=1))
    first = clock.at + HALF_A_SECOND
    later = first + timedelta(hours=1)
    settled = absurd_engine.sdk_pin_the_park
    claims: list[str] = []
    seen: list[int] = []
    racers: list[threading.Thread] = []

    def wf():
        one = yield from await_until(event, _Ack, deadline=first)
        two = yield from await_until(event, _Ack, deadline=later)
        return {"first": type(one).__name__, "second": type(two).__name__}

    def body(params, ctx):
        claims.append(ctx._task["run_id"])
        return DurableHandler(ConcurrentAbsurdCtx(ctx), _NoDomain()).run(wf)

    for each in (app, other):
        each.register_task(name, default_max_attempts=4)(body)

    def held(sdk, key, at):
        if seen:
            return settled(sdk, key, at)
        seen.append(1)
        rounded = sdk._conn.execute(
            t"SELECT timeout_at FROM absurd.w_default WHERE event_name = {event.stored()}"
        ).fetchone()[0]
        # The second worker's clock has passed the rounding, so the first wait is claimable to it.
        HeldClock(rounded + HALF_A_SECOND / 2).hold(other.app._conn)
        racer = threading.Thread(target=other.work_batch)
        racer.start()
        racers.append(racer)
        _until(lambda: _waiting_on_a_lock(conn, claiming) or not racer.is_alive())
        claimed_before_the_commit.append(len(claims))
        settled(sdk, key, at)

    claimed_before_the_commit: list[int] = []
    task_id = app.spawn(name, {"run_id": "r1"})
    with clock.on_absurd(app), monkeypatch.context() as patched:
        patched.setattr(absurd_engine, "sdk_pin_the_park", held)
        app.work_batch()
        for racer in racers:
            racer.join(timeout=SAFETY_NET)
    other.work_batch()

    assert claimed_before_the_commit == [1], (
        f"a claim reached the run while its park was half written: {claimed_before_the_commit}"
    )
    parked = conn.execute(
        t"SELECT timeout_at FROM absurd.w_default WHERE task_id = {task_id}::uuid"
    ).fetchall()
    assert len(parked) == 1, f"the second ask did not park alone: {parked}"
    assert abs(parked[0][0] - later) < TOLERANCE, (
        f"the second wait ends at {parked[0][0]}, not its own deadline {later}"
    )


@pytest.mark.parametrize(("offset", "answer"), [(-1, "Arrived"), (0, "Expired"), (1, "Expired")])
def test_an_event_at_the_deadline_is_late_to_the_microsecond(offset: int, answer: str):
    """The deadline tie, read at the resolution the engine actually stores.

    `absurd.emit_event` deletes the waits whose `timeout_at <= now` and wakes the rest, so the
    deadline itself belongs to the timeout. Both instants are the ENGINE's here: the emit runs
    under a held clock at the deadline plus `offset` microseconds, which is why no margin appears
    and a loaded host cannot move the answer.
    """
    app, emitter = absurd(), absurd()
    name = f"tie-{uuid.uuid4().hex[:8]}"
    event = compose_key(t"shared:{Segment(name)}")
    deadline = datetime.now().astimezone() + timedelta(hours=1)

    def wf():
        outcome = yield from await_until(event, _Ack, deadline=deadline)
        return {"answer": type(outcome).__name__}

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx):
        return DurableHandler(ConcurrentAbsurdCtx(ctx), _NoDomain()).run(wf)

    task_id = app.spawn(name, {"run_id": "r1"})
    app.work_batch()  # the wait parks, on the deadline itself
    with HeldClock(deadline + timedelta(microseconds=offset)).on_absurd(app) as clock:
        clock.hold(emitter.app._conn)
        emitter.emit_event(event, {"ok": True})
        clock.advance_to(deadline + timedelta(seconds=1))
        for _ in range(8):
            snap = app.fetch_task_result(task_id)
            if snap is not None and snap.state in ("completed", "failed"):
                break
            app.work_batch()
    snap = app.fetch_task_result(task_id)
    assert snap is not None, "the task produced no snapshot"
    assert snap.state == "completed", snap
    assert snap.result == {"answer": answer}, f"{offset} us from the deadline: {snap.result}"
