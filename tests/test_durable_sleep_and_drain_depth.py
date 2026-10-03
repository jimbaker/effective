"""A top-level durable sleep and a drain loop's queue depth, on the real engine.

1. **`SdkCtx.sleep_until` forwards two arguments** (`handlers/absurd.py`). The Absurd SDK's ctx
   takes `sleep_until(step_name, wake_at)`, and `SdkCtx`, which `_adapt_ctx` wraps around every
   raw SDK ctx and so around every deployed `DurableHandler`, must pass the step name first. A
   one-argument forward raises `TypeError`, and the task retries to death. The conformance
   workflow `_conformance.make_gather_sleep_wf` cannot see this: it sleeps *inside a gather*,
   which routes to `_branch_sleep`, a pure clock compare that calls neither ctx.

2. **A drain loop's `queue_depth` predicate** counts `state IN ('pending','sleeping')`, the
   engine's own claim shape (`absurd.sql:937-940`). A run whose sleep has expired is claimable,
   and a predicate on `state = 'pending'` alone misses it, so the drain exits with work queued.

The first hazard masks the second: a top-level sleep that never happens leaves no `sleeping` run
to observe. Both divergences are Absurd-side, so a port is verified on the real engine and these
tests are pg-gated.
"""

import datetime as dt
import time
import uuid
from uuid import UUID

import psycopg
import pytest
from _durable import DSN, absurd, clock_at, pg_ready, run_until_result

from effective.api import sleep_until
from effective.bridge_absurd import read_absurd_task
from effective.handlers.absurd import DurableHandler

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")

# Both count queries are SCOPED TO ONE TASK NAME. The queue is shared with the rest of the
# pg-gated suite, so a global count is not a measurement: it reads whatever else is mid-run, and
# a global count here can break an unrelated test in the full-suite run.
#
# PEP 750 t-string SQL (psycopg >= 3.3): `{value}` is a bound parameter, never concatenation
# — the house idiom, `bridge_absurd.py:31-44`.


def _depth_fixed(task_name: str) -> int:
    """A drain loop's `queue_depth` predicate: the engine's own claim shape."""
    with psycopg.connect(DSN) as conn:
        row = conn.execute(
            t"SELECT count(*) FROM absurd.r_default r "
            t"JOIN absurd.t_default t USING (task_id) WHERE t.task_name = {task_name} "
            t"AND r.state IN ('pending', 'sleeping') AND r.available_at <= now()"
        ).fetchone()
        return row[0] if row else 0


def _depth_old(task_name: str) -> int:
    """The `state = 'pending'` predicate, which misses a due sleeper."""
    with psycopg.connect(DSN) as conn:
        row = conn.execute(
            t"SELECT count(*) FROM absurd.r_default r "
            t"JOIN absurd.t_default t USING (task_id) WHERE t.task_name = {task_name} "
            t"AND r.state = 'pending' AND r.available_at <= now()"
        ).fetchone()
        return row[0] if row else 0


def _states(task_name: str) -> list[tuple[str, str]]:
    with psycopg.connect(DSN) as conn:
        return [
            (r[0], r[1])
            for r in conn.execute(
                t"SELECT r.state, r.available_at::text FROM absurd.r_default r "
                t"JOIN absurd.t_default t USING (task_id) WHERE t.task_name = {task_name}"
            ).fetchall()
        ]


def _checkpoint_names(task_id: UUID) -> list[str]:
    """Every committed checkpoint name Absurd wrote for this task.

    **`exclude=()` is the whole point of calling it this way.** The reader's default projection
    is the fork SEED's, which drops `ENGINE_INTERNAL` — and `sleep:` is in that tuple, so a
    sleep assertion made at the default passes VACUOUSLY, having looked at nothing. The
    parameter exists for exactly this case (`read_sqlite_conn`: *"Pass `exclude=()` for the raw
    sequence"*), which is why this is a projection choice rather than a reason to hand-roll SQL.

    Names come back through `Key.parse`, which validates nothing today — so a malformed name
    survives verbatim and is still visible to the assertions below."""
    with psycopg.connect(DSN) as conn:
        return [c.key.stored() for c in read_absurd_task(conn, task_id, exclude=())]


class _NoDomain:
    """The workflow under test yields only `SleepUntil`, so no DomainOp ever reaches a
    domain — but `DurableHandler` types the parameter, and a bare `{}` does not satisfy
    `DomainInterpreter`. Failing loudly beats a dict that silently satisfies nothing."""

    def run(self, op: object) -> object:
        raise AssertionError(f"no domain op expected in this workflow, got {op!r}")


def _nap_task(app, name: str, wake_at: dt.datetime):
    """A workflow whose only op is a durable sleep — the path `make_gather_sleep_wf` misses.

    `wake_at` is decided by the TEST, never read inside the workflow: a clock read between
    yields would breach the determinism boundary.
    """

    def nap(run_id: str):
        yield from sleep_until(wake_at)
        return {"woke": True}

    @app.register_task(name)
    def task(params, ctx):
        return DurableHandler(ctx, _NoDomain(), ledger=None).run(lambda: nap(params["run_id"]))

    spawned = app.spawn(name, {"run_id": f"r-{name}"})
    return spawned["task_id"] if isinstance(spawned, dict) else spawned


def test_top_level_durable_sleep_completes_on_absurd():
    """A top-level durable sleep parks `sleeping` and completes; a one-argument forward would
    raise TypeError and fail every attempt."""
    app = absurd()
    name = f"nap-{uuid.uuid4().hex[:8]}"
    wake_at = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=3)

    task_id = _nap_task(app, name, wake_at)
    app.work_batch()  # runs the workflow -> parks as a durable sleep

    states = _states(name)
    assert states, "no run row for the napping task"
    assert states[0][0] == "sleeping", (
        f"expected the run to be sleeping, got {states!r} — if this says 'failed', "
        "SdkCtx.sleep_until is not forwarding the SDK's two-arg form"
    )

    with clock_at(app, wake_at + dt.timedelta(seconds=1)):
        # `run_until_result`, not one `work_batch()`: an advanced clock makes every dormant
        # leftover in the shared queue claimable at once (measured: 1 -> 3), so a single
        # claim may land on somebody else's task. Drain until THIS one is terminal.
        run_until_result(app, task_id)

    snapshot = app.fetch_task_result(task_id)
    assert snapshot is not None, "no snapshot for the napping task"
    assert snapshot.state == "completed"
    assert snapshot.result == {"woke": True}


def test_drain_depth_sees_a_run_whose_sleep_expired():
    """A run whose sleep expired counts toward depth; the pending-only predicate counts 0 here.

    **The one test here that still sleeps for real, and deliberately.** Its subject is
    `queue_depth`'s SQL predicate, which reads `now()` on its own connection — the
    transaction clock, not `absurd.current_time()`. `clock_at` moves the two clocks the
    *engine* consults; it cannot move `now()`, and reaching into the predicate to make it
    fakeable would be faking the thing under test. So the wait stays real and the MARGIN is
    small: a second is still an order of magnitude above the gap between parking and the first
    assertion."""
    app = absurd()
    name = f"nap-{uuid.uuid4().hex[:8]}"
    wake_at = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=1)

    task_id = _nap_task(app, name, wake_at)
    app.work_batch()

    # Before the wake time: not due, so neither predicate counts it — the drain may exit,
    # and that is the scale-to-zero win, not a bug.
    assert _depth_fixed(name) == 0

    time.sleep((wake_at - dt.datetime.now(dt.UTC)).total_seconds() + 0.2)

    # After the wake time the run is claimable, and the drain must see it.
    assert _depth_fixed(name) >= 1, "the fixed predicate must see a due sleeping run"
    assert _depth_old(name) == 0, (
        "the state='pending' predicate is expected to miss it: this assertion documents "
        "the hazard; if it starts counting, Absurd changed how an expired sleep is stored"
    )

    # Leave nothing sleeping in the shared queue: an abandoned run gets claimed by whichever
    # test calls work_batch() next, whose app has never heard of this task. `clock_at` is
    # belt-and-braces here rather than load-bearing — real time has already passed `wake_at`,
    # so both clocks agree without help. Measured: breaking `clock_at` either way reddens the
    # three tests above and leaves this one green.
    with clock_at(app, wake_at + dt.timedelta(seconds=1)):
        app.work_batch()
    snapshot = app.fetch_task_result(task_id)
    assert snapshot is not None, "the napping task vanished"
    assert snapshot.state == "completed"


# --- positional identity, on the real engine ---------------------------------------------
#
# `op_key(SleepUntil)` keys a sleep by its wake time, so two sleeps to one instant compose one
# name and the Absurd SDK's `#k` occurrence counter is the only thing separating them. These two
# attacks pin what that costs on the DURABLE path, where a design-only or SQLite-only pass sees
# nothing: SQLite writes no sleep checkpoint at all, so only Absurd has rows to be wrong about.


@pytest.mark.adversarial
def test_two_sleeps_to_one_instant_get_two_durable_names():
    """The attack: sleep twice to one instant and read what Absurd actually stored.

    Two assertions, and the second holds only because the first does. The names must be the
    frame's two ordinals — that is the property under test. And no name may carry a `Key` REPR:
    `Key` has no `__str__`, so an f-string over one renders `Key(_value=…)`, and the SDK's
    duplicate-name rule is exactly such an f-string. It fires only when two ops share a name, so
    distinct ordinals keep it out of reach — which makes the repr assertion the tell that the
    names have collided, and worth keeping for that.

    What a collision costs, measured: `is_step_checkpoint` reads the malformed name as a Step, so
    it enters the positionally indexed measured prefix as a phantom step and a fork refuses the
    seed."""
    app = absurd()
    name = f"nap2-{uuid.uuid4().hex[:8]}"
    wake_at = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=2)

    def nap_twice(run_id: str):
        yield from sleep_until(wake_at)
        yield from sleep_until(wake_at)  # SAME instant -> the SAME composed name
        return {"woke": 2}

    @app.register_task(name)
    def task(params, ctx):
        return DurableHandler(ctx, _NoDomain(), ledger=None).run(
            lambda: nap_twice(params["run_id"])
        )

    task_id = app.spawn(name, {"run_id": f"r-{name}"})
    task_id = task_id["task_id"] if isinstance(task_id, dict) else task_id
    app.work_batch()
    with clock_at(app, wake_at + dt.timedelta(seconds=1)):
        run_until_result(app, task_id)

    snapshot = app.fetch_task_result(task_id)
    assert snapshot is not None, "the napping task vanished"
    assert snapshot.state == "completed", snapshot
    names = _checkpoint_names(task_id)
    assert names, "no checkpoint rows for the napping task"
    assert sorted(n for n in names if "sleep" in n) == ["sleep:0", "sleep:1"], names
    assert not any("Key(_value=" in n for n in names), (
        f"a Key repr reached a durable checkpoint name: {names}"
    )


@pytest.mark.adversarial
def test_a_scoped_sleep_is_frame_qualified_on_the_durable_path():
    """The attack: put a sleep inside `scoped(...)` and ask the engine what it called it.

    `_PrefixedCtx` applies `name.prefixed(self._prefix)` in its `step`, `await_event` and
    `peek_event` arms — and cannot in `sleep_until`, because the one-arg protocol carries no
    name to prefix. So the recorder records `rec:0;sleep:…` and Absurd stores bare `sleep:…`:
    two interpreters disagreeing about one op's name. It is invisible today because
    `ENGINE_INTERNAL` filters `sleep:` out of every durable reader before any parity check."""
    from effective.api import scoped
    from effective.keys import compose_key

    app = absurd()
    name = f"nap3-{uuid.uuid4().hex[:8]}"
    wake_at = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=2)

    def body():
        yield from sleep_until(wake_at)
        return {"woke": True}

    def scoped_nap(run_id: str):
        return (yield from scoped(compose_key(t"rec:{0}"), body))

    @app.register_task(name)
    def task(params, ctx):
        return DurableHandler(ctx, _NoDomain(), ledger=None).run(
            lambda: scoped_nap(params["run_id"])
        )

    task_id = app.spawn(name, {"run_id": f"r-{name}"})
    task_id = task_id["task_id"] if isinstance(task_id, dict) else task_id
    app.work_batch()
    with clock_at(app, wake_at + dt.timedelta(seconds=1)):
        run_until_result(app, task_id)

    snapshot = app.fetch_task_result(task_id)
    assert snapshot is not None, "the napping task vanished"
    assert snapshot.state == "completed", snapshot
    sleeps = [n for n in _checkpoint_names(task_id) if "sleep:" in n]
    assert sleeps, "no sleep checkpoint row — the durable path wrote nothing to be wrong about"
    assert all(n.startswith("rec:0;") for n in sleeps), (
        f"a scoped sleep's durable name dropped its frame: {sleeps}"
    )
