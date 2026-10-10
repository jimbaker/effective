"""The SQLite engine's shared connection under concurrent gather branches.

`SqliteApp` hands every branch of a concurrent `gather` the SAME `sqlite3.Connection` and
relies on one `write_lock` to serialize the I/O. This module is the pin on that claim, and it
exists because the claim was false for months in one spot: `peek_event` — the read every
PARKING branch makes — was outside the guard, and the sqlite gather flake was its four faces.

**Role: adversarial.** A pass means only that an attack failed, so these tests owe a mutation
check, and they have one: drop the `with self._guard:` from `SqliteTaskContext.peek_event` and
`test_a_concurrent_peek_never_answers_with_another_branchs_row` reddens on the first run. That
is the property to preserve when editing them — a test here that survives the unguarded read
is measuring nothing.

**Cover the DRIVER, not just the ctx, and that omission cost a blocker.** The first version of
this module drove `peek_event` and `step` only, because the fix had been scoped by reading
`SqliteTaskContext`. `SqliteApp.spawn` was equally unguarded and equally reachable from a branch
thread (`effective.interpreters.tools.spawn_tool` is wired to the app's own connection), so the
instrument could not see the second half of its own bug; a fresh-eyes review found it. Ask of
anything added here **"what else can a branch thread reach?"**

Why the seam and not the workflow: the conformance-lane version of this flake reproduced at a
few percent, drifting with machine state (see `scripts/sqlite_flake_sweep.py`). Driving the
connection directly turns the same defect into thousands of observations per second, which is
the difference between a rate nobody can act on and a fix anyone can verify.
"""

import json
import threading
from collections import Counter
from uuid import UUID, uuid4

import pytest

from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.keys import Key

pytestmark = pytest.mark.adversarial

# Enough iterations that an unguarded touch fails on the first run rather than sometimes.
# Unguarded, the peek probe produces hundreds of exceptions and hundreds of wrong answers at
# this count and the guarded one produces zero — the counts move by 2-3x between runs and
# machines, so size this to clear the noise floor of zero, not to hit a number.
ITERATIONS = 2000

APPROVED = Key.parse("gather:0,0;approve:r-1")
UNAPPROVED = Key.parse("gather:0,1;approve:r-1")


def _ctx(sqlite_app) -> tuple[SqliteApp, SqliteTaskContext]:
    """A task ctx with a real `write_lock` — i.e. the concurrent-gather configuration.

    A `None` lock is the single-threaded default and would make every test here vacuous, so
    the lock is passed explicitly rather than defaulted into.
    """
    app = sqlite_app(":memory:")
    return app, SqliteTaskContext(app.conn, uuid4(), app.write_lock)


def _run_concurrently(*targets) -> None:
    """Run each callable on its own thread and join, failing loudly on a hang.

    The `join(timeout=…)` + liveness assert is the point: a lock bug that DEADLOCKS would
    otherwise hang the suite instead of failing it, and a test that can hang forever is not a
    gate. Every test here races real threads on one connection, so this is the shared shape.
    """
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not [t for t in threads if t.is_alive()], "a thread never finished — deadlock"


def test_a_concurrent_peek_never_answers_with_another_branchs_row(sqlite_app):
    """Two branches peek their OWN approval while a third commits checkpoints.

    The gather shape this models: branch 0's approval has been emitted, branch 1's has not.
    Branch 0 must see its payload and branch 1 must see nothing, and neither may see the
    other's row: one emission must never approve a sibling.

    Both failure modes are asserted because they are the same defect in two directions, and
    only one of them is loud. An exception fails the run; a wrong answer COMPLETES it with a
    forged approval or a spurious denial.
    """
    _app, ctx = _ctx(sqlite_app)
    ctx.emit_event(APPROVED.stored(), {"decision": "approve"})

    errors: Counter[str] = Counter()
    wrong: Counter[str] = Counter()

    def peek(name: Key, found_expected: bool, payload_expected: object) -> None:
        for _ in range(ITERATIONS):
            try:
                found, payload = ctx.peek_event(name)
            except Exception as exc:  # the exception SHAPE is the finding, so catch broadly
                errors[f"{name.stored()}: {type(exc).__name__}: {exc}"] += 1
                continue
            if (found, payload) != (found_expected, payload_expected):
                wrong[f"{name.stored()} -> {found!r}, {payload!r}"] += 1

    def commit_checkpoints() -> None:
        # A writer on the same connection: this is what the branch's read races against.
        for i in range(ITERATIONS):
            ctx.step(Key.parse(f"gather:0,2;step;tool:t{i}"), lambda i=i: {"i": i})

    _run_concurrently(
        lambda: peek(APPROVED, True, {"decision": "approve"}),
        lambda: peek(UNAPPROVED, False, None),
        commit_checkpoints,
    )

    assert not errors, dict(errors.most_common(5))
    assert not wrong, dict(wrong.most_common(5))


def test_concurrent_steps_commit_every_checkpoint_exactly_once(sqlite_app):
    """The write side, for symmetry — and because a lock that serialized reads by dropping
    writes would pass the test above.

    Each thread commits a disjoint set of names (the structural `gather:{g},{i};` prefix is
    what makes them disjoint in the real engine), so a correct run leaves exactly
    ``threads * ITERATIONS`` checkpoint rows with the right payloads."""
    app, ctx = _ctx(sqlite_app)
    branches = 3
    per_branch = ITERATIONS // 4

    def commit(branch: int) -> None:
        for i in range(per_branch):
            ctx.step(Key.parse(f"gather:0,{branch};step;tool:t{i}"), lambda i=i: {"i": i})

    _run_concurrently(*[(lambda b=b: commit(b)) for b in range(branches)])

    rows = app.conn.execute("SELECT name, state FROM checkpoints").fetchall()

    assert len(rows) == branches * per_branch
    # No occurrence suffix anywhere: a `name#2` would mean two branches collided on one name,
    # which is the failure the structural prefix exists to prevent.
    assert not [name for name, _ in rows if "#" in name]
    assert {json.loads(state)["i"] for _, state in rows} == set(range(per_branch))


def test_a_lockless_context_still_serves_the_single_threaded_case(sqlite_app):
    """`write_lock=None` is the default for a task with no gather, and must stay a plain
    `nullcontext` — this pins that the guard added for concurrency did not make the lock
    mandatory."""
    app = sqlite_app(":memory:")
    ctx = SqliteTaskContext(app.conn, uuid4(), None)

    assert ctx.concurrent_safe is False
    ctx.emit_event(APPROVED.stored(), {"decision": "approve"})
    assert ctx.peek_event(APPROVED) == (True, {"decision": "approve"})
    assert ctx.peek_event(UNAPPROVED) == (False, None)
    assert ctx.step(Key.parse("step;tool:t"), lambda: {"i": 1}) == {"i": 1}


def test_concurrent_spawns_never_alias_two_idempotency_keys_onto_one_task(sqlite_app):
    """The DRIVER half: concurrent spawns from gather branches.

    A workflow may `spawn` from inside a gather branch (`effective.interpreters.tools.spawn_tool`,
    whose only wiring hands it the app's own connection), and a branch thunk runs off-lock.
    Unguarded, `SqliteApp.spawn` races on the shared connection and produces `InterfaceError`, a
    `None` returned where the signature says `UUID`, lost task rows, and **two different
    `idempotency_key`s answered with one `task_id`**.

    That last one is the reason this test is `adversarial` rather than a smoke test: it breaks
    the at-most-once property `SqliteApp.spawn`'s docstring promises, by a different route.
    Distinct keys must mean distinct tasks.
    """
    app, ctx = _ctx(sqlite_app)

    @app.register_task("child")
    def _child(params, task_ctx):
        return 1

    errors: Counter[str] = Counter()
    ids: dict[str, set[str]] = {}
    lock = threading.Lock()
    per_thread = ITERATIONS // 4

    def spawn_many(branch: int) -> None:
        for i in range(per_thread):
            key = f"b{branch}-{i}"
            try:
                task_id = app.spawn("child", {"n": i}, idempotency_key=key)
            except Exception as exc:  # the exception SHAPE is the finding, so catch broadly
                errors[f"{type(exc).__name__}: {exc}"] += 1
                continue
            if not isinstance(task_id, UUID):
                errors[f"spawn returned {type(task_id).__name__}, not UUID: {task_id!r}"] += 1
                continue
            with lock:
                ids.setdefault(str(task_id), set()).add(key)

    def commit_checkpoints() -> None:
        for i in range(per_thread):
            ctx.step(Key.parse(f"gather:0,9;step;tool:t{i}"), lambda i=i: {"i": i})

    _run_concurrently(lambda: spawn_many(0), lambda: spawn_many(1), commit_checkpoints)

    # Aliasing FIRST, deliberately. The loud `InterfaceError`s always fire too, so asserting
    # them first would make the exception the observed red every time and the silent harm — the
    # at-most-once violation this test exists for — would never be the thing you read.
    aliased = {task_id: keys for task_id, keys in ids.items() if len(keys) > 1}
    assert not aliased, f"one task_id answered several idempotency keys: {aliased}"
    assert not errors, dict(errors.most_common(5))
    committed = app.conn.execute("SELECT COUNT(*) FROM tasks WHERE name='child'").fetchone()[0]
    assert committed == 2 * per_thread


def test_a_task_emitting_from_inside_its_own_run_serializes_with_its_siblings(sqlite_app):
    """`SqliteTaskContext.emit_event` — the INSIDE-OUT emit, and the guard this module did not
    cover until a review counted them.

    This is how a spawned child answers its parent, so it is branch-thread reachable exactly like
    `step` and `peek_event`, and it was the one guarded method here with no test of its own. It is
    also `_deliver` again, so an unguarded version loses rows rather than merely raising.

    Kept separate from the driver-side emit test on purpose: `SqliteApp.emit_event` and this method
    are two call paths to one helper, and a single test covering both would let a regression in
    either be masked by the other reddening.
    """
    app, ctx = _ctx(sqlite_app)
    per_thread = ITERATIONS // 2
    errors: Counter[str] = Counter()

    def emit_from_inside(branch: int) -> None:
        for i in range(per_thread):
            try:
                ctx.emit_event(f"gather:0,{branch};reply:e{i}", {"n": i, "b": branch})
            except Exception as exc:  # the exception SHAPE is the finding, so catch broadly
                errors[f"emit {type(exc).__name__}: {exc}"] += 1

    def commit_checkpoints() -> None:
        for i in range(per_thread):
            ctx.step(Key.parse(f"gather:0,8;step;tool:t{i}"), lambda i=i: {"i": i})

    _run_concurrently(lambda: emit_from_inside(0), lambda: emit_from_inside(1), commit_checkpoints)

    assert not errors, dict(errors.most_common(5))
    delivered = app.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert delivered == 2 * per_thread


def test_an_outside_in_emit_never_tears_against_a_running_branch(sqlite_app):
    """`SqliteApp.emit_event` is the delivery path a HITL approve arrives on, from a web
    thread, while a gather is running. `_deliver` is two statements (write the event, wake the
    task parked on it), so an unguarded emit can be observed half-done, and the peek that
    observes it is a branch deciding whether it was approved.

    Every emitted event must be readable with its payload intact, and no emit may be lost.

    **TWO emitters and TWO readers.** One emitter against one reader PASSES with the guard
    removed: an instrument that cannot fail. At this width the unguarded version loses hundreds
    of events and raises `InterfaceError` in the emitter.
    If you shrink this, re-run the mutation check in the module docstring first.
    """
    app, ctx = _ctx(sqlite_app)
    per_thread = ITERATIONS // 2
    errors: Counter[str] = Counter()
    torn: Counter[str] = Counter()

    def emit_all(branch: int) -> None:
        for i in range(per_thread):
            try:
                app.emit_event(f"gather:0,{branch};ev:e{i}", {"n": i, "b": branch})
            except Exception as exc:  # the exception SHAPE is the finding, so catch broadly
                errors[f"emit {type(exc).__name__}: {exc}"] += 1

    def commit_checkpoints() -> None:
        for i in range(per_thread):
            ctx.step(Key.parse(f"gather:0,7;step;tool:t{i}"), lambda i=i: {"i": i})

    def read_back(branch: int) -> None:
        for i in range(per_thread):
            name = Key.parse(f"gather:0,{branch};ev:e{i}")
            try:
                found, payload = ctx.peek_event(name)
            except Exception as exc:  # ditto
                errors[f"peek {type(exc).__name__}: {exc}"] += 1
                continue
            if found and payload != {"n": i, "b": branch}:
                torn[f"{name.stored()} -> {payload!r}"] += 1

    _run_concurrently(
        lambda: emit_all(0),
        lambda: emit_all(1),
        lambda: read_back(0),
        lambda: read_back(1),
        commit_checkpoints,
    )

    assert not errors, dict(errors.most_common(5))
    assert not torn, dict(torn.most_common(5))
    # Every emit landed exactly once, so nothing was lost to the race either.
    delivered = app.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert delivered == 2 * per_thread
