"""Shared fixtures for the durable-replay tests (run against the Podman test PG).

Not a test module (leading underscore) — just the common machinery the
`test_replay_*` files reuse: the DSN + readiness gate, an Absurd app factory,
the `FaultCtx` crash injector, and ledger/result helpers. Domain-free: the canned
domain the durable tests drive lives in ``_approval_domain.py``.
"""

import contextlib
import datetime as dt
import os
from collections.abc import Iterator
from typing import Any, Protocol

import psycopg

from effective.keys import Key

DSN = os.environ.get("DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective")

# Immediate retry — no backoff — so a tight work_batch() loop claims the retry at once.
IMMEDIATE_RETRY = {"kind": "fixed", "base_seconds": 0}


def pg_ready() -> bool:
    """True iff the Podman test Postgres (Absurd schema + ledger) is reachable."""
    try:
        with psycopg.connect(DSN, connect_timeout=2) as conn:
            conn.execute("SELECT 1 FROM ledger LIMIT 0")
            conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='absurd'")
        return True
    except Exception:
        return False


class FaultInjected(Exception):
    """A simulated crash raised before the k-th op commits."""


class Fault:
    """One-shot crash before the k-th ctx op; persists across retry attempts."""

    def __init__(self, k: int | None) -> None:
        self.k = k  # crash before the k-th op (1-based); None/0 => never
        self.count = 0
        self.armed = k is not None


class FaultCtx:
    """Proxies an Absurd ctx, crashing once before the k-th step/await/sleep."""

    def __init__(self, ctx: Any, fault: Fault) -> None:
        self._ctx: Any = ctx  # the Absurd TaskContext — duck-typed (step/await_event/sleep_until)
        self._fault = fault

    def _tick(self) -> None:
        f = self._fault
        f.count += 1
        if f.armed and f.count == f.k:
            f.armed = False
            raise FaultInjected(f"crash before op #{f.k}")

    def step(self, name, fn):  # duck-typed ctx
        self._tick()
        return self._ctx.step(name, fn)

    def await_event(self, name):
        self._tick()
        return self._ctx.await_event(name)

    def sleep_until(self, when, *, name: Key | None = None):
        self._tick()
        return self._ctx.sleep_until(when, name=name)

    def repark(self, name):
        # The no-burn wake-race reschedule is a ctx-op touch too (kept aligned
        # with tests/_conformance.py's FaultCtx; no gather workflow runs through
        # this suite today, so the touch is latent until one does).
        self._tick()
        return self._ctx.repark(name)

    def __getattr__(self, name):  # delegate the rest
        return getattr(self._ctx, name)


def absurd() -> Any:
    from effective.absurd_worker import absurd_worker

    return absurd_worker(DSN)


@contextlib.contextmanager
def clock_at(app: Any, when: dt.datetime) -> Iterator[None]:
    """Hold BOTH clocks a durable sleep consults at `when`, so a park wakes without waiting.

    A parked sleep resumes only if two independent conditions agree, and they are read by
    different processes:

    - **server**, in `absurd.claim_task`: `r.available_at <= absurd.current_time()` decides
      whether the run is even a claim candidate. `absurd.current_time()` returns the
      `absurd.fake_now` GUC when it is set, which is the hook Absurd ships for this.
    - **client**, in the SDK: on resume the workflow re-executes `sleep_until`, and the SDK
      re-derives `actual_wake_at - _get_current_time()`. A positive remainder suspends again.

    **Moving one without the other is worse than moving neither**, which is why this is a
    single context manager and not two helpers. Advance only the server clock and the run
    does not merely fail to wake: it is claimed, re-parks, and reschedules itself to
    `absurd.current_time() + remaining` — so `available_at` lands *beyond* the frozen fake
    clock, and no later advance to that same instant can reach it. The run is then
    unreachable until the fake clock is moved again, further. Measured while building this,
    on an isolated queue.

    Restores both on exit; an empty `absurd.fake_now` is how the SQL asks for the real clock
    (`absurd.sql` tests `length(trim(v_fake)) > 0`).
    """
    import absurd_sdk

    def frozen() -> dt.datetime:
        return when

    real = absurd_sdk._get_current_time
    # `setattr`, not a plain assignment: patching a module-level function is a dynamic act the
    # checker models as rebinding that exact function object, so the assignment form is an
    # `invalid-assignment` error no signature can satisfy. The SDK invites the patch — its
    # docstring reads "can be monkeypatched in tests" — and `frozen` matches the signature.
    setattr(absurd_sdk, "_get_current_time", frozen)  # noqa: B010
    app._conn.execute("SELECT set_config('absurd.fake_now', %s, false)", (when.isoformat(),))
    try:
        yield
    finally:
        setattr(absurd_sdk, "_get_current_time", real)  # noqa: B010
        app._conn.execute("SELECT set_config('absurd.fake_now', '', false)")


def run_until_result(app: Any, task_id: Any, max_batches: int = 24) -> Any:
    for _ in range(max_batches):
        snap = app.fetch_task_result(task_id)
        if snap and snap.state in ("completed", "failed", "cancelled"):
            return snap
        app.work_batch()
    return app.fetch_task_result(task_id)


class Executes(Protocol):
    """The one method the harness's own writes use, so a test can hand them a fake."""

    def execute(self, query: Any, params: Any = None) -> Any: ...


def is_disposable(conn: Executes) -> bool:
    """Whether this database has declared itself a throwaway fixture (`scripts/pgtest_up.sh`).

    Every row the test harness writes on its own account is gated on it: `DATABASE_URL` cannot
    tell the pg-test container from a dev database, and pytest may run against a production
    database, where a run parked for a human sign-off is non-terminal and must be kept."""
    found = conn.execute("SELECT to_regclass('_pgtest_disposable')").fetchone()
    return found is not None and found[0] is not None


def cancel_leftover_runs(conn: Executes) -> int:
    """Cancel every task with a non-terminal run on the `default` queue, returning how many.
    Refused on a database that has not declared itself disposable.

    Joined to its task, because a run can outlive a task row deleted by hand."""
    if not is_disposable(conn):
        raise PermissionError("this database is not marked disposable; cancelling nothing")
    return conn.execute(
        "SELECT absurd.cancel_task('default', task_id) FROM ("
        "SELECT DISTINCT task_id FROM absurd.r_default JOIN absurd.t_default USING (task_id)"
        " WHERE r_default.state IN ('pending', 'running', 'sleeping')) AS leftover"
    ).rowcount


def ledger_kinds(run_id: str) -> list[str]:
    with psycopg.connect(DSN) as conn:
        rows = conn.execute(
            "SELECT kind FROM ledger WHERE workflow_run_id = %s ORDER BY seq", (run_id,)
        ).fetchall()
    return [r[0] for r in rows]
