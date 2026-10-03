"""A worker that dies at a designed cut, under a gather or a race.

| scenario | the program                                | the cut                               |
|----------|--------------------------------------------|---------------------------------------|
| `gather` | two branches append `a0 a1` and `b0 b1`;   | after `b1`'s row lands and before its |
|          | a `Turnstile` forces `a0 b0 b1 a1`         | checkpoint, while `a1` is held        |
| `race`   | a winner appends `w0`; a loser appends     | after `l0`'s row lands and before its |
|          | `l0`, held until the choice is settled,    | checkpoint: admitted before the flag, |
|          | then `l1`                                  | interrupted after the choice          |

`work` is the child's entry point, run in its own interpreter, so the death takes every branch
thread at once.
"""

import os
import threading
from typing import Any

from _conformance import Fault, FaultCtx, FaultInjected, FaultPosition
from _schedules import Turnstile, ledger_path

from effective.api import Effect, append_ledger, gather, race
from effective.choice import Answer
from effective.handlers.absurd import DurableHandler
from effective.keys import Key, Run, Segment, compose_key
from effective.ops import LedgerRow

ORDER = ("a0", "b0", "b1", "a1")
CUT = ",b1"
"""Only `b1`'s row id contains this, so only `b1`'s append trips the fault."""

DIED = 9
"""The child's exit status when the cut fires."""


def row_id(run_id: str, path: str) -> Key:
    return compose_key(t"death-row:{Run(run_id)},{Segment(path)}")


def appends(run_id: str, branch: str) -> Effect[str]:
    for path in (f"{branch}0", f"{branch}1"):
        yield from append_ledger(LedgerRow(event_id=row_id(run_id, path), kind="death", path=path))
    return branch


def program(run_id: str) -> Effect[list[str]]:
    return (yield from gather([lambda: appends(run_id, "a"), lambda: appends(run_id, "b")]))


class Nothing:
    """A domain for a run that asks for nothing."""

    def run(self, op: Any) -> Any:
        raise AssertionError(f"the death program asks for nothing: {op!r}")


def cut() -> Fault:
    return Fault(on_name=CUT, position=FaultPosition.AFTER_THUNK)


RACE_CUT = ",l0"
"""Only the loser's first row id contains this."""

WINNER = ",w0"
"""Only the winner's row id contains this."""


def race_program(run_id: str) -> Effect[list[Any]]:
    def winner() -> Effect[str]:
        yield from append_ledger(LedgerRow(event_id=row_id(run_id, "w0"), kind="death", path="w0"))
        return "w"

    def loser() -> Effect[str]:
        for path in ("l0", "l1"):
            row = LedgerRow(event_id=row_id(run_id, path), kind="death", path=path)
            yield from append_ledger(row)
        return "l"

    answer: Answer[str] = yield from race([winner, loser])
    return [[type(ending).__name__, ending.index] for ending in answer.endings]


class HeldUntilSettled:
    """Holds the loser's first append inside the ledger write until the race's choice is settled,
    and the winner's append until that write has started, so the loser's op is in flight, past
    every check, when the flag goes up."""

    def __init__(self, ctx: Any) -> None:
        self._ctx, self._settled, self._started = ctx, threading.Event(), threading.Event()

    def settle(self, name: Key, value: Any) -> Any:
        stored = self._ctx.settle(name, value)
        self._settled.set()
        return stored

    def step(self, name: Key, thunk: Any) -> Any:
        if WINNER not in name.stored():
            return self._ctx.step(name, thunk)

        def waits() -> Any:
            assert self._started.wait(30), "the loser's first append never started"
            return thunk()

        return self._ctx.step(name, waits)

    def step_resolved(self, name: Key, thunk: Any) -> Any:
        """`step`'s wait, for a step whose thunk is handed the name the engine resolved."""
        if WINNER not in name.stored():
            return self._ctx.step_resolved(name, thunk)

        def waits(resolved: Key) -> Any:
            assert self._started.wait(30), "the loser's first append never started"
            return thunk(resolved)

        return self._ctx.step_resolved(name, waits)

    def holding(self, ledger: Any) -> Any:
        """`ledger`, holding the loser's first append until the choice is settled."""
        return _HeldLedger(ledger, self._started, self._settled)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


class _HeldLedger:
    def __init__(self, ledger: Any, started: threading.Event, settled: threading.Event) -> None:
        self._ledger, self._started, self._settled = ledger, started, settled

    def append(self, row: LedgerRow, **kwargs: Any) -> Any:
        if RACE_CUT in row.event_id.stored():
            self._started.set()
            assert self._settled.wait(30), "the choice was never settled"
        return self._ledger.append(row, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ledger, attr)


class Dies(FaultCtx):
    """`FaultCtx`, ending the process where it would raise."""

    def _trip(self, name: Key | str) -> None:
        try:
            super()._trip(name)
        except FaultInjected:
            os._exit(DIED)


def handler(ctx: Any, ledger: Any, *, dying: bool) -> DurableHandler:
    """The gather death run's handler when `dying`, and the recovery run's otherwise."""
    if not dying:
        return DurableHandler(ctx, Nothing(), ledger=ledger)
    turnstile = Turnstile(ORDER, ledger_path)
    return DurableHandler(
        Dies(ctx, cut()), Nothing(), ledger=ledger, op_layers=(turnstile.layer(),)
    )


def race_handler(ctx: Any, ledger: Any, *, dying: bool) -> DurableHandler:
    """The race death run's handler when `dying`, and the recovery run's otherwise."""
    if not dying:
        return DurableHandler(ctx, Nothing(), ledger=ledger)
    held = HeldUntilSettled(ctx)
    dies = Dies(held, Fault(on_name=RACE_CUT, position=FaultPosition.AFTER_THUNK))
    return DurableHandler(dies, Nothing(), ledger=held.holding(ledger))


SCENARIOS = {"gather": (program, handler), "race": (race_program, race_handler)}


def work(engine: str, where: str, task: str, dsn: str, scenario: str = "gather") -> None:
    """Claim `task` and run `scenario` until the cut ends this process."""
    program, handler = SCENARIOS[scenario]
    if engine == "sqlite":
        import effective.sqlite as sqlite

        sqlite.CLAIM_LEASE_SECONDS = 0.0  # the lease is spent the moment it is taken
        app = sqlite.SqliteApp(where)

        @app.register_task(task)
        def dying(params: Any, ctx: Any) -> Any:
            ledger = sqlite.SqliteLedger(app.conn, params["run_id"], app.write_lock)
            return handler(ctx, ledger, dying=True).run(lambda: program(params["run_id"]))

        app.work_batch()
    else:
        from effective.absurd_worker import absurd_worker
        from effective.handlers.absurd import ConcurrentAbsurdCtx
        from effective.ledger import PostgresLedger

        app: Any = absurd_worker(dsn, queue_name=where)

        @app.register_task(task)
        def dying(params: Any, ctx: Any) -> Any:
            ledger = PostgresLedger(dsn, workflow_run_id=params["run_id"])
            live = handler(ConcurrentAbsurdCtx(ctx), ledger, dying=True)
            return live.run(lambda: program(params["run_id"]))

        app.work_batch(claim_timeout=1)
    raise AssertionError("the worker outlived its cut")
