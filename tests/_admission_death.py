"""The process pin A1 runs (`tests/test_race_admission.py`): a race whose loser's layer injects an
`audit` call after the loser's own op, on SQLite.

`dying`: the domain holds `audit` until the race's choice is saved, logs the call, and ends the
process with `os._exit(9)` before its checkpoint lands. `recovering`: the same program over the
same file, with a domain that logs any `audit` it is asked to make.
"""

import os
import sys
import threading
from typing import Any
from uuid import UUID

from effective.api import call_tool, race
from effective.domain import CallTool
from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.handlers.durable import DurableHandler
from effective.keys import Key, race_choice
from effective.ops import Step

DIED = 9


class _Ctx:
    """The claimed ctx, marking when the race's choice has been saved."""

    def __init__(self, ctx: Any, chosen: threading.Event) -> None:
        self._ctx, self._chosen = ctx, chosen

    def settle(self, name: Key, value: Any) -> Any:
        stored = self._ctx.settle(name, value)
        if name == race_choice(0):
            self._chosen.set()
        return stored

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._ctx, attr)


class _Domain:
    """Logs each `audit` it runs; when `dying`, holds it for the choice and ends the process."""

    def __init__(self, calls: str, dying: bool, chosen: threading.Event) -> None:
        self._calls, self._dying, self._chosen = calls, dying, chosen
        self._started = threading.Event()

    def run(self, op: Any) -> Any:
        if op.name == "winner" and self._dying:
            assert self._started.wait(10)
        if op.name == "audit":
            self._started.set()
            if self._dying:
                assert self._chosen.wait(10)
            with open(self._calls, "a") as log:
                log.write("audit\n")
            if self._dying:
                os._exit(DIED)
        return op.name


def _audits(op: Any) -> Any:
    value = yield op
    if isinstance(op, Step) and op.op.name == "loser":
        yield Step(name="audit", op=CallTool(name="audit", args={}, result_schema=str))
    return value


def _program() -> Any:
    return (
        yield from race(
            [lambda: call_tool("winner", {}, str), lambda: call_tool("loser", {}, str)]
        )
    )


def main(db: str, task: str, calls: str, mode: str) -> None:
    app, chosen = SqliteApp(db), threading.Event()
    ctx: Any = _Ctx(SqliteTaskContext(app.conn, UUID(task), app.write_lock), chosen)
    try:
        DurableHandler(ctx, _Domain(calls, mode == "dying", chosen), op_layers=[_audits]).run(
            _program
        )
    finally:
        app.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
