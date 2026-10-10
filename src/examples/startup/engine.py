"""The scripted world the startup examples run against, and one run of a workflow on SQLite.

A `World` answers every model, tool and judgment call from Python functions and counts them, so
a run needs no keys and no network. `run` drives a workflow as a task on the embedded engine,
delivering each event it parks on, and returns the result with the op keys the store holds.
Each delivery starts a new attempt that replays the recorded ops, which the counts show: an op
the store holds is never asked again.
"""

import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from effective import Contract, MeteredInterpreter, Usage
from effective.api import Effect
from effective.checkpoints import keys, read_sqlite_task
from effective.domain import AskLLM, CallTool, Judge
from effective.engines.sqlite import SqliteApp
from effective.handlers.durable import DurableHandler
from effective.parked import pending_key, read_sqlite_parked_conn

ENDED = frozenset({"completed", "failed", "cancelled"})


@dataclass
class World:
    """What the ops reach, as functions: `llm` reads a prompt's text, `tools` maps a tool name to
    its body over the call's arguments, and `judge` answers a judgment's state."""

    llm: Callable[[str], Any]
    tools: Mapping[str, Callable[[Mapping[str, Any]], Any]]
    judge: Callable[[Judge[Any]], Mapping[str, Any]] = lambda op: {}
    calls: Counter[str] = field(default_factory=Counter)

    def _model(self, op: AskLLM[Any]) -> tuple[Any, Usage]:
        self.calls["llm"] += 1
        return self.llm(str(op.messages)), Usage()

    def _tool(self, op: CallTool[Any]) -> Any:
        self.calls[op.name] += 1
        return self.tools[op.name](op.args)

    def _judged(self, op: Judge[Any]) -> tuple[Any, Usage]:
        self.calls["judge"] += 1
        return self.judge(op), Usage()

    def domain(self) -> MeteredInterpreter:
        return MeteredInterpreter(llm=self._model, tools=self._tool, judge=self._judged)


@dataclass(frozen=True)
class Ran:
    result: Any
    keys: tuple[str, ...]
    delivered: tuple[str, ...]
    attempts: int
    timeline: tuple[str, ...] = ()
    """`keys` with each await the run parked on placed where it parked. The store holds no
    checkpoint for an await, so the park reader is what places it."""


def run(
    program: Callable[[], Effect[BaseModel]],
    world: World,
    deliver: Callable[[str], BaseModel] | None = None,
    *,
    store: Path | None = None,
) -> Ran:
    """Run `program` to its end, answering each park with `deliver(event name)`.

    `store` is a database file earlier runs may share, so an event they were sent is already
    delivered; without one, the run gets a fresh file."""
    with tempfile.TemporaryDirectory() as directory:
        return _run(program, world, deliver, store or Path(directory) / "startup.db")


def _run(
    program: Callable[[], Effect[BaseModel]],
    world: World,
    deliver: Callable[[str], BaseModel] | None,
    db: Path,
) -> Ran:
    attempts, delivered, parks = 0, [], []
    app = SqliteApp(str(db))

    @app.register_task("startup")
    def task(params: dict[str, Any], ctx: Any) -> Any:
        nonlocal attempts
        attempts += 1
        return DurableHandler(ctx, world.domain(), ledger=None, contract=Contract.V1).run(program)

    try:
        spawned = app.spawn("startup", {})
        while (done := app.run_until_result(spawned)) is not None and done.state not in ENDED:
            parked = [p for p in read_sqlite_parked_conn(app.conn) if p.task_id == spawned]
            if not parked:
                raise RuntimeError(f"the run is {done.state} with no event to answer")
            if deliver is None:
                raise RuntimeError("the run parked, and nobody answers it")
            recorded = len(keys(read_sqlite_task(db, spawned)))
            for waiting in parked:
                delivered.append(waiting.wake_event)
                parks.append((recorded, pending_key(waiting).display()))
                app.emit_event(waiting.wake_event, deliver(waiting.wake_event).model_dump())
        if done is None or done.state != "completed":
            raise RuntimeError(f"the run ended {done}")
        recorded = keys(read_sqlite_task(db, spawned))
        timeline = list(recorded)
        for placed, (at, await_key) in enumerate(parks):
            timeline.insert(at + placed, await_key)
        return Ran(done.result, recorded, tuple(delivered), attempts, tuple(timeline))
    finally:
        app.close()
