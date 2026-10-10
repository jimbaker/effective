"""Durable engines: the stores `DurableHandler` runs a workflow on, opened by URL.

| URL                         | engine                                  |
|-----------------------------|-----------------------------------------|
| `sqlite:///relative/file`   | `effective.engines.sqlite.SqliteApp`    |
| `sqlite:////absolute/file`  | `SqliteApp`                             |
| `sqlite://`                 | `SqliteApp` in memory                   |
| `postgresql://...`          | `effective.engines.absurd.AbsurdEngine` |

`open` imports a driver only in its own arm, so a SQLite process loads neither psycopg nor the
Absurd SDK. An engine runs tasks (`Engine`); one this process drives batch by batch, as a test or a
command-line run does, also `Drives`. A spawn's idempotency key is the task's stable identity: a
repeat returns the task already spawned under it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit
from uuid import UUID

if TYPE_CHECKING:
    from effective.engines.absurd import AbsurdEngine
    from effective.engines.sqlite import SqliteApp
    from effective.parked import ParkedTask

type TaskFn = Callable[[dict[str, Any], Any], Any]


class TaskState(StrEnum):
    """Where a task is, on any engine.

    | state     | the task                                     |
    |-----------|----------------------------------------------|
    | pending   | is claimable now or once its retry comes due |
    | running   | is held by a worker's claim                  |
    | sleeping  | waits for a time                             |
    | waiting   | waits for an event, with or without a time   |
    | completed | returned a result                            |
    | failed    | failed for good                              |
    | cancelled | was cancelled before it finished             |
    """

    PENDING = "pending"
    RUNNING = "running"
    SLEEPING = "sleeping"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)


@dataclass(frozen=True)
class Failure:
    """What a task failed of: its failing error's type name, `None` for a failure the engine wrote
    itself, and the engine's text for it, which names the type and the notes beside it."""

    kind: str | None
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class TaskSnapshot:
    """A task's state, with its result once completed and its failure once failed."""

    state: TaskState
    result: Any = None
    failure: Failure | None = None


class Engine(Protocol):
    """A durable engine running tasks: what a consumer of either engine calls."""

    def register_task(
        self, name: str, *, default_max_attempts: int | None = None
    ) -> Callable[[TaskFn], TaskFn]: ...

    def spawn(
        self,
        name: str,
        params: dict[str, Any],
        max_attempts: int | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> UUID: ...

    def emit_event(self, name: str, payload: Any) -> None: ...

    def cancel(self, task_id: UUID) -> None:
        """Cancel a task that has not finished, and deliver `Cancelled` to the parent that
        spawned it. A finished task stays as it ended; an unknown one raises `LookupError`. The
        first delivery on a done event stands, so a running task that delivered its ending before
        the cancel ends cancelled while its parent has that ending."""
        ...

    def fetch_task_result(self, task_id: UUID) -> TaskSnapshot | None: ...

    def parked(self) -> tuple[ParkedTask, ...]: ...

    def close(self) -> None: ...


class Drives(Protocol):
    """An engine this process drives batch by batch."""

    def work_batch(self) -> bool: ...

    def run_until_result(self, task_id: UUID, max_batches: int = 64) -> TaskSnapshot | None: ...


def drain(
    fetch: Callable[[UUID], TaskSnapshot | None],
    work_batch: Callable[[], bool],
    task_id: UUID,
    max_batches: int,
) -> TaskSnapshot | None:
    """Work batches until the task is terminal or nothing is claimable, as when it is parked."""
    for _ in range(max_batches):
        snapshot = fetch(task_id)
        if snapshot is not None and snapshot.state.terminal:
            return snapshot
        if not work_batch():
            break
    return fetch(task_id)


class NoDriver(ValueError):
    """A URL naming an engine this installation has no driver for."""


def open(
    url: str,
    *,
    queue: str = "default",
    default_max_attempts: int = 5,
    require_wal: bool = True,
) -> SqliteApp | AbsurdEngine:
    """The engine at `url`. `queue` is Absurd's; `require_wal` is SQLite's."""
    parts = urlsplit(url)
    match parts.scheme:
        case "sqlite" if parts.netloc or parts.query or parts.fragment:
            raise ValueError(f"a SQLite URL names a file path and nothing else: {url!r}")
        case "sqlite":
            from effective.engines.sqlite import SqliteApp

            return SqliteApp(
                parts.path.removeprefix("/") or ":memory:",
                require_wal=require_wal,
                default_max_attempts=default_max_attempts,
            )
        case "postgresql" | "postgres":
            from effective.engines.absurd import AbsurdEngine

            return AbsurdEngine(url, queue=queue, default_max_attempts=default_max_attempts)
        case scheme:
            raise NoDriver(f"no engine driver for {scheme!r} URLs")
