"""Two attempts of one task: the first's lease expires with an ERP op in flight, and a second
worker reclaims the task while that op waits.

Both engines fence a lost claim at its next checkpoint write, so the stale attempt lands the op it
holds and nothing after it. A schedule of the two attempts therefore varies two things:

| choice   | what it is                                                                |
|----------|---------------------------------------------------------------------------|
| `at`     | the stale attempt's ERP op, counted from one, that the lease expires at   |
| `before` | how many ERP calls the reclaiming attempt makes before the stale op lands |

A `before` past the reclaiming attempt's last call lands the stale op after the run completes.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from threading import Event, Thread, get_ident
from typing import Any
from uuid import UUID

import psycopg
from _conformance import PG_DSN

from examples.agent_firm.erp.port import Entry, Erp, ErpOp, ErpResult

TIMEOUT = 10.0
ENDED = ("completed", "failed")


def _unwatched(_: int) -> None:
    return None


@dataclass
class Watched:
    """An ERP that records each op it performs with its key, calling `before` ahead of each op and
    `after` once it lands, each with the count of ops it performed earlier."""

    erp: Erp
    before: Callable[[int], None] = _unwatched
    after: Callable[[int], None] = _unwatched
    performed: list[tuple[ErpOp, str]] = field(default_factory=list)

    def perform(self, op: ErpOp, key: str) -> ErpResult:
        earlier = len(self.performed)
        self.before(earlier)
        self.performed.append((op, key))
        try:
            return self.erp.perform(op, key)
        finally:
            self.after(earlier)

    def entries(self, invoice: str) -> list[Entry]:
        return self.erp.entries(invoice)

    def residual(self, document: str) -> Decimal:
        return self.erp.residual(document)

    def keyed(self, invoice: str) -> int:
        return self.erp.keyed(invoice)


@dataclass
class Overlap:
    """The two attempts' ERPs over one set of books, interleaved at `at` and `before`."""

    books: Erp
    at: int
    before: int
    reached: Event = field(default_factory=Event)
    """Set once the stale attempt holds its op, or once it ended without reaching it."""
    go: Event = field(default_factory=Event)
    holding: bool = False
    landed: Event = field(default_factory=Event)
    starts: list[int] = field(default_factory=list)
    """The thread each attempt of the task started on, in order."""
    overlapping: bool = False
    """Whether the second attempt started while the stale op was held, not yet landed."""
    abandoned: bool = False
    """Whether the hold waited out its release, so the op it held never landed."""

    def __post_init__(self) -> None:
        self.stale = Watched(self.books, self._hold, self._landed)
        self.reclaiming = Watched(self.books, self._release)

    def _hold(self, earlier: int) -> None:
        if earlier + 1 == self.at:
            self.holding = True
            self.reached.set()
            if not self.go.wait(TIMEOUT):
                self.abandoned = True
                raise AssertionError(f"nothing let the stale op {self.at} go")

    def _landed(self, earlier: int) -> None:
        if earlier + 1 == self.at:
            self.landed.set()

    def _release(self, earlier: int) -> None:
        if earlier == self.before:
            self.go.set()
            assert self.landed.wait(TIMEOUT), "the stale op never landed"

    def attempt(self, build: Callable[[Erp], Any]) -> Callable[[], Any]:
        """A backend's `fresh`: the first attempt's domain built over the stale ERP and every later
        one's over the reclaiming ERP, so only the first attempt can hold the op."""

        def fresh() -> Any:
            self.starts.append(get_ident())
            if len(self.starts) == 2:
                held = self.holding and not self.abandoned
                self.overlapping = held and not self.landed.is_set()
            return build(self.stale if len(self.starts) == 1 else self.reclaiming)

        return fresh

    @property
    def performed(self) -> list[tuple[ErpOp, str]]:
        return self.stale.performed + self.reclaiming.performed


def expire(backend: Any, task: UUID) -> None:
    """End the running attempt's lease, as a worker that stopped renewing it would."""
    match backend.name:
        case "sqlite":
            with backend.app.write_lock:
                backend.app.conn.execute(
                    "UPDATE tasks SET claim_expires_at=0 WHERE task_id=?", (task,)
                )
        case "postgres":
            with psycopg.connect(PG_DSN, autocommit=True) as conn:
                conn.execute(
                    t"UPDATE absurd.r_default SET claim_expires_at = now() - interval '1 hour' "
                    t"WHERE task_id = {task}::uuid AND state = 'running'"
                )
        case other:
            raise ValueError(f"no lease to expire on {other!r}")


def overlapped(backend: Any, task: UUID, overlap: Overlap) -> bool:
    """Run the task's first attempt to its held op, expire its lease, and reclaim the task with a
    second worker; `False` when the first worker completed the task without holding the op.

    A worker claims whatever the queue holds, so each worker works until it has run `task`, and an
    attempt counts as started once the backend builds its domain through `Overlap.attempt`."""
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(24):
                backend.app.work_batch()
                snapshot = backend.engine.fetch_task_result(task)
                if overlap.holding or (snapshot is not None and snapshot.state in ENDED):
                    break
        except BaseException as error:
            errors.append(error)
        finally:
            overlap.reached.set()

    stale = Thread(target=work, daemon=True)
    stale.start()
    try:
        assert overlap.reached.wait(TIMEOUT), "the first attempt neither reached its op nor ended"
        if not overlap.holding:
            stale.join(TIMEOUT)
            assert not errors, errors
            assert backend.run_until_result(task).state == "completed", "no op to hold, nor done"
            return False
        assert overlap.starts == [stale.ident], "the first worker started more than one attempt"
        expire(backend, task)
        for _ in range(24):
            backend.app.work_batch()
            if len(overlap.starts) >= 2:
                break
        assert overlap.starts == [stale.ident, get_ident()], "no second worker reclaimed the task"
        assert overlap.overlapping, "the second attempt started with no stale op held"
    finally:
        overlap.go.set()
        stale.join(TIMEOUT)
    assert not stale.is_alive(), "the stale attempt never ended"
    assert not errors, errors
    assert overlap.landed.is_set(), "the held op never landed"
    return True
