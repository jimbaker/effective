"""Is this run alive, done, or dead — and if dead, why. The read every surface needs and none had.

**Written because a run view could not tell a FAILED run from an empty one.** The terminal surface
rendered a run whose workflow had raised as a tree with no nodes, a status line reading `0 ops · 0
parked`, and nothing else — which is exactly what a run that did nothing looks like. Neither
bookkeeper can answer it: the checkpoint tape holds what a run *did*, and a run that died before
committing anything did nothing, truthfully. The engine's task row is where the difference lives.

**Normalized the way `ParkedTask.state` is, but only one engine is read yet.** SQLite says
`waiting` for an event park and Absurd says `sleeping` for both an event park and a timed one; a
consumer that had to branch on which engine answered would be reproducing the divergence rather
than reading past it. `RunState` is the vocabulary that split is normalized into and `raw_state`
keeps the engine's own word for anyone debugging the engine rather than the run.

**The Absurd half is NOT here.** A first attempt shipped one and it had never been executed: it
read `t.name`, `r.error` and `t.created_at`, where the columns are `task_name`, `failure_reason`
and `enqueue_at` on the other table, and `_state` refused `pending`, the ordinary first state of
every Absurd task.

Writing it is open work, and what makes it a task rather than a patch is not the
column names. `r_{queue}.failure_reason` is **jsonb** where SQLite's `tasks.failure` is
`repr(exc)`, so `failure` would carry a different shape depending on which engine answered, which
is the divergence this type exists to hide. And Absurd spells a timed sleep and an event park with
one word, `sleeping`, so telling PARKED from READY needs the run-table join and a `wake_event`
discriminator rather than a state lookup. Both are rulings about the shared vocabulary.
"""

import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from effective.sqlite import connect


class RunState(StrEnum):
    """What a run is doing, in the vocabulary an engine's own word is read into.

    Five, and the split that matters is `PARKED` against `READY`: a parked run is waiting on
    somebody, and a ready one is waiting on a drain. Those are different obligations and a surface
    that showed one word for both would tell a reader to sit still when the answer was to run a
    process."""

    READY = "ready"
    RUNNING = "running"
    PARKED = "parked"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def live(self) -> bool:
        """Is there more to come? A terminal run's picture is final; a live one's is a snapshot."""
        return self in {RunState.READY, RunState.RUNNING, RunState.PARKED}


@dataclass(frozen=True)
class RunStatus:
    """One task row, read as a run.

    `failure` is the field this type exists for and it is `str | None` rather than a bool: "it
    failed" without the message sends a reader to a database, which is the drop-out that produced
    this module."""

    task_id: UUID
    task_name: str
    state: RunState
    raw_state: str
    failure: str | None = None
    waiting_on: str | None = None
    params: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        """The LEDGER's id, which is not the task id: a caller's convention living in `params`.

        Falls back to the task id when params carry no `run_id`, which suits a task nobody
        scoped: it matches no ledger rows, and there are none to match."""
        return str((self.params or {}).get("run_id", self.task_id))


_SQLITE_STATES = {
    "ready": RunState.READY,
    "running": RunState.RUNNING,
    "waiting": RunState.PARKED,
    "sleeping": RunState.PARKED,
    "completed": RunState.COMPLETED,
    "failed": RunState.FAILED,
}
"""This engine's words, mapped. A word not in here is a divergence rather than a default — see
`_state`, which refuses instead of guessing."""


def _state(raw: str) -> RunState:
    """Normalize, or REFUSE.

    A `.get(raw, RunState.READY)` would turn an engine word nobody mapped into a confident wrong
    answer, and the wrong answer would be the reassuring one — a surface saying "ready" about a run
    in a state this module has never heard of. The engines are ours and their vocabularies are
    small, so an unmapped word is a fact worth raising on."""
    if raw not in _SQLITE_STATES:
        raise ValueError(f"unmapped engine state {raw!r} — add it to _SQLITE_STATES deliberately")
    return _SQLITE_STATES[raw]


_SQLITE_RUNS = (
    "SELECT task_id, name, state, failure, "
    "  CASE WHEN state = 'waiting' THEN waiting_event END, params "
    "FROM tasks ORDER BY rowid DESC"
)
"""Every run, newest first, with `waiting_on` answered only where the task is parked.

**A park is a STATE.** `tasks.waiting_event` names the event a park registered under, and the
claim that takes the task out of `waiting` is what ends that park, so a sleep, a retry or a
completion leaves the column standing until then. Reported raw it said a sleeping task was PARKED
on an event nobody was waiting for, which is the one thing this view exists to tell an operator
correctly.

**Deliberately a different question from `parked.read_sqlite_parked`**, which also asks whether
the deadline is still ahead. That relation is *what an emit can still answer*; this column is *what
this run is parked on*, and a run whose deadline is running out is parked on it right up to the
claim that ends it. Showing the name is how an operator sees the wait they are about to lose."""


def read_sqlite_runs(path: str | Path) -> tuple[RunStatus, ...]:
    """Every run in a SQLite engine file, newest first, opened READ-ONLY.

    `file:…?mode=ro` rather than a `SqliteApp`, whose constructor runs `executescript(_SCHEMA)` —
    a read-only observer that reached for the driver would issue DDL against somebody else's store
    to answer a GET. The same rule `read_sqlite_parked` follows, and for the same reason."""
    conn = connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        return read_sqlite_runs_conn(conn)
    finally:
        conn.close()


def read_sqlite_runs_conn(conn: sqlite3.Connection) -> tuple[RunStatus, ...]:
    """`read_sqlite_runs` over an ALREADY-OPEN connection — an `:memory:` store has no path to
    reopen. Mirrors the pair in `checkpoints` and `parked`."""
    return tuple(
        RunStatus(
            task_id=task_id,
            task_name=name,
            state=_state(state),
            raw_state=state,
            failure=failure,
            waiting_on=waiting_event,
            params=json.loads(params) if params else None,
        )
        for task_id, name, state, failure, waiting_event, params in conn.execute(_SQLITE_RUNS)
    )


__all__ = [
    "RunState",
    "RunStatus",
    "read_sqlite_runs",
    "read_sqlite_runs_conn",
]
