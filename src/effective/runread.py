"""What a run view reads from a SQLite store, composed from the substrate's readers.

The shaping lives in `effective.runview` (the joins) and `effective.graphview` (the projections);
this module says which readers a surface calls and in what order, so a surface holds no database
code. A consumer renders a run with `to_markdown(view(db, task_id))` and no UI installed.

Every connection opens read-only and closes through `closing`: a `sqlite3.Connection` used as a
context manager ends the transaction and leaves the connection open.
"""

from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from uuid import UUID

from effective.checkpoints import Checkpoint, read_sqlite_task
from effective.graphview import RunGraph, from_keys
from effective.ledgerread import sqlite_payloads
from effective.parked import ParkedTask, pending_key, read_sqlite_parked
from effective.runs import RunStatus, read_sqlite_runs
from effective.runview import RunView, run_view
from effective.sqlite import connect
from effective.telemetry import Measurements


def runs(db_path: str | Path) -> tuple[RunStatus, ...]:
    """Every run in the store, newest first, with its state and its failure."""
    return read_sqlite_runs(db_path)


def tape(db_path: str | Path, task_id: UUID) -> tuple[Checkpoint, ...]:
    """The committed checkpoints, engine-internal names included.

    A viewer must see what the record holds: `ViewingCtx` compares against this set, and a tape
    narrower than the record would make it refuse ops that are recorded."""
    return read_sqlite_task(str(db_path), task_id, exclude=())


def graph(
    db_path: str | Path,
    task_id: UUID,
    *,
    checkpoints: Sequence[Checkpoint] | None = None,
    telemetry: Measurements | None = None,
) -> tuple[RunGraph, tuple[ParkedTask, ...]]:
    """The unrolled graph plus the parks, joined on the one node that has no producer.

    No engine checkpoints an await while it is pending, so a parked run's pending node arrives
    through `from_keys(pending=...)`. `telemetry` gives each node its cost and time, and
    `telemetry.sidecar_measurements` reads one from a sidecar; a node it misses is unmeasured."""
    committed = tuple(checkpoints if checkpoints is not None else tape(db_path, task_id))
    parked = parks(db_path, task_id)
    return (
        from_keys(
            str(task_id),
            [c.key.display() for c in committed],
            pending=pending_key(parked[0]) if parked else None,
            telemetry=telemetry,
        ),
        parked,
    )


def parks(db_path: str | Path, task_id: UUID) -> tuple[ParkedTask, ...]:
    """This run's parks."""
    return tuple(p for p in read_sqlite_parked(str(db_path)) if p.task_id == task_id)


def status(db_path: str | Path, task_id: UUID) -> RunStatus | None:
    """This run's engine row, or `None` for a task id the store does not hold."""
    return next((r for r in read_sqlite_runs(db_path) if r.task_id == task_id), None)


def ledger(db_path: str | Path, run_id: str) -> tuple[dict, ...]:
    """The canonical ledger rows of `run_id`, keyed by the run id and not the task id."""
    with closing(connect(f"file:{Path(db_path)}?mode=ro", uri=True)) as conn:
        return sqlite_payloads(conn, run_id)


def view(
    db_path: str | Path,
    task_id: UUID,
    *,
    project: Callable[[RunGraph], RunGraph] | None = None,
    telemetry: Measurements | None = None,
) -> RunView:
    """The whole shape a pane draws: graph, parks, recorded results, and the ledger join.

    `project` applies between the read and the join, since the joins are computed from node keys:
    a projection applied afterwards would discard a join made against the unrolled graph."""
    committed = tape(db_path, task_id)
    run_graph, parked = graph(db_path, task_id, checkpoints=committed, telemetry=telemetry)
    run = status(db_path, task_id)
    return run_view(
        project(run_graph) if project else run_graph,
        ledger=ledger(db_path, run.run_id if run else str(task_id)),
        parked=parked,
        recorded={c.key.display(): c.state for c in committed},
    )


__all__ = ["graph", "ledger", "parks", "runs", "status", "tape", "view"]
