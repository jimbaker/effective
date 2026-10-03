"""Which durable tasks are parked on an event, and how to answer one.

The checkpoint readers (`effective.checkpoints.read_sqlite_task`,
`effective.bridge_absurd.read_absurd_task`) answer *where a run has been*. This one answers
**where the fleet is waiting**. That is the read every interactive surface starts from: a list
page, a run view's "it is here now", and, because the wake registration is the address you emit
on, action discovery.

**The write belongs beside the read, and the reason is the coordinates.** A park's wake
registration carries whatever the walk applied: the enclosing `gather:{g},{i};` and `scoped(...)`
frames, and `Key.occurrence`'s `#N` for a `Scope.SETTLEMENT` namespace at its second ask.
`qualified_event_name` covers the frames and takes no occurrence, so a caller naming that second
ask asks the composed key for it: `qualified_event_name(*frames, name=...).occurrence(2)`.
Appending `#2` to the name itself is refused at the author's door. Reading the registration needs
to know none of it. Every other statement of this rule in this module points
back here rather than repeating it.

**It reports the wake registration RAW, exactly as the engine stores it**: not `event:{name}`,
not `$awaitEvent:{name}`, not a synthesized graph key. Raw is what the *action* path needs. The
*view* path needs a graph key instead, which is `pending_key` below: a named function beside the
record rather than a default buried in a reader.

**Unfiltered, and it does not classify.** A domain reader that is Absurd-only and narrows to `LIKE
'approve%'` makes a `budget-grant:` or `govern:` park invisible. Deciding what a park *means* from
its event name would be a denylist, bounded by the names it enumerates, and what the scope model
should be is an open decision rather than a `kind=` parameter.

Engine differences the record carries rather than papers over are in `ParkedTask`.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import psycopg

import effective.pgkeys  # noqa: F401  (registers the `Key` psycopg dumper)
from effective.graphview import PARKED
from effective.keys import Key, compose_key
from effective.sqlite import SqliteApp, connect

# ── the shared record ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParkedTask:
    """One durable task suspended on a named event, as either engine reports it.

    A **task**, not a run: `run_id` is a caller's convention living in `params`
    (`params["run_id"]`, `child_run_id`, `immutable_id`, …) rather than a substrate column, so the
    record hands the caller the whole `params` and lets it project. A name like `ParkedRun` would
    assert a key the substrate does not have.

    Three fields are the same thing on both engines and one is not.

    - `task_id`: a `UUID` on both, because both engines MINT one. Absurd's `t_{queue}.task_id` is
      a `uuid` column psycopg hydrates directly; `SqliteApp.spawn` mints `uuid7` and the driver
      parses it back (`effective.sqlite.connect`). Normalizing down to `str` would cost the
      validating parse (`UUID(text)` refuses a malformed id where a `str` accepts anything) and the
      time-ordering. A caller round-tripping this through a URL still does not have to know which
      engine answered: FastAPI coerces a `UUID` path parameter natively and 422s a malformed one.
    - `task_name`: the registered task name, the same string on both.
    - `wake_event`: the wake registration, RAW (see the module docstring). Never `None`, because a
      task in this list is parked on an *event*, which is the actionable park.
    - `params`: the immutable spawn params, decoded to a Python dict on both.
    - `state`: **normalized to the substrate's own vocabulary** (`graphview.PARKED`), not each
      engine's. SQLite spells an event park `'waiting'` and keeps `'sleeping'` for a timed one;
      Absurd spells both `'sleeping'`. Carrying either word would ship a field whose meaning
      depends on which engine answered, into a record whose job is to be engine-neutral: a
      consumer would have to branch on the engine to read it. The tell was that the field could
      not go in the conformance suite.

      Every row in this relation is `PARKED` today, because the relation *is* "parked on an
      event", so the field is currently redundant. It is carried because it is the word the view
      consumes directly (`from_keys(states=…)`, `Node.state`), so a caller never translates, and
      because the vocabulary has room the moment the relation widens: a timed park (an open
      decision) is a second word here, not a second meaning for this one.
    - `parked_since`: **`None` on SQLite, which has no such column.** `effective.sqlite`'s `tasks`
      table records no park timestamp at all, so the honest answer is absence. On Absurd it is the
      current run's `started_at`.
      That is when the attempt *started*, an upper bound on when it parked; `w_{queue}.created_at`
      is the exact instant if a consumer ever needs it, at the cost of a third join.
    """

    task_id: UUID
    task_name: str
    wake_event: str
    params: dict[str, Any] = field(default_factory=dict)
    state: str = ""
    parked_since: datetime | None = None


# ── the bridge to the view: one park, one pending node ───────────────────────


def pending_key(park: ParkedTask) -> Key:
    """The graph key of the node a parked task shows up as: `event;{wake_event}`.

    The **synthesized** node, which neither bookkeeper holds while a run is parked: SQLite
    writes no checkpoint for a plain await (the park lives in `tasks.waiting_event`) and
    Absurd freezes `$awaitEvent:{name}` only on *delivery*. So the view mints it, which makes the
    spelling a decision rather than a discovery.

    `event;{wake_event}` is exactly `op_key(AwaitEvent(name=…))` (`handlers/base.py`): the pending
    node keyed as the op it is waiting on, so nothing new enters the key grammar. Three properties
    carry it:

    - **`kind_of` reads it as an `await`**, so `graphview` draws the hexagon. A node keyed on the
      raw registration (`review:m1`) is a `step`: a rectangle, rendered fine and wrong.
    - **`event` is an ARM TAG**, and a Step's key opens with its own arm, so `op_key` cannot author
      a `Step` named `event:…`. What a synthesized key CAN meet is the slot a bounded wait settles
      its outcome in, which both engines write under the op's placement — and meeting it is right:
      the two name the same await, so `to_mermaid` drawing one node is the wanted reading. A
      second ask of the name carries an occurrence and is a node of its own.
    - **it is engine-neutral**, which is what `from_keys` already claims about its input. The
      alternative `$awaitEvent:{wake_event}` is one engine's SDK spelling for a row that exists
      only *after* the answer arrives, and `ENGINE_INTERNAL` filters it out of the default view.

    **`wake_event` goes in as the terminal hole, verbatim**, including a gather branch's
    `gather:{g},{i};` prefix, which the engines carry *inside the awaited name* rather than around
    the key (`_PrefixedCtx.await_event`). A parked branch composes `event;gather:0,0;ev:r1` with
    the branch coordinate INSIDE the `event` arm, where `graphview.strip_branches` (anchored
    `^gather:`) cannot see it. It is the one place a branch coordinate is not outermost, pinned by
    `tests/test_graphview.py::test_a_parked_gather_branch_keeps_its_branch_coordinate_inside_the_tag`."""
    # An ADDRESS, and it can say so because `grammar.admits` looks PAST a frame: the engines carry
    # the branch coordinate INSIDE the awaited name (the paragraph above), so this arrives as
    # `gather:0,0;ev0:r1`, an address wearing a frame, whose leading tag is an arm's.
    return compose_key(t"event;{Key.parse(park.wake_event):domain=address}")


# ── the 0↔1 half: the embedded SQLite engine ─────────────────────────────────

_SQLITE_PARKED = (
    "SELECT task_id, name, params, waiting_event FROM tasks "
    "WHERE state = 'waiting' AND waiting_event IS NOT NULL "
    "  AND (available_at = 0 OR available_at > unixepoch('subsec')) "
    "ORDER BY rowid"
)
"""Every event-parked task a human can still answer, oldest first.

**Three clauses and each earns its place**, which is worth saying because a second reader once
asked only the first: `waiting` is what a park IS, a name is what an emitter aims at, and a
deadline still ahead is what makes the answer land. `bridge_sqlite.park_name` asks the same
question of one task and PROJECTS this relation rather than restating it.

`state = 'waiting'` is the definition of the relation, not a filter over it: `SqliteApp.work_batch`
sets exactly that state when a `_Suspend` carries an event, and `'sleeping'` when it carries a
deadline alone. A `_Suspend` carrying both is `'waiting'` with its deadline in `available_at`, and
the clause above keeps such a task listed while its deadline is ahead: answering it still does
something. Once the deadline passes the task is *claimable*, and a claimable task is nobody's to
answer, which is the same reason a `'sleeping'` task whose time has come stays out. A timed park
with no event is out entirely: `wake_event` would have to go optional to carry it.

`ORDER BY rowid` is spawn order, which is the closest thing this engine has to "oldest first"
(there is no timestamp; see `ParkedTask.parked_since`). Hole-free, so it stays a plain literal:
`effective.sql.bind` composes a statement that has data to keep separate, and this one has none."""


def read_sqlite_parked(path: str | Path) -> tuple[ParkedTask, ...]:
    """Every event-parked task in a SQLite engine file, oldest first, opened READ-ONLY.

    A second process holding only the file. It opens `file:…?mode=ro` rather than constructing a
    `SqliteApp`, whose constructor runs `executescript(_SCHEMA)`, so a read-only observer that
    reached for the driver would issue DDL against somebody else's store to answer a GET.

    Use `read_sqlite_parked_conn` for an `:memory:` engine, which has no path to reopen."""
    conn = connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        return read_sqlite_parked_conn(conn)
    finally:
        conn.close()


def read_sqlite_parked_conn(conn: sqlite3.Connection) -> tuple[ParkedTask, ...]:
    """`read_sqlite_parked` over an ALREADY-OPEN connection: the same read, minus the file.

    Mirrors the `read_sqlite_task` / `read_sqlite_conn` pair in `effective.checkpoints`: an
    `:memory:`
    store dies with its connection, so the conformance harness and any in-process caller need this
    entry point and the path form is that plus a read-only open."""
    return tuple(
        ParkedTask(
            task_id=task_id,
            task_name=task_name,
            wake_event=waiting_event,
            params=json.loads(params),
            state=PARKED,  # normalized: this engine's own word is 'waiting'
            parked_since=None,  # this engine records none; see ParkedTask.parked_since
        )
        for task_id, task_name, params, waiting_event in conn.execute(_SQLITE_PARKED)
    )


def read_absurd_parked(
    conn: psycopg.Connection[Any], *, queue: str = "default"
) -> tuple[ParkedTask, ...]:
    """Every event-parked task on the queue, oldest first. The 0↔N half of the park reader pair
    (`effective.parked.read_sqlite_parked` is the 0↔1 half). Read-only; unfiltered.

    `park_name` generalized from one task to the fleet, and it is `park_name`'s join that gets
    generalized. That matters:

    - **`t.last_attempt_run = r.run_id`, not `USING (task_id)`.** `r_{queue}` holds one row per
      *attempt*, so `USING (task_id)` returns one row per sleeping ATTEMPT while this returns one
      row per parked TASK, by construction. No divergence is reproduced today, because a parked
      run's earlier attempts are terminal, so both queries agree. But a fleet list keyed on task
      identity should not depend on that holding, and a duplicate here would offer a human an
      emit against a superseded run.
    - **the `LIKE 'approve%'` narrowing is gone.** This lists every park, `budget-grant:` and
      `govern:` and `fork:` included. What a park *means* is not decided here (see
      `effective.parked`).

    `r.state = 'sleeping' AND r.wake_event IS NOT NULL` is the definition of an event park on this
    engine: a timed park is `'sleeping'` with a NULL `wake_event` (Absurd has one state word for
    both, where SQLite has two; a divergence `ParkedTask.state` normalizes away rather than
    carries, so no consumer has to branch on which engine answered).
    The run's state is the authority, not the task's: `t.state` is set `'sleeping'` alongside it,
    but the run row is where the wake registration lives.
    """
    r_tbl, t_tbl = f"r_{queue}", f"t_{queue}"
    rows = conn.execute(
        t"SELECT t.task_id, t.task_name, t.params, r.wake_event, r.started_at "
        t"FROM absurd.{r_tbl:i} r "
        t"JOIN absurd.{t_tbl:i} t ON t.last_attempt_run = r.run_id "
        t"WHERE r.state = 'sleeping' AND r.wake_event IS NOT NULL "
        t"ORDER BY r.started_at, t.task_id"
    ).fetchall()
    return tuple(
        # No `str(task_id)`: this column is a `uuid` psycopg hydrates as a `UUID`, and SQLite
        # mints and parses one too, so the two engines agree on the VALUE and there is nothing to
        # normalize. A stringify would normalize DOWN to the weaker of the two types; see
        # `ParkedTask.task_id`.
        ParkedTask(
            task_id=task_id,
            task_name=task_name,
            wake_event=wake_event,
            params=params,  # jsonb -> a Python dict already
            state=PARKED,  # normalized: this engine's own word is 'sleeping'
            parked_since=started_at,
        )
        for task_id, task_name, params, wake_event, started_at in rows
    )


# ── the action: answering a park you read ────────────────────────────────────
#
# **`answer` is not `emit_event` with a nicer argument. They are two operations**, and one name
# for both is what let them be conflated:
#
#   `emit_event(name, payload)`   this happened. The name comes from a spawn param or an author,
#                                 no waiter need exist, and emitting before the await is normal:
#                                 a fork child emits its done-event when it finishes.
#   `answer(park, payload)`       settle THIS park. The name comes from the registration you read,
#                                 so the caller never composes one.
#
# What makes the second worth a verb is the coordinates argument in this module's docstring.
#
# ONE VERB, and the engine difference is absorbed here rather than published. Two verbs, one per
# engine, do not remove the branch on the engine: they move it out to every caller, which is what
# `ParkedTask.state` calls the thing parity exists to prevent.


def answer(app: SqliteApp | BroadcastEmitter, park: ParkedTask, payload: Any) -> str:
    """Settle the park you READ, and return the name answered.

    Takes a `ParkedTask` rather than a name: the registration is already in hand, so nothing has
    to be recomposed to reach the ask it belongs to.

    **First write wins.** Events are immutable once emitted on both engines, so answering an
    already-answered name is a silent no-op rather than an overwrite: the second answer is
    discarded, not applied. A surface that must know whether ITS answer settled the park has to
    read the park back, not trust this return value.

    **The reach differs by engine, and it is the name that has to carry the difference.** On
    SQLite delivery is addressed, so this settles exactly the task named. On Absurd it is
    broadcast: this settles EVERY task registered on `park.wake_event` in the queue, and the event
    stays set, satisfying any later await on that name too (both measured,
    `tests/test_conformance.py`). One answer settling one occurrence therefore rests on the NAME
    carrying enough coordinates, which is what the generation coordinate on `depth-grant:`,
    `govern:` and `approve:` is for, and why two hand-spawned tasks sharing a `run_id` alias by
    design (`tests/test_grant_aliasing.py`).

    Marks the task claimable; it does NOT resume it. A drain must still run, because nothing polls.
    """
    app.emit_event(park.wake_event, payload)
    return park.wake_event


class BroadcastEmitter(Protocol):
    """The one method `answer` needs on a broadcast engine: `Absurd.emit_event`, structurally.

    Declared BELOW its use in `answer`'s annotation, which is ordinary in 3.14: PEP 649 defers
    annotation evaluation, so the two read in the order they matter rather than in dependency
    order.

    A Protocol rather than the concrete class, for the reason `TaskContext` is one: `effective`
    does not import `absurd_sdk` at module level (only `handlers/absurd.py` does, at its one use,
    with a recorded reason in the lazy-import allowlist). Stating the shape we depend on is a
    smaller claim than importing a vendor and is checkable at the call site either way."""

    def emit_event(self, event_name: str, payload: Any = ..., /) -> None: ...


__all__ = [
    "ParkedTask",
    "answer",
    "pending_key",
    "read_absurd_parked",
    "read_sqlite_parked",
    "read_sqlite_parked_conn",
]
