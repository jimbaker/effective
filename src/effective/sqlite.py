"""Embedded SQLite durable engine: the 0↔1 regime.

The primary durable backend is Absurd/Postgres (0↔N). This is its
single-writer sibling: durable execution from one SQLite file, no server: a
laptop without Postgres, or a Cloudflare Durable Object (same engine, different
home).

There is no ``SqliteHandler``: ``DurableHandler`` is a generic ``TaskContext``
interpreter (it requires ``step``/``await_event``/``sleep_until`` and discovers the
optional capabilities, ``peek_event`` and ``repark`` among them, by ``getattr``), so the
SQLite side is an *engine*. ``SqliteApp`` is the driver (spawn / claim-with-lease / work-loop),
``SqliteTaskContext`` is the durable ctx, ``SqliteLedger`` the append-only
canonical record. Run a workflow with::

    app = SqliteApp(":memory:")
    @app.register_task("ticket")
    def task(params, ctx):
        ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        return DurableHandler(ctx, domain, ledger=ledger).run(lambda: workflow(...))
    tid = app.spawn("ticket", {"run_id": "r1"})
    snap = app.run_until_result(tid)

Durability is by *replay*, never a captured continuation: on resume the task fn
re-runs from scratch; ``step`` returns committed checkpoints without re-running
their thunks, ``await_event`` re-binds the delivered payload by name.
Single-writer is the *definition* of 0↔1: parallel ``gather`` branches key
distinct checkpoint rows, so they serialize without conflict.

Modern SQLite only (3.45+, what Python 3.14 bundles): upsert (3.24), ``JSON``
operators (3.38), the append-only trigger via ``RAISE(ABORT)``.
"""

import itertools
import json
import os
import sqlite3
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid7

from pydantic_core import to_jsonable_python

from effective.handlers.base import Attempt, EngineSignal, failing_leaf
from effective.keys import Key
from effective.ops import (
    DONE_EVENT_PARAM,
    LedgerRow,
    WaitOutcome,
    Writer,
    noted,
    refuse_placed_writer_collision,
    settled_wait,
)
from effective.spawning import failure_answer
from effective.sql import bind

CLAIM_LEASE_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 3

# A task id is a `UUID` OBJECT at every seam — never a string the caller might parse.
# SQLite has no uuid type, so the driver carries the conversion: the adapter writes the
# canonical text on the way in, and `PARSE_DECLTYPES` + this converter parse it back on the way
# out, keyed on the `UUID` *declared* type in `_SCHEMA`. That is where "we control the sqlite
# driver" cashes out — no caller ever sees a `str`, so the two engines agree on
# the VALUE (Absurd's `t_{queue}.task_id` is a `uuid` column psycopg hydrates the same way) and
# `UUID(text)` is a validating parse rather than a cast.
#
# Registration is process-wide, which is correct here rather than merely convenient: it is keyed
# on a stdlib type and a declared column type this module owns, so importing the engine is what
# makes its own ids round-trip. The converter must accept `bytes` — sqlite3 hands converters the
# raw stored value.
sqlite3.register_adapter(UUID, str)
sqlite3.register_converter("UUID", lambda raw: UUID(raw.decode()))

# The SERDE half of the two positions allowed to treat a `Key` as text (`effective.keys.Key`).
# Registered against the driver rather than called at a site, so binding a checkpoint name into
# SQL needs no conversion at all — the consumer that most legitimately wants text never asks.
# There is deliberately NO converter back: a column holding a key is read as text and re-enters
# the typed world through `Key.parse`, which is the named read boundary.
sqlite3.register_adapter(Key, lambda key: key.stored())


TRACK_CONNECTIONS = "EFFECTIVE_TRACK_CONNECTIONS"
"""Env var enabling leak accounting in `connect`. Off by default; the test session turns it on.

Opt-in because the alternative gate does not work. An unclosed `sqlite3.Connection` announces
itself as a `ResourceWarning` **when the garbage collector gets to it**, which lands the warning
in whatever test happened to be running — so `-W error::ResourceWarning` reddens an innocent
test, nondeterministically, and the report cannot name the leak. Counting explicit closes
instead makes a leak an exact fact with a creation site, at the cost of a dict entry per open
connection while it is on.
"""

_unclosed: dict[int, str] = {}
_track_seq = itertools.count()


class _TrackedConnection(sqlite3.Connection):
    """A connection that records its own release. Only `close` is overridden — the C-level
    finalizer that runs on garbage collection does NOT route through here, which is exactly what
    makes the leftover entry mean "nobody closed this" rather than "this is still alive"."""

    def close(self) -> None:
        _unclosed.pop(getattr(self, "_track_seq", -1), None)
        super().close()


def _creation_site() -> str:
    """The first frame outside this module — where a caller opened the connection."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        if not frame.filename.endswith("effective/sqlite.py"):
            return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return "<unknown>"


def unclosed_connections() -> tuple[str, ...]:
    """Creation sites of connections opened through `connect` and never explicitly closed.

    Empty unless `TRACK_CONNECTIONS` is set. Sorted and de-duplicated with counts by the caller;
    this returns one entry per leaked connection so a site leaking 40 of them reads as 40.
    """
    return tuple(_unclosed.values())


def connect(path: str = ":memory:", *, uri: bool = False) -> sqlite3.Connection:
    """Open a connection to an Effective SQLite store — the ONE place the driver is configured.

    Every reader of this store goes through here, not through `sqlite3.connect`, because
    ``detect_types`` is what makes a `task_id` come back as a `UUID`: the converter above is
    keyed on the declared column type, and a connection that skips ``PARSE_DECLTYPES`` silently
    yields the text instead. That is a *quiet* divergence — the read still works, and the caller
    gets the weaker type from the engine we control — so the fix is a shared opener rather than a
    rule each reader remembers. (Third requirement, in the SELECTs: name the **bare** column;
    ``SELECT COALESCE(task_id, task_id)`` has no declared type and yields text again.)

    ``uri=True`` is for the read-only observers (`effective.parked.read_sqlite_parked`,
    `effective.checkpoints.read_sqlite_task`), which open ``file:…?mode=ro`` rather than
    constructing a `SqliteApp` — a reader must not issue this module's DDL against somebody
    else's store to answer a GET.
    """
    tracking = os.environ.get(TRACK_CONNECTIONS) == "1"
    conn = sqlite3.connect(
        path,
        isolation_level=None,
        check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES,
        uri=uri,
        factory=_TrackedConnection if tracking else sqlite3.Connection,
    )
    if tracking:
        # Keyed on a counter rather than `id(conn)`: a leaked connection's address is reused, and
        # that would silently overwrite one leak's record with another's — undercounting exactly
        # when the count matters.
        seq = next(_track_seq)
        # The suppression below is load-bearing, not laziness: `factory=` makes this a
        # `_TrackedConnection`, but the stdlib types `sqlite3.connect` as returning the base
        # `Connection`, so the attribute the subclass reads back in `close` is invisible to the
        # checker at the point it is set. Keep the reason on its own line and the directive
        # inline — ty parses a suppression anywhere it appears in a comment, including inside
        # prose describing one, so an explanation that quotes the syntax breaks the build.
        conn._track_seq = seq  # ty: ignore[unresolved-attribute]
        _unclosed[seq] = _creation_site()
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _is_fresh_store(conn: sqlite3.Connection) -> bool:
    """No ``tasks`` table yet — this open is CREATING the store rather than visiting one."""
    return not {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}


def enable_wal(conn: sqlite3.Connection) -> str:
    """Put a FILE-backed store into WAL, and return the journal mode now in force.

    Separate from `connect` on purpose, and the distinction is not cosmetic: `connect`
    configures the **driver** — per-connection settings that die with the connection — whereas a
    journal mode is a property of the **database file**, persistent, and therefore a decision
    that belongs to whoever owns the store.

    **`SqliteApp` calls this only when it is CREATING a store**, never when opening one that
    already exists, and the difference is worth the extra condition. WAL is a header write, so
    imposing it on an existing file mutates somebody else's data to answer their own question,
    the hazard the read-only observers exist to avoid. The concrete case, measured 2026-08-09
    over `agent.bank.all_stores()`: the banked bench corpus held 325 `task.db` stores, **all**
    `journal_mode=delete`, 194 of them pre-uuid7. Those stores are opened READ-WRITE on a path
    that only reads (`contrastbench.replay_combinator` constructs a `SqliteApp` over each one,
    driven by `test_banked_corpus.py::test_current_era_stores_still_replay`), so a "convert
    unless legacy" rule would have silently rewritten the other **131** paid files on first open.

    An existing store therefore keeps its mode, and converting one is a deliberate act: call
    this directly. That is the right default for a substrate whose stores outlive its versions.

    **The cost of WAL, stated because it is a real narrowing.** A WAL store cannot be read from
    a READ-ONLY DIRECTORY: the flag lives in the file header, so SQLite insists on creating the
    `-shm` sidecar even when no `-wal` exists and even for `mode=ro`, and the open fails with
    `attempt to write a readonly database` where a rollback store reads fine. That reaches the
    observers, which only ever open read-only — so an archived store on an immutable mount or a
    read-only container layer is affected. No current caller does this (today's corpus is all
    `delete`), but every store created from here on is WAL. Measured: a `delete` store reads
    fine there, a WAL store fails, and `?mode=ro&immutable=1` reads BOTH — so that is the escape
    if the case ever arrives. It is not the default because `immutable=1` promises SQLite the
    file cannot change underneath it, which is true of an archived artifact and false of a live
    store; getting that wrong buys silent stale reads instead of a loud error.

    Why WAL for a store we do own: the engine's read-only observers (`effective.parked`,
    `effective.checkpoints`) open a *second* connection to the same file, and under the
    rollback-journal default a writer locks them out — a 5s `busy_timeout` stall, then
    `SQLITE_BUSY`. The usual objection to WAL does not bite here: a `file:…?mode=ro`
    open succeeds against a WAL database with no writer holding it, and also after an *unclean*
    exit that left `-wal`/`-shm` behind, which SQLite recovers. The one case that does fail is the
    read-only directory above, and it is NOT excused by "a writable open would fail there too",
    since these observers never attempt one.

    `synchronous` stays at the `FULL` default rather than dropping to the `NORMAL` that WAL
    guidance usually suggests. `NORMAL` under WAL can lose the last commits on power loss, and
    a checkpoint that is durable the moment its step commits is the property crash-replay rests
    on. Slower and correct is the right end of that trade for a durable engine.
    """
    return conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]


def _failure_text(error: BaseException) -> str:
    """A failed task's `failure`: the error's `repr`, then its notes, a line each."""
    return "\n".join([repr(error), *getattr(error, "__notes__", ())])


class _Suspend(EngineSignal):
    """Raised by the ctx to park a task durably: waiting on an event, or sleeping.

    A control signal, not an error condition (hence the non-``Error`` name).

    Propagates cleanly through ``DurableHandler.run`` (which catches only
    ``Refused``) out to ``SqliteApp.work_batch``, which records the park.
    """

    def __init__(self, *, event: Key | None = None, until: float | None = None) -> None:
        # `event` is a `Key`; it reaches the `waiting_event` column through the registered
        # sqlite3 adapter, so the stored text is unchanged and an emitter's `str` still matches.
        # BOTH may be set: whichever comes first ends the park.
        self.event = event
        self.until = until


_IDEMPOTENCY_INDEX = """
-- At-most-once lives HERE, on its own column, and not on the task id. Composing the id from
-- `name` and the key put two shapes on one delimiter and made `spawn("w", …)` collide with
-- `spawn("w", …, idempotency_key="1")`; a unique index states the property the id was being
-- overloaded to imply. SQLite treats NULLs as DISTINCT in a unique index, so keyless spawns are
-- unconstrained by it — which is the wanted reading: no key, no at-most-once claim.
CREATE UNIQUE INDEX IF NOT EXISTS tasks_idempotency ON tasks (name, idempotency_key);
"""
"""Applied only where the column exists — i.e. every store except a pre-uuid7 one, which this
engine opens READ-ONLY (see `SqliteApp.__init__`). Kept out of `_SCHEMA` because a
`CREATE INDEX` naming an absent column fails outright, and the whole point is that opening a
banked store must not."""


_SCHEMA = """
-- `available_at` is the instant a task became, or becomes, claimable, and `_claim_locked`
-- orders the queue by it so the work that came due first runs first:
--
--   ready     when it was enqueued, woken or re-queued
--   sleeping  when its sleep ends
--   waiting   its deadline, or 0 for a wait that named none
--   running   what it held before the claim; the lease decides this one
CREATE TABLE IF NOT EXISTS tasks (
  task_id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  params TEXT NOT NULL,
  state TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 1,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  available_at REAL NOT NULL DEFAULT 0,
  waiting_event TEXT,
  claimed_by TEXT,
  claim_expires_at REAL,
  result TEXT,
  failure TEXT,
  idempotency_key TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
  task_id UUID NOT NULL,
  name TEXT NOT NULL,
  state TEXT,
  PRIMARY KEY (task_id, name)
);
-- Delivery is BROADCAST: keyed by name alone, queue-wide, matching the reference engine's
-- `e_{queue}(event_name text primary key, payload jsonb, emitted_at timestamptz)`.
--
-- There is no `task_id`: it would name an ADDRESSEE, and a broadcast engine has none. The
-- reference keeps waiter provenance in a separate table (`w_{queue}`), which on this engine is
-- `tasks.waiting_event`.
--
-- `emitted_at` is carried for parity with that reference and NOTHING READS IT YET. What a bounded
-- wait reads instead is `tasks.waiting_event`, which says whether the CLOCK or the event woke this
-- task — the question the SDK answers from `wake_event`, and the one emission time cannot: the
-- reference lets an event emitted long past a deadline answer a wait that never parked. Recorded
-- here rather than deferred because `CREATE TABLE IF NOT EXISTS` means a column added later never
-- reaches an existing store, so this is the one commit where it is free. It is not evidence of
-- anything: events are disposable execution state, and the ledger is the canonical bookkeeper
-- with its own `recorded_at`.
CREATE TABLE IF NOT EXISTS events (
  name TEXT NOT NULL PRIMARY KEY,
  payload TEXT,
  emitted_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
);
CREATE TABLE IF NOT EXISTS ledger (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  workflow_run_id TEXT,
  payload TEXT NOT NULL,
  recorded_at REAL NOT NULL,
  hypothetical INTEGER NOT NULL DEFAULT 0, -- counterfactual (D); canonical view filters it
  -- WHO wrote the row (`effective.ops.Writer`). Nullable: a direct writer append (a fork's
  -- genesis/seal) has no placement, and rows written before this column existed have none
  -- either — both must keep meaning 'unknown', which the collision check reads as allow.
  writer_task TEXT,
  writer_placement TEXT
);
-- The ledger is append-only (two bookkeepers): block UPDATE/DELETE in the DB,
-- the SQLite analog of the Postgres plpgsql trigger.
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (UPDATE blocked)'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
  BEGIN SELECT RAISE(ABORT, 'ledger is append-only (DELETE blocked)'); END;
"""


class IncompatibleStore(RuntimeError):
    """An on-disk store this engine cannot open — raised at construction, never mid-run."""


def _is_pre_uuid7_store(conn: sqlite3.Connection) -> bool:
    """Whether this file was written before task ids were minted — READ-ONLY to this engine.

    Such a store has `task_id` declared `TEXT` holding composed ids (`"w-1"`) and no
    `idempotency_key` column. The schema change is **not backward compatible for WRITING**, and
    the distinction between writing and reading is the whole point of this predicate.

    **Reading one must keep working, and that is not a nicety.** A bench `task.db` is a PAID run
    kept so scoring can replay for free (`contrastbench.replay_structured` / `replay_combinator`
    reopen an existing file and drive `DurableHandler` with an exploding LLM). A bench host keeps
    hundreds of them. An earlier version of this refused at construction, which would have made
    every one of them unreplayable and their regeneration a model bill — exactly the cost the
    replay-for-free design exists to avoid. Replay touches no `tasks` INSERT, so nothing about the
    old shape actually obstructs it: SQLite is dynamically typed, the checkpoint lookup binds a
    `Key` as text through the adapter, and the old TEXT names match.

    So the refusal moved to `spawn` — the one operation that genuinely needs the column. A
    migration is still rejected: `ALTER TABLE` can add the column but CANNOT change `task_id`'s
    declared type, so `PARSE_DECLTYPES` would keep yielding text from a store this engine
    believes hands back `UUID`s — trading a loud failure for a silent type divergence, which is
    the wrong direction for the defect this whole change is about. Backfilling the key by parsing
    the old ids is worse still: the delimiter is ambiguous, which was the original bug.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    return bool(columns) and "idempotency_key" not in columns


def _deliver(conn: sqlite3.Connection, name: str, payload: Any) -> None:
    """Write an event and wake EVERY task parked on it — the one place delivery is spelled.

    Shared by the driver (`SqliteApp.emit_event`, an outside-in delivery) and the ctx
    (`SqliteTaskContext.emit_event`, a task emitting from inside its own run), so the
    first-write-wins rule and the wake are one implementation rather than two.

    **Delivery is BROADCAST, keyed by name alone**: the reference engine's model
    (`e_{queue}(event_name primary key)`, `infra/absurd/absurd.sql`). An addressed key would let a
    green SQLite test assert an isolation production lacks, one approval settling one task. The
    protection lives where both engines can express it, the NAME: that is what the generation
    coordinate on `govern:`/`approve:`/`depth-grant:` is.

    `UPDATE` with no task predicate: one emit wakes every waiter on the name. First write still
    wins — events are immutable once emitted on both engines — so a second emit on a settled name
    changes nothing, including for a task that parks on it later and finds it already set.

    **The row and the wake commit together**, so a waiter wakes exactly when its event becomes
    durable. The connection is `isolation_level=None`, which makes each `execute` its own
    transaction, so the two are wrapped here; a partial commit would strand a waiter beside an
    event it can already see, and only the source could repair it, having perhaps dropped the
    entry. `BEGIN IMMEDIATE` takes the write lock up front, so a second emitter waits its turn on
    the 5s busy timeout. Postgres commits both inside `emit_event`, which is what this conforms
    to."""
    payload_json = json.dumps(to_jsonable_python(payload))
    conn.execute("BEGIN IMMEDIATE")
    now = time.time()
    try:
        conn.execute(
            *bind(
                t"INSERT INTO events (name, payload) VALUES ({name}, {payload_json}) "
                t"ON CONFLICT(name) DO NOTHING"
            )
        )
        # A wake is owed to the waits the clock has not passed. `absurd.emit_event` deletes the
        # expired ones and wakes `timeout_at is null or timeout_at > now`; a waiter left here is
        # claimable by its own deadline, so it still ends — with the expiry it is owed.
        conn.execute(
            *bind(
                t"UPDATE tasks SET state='ready', available_at={now}, waiting_event=NULL "
                t"WHERE state='waiting' AND waiting_event={name} "
                t"  AND (available_at = 0 OR available_at > {now})"
            )
        )
        conn.execute("COMMIT")
    except BaseException:
        # `COMMIT` is inside the try because a transaction fails at both ends, and a commit that
        # raises would otherwise leave the connection inside it for good: later emits refuse to
        # begin one, and later autocommit writes join it instead of committing. `in_transaction`
        # because a statement error aborts the transaction itself, and a bare ROLLBACK then
        # raises over the top of the error that caused it.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


@dataclass(frozen=True)
class TaskSnapshot:
    state: str
    result: Any = None
    failure: str | None = None


class ClaimLost(RuntimeError):
    """A checkpoint write from a claim the task has moved past.

    A claim is stale once its task is no longer running as the attempt it claimed: another worker
    reclaimed an expired lease, which starts the next attempt, or the claim sweep failed the task
    on its last attempt. Absurd's `set_task_checkpoint_state` raises for the same run."""


class SqliteTaskContext:
    """A durable ``TaskContext`` over SQLite (step/await_event/sleep_until).

    A ``write_lock`` makes it safe for concurrent ``gather`` branches: the lock is held
    around **every touch of the connection** (the checkpoint read and write, the event
    peek, the emit) and never around the ``thunk``, so the tool/domain work runs lock-free
    and **branches' tools overlap while their connection I/O serializes** (sequential
    consistency → a *partial* order on the ledger). The structural
    ``gather:{g},{i};`` frames mean branches never write the same row, so serialized writes are
    conflict-free. ``concurrent_safe`` advertises this to the handler (``None`` lock →
    single-threaded, the default for one task).

    **Reads are guarded as well as writes.** A serialized-mode ``sqlite3`` connection is safe to
    share, but two threads driving one connection's statement cache are not: an unguarded peek
    raises ``InterfaceError('bad parameter or other API misuse')`` or returns a wrong answer, such
    as a branch seeing its sibling's approval. ``SqliteApp``'s driver methods are guarded for the
    same reason, since a workflow can ``spawn`` from inside a gather branch.

    ``claimed_as`` is the attempt the worker's claim runs as. A checkpoint write lands only while
    the task is running as that attempt, raises ``ClaimLost`` otherwise, and extends the claim's
    lease as Absurd's checkpoint write does. A ctx built outside a claim passes ``None`` and
    writes unfenced.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        task_id: UUID,
        write_lock: threading.Lock | None = None,
        *,
        claimed_as: int | None = None,
    ) -> None:
        self.conn = conn
        self.task_id = task_id
        self.claimed_as = claimed_as
        self._guard: AbstractContextManager[Any] = write_lock or nullcontext()
        self.concurrent_safe = write_lock is not None
        # Wakes this execution has already spent, in memory and per claim: a wake belongs to the
        # attempt it woke, and the claim is what ends one. See `_clock_woke_us`.
        self._spent_wakes: set[str] = set()

    @property
    def attempt(self) -> Attempt:
        """The execution of this task this ctx runs as: its claim's, or the row's outside one."""
        task_id = self.task_id
        with self._guard:
            row = self.conn.execute(
                *bind(t"SELECT attempt, max_attempts FROM tasks WHERE task_id={task_id}")
            ).fetchone()
        if row is None:
            raise LookupError(f"no task {task_id} in this store to read an attempt from")
        return Attempt(number=row[0] if self.claimed_as is None else self.claimed_as, limit=row[1])

    def step(self, name: Key, thunk: Callable[[], Any], /) -> Any:
        with self._guard:
            task_id = self.task_id
            row = self.conn.execute(
                *bind(t"SELECT state FROM checkpoints WHERE task_id={task_id} AND name={name}")
            ).fetchone()
        if row is not None:  # already committed on a prior attempt — do NOT re-run the thunk
            return json.loads(row[0]) if row[0] is not None else None
        result = thunk()  # OUTSIDE the lock: tool work overlaps across branches
        with self._guard:
            task_id, state, claimed_as = self.task_id, json.dumps(result), self.claimed_as
            written = self.conn.execute(
                *bind(
                    t"INSERT INTO checkpoints (task_id, name, state) "
                    t"SELECT {task_id}, {name}, {state} "
                    t"WHERE {claimed_as} IS NULL OR EXISTS (SELECT 1 FROM tasks "
                    t"WHERE task_id={task_id} AND attempt={claimed_as} AND state='running') "
                    t"ON CONFLICT(task_id, name) DO UPDATE SET state=excluded.state"
                )
            ).rowcount
            if written and claimed_as is not None:
                self._extend_claim()
        if written == 0:
            raise ClaimLost(
                f"task {task_id} is no longer running as attempt {claimed_as}, so its "
                f"checkpoint {name.stored()!r} was not written"
            )
        return result

    def _extend_claim(self) -> None:
        """Push the claim's lease out by a full lease while the task still runs as its attempt;
        called under the guard, right after a fenced checkpoint write landed."""
        task_id, claimed_as = self.task_id, self.claimed_as
        expires_at = time.time() + CLAIM_LEASE_SECONDS
        self.conn.execute(
            *bind(
                t"UPDATE tasks SET claim_expires_at={expires_at} WHERE task_id={task_id} "
                t"AND attempt={claimed_as} AND state='running'"
            )
        )

    def peek_step(self, name: Key, /) -> tuple[bool, Any]:
        """A checkpoint under its full name, read without counting an occurrence."""
        task_id = self.task_id
        with self._guard:
            row = self.conn.execute(
                *bind(t"SELECT state FROM checkpoints WHERE task_id={task_id} AND name={name}")
            ).fetchone()
        return (False, None) if row is None else (True, json.loads(row[0]))

    def settle(self, name: Key, value: Any, /) -> Any:
        """Write `value` under `name` unless the store holds one, then return what it holds.

        The write carries `step`'s fence and extends the claim's lease as `step` does, and one
        critical section covers the write and the read, so the value returned is the one every
        later reader sees."""
        task_id, state, claimed_as = self.task_id, json.dumps(value), self.claimed_as
        with self._guard:
            written = self.conn.execute(
                *bind(
                    t"INSERT INTO checkpoints (task_id, name, state) "
                    t"SELECT {task_id}, {name}, {state} "
                    t"WHERE {claimed_as} IS NULL OR EXISTS (SELECT 1 FROM tasks "
                    t"WHERE task_id={task_id} AND attempt={claimed_as} AND state='running') "
                    t"ON CONFLICT(task_id, name) DO NOTHING"
                )
            ).rowcount
            if written and claimed_as is not None:
                self._extend_claim()
            row = self.conn.execute(
                *bind(t"SELECT state FROM checkpoints WHERE task_id={task_id} AND name={name}")
            ).fetchone()
        if row is None:
            raise ClaimLost(
                f"task {task_id} is no longer running as attempt {claimed_as}, so "
                f"{name.stored()!r} was not settled"
            )
        return json.loads(row[0])

    def await_event(self, name: Key, /) -> Any:
        found, payload = self.peek_event(name)
        if found:
            return payload
        raise _Suspend(event=name)  # park; an external emit_event resumes us

    def await_until(self, name: Key, deadline: float, decided: Key, /) -> WaitOutcome[Any]:
        """Park until ``name`` is delivered or ``deadline`` passes, and answer once.

        | the wait                                  | answer                             |
        |-------------------------------------------|------------------------------------|
        | is settled                                | what it settled, whatever since    |
        | is open, the clock woke this task         | ``Expired``, settled now           |
        | is open, its event is stored              | ``Arrived``, settled now           |
        | is open, no event, the deadline passed    | ``Expired``, settled now           |
        | is open, no event, the deadline is ahead  | park on both                       |

        Row one is what makes this replayable: the settle is the record that survives, which a
        wait with one possible answer can do without and a wait with two cannot. **What enforces
        it is ``settle`` itself** — insert-or-ignore, returning what the store holds — so the two
        deciding rows below reach the same answer whether or not the read above happened.
        Measured by mutation: deleting the read leaves every pin green. It is here to spare a
        settled wait a write and its claim fence, and to put the table's first row where a reader
        meets it, not because the rows beneath it would otherwise decide twice.

        **Row two is the wake reason, which is what the reference decides on.** The SDK raises its
        timeout from ``wake_event`` before it ever reads the events table, so a task its deadline
        woke expires even though the event has since landed — and a wait on its FIRST call has no
        wake to read, so there the store answers whatever the clock says. ``waiting_event`` is
        that fact here: a park writes it, an emit clears it for the waiters it wakes, and a task
        the clock woke still carries it. Consumed when it fires, as the SDK consumes its own, so
        a later ask of the same name decides for itself.

        ``decided`` is the slot that record lives in, handed down by the walk because only the
        walk knows which ask is asking. Two waits on one name are one address and two questions.

        ``deadline`` is absolute, and the caller chooses it while this only reads it, so every
        attempt parks on the instant the first one did. An optional ctx capability, found by
        ``getattr`` as ``peek_event`` is.
        """
        settled, stored = self.peek_step(decided)
        if settled:
            return settled_wait(stored)
        if self._clock_woke_us(name):
            return settled_wait(self.settle(decided, ["expired"]))
        found, payload = self.peek_event(name)
        if found:
            return settled_wait(self.settle(decided, ["arrived", payload]))
        if time.time() >= deadline:
            return settled_wait(self.settle(decided, ["expired"]))
        raise _Suspend(event=name, until=deadline)

    def _clock_woke_us(self, name: Key, /) -> bool:
        """Whether this claim is the deadline's wake for a park on ``name``, consuming it.

        A park writes ``waiting_event``, ``_deliver`` clears it for the waiters it wakes, and the
        CLAIM clears it for every row it took out of anything but ``waiting``. So the
        column still naming ``name`` here means the clock got there first, which is the SDK's
        ``wake_event`` test written against the fact this engine keeps. The claim is what makes
        that an invariant rather than an inference, and the inference is what failed: a park
        ended by a sleep left its registration standing, and a wait three claims later was
        answered by a wake the first one had spent.

        CONSUMED, or one wake would answer every later ask of the name on this claim. IN MEMORY,
        because a wake does not outlive the attempt it woke: the claim clears the column for every
        row it takes out of anything but ``waiting``, so the next attempt reads no wake at all,
        which is what ``absurd.fail_run`` gives the next run. The in-memory record covers the one
        span the column cannot, from this claim's first ask to its last."""
        task_id, spelled = self.task_id, name.stored()
        # The whole test-and-set under the guard: two readers finding "not spent" would both
        # spend it. Unreachable while a branch's names carry its `gather:{g},{i};` frame, so no
        # two branches ask one name, and the guard costs a reasoning step rather than a lock.
        with self._guard:
            if spelled in self._spent_wakes:
                return False
            row = self.conn.execute(
                *bind(t"SELECT waiting_event FROM tasks WHERE task_id={task_id}")
            ).fetchone()
            if row is None or row[0] != spelled:
                return False
            self._spent_wakes.add(spelled)
        return True

    def peek_event(self, name: Key, /) -> tuple[bool, Any]:
        """Non-suspending probe (the gather-branch surface): exactly the read
        ``await_event`` performs, minus the park. Mirrors the engine's own await
        semantics: the events table is re-read on every run, no checkpoint.

        Guarded, and this is the branch-thread read that made the guard non-optional: a
        parking gather branch reaches here concurrently with its siblings (``_branch_await``
        in ``handlers.absurd``), and an unguarded read on the shared connection both raises
        ``InterfaceError`` and — worse, because it is silent — answers with another branch's
        row. See the class docstring for the measurement."""
        with self._guard:
            # By NAME alone — the read side of broadcast delivery. This narrowed by `task_id`
            # until parity, which is why a second task awaiting a name another task had answered
            # found nothing here and parked forever on the reference engine's semantics.
            row = self.conn.execute(
                *bind(t"SELECT payload FROM events WHERE name={name}")
            ).fetchone()
        if row is None:
            return False, None
        return True, json.loads(row[0]) if row[0] is not None else None

    def sleep_until(self, when: datetime, /, *, name: Key) -> None:
        # `name` is unused here, like `repark`'s: a SQLite sleep is NAMELESS — the wake time
        # lands on `tasks.available_at` and no checkpoint row is written at all, so there is
        # nothing for an identity to key. The parameter exists because the seam is shared with
        # Absurd, where the timer really does have a checkpoint of its own.
        if time.time() >= when.timestamp():
            return None
        raise _Suspend(until=when.timestamp())

    def emit_event(self, name: str, payload: Any, /) -> None:
        """Emit an event from INSIDE a running task — how a spawned child answers its parent.

        Delivery is BROADCAST, keyed by name alone across the queue, the same as the reference
        engine — see `_deliver`, which spells it once for this path and the driver's.

        The signature now matches the Absurd ctx's exactly — `(name, payload)` — because there is
        nothing left to differ about. `to_task` is gone with the addressing it served.

        A broadcast namespace keeps sibling answers apart because a done event names the task and
        placement that spawned it (`spawn_done_name`), so no spawn param addresses the parent.

        Wrapped in the write lock like every other write on this connection, so a child emitting
        while its own gather branches commit stays serialized."""
        with self._guard:
            _deliver(self.conn, name, payload)

    def repark(self, name: str, /) -> None:
        """Park so the task re-queues immediately, burning no attempt (the
        optional capability — ``TaskContext``'s docstring). Raises ``_Suspend``
        directly rather than routing through ``sleep_until``, whose already-due
        guard would turn an at-``now`` park into a silent no-op — the exact
        opposite of parking. The ``_Suspend`` arm of ``work_batch`` marks the
        task ``sleeping``/``available_at=now`` without touching ``attempt``
        (only the exception arm increments), so the next ``work_batch`` claims
        it for a fresh replay. ``name`` is unused here — SQLite sleeps are
        nameless (no wake-time checkpoint) — but kept for the cross-engine
        capability signature."""
        raise _Suspend(until=time.time())


class SqliteLedger:
    """Append-only canonical ledger over SQLite; idempotent by ``event_id``.

    Shares the task's ``write_lock`` so a concurrent ``gather`` branch's append serializes
    with the checkpoint writes on the one connection — sequential consistency. Branch
    events keep *within*-branch order; cross-branch interleaving is a race (partial order).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        workflow_run_id: str | None,
        write_lock: threading.Lock | None,
        *,
        hypothetical: bool = False,
    ) -> None:
        """``write_lock`` is REQUIRED — pass ``app.write_lock``, or ``None`` to say out loud that
        this ledger is single-threaded.

        `append` is one `INSERT` on the app's shared connection, so an unlocked ledger racing a
        gather branch loses rows on the **canonical append-only record**. Measured with three
        appender threads against a guarded `step`: ~1000 `InterfaceError`s and **~25% of appends
        silently missing** (4500 expected, 3318-3370 written); guarded, 4500 of 4500. It
        reproduces only once the payload is large enough for the insert to interleave, so a
        small-row smoke test does not see it. A required parameter makes an omitted lock a type
        error rather than a convention.
        """
        self.conn = conn
        self.workflow_run_id = workflow_run_id
        self._guard: AbstractContextManager[Any] = write_lock or nullcontext()
        self.hypothetical = hypothetical  # marks the whole lineage, parity with PG

    def append(self, row: LedgerRow, *, writer: Writer | None = None) -> None:
        with self._guard:
            event_id, kind = row.event_id, row.kind
            run_id, payload = self.workflow_run_id, json.dumps(to_jsonable_python(row))
            recorded_at, hypothetical = time.time(), int(self.hypothetical)
            task = None if writer is None else writer.task
            placement = None if writer is None else writer.placement.stored()
            inserted = self.conn.execute(
                *bind(
                    t"INSERT INTO ledger "
                    t"(event_id, kind, workflow_run_id, payload, recorded_at, hypothetical, "
                    t"writer_task, writer_placement) "
                    t"VALUES ({event_id}, {kind}, {run_id}, {payload}, {recorded_at}, "
                    t"{hypothetical}, {task}, {placement}) ON CONFLICT(event_id) DO NOTHING "
                    t"RETURNING event_id"
                )
            ).fetchone()
            if inserted is not None:
                return
            # The append LOST a conflict. `DO NOTHING` returns no row, so read back the winner —
            # in the SAME critical section, the rule `SqliteApp.spawn` states and for the same
            # reason: split the two statements and the row you read is not the one you lost to.
            #
            # This must NOT filter `hypothetical`, and filtering it would be the defect rather
            # than the fix. That predicate exists so a fork's row cannot join the canonical FOLD;
            # this is not a fold — it is a lookup keyed on the UNIQUE column, asking one question:
            # who holds this exact id? `ForkLedger` rescopes to `hyp:{child};{id}` BEFORE the
            # store, so a hypothetical row and its canonical counterpart have different ids and
            # cannot answer for each other. Adding `AND NOT hypothetical` would make the read miss
            # a hypothetical holder, return `None`, and ALLOW — silently restoring the lost row
            # inside forks, which is the case that made `sealed => valid marginal` false.
            held = self.conn.execute(
                *bind(
                    # lint: ledger-read-not-canonical: a lookup on the UNIQUE id, never a fold
                    t"SELECT writer_task, writer_placement FROM ledger WHERE event_id={event_id}"
                )
            ).fetchone()
        refuse_placed_writer_collision(row.event_id, writer, held)


type TaskFn = Callable[[dict[str, Any], SqliteTaskContext], Any]


class SqliteApp:
    """The 0↔1 durable engine: register tasks, spawn, and drain the work-loop.

    Single connection, single writer (``isolation_level=None`` autocommit so each
    checkpoint is durable the moment its step commits — the property crash-replay
    needs). Task ids are opaque time-ordered ``UUID``\\ s (uuid7); replay-safety
    across a spawn comes from its ``idempotency_key``, which the spawning handler names,
    never from the id (see ``spawn``). Pull-only, like Absurd: ``work_batch`` claims one
    ready task.

    ``check_same_thread=False`` + a ``write_lock`` let concurrent ``gather`` branches
    share the one connection safely: the lock (held around every connection touch, read
    as well as write — see ``SqliteTaskContext``) serializes the I/O while tool work
    overlaps. ``check_same_thread=False`` buys the right to *pass* the connection between
    threads; it buys nothing about driving it from two at once, which is the lock's job.
    """

    def __init__(self, path: str = ":memory:", *, require_wal: bool = True) -> None:
        """Open (or create) a store. ``require_wal=False`` accepts a rollback journal.

        The flag exists because raising is a NARROWING and a narrowing needs an exit. A new file
        store is put into WAL and, by default, a store that *refuses* WAL is an error rather than
        a silent downgrade: `PRAGMA journal_mode` reports the mode it ended up in instead of
        failing, so without the check a filesystem with no shared-memory support (NFS and some
        container mounts are the usual suspects) yields a store that looks ordinary while quietly
        lacking the concurrent-reader property the observers rely on.

        This engine aims at laptops, where refusing to construct on such a filesystem would strand
        a working case, so ``require_wal=False`` opens it with a rollback journal and the mode left
        visible in ``journal_mode``. It is not the
        default, because losing the concurrent-reader property silently is the failure the check
        surfaces.
        """
        self.conn = connect(path)
        self.path = path
        # A pre-uuid7 store opens READ-ONLY rather than being refused: a banked bench `task.db` is
        # a paid run kept for free replay, and replay writes no task row. The index is the one
        # piece of DDL that names the missing column, so it is applied conditionally; everything
        # in `_SCHEMA` is `IF NOT EXISTS` and inert on an existing file.
        self.legacy_store = _is_pre_uuid7_store(self.conn)
        # A file store gets WAL at CREATION and never on a later open — journal mode is a
        # persistent property of the file, so converting one we are merely visiting would rewrite
        # a banked corpus nobody asked us to touch. `:memory:` has no journal to set (the pragma
        # answers `memory`). `enable_wal` carries the reasoning and the escape hatch.
        # Raise rather than record a silent downgrade: `PRAGMA journal_mode` REPORTS the mode it
        # ended up in instead of failing, so a store that refuses WAL (a filesystem without
        # shared-memory support is the usual reason) would otherwise come up looking ordinary
        # while quietly lacking the concurrent-reader property this engine now assumes.
        if (
            path != ":memory:"
            and _is_fresh_store(self.conn)
            and (mode := enable_wal(self.conn)) != "wal"
            and require_wal
        ):
            # Close before raising: a constructor that fails still owns the connection it opened,
            # and a caller who never got an object has nothing to call `close()` on. The leak
            # accounting caught this one on its first run, which is the gate paying for itself.
            self.conn.close()
            raise RuntimeError(
                f"{path!r} refused WAL and stayed in {mode!r} journal mode. The read-only "
                "observers open a second connection to this file and a rollback journal "
                "locks them out; check the filesystem supports shared memory, or pass "
                "`require_wal=False` to accept a rollback journal on this store."
            )
        self.journal_mode: str = self.conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.conn.executescript(_SCHEMA)
        if not self.legacy_store:
            self.conn.executescript(_IDEMPOTENCY_INDEX)
        self.write_lock = threading.Lock()
        self._tasks: dict[str, TaskFn] = {}

    def register_task(self, name: str) -> Callable[[TaskFn], TaskFn]:
        def deco(fn: TaskFn) -> TaskFn:
            self._tasks[name] = fn
            return fn

        return deco

    def spawn(
        self,
        name: str,
        params: dict[str, Any],
        max_attempts: int | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> UUID:
        """Enqueue a task. With an `idempotency_key`, enqueue it AT MOST ONCE.

        `max_attempts` is how many executions the task may have; `None` takes the engine's
        default of 3.

        The key is its own column under `UNIQUE(name, idempotency_key)`, and the insert is
        `DO NOTHING`, so a caller that spawns the same child twice gets the same task id and one
        task. This is what makes a spawn safe from INSIDE a durable run
        (`effective.fork.spawn_fork`, the subagent path): the spawn is wrapped in a `Step`, so a
        replay normally returns the checkpointed child id without re-spawning — but a crash in
        the window *between* the enqueue and that checkpoint's commit would otherwise enqueue a
        second child on the retry. Absurd's `spawn` takes the same argument for the same reason;
        this is the 0<->1 counterpart, so a cross-engine spawn story does not have a hole on one
        side.

        **The id is minted, never composed.** The identity and the key are separate columns, so
        two spawns share a task only by sending one key, and no spelling of a name or a key can
        reach the id.

        Returns the id of the task that is now enqueued under this key, which on a conflict is
        the EXISTING one — the caller must be able to await the task it deduplicated against,
        not a uuid that was never inserted."""
        if self.legacy_store:
            raise IncompatibleStore(
                f"{self.path!r} was written by an older engine — task ids were composed strings "
                "(`name-seq`) and at-most-once rode the id instead of its own "
                "`idempotency_key` column. It opens READ-ONLY (replaying a banked run works), "
                "but it cannot be spawned into: the two shapes cannot be migrated between (see "
                "`_is_pre_uuid7_store`). Spawn into a fresh file."
            )
        params_json = json.dumps(params)
        task_id = uuid7()
        limit = DEFAULT_MAX_ATTEMPTS if max_attempts is None else max_attempts
        # Under the lock, and the INSERT and its read-back together rather than separately: a
        # workflow can spawn from inside a `gather` BRANCH
        # (`effective.interpreters.tools.spawn_tool`, wired to this very connection), and a branch
        # thunk runs off-lock by design. Unguarded, this method reproduced the whole family:
        # `InterfaceError`, a `None` task id where the signature says `UUID`, lost task rows, and
        # two DIFFERENT `idempotency_key`s handed the same `task_id`, which is precisely the
        # at-most-once defect the docstring above says was structurally removed. Splitting the two
        # statements would leave that last one: the winner must be read back in the same critical
        # section that lost the race.
        spawned_at = time.time()
        with self.write_lock:
            inserted = self.conn.execute(
                *bind(
                    t"INSERT INTO tasks "
                    t"(task_id, name, params, state, attempt, max_attempts, available_at, "
                    t"idempotency_key) "
                    t"VALUES ({task_id}, {name}, {params_json}, 'ready', 1, {limit}, "
                    t"{spawned_at}, {idempotency_key}) "
                    t"ON CONFLICT(name, idempotency_key) DO NOTHING "
                    t"RETURNING task_id"
                )
            ).fetchone()
            if inserted is not None:
                return inserted[0]
            # The key was already spawned. `DO NOTHING` returns no row, so read back the winner
            # rather than returning the uuid we minted and discarded.
            existing = self.conn.execute(
                *bind(
                    t"SELECT task_id FROM tasks "
                    t"WHERE name={name} AND idempotency_key={idempotency_key}"
                )
            ).fetchone()
        return existing[0]

    def emit_event(self, name: str, payload: Any) -> None:
        """Deliver an event and wake EVERY task parked on it (the resume side of await).

        Events are immutable once emitted — first write wins, mirroring the
        Absurd engine's ``emit_event`` (its SQL states the rule verbatim). The
        engine replays an await by RE-READING the events table, so a mutable
        payload would let a re-emission rewrite history: a step committed
        against the first payload while a later replay re-binds the second.

        **`(name, payload)`, matching the ctx path and the Absurd driver.** Delivery is broadcast,
        so an emit reaches every waiter on the name and there is no recipient to select.

        Guarded like every other touch of this connection. This is the OUTSIDE-IN delivery path,
        so the racing thread is the caller's (a HITL approve arriving on a web thread while a
        gather runs), and `_deliver` is two statements (the insert and the wake) that a reader
        must not see half of."""
        with self.write_lock:
            _deliver(self.conn, name, payload)

    def _claim(self) -> tuple[UUID, str, str, int, int] | None:
        # The whole claim under one lock, not three: reap, select and mark-running are a single
        # decision, and two workers interleaving them would both claim the row the other just
        # marked. The lock is released before `work_batch` runs the body — see there.
        with self.write_lock:
            return self._claim_locked()

    def _claim_locked(self) -> tuple[UUID, str, str, int, int] | None:
        now = time.time()
        # A 'running' task whose lease has expired lost its worker mid-execution, and that
        # execution counts. On its last attempt the task fails here, as Absurd's claim sweep
        # fails it through `fail_run`; otherwise the reclaim below starts the next attempt.
        self.conn.execute(
            *bind(
                t"UPDATE tasks SET state='failed', claimed_by=NULL, "
                t"failure='worker repeatedly died; reclaim exceeded max_attempts' "
                t"WHERE state='running' AND claim_expires_at IS NOT NULL "
                t"  AND claim_expires_at <= {now} AND attempt >= max_attempts"
            )
        )
        # Claimable: a ready/sleeping task whose time has come, OR a 'running' task whose
        # lease has expired — the worker that held it died mid-run (no `except` fired, so it
        # never re-queued). Reclaiming it re-runs from scratch, replaying committed
        # checkpoints: worker-death + fresh-worker-resume, the durability story (the
        # Cloudflare-DO eviction case). The lease bounds how long a dead worker holds a task.
        # CLAIM ONLY WHAT THIS WORKER CAN RUN. A task whose name is not registered here is not
        # this worker's to touch: claiming it and then failing on the registry lookup conflates two
        # different things — "the task BODY raised" (a workflow failure; retry, burn an attempt)
        # and "I do not know this task name" (an operator/deployment fact about THIS process).
        #
        # A drain triggered from a process without the registry (a minimal web host is one) would
        # otherwise claim an ANSWERED run, burn an attempt per call, and fail it permanently on the
        # third, ledger empty: a human clicking approve destroys the run, and every assertion the
        # caller can make stays green.
        #
        # `json_each` rather than a generated `IN (?,?,…)`: one hole, arity-free — the shape
        # `effective.sql` steers a container toward, since a placeholder binds one scalar.
        names_json = json.dumps(sorted(self._tasks))
        row = self.conn.execute(
            *bind(
                t"SELECT task_id FROM tasks "
                t"WHERE name IN (SELECT value FROM json_each({names_json})) "
                t"  AND ((state IN ('ready', 'sleeping') AND available_at <= {now}) "
                t"   OR (state = 'waiting' AND available_at > 0 AND available_at <= {now}) "
                t"   OR (state = 'running' AND claim_expires_at IS NOT NULL "
                t"       AND claim_expires_at <= {now})) "
                t"ORDER BY available_at, task_id LIMIT 1"
            )
        ).fetchone()
        if row is None:
            return None
        # A reclaimed running task starts its next attempt; a ready or sleeping one continues the
        # attempt it has. The claim returns the attempt this execution runs as.
        #
        # `waiting_event` survives ONLY a claim that took a `waiting` row, which is the one the
        # deadline disjunct above makes — so after any claim the column naming an event means the
        # CLOCK got here first, and `SqliteTaskContext._clock_woke_us` can read it as that.
        #
        # A wake belongs to the attempt that was woken, and only a `waiting` row carries one into
        # this claim: the SELECT's waiting disjunct is a deadline that has passed, so the clock is
        # what got here. Every other state clears, the reclaim included — which is the reference's
        # own rule, not a choice: `absurd.claim_task` takes only `pending` and `sleeping` runs, an
        # expired lease goes through `absurd.fail_run`, and that inserts the next attempt's run
        # with `wake_event` NULL.
        #
        # This is the only place that has to know: a READER asking "is this parked" asks for
        # `state='waiting'` instead, so a registration a park left behind answers nobody.
        expires_at, claimed = now + CLAIM_LEASE_SECONDS, row[0]
        return self.conn.execute(
            *bind(
                t"UPDATE tasks SET state='running', claimed_by='w', "
                t"claim_expires_at={expires_at}, attempt = attempt + (state='running'), "
                t"waiting_event = CASE WHEN state='waiting' THEN waiting_event ELSE NULL END "
                t"WHERE task_id={claimed} "
                t"RETURNING task_id, name, params, attempt, max_attempts"
            )
        ).fetchone()

    def _fail(
        self,
        task_id: UUID,
        claimed_as: int,
        params: dict[str, Any],
        leaf: BaseException,
        raised: BaseException,
    ) -> None:
        """Fail a task for good, reported as `leaf`, and answer the parent that spawned it.

        The row is failed only while the claim still runs it. The parent is answered while the
        task is still at this claim's attempt, which includes a task the claim sweep failed under
        it, as Absurd's worker answers for the task's latest run."""
        failure = _failure_text(noted(leaf, raised))
        with self.write_lock:
            self.conn.execute(
                *bind(
                    t"UPDATE tasks SET state='failed', "
                    t"failure={failure}, claimed_by=NULL WHERE task_id={task_id} "
                    t"AND attempt={claimed_as} AND state='running'"
                )
            )
            latest = self.conn.execute(
                *bind(t"SELECT 1 FROM tasks WHERE task_id={task_id} AND attempt={claimed_as}")
            ).fetchone()
            if latest is not None and DONE_EVENT_PARAM in params:
                _deliver(self.conn, params[DONE_EVENT_PARAM], failure_answer(leaf, raised))

    def work_batch(self) -> bool:
        """Claim and run one ready task. Returns False when nothing is claimable.

        Every write that ends the claim holds only while the task is still running as the
        attempt this claim took, so a claim another worker reclaimed changes nothing in the task's
        row or checkpoints."""
        claimed = self._claim()
        if claimed is None:
            return False
        task_id, name, params_json, attempt, max_attempts = claimed
        params = json.loads(params_json)
        ctx = SqliteTaskContext(self.conn, task_id, self.write_lock, claimed_as=attempt)
        try:
            result = self._tasks[name](json.loads(params_json), ctx)
        except _Suspend as s:
            if s.event is not None:
                # One statement reads the event and records the park, so an emit from another
                # connection lands either before it (the task is ready) or after it (the emit's
                # wake finds the task waiting).
                event = s.event
                # 0 is "no deadline" for a `waiting` row, which is what `_claim_locked` and
                # `read_sqlite_parked` test for. `woken` is the other reading of this column:
                # the instant the task became claimable, written when the event is already here.
                deadline, woken = s.until or 0.0, time.time()
                with self.write_lock:
                    self.conn.execute(
                        *bind(
                            t"UPDATE tasks SET "
                            t"state=CASE WHEN EXISTS (SELECT 1 FROM events WHERE name={event}) "
                            t"THEN 'ready' ELSE 'waiting' END, "
                            t"waiting_event=CASE WHEN EXISTS "
                            t"(SELECT 1 FROM events WHERE name={event}) "
                            t"THEN NULL ELSE {event} END, "
                            t"available_at=CASE WHEN EXISTS "
                            t"(SELECT 1 FROM events WHERE name={event}) "
                            t"THEN {woken} ELSE {deadline} END, "
                            t"claimed_by=NULL WHERE task_id={task_id} "
                            t"AND attempt={attempt} AND state='running'"
                        )
                    )
            else:
                until = s.until
                with self.write_lock:
                    self.conn.execute(
                        *bind(
                            t"UPDATE tasks SET state='sleeping', available_at={until}, "
                            t"claimed_by=NULL WHERE task_id={task_id} "
                            t"AND attempt={attempt} AND state='running'"
                        )
                    )
        except Exception as exc:
            # A refused write means the task moved past this claim, and whoever holds it ends it,
            # so it answers nobody. A real failure beside one in a group still ends the claim.
            match exc:
                case ClaimLost():
                    return True
                case ExceptionGroup():
                    _, real = exc.split(ClaimLost)
                    if real is None:
                        return True
                    exc = real
            match failing_leaf(exc, Attempt(number=attempt, limit=max_attempts)):
                case None:
                    # Re-queue as the next attempt, as `fail_run` does: committed checkpoints
                    # persist, so the retry resumes mid-task.
                    following, requeued = attempt + 1, time.time()
                    with self.write_lock:
                        self.conn.execute(
                            *bind(
                                t"UPDATE tasks SET state='ready', attempt={following}, "
                                t"available_at={requeued}, claimed_by=NULL "
                                t"WHERE task_id={task_id} "
                                t"AND attempt={attempt} AND state='running'"
                            )
                        )
                case leaf:
                    self._fail(task_id, attempt, params, leaf, exc)
        else:
            # `to_jsonable_python`, not a bare `json.dumps`: a task result is Pydantic-shaped as
            # often as a checkpoint is (a `ForkOutcome`, a model a workflow returns), and the
            # substrate's serde policy is one policy — the same round-trip the ledger append and
            # the Absurd checkpoints use. A bare dumps made a returned model an engine-specific
            # TypeError, which is exactly the kind of 0<->1/0<->N divergence the conformance
            # suite exists to keep out.
            result_json = json.dumps(to_jsonable_python(result))
            with self.write_lock:
                self.conn.execute(
                    *bind(
                        t"UPDATE tasks SET state='completed', result={result_json}, "
                        t"claimed_by=NULL WHERE task_id={task_id} "
                        t"AND attempt={attempt} AND state='running'"
                    )
                )
        return True

    def fetch_task_result(self, task_id: UUID) -> TaskSnapshot | None:
        with self.write_lock:
            row = self.conn.execute(
                *bind(t"SELECT state, result, failure FROM tasks WHERE task_id={task_id}")
            ).fetchone()
        if row is None:
            return None
        state, result, failure = row
        return TaskSnapshot(state, json.loads(result) if result is not None else None, failure)

    def run_until_result(self, task_id: UUID, max_batches: int = 64) -> TaskSnapshot | None:
        """Drain until the task is terminal or nothing is claimable (e.g. parked)."""
        for _ in range(max_batches):
            snap = self.fetch_task_result(task_id)
            if snap is not None and snap.state in ("completed", "failed", "cancelled"):
                return snap
            if not self.work_batch():
                break
        return self.fetch_task_result(task_id)

    def close(self) -> None:
        self.conn.close()
