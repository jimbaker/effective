"""SQLite engine paths that don't fit the cross-backend conformance shape.

Engine-specific durable behavior (the terminal-failure branch and the durable timer),
exercised infra-free against `effective.sqlite`. The
cross-backend *properties* live in `test_conformance.py`; these pin the SQLite engine's
own control flow.
"""

import json
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

import effective
from effective.api import await_until as api_await_until
from effective.api import sleep_until
from effective.domain import CallTool, DomainOp
from effective.handlers.absurd import DurableHandler
from effective.keys import Key, compose_key
from effective.ops import Arrived, Expired
from effective.parked import read_sqlite_parked
from effective.sqlite import (
    ClaimLost,
    IncompatibleStore,
    SqliteApp,
    SqliteTaskContext,
    _Suspend,
    enable_wal,
)


class _Tool:
    """Minimal domain: every CallTool returns 1."""

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        return 1


def test_task_fails_after_max_attempts():
    """A task that keeps raising lands in 'failed' with the error recorded — the
    work_batch terminal branch (no retries left)."""
    app = SqliteApp()

    @app.register_task("boom")
    def task(params, ctx):
        raise ValueError("kaboom")

    snap = app.run_until_result(app.spawn("boom", {"run_id": "r"}, max_attempts=1))
    assert snap is not None
    assert snap.state == "failed"
    assert "ValueError" in (snap.failure or "")
    app.close()


def test_expired_lease_running_task_is_reclaimed():
    """Worker-death + fresh-worker-resume (the shape of a Cloudflare Durable Object eviction):
    a task left in 'running' with an expired lease is reclaimed, re-run from its committed
    checkpoints, and completed, with the reclaim counted as a fresh attempt. Without the reclaim
    the lease columns are dead state and such a task wedges forever."""
    app = SqliteApp()

    @app.register_task("t")
    def task(params, ctx):
        return "ok"

    tid = app.spawn("t", {"run_id": "r"})
    spawned = app.conn.execute("SELECT attempt FROM tasks WHERE task_id=?", (tid,)).fetchone()[0]
    # simulate a worker that claimed then died mid-run: 'running' with a lease already expired.
    app.conn.execute(
        "UPDATE tasks SET state='running', claimed_by='dead', claim_expires_at=? WHERE task_id=?",
        (time.time() - 100, tid),
    )
    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result == "ok"
    attempt = app.conn.execute("SELECT attempt FROM tasks WHERE task_id=?", (tid,)).fetchone()[0]
    assert attempt == spawned + 1  # the lease-steal counted as a fresh attempt
    app.close()


def test_poison_task_reclaim_is_capped_at_max_attempts():
    """A worker that repeatedly dies with no exception reaching work_batch (process kill / DO
    eviction) leaves the task 'running'; the reclaim must NOT loop forever. Once attempts are
    exhausted the reclaim path fails it terminally. (Fresh-eyes review #4 reproduced an uncapped
    reclaim — attempt climbing past max_attempts, never terminal.)"""
    app = SqliteApp()

    @app.register_task("t")
    def task(params, ctx):
        return "ok"

    tid = app.spawn("t", {"run_id": "r"}, max_attempts=2)
    # simulate repeated worker-death: each round leave it 'running' with an expired lease.
    final = None
    for _ in range(6):
        app.conn.execute(
            "UPDATE tasks SET state='running', claim_expires_at=? WHERE task_id=?",
            (time.time() - 100, tid),
        )
        app._claim()
        final = app.conn.execute(
            "SELECT state, attempt FROM tasks WHERE task_id=?", (tid,)
        ).fetchone()
        row = final
        if row[0] == "failed":
            break
    assert final is not None
    assert final[0] == "failed"  # terminal, not looping
    assert final[1] <= 3  # capped near max_attempts, not climbing to 6
    app.close()


def test_poison_body_runs_max_attempts_regardless_of_death_mode():
    """`max_attempts` must mean the same number of body executions whether the worker fails
    cleanly (exception reaches work_batch) or dies silently (left 'running' + expired). Fixes
    an off-by-one where silent death ran the body one extra time. (Fresh-eyes review #5.)"""

    def clean_failure_execs(max_attempts: int) -> int:
        app = SqliteApp()
        n = [0]

        @app.register_task("t")
        def task(params, ctx):
            n[0] += 1
            raise RuntimeError("boom")

        app.run_until_result(app.spawn("t", {"run_id": "r"}, max_attempts=max_attempts))
        app.close()
        return n[0]

    def silent_death_execs(max_attempts: int) -> int:
        app = SqliteApp()
        n = [0]

        @app.register_task("t")
        def task(params, ctx):
            n[0] += 1
            raise RuntimeError("boom")

        tid = app.spawn("t", {"run_id": "r"}, max_attempts=max_attempts)
        for _ in range(20):
            snap = app.fetch_task_result(tid)
            if snap is not None and snap.state in ("failed", "completed"):
                break
            row = app._claim()  # claim, "run" the body once, then leave 'running' (silent death)
            if row is None:
                break
            n[0] += 1  # the body executed once (then the worker died with no exception)
            app.conn.execute(
                "UPDATE tasks SET state='running', claim_expires_at=? WHERE task_id=?",
                (time.time() - 100, tid),
            )
        app.close()
        return n[0]

    for m in (1, 2, 3):
        assert clean_failure_execs(m) == m
        assert silent_death_execs(m) == m  # symmetric: no off-by-one extra run


def test_valid_lease_running_task_is_not_stolen():
    """A task whose lease is still in the future must NOT be reclaimed — only an *expired*
    lease is stealable."""
    app = SqliteApp()

    @app.register_task("t")
    def task(params, ctx):
        return "ok"

    tid = app.spawn("t", {"run_id": "r"})
    app.conn.execute(
        "UPDATE tasks SET state='running', claim_expires_at=? WHERE task_id=?",
        (time.time() + 100, tid),
    )
    assert app.work_batch() is False  # nothing claimable
    snap = app.fetch_task_result(tid)
    assert snap is not None
    assert snap.state == "running"  # still held
    app.close()


def test_sleep_until_past_returns_immediately():
    """A sleep whose deadline already passed is a no-op — the worker runs to completion."""
    app = SqliteApp()

    @app.register_task("sp")
    def task(params, ctx):
        def wf():
            yield from sleep_until(datetime(2000, 1, 1, tzinfo=UTC))
            return "done"

        return DurableHandler(ctx, _Tool()).run(wf)

    snap = app.run_until_result(app.spawn("sp", {"run_id": "r"}))
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result == "done"
    app.close()


def test_sleep_until_future_parks_then_resumes():
    """A future deadline parks the task ('sleeping', worker released); once the deadline
    passes a later work_batch claims and completes it — durable timer across the gap."""
    app = SqliteApp()
    when = datetime.fromtimestamp(time.time() + 0.15, tz=UTC)

    @app.register_task("sf")
    def task(params, ctx):
        def wf():
            yield from sleep_until(when)
            return "woke"

        return DurableHandler(ctx, _Tool()).run(wf)

    tid = app.spawn("sf", {"run_id": "r"})
    parked = app.run_until_result(tid)  # stops: the sleeping task isn't claimable yet
    assert parked is not None
    assert parked.state == "sleeping"

    time.sleep(0.2)
    done = app.run_until_result(tid)
    assert done is not None
    assert done.state == "completed"
    assert done.result == "woke"
    app.close()


class _Statements:
    """The worker's statements counted from the missed read, and whether the emit has fired."""

    after_miss: int | None = None
    fired: bool = False


def _emit_before_statement(where: str, sqlite_app: Any, k: int) -> tuple[bool, tuple[Any, ...]]:
    """Park a task whose read missed, emitting from a second connection just before the worker's
    k-th statement after the miss. Returns whether the emit fired inside the park, and the row."""
    app, bridge = sqlite_app(where), sqlite_app(where)
    seen = _Statements()

    def before(_sql: str) -> None:
        if (n := seen.after_miss) is None or seen.fired:
            return
        if n == k:
            seen.fired = True
            bridge.emit_event("arrived", {"id": "A"})
        seen.after_miss = n + 1

    @app.register_task("consumer")
    def consumer(params: Any, ctx: Any) -> Any:
        try:
            return ctx.await_event(Key.parse("arrived"))
        except _Suspend:
            seen.after_miss = 0
            raise

    task = app.spawn("consumer", {})
    app.conn.set_trace_callback(before)
    app.run_until_result(task)
    app.conn.set_trace_callback(None)
    if not seen.fired:
        bridge.emit_event("arrived", {"id": "A"})
    app.run_until_result(task)  # a task the wake missed stays unfinished through this drain
    row = app.conn.execute(
        "SELECT state, result, waiting_event FROM tasks WHERE task_id=?", (task,)
    ).fetchone()
    return seen.fired, tuple(row)


def test_an_event_emitted_at_any_statement_of_the_park_wakes_the_task(tmp_path, sqlite_app):
    """An emit from another connection, as an outside bridge would send it, lands before each
    statement the worker runs after `await_event` reads nothing, until it lands after them all.
    Every task finishes with the event, and no finished row still names what it waited on."""
    finished = ("completed", json.dumps({"id": "A"}), None)
    wrong = {}
    for k in range(64):
        fired, row = _emit_before_statement(str(tmp_path / f"gap{k}.db"), sqlite_app, k)
        if row != finished:
            wrong[k] = row
        if not fired:
            break
    else:
        pytest.fail("the park never ran out of statements")
    assert wrong == {}


# --- the task id is minted, never composed ----------------------------------------------------


def test_an_idempotency_key_does_not_collide_with_the_spawn_that_precedes_it():
    """A task id composed from `name` and either a counter or the `idempotency_key` is two shapes
    on one delimiter with a free leading hole. `spawn("w", {"n": 1})` would yield `w-1` from the
    counter and `spawn("w", {"n": 2}, idempotency_key="1")` `w-1` from the key; `ON CONFLICT DO
    NOTHING` swallows the second insert and the caller gets the FIRST task's id back with the
    first task's params on the row. Nothing raises, and one spawn vanishes on live paths
    (`fork.spawn_fork` passes `idempotency_key=str(child)`, and task names carry `-`).

    Two independent spawns, two tasks, each with its own params. The id is opaque, so the
    assertion is about DISTINCTNESS rather than about any spelling."""
    app = SqliteApp()

    @app.register_task("w")
    def _w(params: dict[str, Any], ctx: Any) -> Any:
        return params["n"]

    plain = app.spawn("w", {"n": 1})
    keyed = app.spawn("w", {"n": 2}, idempotency_key="1")

    assert plain != keyed
    rows = dict(app.conn.execute("SELECT task_id, params FROM tasks").fetchall())
    assert set(rows) == {plain, keyed}
    assert json.loads(rows[plain])["n"] == 1
    assert json.loads(rows[keyed])["n"] == 2  # NOT the first task's params
    app.close()


def test_the_same_idempotency_key_still_deduplicates_to_the_one_task():
    """The other half: at-most-once is the property the key exists for, and moving it off the id
    onto `UNIQUE(name, idempotency_key)` must not lose it. The second spawn returns the FIRST
    task's id — the caller has to be able to await the task it deduplicated against, not the
    uuid `spawn` minted and discarded."""
    app = SqliteApp()

    @app.register_task("w")
    def _w(params: dict[str, Any], ctx: Any) -> Any:
        return params["n"]

    first = app.spawn("w", {"n": 1}, idempotency_key="k")
    second = app.spawn("w", {"n": 2}, idempotency_key="k")

    assert first == second
    (count,) = app.conn.execute("SELECT count(*) FROM tasks").fetchone()
    assert count == 1
    snap = app.run_until_result(first)
    assert snap is not None
    assert snap.result == 1  # the FIRST spawn's params won
    app.close()


def test_two_keyless_spawns_of_one_name_are_two_tasks():
    """NULLs are distinct in a SQLite unique index, and that is the wanted reading: no
    idempotency key means no at-most-once claim, so `UNIQUE(name, idempotency_key)` must not
    quietly deduplicate keyless spawns of the same task name."""
    app = SqliteApp()

    @app.register_task("w")
    def _w(params: dict[str, Any], ctx: Any) -> Any:
        return params["n"]

    ids = {app.spawn("w", {"n": i}) for i in range(3)}
    assert len(ids) == 3
    (count,) = app.conn.execute("SELECT count(*) FROM tasks").fetchone()
    assert count == 3
    app.close()


def test_a_task_id_is_a_time_ordered_uuid_object_on_the_way_out_and_back():
    """The id is a `UUID` OBJECT at every seam, and the driver, not the caller, carries the
    conversion. `PARSE_DECLTYPES` plus the `UUID`-declared column is what makes the read side
    hydrate one; without it the column comes back as text and the engine we control would hand
    out the weaker type.

    uuid7 is time-ordered, so spawn order is recoverable from the ids themselves — which is
    also the closest thing this engine has to a park timestamp, a column it does not have."""
    app = SqliteApp()

    @app.register_task("w")
    def _w(params: dict[str, Any], ctx: Any) -> Any:
        return "ok"

    minted = [app.spawn("w", {"n": i}) for i in range(5)]
    assert all(isinstance(tid, UUID) and tid.version == 7 for tid in minted)
    assert minted == sorted(minted)  # time-ordered, so spawn order survives in the id

    read_back = [r[0] for r in app.conn.execute("SELECT task_id FROM tasks ORDER BY rowid")]
    assert read_back == minted  # a UUID out of the driver, not the canonical text
    app.close()


def _legacy_store(path) -> None:
    """A `tasks`/`checkpoints` pair exactly as the pre-uuid7 engine wrote them."""
    old = sqlite3.connect(str(path))
    old.executescript(
        "CREATE TABLE tasks (task_id TEXT PRIMARY KEY, name TEXT NOT NULL, params TEXT NOT NULL,"
        " state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,"
        " max_attempts INTEGER NOT NULL DEFAULT 3, available_at REAL NOT NULL DEFAULT 0,"
        " waiting_event TEXT, claimed_by TEXT, claim_expires_at REAL, result TEXT, failure TEXT);"
        " CREATE TABLE checkpoints (task_id TEXT NOT NULL, name TEXT NOT NULL, state TEXT,"
        " PRIMARY KEY (task_id, name));"
        " INSERT INTO tasks VALUES ('trial-1','trial','{}','completed',0,3,0,NULL,NULL,NULL,"
        "'\"ok\"',NULL);"
        " INSERT INTO checkpoints VALUES ('trial-1','react:turn,0','\"turn-0\"');"
    )
    old.commit()
    old.close()


def test_a_store_from_the_previous_engine_still_REPLAYS(tmp_path):
    """A banked store opens read-only and replays — the capability the refusal nearly cost.

    A bench `task.db` is a PAID run kept so scoring can re-run for free
    (`contrastbench.replay_structured` reopens the file and drives `DurableHandler` with an
    exploding LLM). The first version of the compat check refused at
    `SqliteApp()` construction, which would have made every one unreplayable and their
    regeneration a model bill — the exact cost replay-for-free exists to avoid.

    Nothing about the old shape actually obstructs a replay: it writes no `tasks` row, SQLite is
    dynamically typed, and the checkpoint lookup binds a `Key` as text through the adapter, which
    matches the old TEXT names. So the check became a PREDICATE and the refusal moved to the one
    operation that genuinely needs the new column."""
    path = tmp_path / "banked.db"
    _legacy_store(path)

    app = SqliteApp(str(path))
    assert app.legacy_store  # recognized, not rejected

    (task_id,) = [r[0] for r in app.conn.execute("SELECT task_id FROM tasks")]
    ctx = SqliteTaskContext(app.conn, task_id, app.write_lock)

    def _must_not_run():
        raise AssertionError("the thunk re-ran — this was supposed to replay from the checkpoint")

    assert ctx.step(Key.parse("react:turn,0"), _must_not_run) == "turn-0"
    app.close()


def test_spawning_into_a_pre_uuid7_store_is_refused_by_name(tmp_path):
    """Read-only means read-only: the incompatibility is about WRITING, and that is where it
    fires — with an instruction, not with `no such column: idempotency_key` three layers down.

    The rejected alternative stays rejected: `ALTER TABLE` could add the column, but `task_id`
    would stay declared `TEXT` and keep hydrating as text out of a store this engine believes
    yields `UUID`s. A loud refusal beats a silent type divergence — the same judgement the id
    change itself rests on."""
    path = tmp_path / "banked.db"
    _legacy_store(path)
    app = SqliteApp(str(path))

    with pytest.raises(IncompatibleStore) as caught:
        app.spawn("w", {"n": 1})
    assert "older engine" in str(caught.value)
    assert "READ-ONLY" in str(caught.value)
    assert "fresh file" in str(caught.value)  # the message says what to DO
    app.close()


# --- journal mode: durability of the FILE, not configuration of the connection -------------


def _store_from_the_rollback_era(path):
    """A store this engine WROTE, then put back into the rollback journal — i.e. a file banked
    before WAL was set at creation, which is what the 325-store corpus actually is.

    Built with `SqliteApp` rather than by running `_SCHEMA` by hand, and the difference is not
    pedantic: `_SCHEMA` alone omits the idempotency index, so the first `SqliteApp` open creates
    it and the file changes for a reason that has nothing to do with journal mode. The
    hand-built version of this fixture failed for exactly that, which is the fixture lying about
    what it models rather than the engine misbehaving.
    """
    app = SqliteApp(str(path))
    assert app.journal_mode == "wal"
    app.conn.execute("PRAGMA journal_mode=DELETE")
    app.close()
    return path


def test_a_file_store_is_created_in_wal(tmp_path, sqlite_app):
    """The 0↔1 durable engine deploys to a file, and a file store wants WAL — the read-only
    observers (`effective.parked`, `effective.checkpoints`) open a SECOND connection to it, and
    under SQLite's rollback-journal default a live writer locks them out."""
    app = sqlite_app(str(tmp_path / "live.db"))
    assert app.journal_mode == "wal"
    assert app.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    # FULL, not the NORMAL that WAL guidance suggests: `NORMAL` can lose the last commits on
    # power loss, and a checkpoint durable the moment its step commits is what replay rests on.
    assert app.conn.execute("PRAGMA synchronous").fetchone()[0] == 2
    app.close()


def test_an_in_memory_store_has_no_journal_to_set(sqlite_app):
    """`:memory:` answers the pragma with `memory` and cannot be put into WAL. Pinned so the
    creation rule stays readable as "file stores get WAL" rather than accidentally depending on
    the pragma being a silent no-op here."""
    app = sqlite_app(":memory:")
    assert app.journal_mode == "memory"


def test_opening_an_existing_ROLLBACK_store_does_not_convert_it(tmp_path, sqlite_app):
    """The condition that earned itself. Journal mode is a persistent property of the FILE, so
    converting a store we are only VISITING rewrites data nobody asked us to touch.

    The concrete loss this prevents: the banked bench corpus is 325 `task.db` files, all
    `journal_mode=delete`, and only 194 of them are pre-uuid7 — so a "convert unless legacy"
    rule silently rewrites the other 131 paid stores on first open. They are reached read-write
    by a path that only reads: `contrastbench.replay_combinator` constructs a `SqliteApp` over
    each one, driven by `test_banked_corpus.py::test_current_era_stores_still_replay`. (An
    earlier version of this docstring credited that module with a byte-stability assertion. It
    has none — the citation failed its own grep.) Converting is still possible; it is just
    deliberate (`enable_wal`)."""
    path = _store_from_the_rollback_era(tmp_path / "banked.db")
    before = path.read_bytes()

    app = sqlite_app(str(path))
    assert app.journal_mode == "delete"  # visited, not converted
    app.close()  # closed HERE, not at teardown: the byte comparison below needs it flushed

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("banked.db-wal"))
    assert not list(tmp_path.glob("banked.db-shm"))


def test_enable_wal_converts_an_existing_store_when_asked(tmp_path, sqlite_app):
    """The escape hatch is real and is the supported way to upgrade a deployed store."""
    path = _store_from_the_rollback_era(tmp_path / "banked.db")

    app = sqlite_app(str(path))
    assert app.journal_mode == "delete"
    assert enable_wal(app.conn) == "wal"
    assert app.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


_CRASHING_WRITER = """
import os, sys
sys.path.insert(0, sys.argv[2])
from effective.sqlite import SqliteApp
app = SqliteApp(sys.argv[1])
print(app.spawn("w", {"n": 1}), flush=True)
os._exit(9)                      # no close, no atexit, no flush: a real unclean exit
"""


def test_a_read_only_observer_still_opens_a_WAL_store_after_an_unclean_exit(tmp_path):
    """The standing objection to WAL, checked rather than assumed.

    A `file:…?mode=ro` connection must read a WAL database **with no writer holding it open** —
    including after a crash left `-wal`/`-shm` behind, which is precisely when someone reaches
    for `read_sqlite_parked`. SQLite recovers them.

    **The writer crashes in a SUBPROCESS, and that is the whole test.** The first version kept
    the writing `SqliteApp` alive in this process and called that "modelling the crash", so a
    live connection was holding the `-shm` the entire time and the test would have passed even
    if the stated property were false — an instrument that could not fail. `os._exit` in a child
    is the only way to leave the sidecars behind with genuinely nobody attached."""
    path = tmp_path / "live.db"
    src = str(Path(effective.__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, "-c", _CRASHING_WRITER, str(path), src],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 9, done.stderr
    task_id = UUID(done.stdout.strip())
    # The sidecars survive precisely because nobody closed the store.
    assert (tmp_path / "live.db-wal").exists()
    assert (tmp_path / "live.db-shm").exists()

    observer = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = observer.execute("SELECT task_id FROM tasks").fetchall()
    observer.close()
    assert [UUID(r[0]) if isinstance(r[0], str) else r[0] for r in rows] == [task_id]

    # And the observer the engine actually ships, over the same crashed store.
    assert read_sqlite_parked(str(path)) == ()


def test_a_store_that_refuses_WAL_is_an_error_not_a_silent_downgrade(tmp_path, monkeypatch):
    """`PRAGMA journal_mode` REPORTS the mode it reached rather than failing, so a filesystem
    with no shared-memory support (NFS, some container mounts) would otherwise hand back a store
    that looks ordinary while quietly lacking the concurrent-reader property the observers rely
    on. Simulated by neutering the pragma, because this box's filesystems all accept WAL."""
    monkeypatch.setattr("effective.sqlite.enable_wal", lambda conn: "delete")

    with pytest.raises(RuntimeError) as caught:
        SqliteApp(str(tmp_path / "nfs.db"))
    assert "refused WAL" in str(caught.value)
    assert "require_wal=False" in str(caught.value)  # the message names the way out


def test_require_wal_false_accepts_the_rollback_journal(tmp_path, monkeypatch, sqlite_app):
    """The escape hatch, and why it is not the default.

    Raising is a NARROWING: a store on a WAL-hostile filesystem works under the rollback
    journal. This engine is aimed at laptops, so refusing to CONSTRUCT there would be a hard
    regression; the flag keeps the rollback journal, with the mode visible on the instance
    rather than silently assumed."""
    monkeypatch.setattr("effective.sqlite.enable_wal", lambda conn: "delete")

    app = sqlite_app(str(tmp_path / "nfs.db"), require_wal=False)
    assert app.journal_mode == "delete"  # honest about what it got
    assert app.spawn("w", {"n": 1}) is not None  # and still a working store


# --- an event and its wake commit together -----------------------------------------------------


def test_an_interrupted_emit_leaves_no_event_without_its_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_deliver` writes the event and wakes its waiters, and the two must be one commit.

    The connection is `isolation_level=None`, so two `execute` calls are two transactions: the
    INSERT lands, the UPDATE is lost, and a task parked on that name waits on an event the store
    already holds. Postgres does both inside `emit_event`, which is what SQLite conforms to.

    The raise stands for any interruption between the statements, a killed emitter included. What
    the pin asserts is the DURABLE state both leave behind, which is the same either way.
    """
    path = str(tmp_path / "wake.db")
    app, emitter = SqliteApp(path), SqliteApp(path)
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            return ctx.await_event(Key.parse("ev:x"))

        task = app.spawn("waiter", {})
        app.work_batch()
        parked = app.fetch_task_result(task)
        assert parked is not None
        assert parked.state == "waiting"

        # Fail the emitter once the event row is in and before its waiters are woken.
        real = emitter.conn.execute

        def fail_the_wake(sql: Any, *args: Any) -> Any:
            if "UPDATE tasks SET state='ready'" in str(sql):
                raise sqlite3.OperationalError("emitter interrupted before the wake")
            return real(sql, *args)

        monkeypatch.setattr(emitter.conn, "execute", fail_the_wake)
        with pytest.raises(sqlite3.OperationalError):
            emitter.emit_event("ev:x", {"v": 1})
        monkeypatch.undo()

        rows = emitter.conn.execute("SELECT count(*) FROM events WHERE name='ev:x'")
        stored = rows.fetchone()[0]
        snap = app.fetch_task_result(task)
        assert snap is not None
        # Either the event is not there and the emitter retries, or it is there and its waiter was
        # woken. A durable event beside a sleeping waiter is the state nothing repairs.
        assert not (stored and snap.state == "waiting"), (
            f"the event is durable ({stored=}) and its waiter is still {snap.state!r}"
        )
    finally:
        emitter.close()
        app.close()


# --- a park may name an event AND a deadline ---------------------------------------------------


def _slot(name: str) -> Key:
    """The slot a bounded wait records its outcome in, as the walk would name it.

    A ctx called directly has no walk above it, so these tests name what `current_placement`
    would. `event;{name}` is the walk's first ask of a name (`Key.occurrence` is byte-preserving
    at n <= 1), which is what every test here makes.
    """
    # lint: terminal-hole — a `Key`, so it needs no wrapper
    return compose_key(t"event;{Key.parse(name):domain=address}")


def _waits_until(app: SqliteApp, name: str, deadline: float, seen: list[Any]) -> UUID:
    """Register a task that parks on `name` until `deadline` and records how its wait ended."""

    @app.register_task("waiter")
    def waiter(params: dict[str, Any], ctx: Any) -> Any:
        outcome = ctx.await_until(Key.parse(name), deadline, _slot(name))
        seen.append(outcome)
        return "done"

    return app.spawn("waiter", {})


def test_a_deadline_already_past_expires_without_parking(tmp_path: Path) -> None:
    """The wait is over before it starts, so it answers at once."""
    app = SqliteApp(str(tmp_path / "past.db"))
    try:
        seen: list[Any] = []
        task = _waits_until(app, "ev:x", time.time() - 1.0, seen)
        app.work_batch()
        snap = app.fetch_task_result(task)
        assert snap is not None
        assert snap.state == "completed"
        assert seen == [Expired()]
    finally:
        app.close()


def test_a_park_records_its_deadline(tmp_path: Path) -> None:
    """A park carrying both an event and a deadline stores the deadline in `available_at`, where
    the claim reads it."""
    app = SqliteApp(str(tmp_path / "records.db"))
    try:
        seen: list[Any] = []
        deadline = time.time() + 3600.0
        task = _waits_until(app, "ev:x", deadline, seen)
        app.work_batch()
        row = app.conn.execute(
            "SELECT state, waiting_event, available_at FROM tasks WHERE task_id=?",
            (str(task),),
        ).fetchone()
        assert row[0] == "waiting"
        assert row[1] == "ev:x"
        assert row[2] == pytest.approx(deadline)
    finally:
        app.close()


def test_a_waiting_task_whose_deadline_passed_is_claimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim half. One clock drives the claim and the body, so the deadline passes for
    both together, and the pin states a claim rule rather than a duration."""
    app = SqliteApp(str(tmp_path / "due.db"))
    try:
        base = time.time()
        clock = [base]
        monkeypatch.setattr(time, "time", lambda: clock[0])
        seen: list[Any] = []
        task = _waits_until(app, "ev:x", base + 1.0, seen)
        app.work_batch()
        assert app.work_batch() is False, "a park whose deadline is future was claimed early"
        clock[0] = base + 2.0
        assert app.work_batch() is True, "a park whose deadline passed was not claimable"
        snap = app.fetch_task_result(task)
        assert snap is not None
        assert snap.state == "completed"
        assert seen == [Expired()]
    finally:
        app.close()


def test_an_event_before_the_deadline_arrives_with_its_payload(tmp_path: Path) -> None:
    """The other end of the same wait: an emit before the deadline wins, and the body is handed
    the payload rather than an expiry."""
    path = str(tmp_path / "arrive.db")
    app, emitter = SqliteApp(path), SqliteApp(path)
    try:
        seen: list[Any] = []
        task = _waits_until(app, "ev:x", time.time() + 30.0, seen)
        app.work_batch()
        parked = app.fetch_task_result(task)
        assert parked is not None
        assert parked.state == "waiting"
        emitter.emit_event("ev:x", {"v": 7})
        app.work_batch()
        snap = app.fetch_task_result(task)
        assert snap is not None
        assert snap.state == "completed"
        assert seen == [Arrived({"v": 7})]
    finally:
        emitter.close()
        app.close()


def test_an_event_park_with_no_deadline_is_not_claimable_by_time(tmp_path: Path) -> None:
    """The regression guard on the claim query. A plain event park stores `available_at=0`, so a
    claim that admitted `waiting` rows on time alone would wake EVERY parked task at once."""
    app = SqliteApp(str(tmp_path / "plain.db"))
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            return ctx.await_event(Key.parse("ev:x"))

        task = app.spawn("waiter", {})
        app.work_batch()
        parked = app.fetch_task_result(task)
        assert parked is not None
        assert parked.state == "waiting"
        assert app.work_batch() is False, "an event park with no deadline was claimed on time"
    finally:
        app.close()


# --- one wait, one answer ----------------------------------------------------------------------


def test_a_wait_that_expired_stays_expired_when_its_event_arrives_late(tmp_path: Path) -> None:
    """A deadline decided is a decision, and a later event leaves it alone.

    Absurd rules this: `emit_event` deletes waits whose `timeout_at` has passed and wakes only
    those still ahead of it, so a late event reaches no expired waiter. SQLite conforms by
    recording the outcome the first time it is reached.

    The divergence needs no crash and no retry: the body acts on `Expired`, parks on a second
    event, and the replay of the first wait answers again.
    """
    path = str(tmp_path / "once.db")
    app, outside = SqliteApp(path), SqliteApp(path)
    try:
        seen: list[Any] = []

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            seen.append(ctx.await_until(Key.parse("ev:a"), time.time() - 1.0, _slot("ev:a")))
            return ctx.await_event(Key.parse("ev:b"))

        task = app.spawn("waiter", {})
        app.work_batch()
        assert seen == [Expired()]

        outside.emit_event("ev:a", {"late": True})
        outside.emit_event("ev:b", {"go": True})
        app.work_batch()

        snap = app.fetch_task_result(task)
        assert snap is not None
        assert snap.state == "completed"
        assert seen == [Expired(), Expired()], f"one wait answered twice: {seen}"
    finally:
        outside.close()
        app.close()


def test_an_event_landing_while_the_park_is_recorded_leaves_the_task_claimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`state='ready'` implies `available_at <= now`, for every writer of the pair.

    The park reads the event, releases the write lock, and re-takes it to record itself. An emit
    inside that window makes the row `ready` with `waiting_event` cleared, so the wake can never
    reach it again and only the deadline would release it.
    """
    path = str(tmp_path / "window.db")
    app, emitter = SqliteApp(path), SqliteApp(path)
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            return ctx.await_until(Key.parse("ev:x"), time.time() + 3600.0, _slot("ev:x"))

        task = app.spawn("waiter", {})
        real = app.conn.execute

        def emit_inside_the_window(sql: Any, *args: Any) -> Any:
            if "UPDATE tasks SET state=CASE WHEN EXISTS" in str(sql):
                monkeypatch.undo()
                emitter.emit_event("ev:x", {"v": 1})
                monkeypatch.setattr(app.conn, "execute", emit_inside_the_window)
            return real(sql, *args)

        monkeypatch.setattr(app.conn, "execute", emit_inside_the_window)
        app.work_batch()
        monkeypatch.undo()

        rows = app.conn.execute(
            "SELECT state, available_at FROM tasks WHERE task_id=?", (str(task),)
        )
        row = rows.fetchone()
        assert row[0] == "ready"
        # BOTH bounds. `<= now` alone is satisfied by 0, which is the value to rule out: a ready
        # task stamped 0 is due, and outranks every waiter the clock has passed.
        assert 0 < row[1] <= time.time(), f"a ready task carries {row[1]} for its due instant"
        assert app.work_batch() is True, "a ready task holding its event was not claimable"
    finally:
        emitter.close()
        app.close()


# --- what the wait change owes its neighbours --------------------------------------------------


def test_a_failed_commit_leaves_no_open_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_deliver` opens a transaction, so both of its ends can fail.

    A `COMMIT` that raises leaves the connection inside the transaction for good: every later
    emit refuses to begin one, and every later autocommit write silently joins it instead of
    committing, which is the durability the engine sells. `SQLITE_BUSY` at commit reaches this on
    a rollback-journal store, and the banked corpus is all rollback-journal.
    """
    app = SqliteApp(str(tmp_path / "commit.db"))
    try:
        real = app.conn.execute

        def fail_the_commit(sql: Any, *args: Any) -> Any:
            if str(sql).strip() == "COMMIT":
                raise sqlite3.OperationalError("database is locked")
            return real(sql, *args)

        monkeypatch.setattr(app.conn, "execute", fail_the_commit)
        with pytest.raises(sqlite3.OperationalError):
            app.emit_event("ev:x", {"v": 1})
        monkeypatch.undo()

        assert app.conn.in_transaction is False, "the connection is stuck inside its transaction"
        app.emit_event("ev:y", {"v": 2})  # a later emit still works
    finally:
        app.close()


def test_a_task_whose_deadline_passed_is_not_reported_as_parked(tmp_path: Path) -> None:
    """`read_sqlite_parked` lists what a human can act on, so a claimable task is out.

    Its own docstring already rules this for a `sleeping` task whose time has come. A `waiting`
    task carrying a deadline that has passed is claimable the same way.
    """
    from effective.parked import read_sqlite_parked

    path = str(tmp_path / "parked.db")
    app = SqliteApp(path)
    try:
        seen: list[Any] = []
        task = _waits_until(app, "ev:x", time.time() + 3600.0, seen)
        app.work_batch()
        assert [p.task_id for p in read_sqlite_parked(path)] == [task], "a live park should list"

        app.conn.execute(
            "UPDATE tasks SET available_at=? WHERE task_id=?", (time.time() - 1.0, str(task))
        )
        assert read_sqlite_parked(path) == (), "a claimable task was reported as parked"
    finally:
        app.close()


def test_await_until_under_a_prefix_parks_on_the_prefixed_name(tmp_path: Path) -> None:
    """A ctx wrapper applies its frame to every wait, or two gather branches share one park.

    `_PrefixedCtx` names its decorated arms explicitly and delegates the rest raw, so a capability
    added without an arm keeps the engine's bare name. `_supports_peek`'s docstring already calls
    this the recurring bug rather than an instance of one.
    """
    from effective.handlers.absurd import _PrefixedCtx

    app = SqliteApp(str(tmp_path / "prefix.db"))
    try:
        parked: list[str] = []

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            inner = _PrefixedCtx(ctx, "gather:0,1;")
            try:
                return inner.await_until(Key.parse("ev:x"), time.time() + 3600.0, _slot("ev:x"))
            except Exception:
                parked.append(str(app.conn.execute("SELECT waiting_event FROM tasks").fetchone()))
                raise

        app.spawn("waiter", {})
        app.work_batch()
        row = app.conn.execute("SELECT waiting_event FROM tasks").fetchone()
        assert row[0] == "gather:0,1;ev:x", f"the frame was dropped: {row[0]!r}"
    finally:
        app.close()


def test_a_claim_takes_the_task_that_came_due_first(tmp_path: Path) -> None:
    """`ORDER BY available_at` ranks by the instant a task became claimable, in every state.

    A ready task and a waiting task whose deadline has passed are both due, so the older instant
    goes first. Absurd writes `available_at = v_now` when it makes a task ready, which is what
    puts its queue in that order; writing 0 instead ranks every ready task ahead of every expired
    waiter, whatever the clock says.
    """
    app = SqliteApp(str(tmp_path / "due.db"))
    try:
        ran: list[str] = []

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            ran.append("waiter")
            return ctx.await_until(Key.parse("ev:x"), params["deadline"], _slot("ev:x"))

        @app.register_task("ready")
        def ready(params: dict[str, Any], ctx: Any) -> Any:
            ran.append("ready")
            return "done"

        parked = app.spawn("waiter", {"deadline": time.time() + 3600.0})
        app.work_batch()
        app.conn.execute(
            "UPDATE tasks SET available_at=? WHERE task_id=?",
            (time.time() - 10.0, str(parked)),
        )
        app.spawn("ready", {})

        ran.clear()
        app.work_batch()
        assert ran == ["waiter"], "the claim took work that came due later"
    finally:
        app.close()


def test_a_woken_task_carries_the_instant_it_was_woken(tmp_path: Path) -> None:
    """The wake is the other writer of `available_at`, and it has to stamp the clock too.

    `test_a_claim_takes_the_task_that_came_due_first` pins the ORDER BY by setting the column
    itself, so it cannot see a writer that leaves 0 there; this asserts the writer. A woken task
    stamped 0 outranks every waiter the clock has passed, which is the defect in full.
    """
    app = SqliteApp(str(tmp_path / "woken.db"))
    try:
        seen: list[Any] = []
        task = _waits_until(app, "ev:x", time.time() + 3600.0, seen)
        app.work_batch()
        assert _state_of(app, task) == "waiting"

        app.emit_event("ev:x", {"n": 1})
        state, available_at = app.conn.execute(
            "SELECT state, available_at FROM tasks WHERE task_id=?", (str(task),)
        ).fetchone()
        assert state == "ready"
        assert 0 < available_at <= time.time(), f"the wake wrote {available_at}"
    finally:
        app.close()


def _state_of(app: SqliteApp, task: UUID) -> str:
    row = app.conn.execute("SELECT state FROM tasks WHERE task_id=?", (str(task),)).fetchone()
    return row[0]


def test_an_emit_past_a_deadline_leaves_the_waiter_where_it_was(tmp_path: Path) -> None:
    """A wake is owed to the waits the clock has not passed, and the ROW is where that shows.

    `absurd.emit_event` deletes those waits and wakes the rest, so the waiter it skips keeps its
    registration. Asserting the outcome instead cannot see this: the task is claimable by its own
    deadline either way, so it reaches `Expired` down two different paths. The row says which.
    """
    app = SqliteApp(str(tmp_path / "skipped.db"))
    try:
        seen: list[Any] = []
        deadline = time.time() + 0.3
        task = _waits_until(app, "ev:x", deadline, seen)
        app.work_batch()
        assert _state_of(app, task) == "waiting"

        while time.time() < deadline + 0.05:
            time.sleep(0.02)
        app.emit_event("ev:x", {"n": 1})

        state, waiting_event = app.conn.execute(
            "SELECT state, waiting_event FROM tasks WHERE task_id=?", (str(task),)
        ).fetchone()
        woke_it = "the emit woke a waiter it had passed"
        assert (state, waiting_event) == ("waiting", "ev:x"), woke_it
    finally:
        app.close()


def test_a_wake_survives_a_claim_that_did_not_settle_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wake is spent for the CLAIM, as the SDK spends its own, not written off the task.

    The SDK clears `wake_event` on its in-memory task dict and raises before reaching the
    statement that would clear the column, so a run that then loses its claim meets the same wake
    on the attempt that follows. A durable clear here would leave that attempt reading the events
    table instead, answering `Arrived` where the reference expires.

    Reachable exactly when the settle does not land, so the settle is made to fail: that is the
    window, and outside it the recorded outcome answers and no wake is consulted at all.
    """
    app = SqliteApp(str(tmp_path / "wake.db"))
    try:
        seen: list[Any] = []
        task = _waits_until(app, "ev:x", time.time() + 3600.0, seen)
        app.work_batch()
        assert _state_of(app, task) == "waiting"
        # The deadline arrives, so the emit that follows wakes nobody and the row keeps its
        # registration — the state a task claimed by its own deadline runs in.
        app.conn.execute(
            "UPDATE tasks SET available_at=? WHERE task_id=?", (time.time() - 1.0, str(task))
        )
        app.emit_event("ev:x", {"n": 1})
        assert _state_of(app, task) == "waiting"

        behind = time.time() - 1.0
        losing = SqliteTaskContext(app.conn, task, app.write_lock)
        monkeypatch.setattr(losing, "settle", _raises_claim_lost)
        with pytest.raises(ClaimLost):
            losing.await_until(Key.parse("ev:x"), behind, _slot("ev:x"))
        monkeypatch.undo()

        retrying = SqliteTaskContext(app.conn, task, app.write_lock)
        outcome = retrying.await_until(Key.parse("ev:x"), behind, _slot("ev:x"))
        read_the_store = f"the next claim read the store instead of its wake: {outcome}"
        assert outcome == Expired(), read_the_store
    finally:
        app.close()


def _raises_claim_lost(name: Key, value: Any, /) -> Any:
    raise ClaimLost("this claim moved on")


def test_a_park_that_ended_stops_reading_as_a_park(tmp_path: Path) -> None:
    """`waiting_event` is the name of a park, and a park is a STATE — so both are the question.

    The three readers are operator-facing and two took the column at its word: `read_sqlite_runs`
    reported a sleeping task as parked on an event, and `bridge_sqlite.park_name` answered with
    it, so a human is told to emit an event at a task that is not waiting for one.
    `read_sqlite_parked` asks for `state='waiting'` and was right all along.

    Asserted over all three at once, because the subject is the relation rather than any column:
    a fourth reader is then a row here, not a writer somebody has to remember.
    """
    from effective.bridge_sqlite import park_name
    from effective.parked import read_sqlite_parked_conn
    from effective.runs import read_sqlite_runs_conn

    app = SqliteApp(str(tmp_path / "ended.db"))
    ends_at = datetime.now(UTC) + timedelta(seconds=0.3)
    _deadline = ends_at.timestamp()
    wake_at = datetime.now(UTC) + timedelta(hours=1)
    try:

        @app.register_task("waiter")
        def waiter(params: dict[str, Any], ctx: Any) -> Any:
            def wf():
                yield from api_await_until(Key.parse("ev:x"), dict, deadline=ends_at)
                yield from sleep_until(wake_at)
                return "done"

            return DurableHandler(ctx, _Tool()).run(wf)

        task = app.spawn("waiter", {})
        app.work_batch()
        assert _state_of(app, task) == "waiting", "the wait must PARK, or it leaves nothing behind"
        while time.time() < _deadline + 0.05:
            time.sleep(0.02)
        app.work_batch()  # the clock ends the park, then the sleep begins

        assert _state_of(app, task) == "sleeping"
        # The READERS, not the column: a registration left standing is untidy, and a reader that
        # reports a park nobody is in is the defect. `read_sqlite_parked` already asks for
        # `state='waiting'`; these two are its siblings, and they did not.
        assert park_name(app.conn, task) is None
        assert [r.waiting_on for r in read_sqlite_runs_conn(app.conn)] == [None]
        assert read_sqlite_parked_conn(app.conn) == ()
    finally:
        app.close()


def test_a_park_a_deadline_has_passed_is_not_a_park_to_key_a_grant_on(tmp_path: Path) -> None:
    """`park_name` names an event a grant is worth emitting to, so the deadline is its question.

    A waiter the clock has passed is claimable, and `_deliver` refuses to wake one, so the grant
    lands in the events table and the task expires regardless. `read_sqlite_parked` excludes those
    for exactly that reason — *a claimable task is nobody's to answer* — and this reader asks the
    same question of one task, so it owes the same answer.

    `read_sqlite_runs` is NOT in this relation: it reports what a run is parked on, which stays
    true while the deadline runs out, so it is asserted here as the one that legitimately differs.
    """
    from effective.bridge_sqlite import park_name
    from effective.parked import read_sqlite_parked_conn
    from effective.runs import read_sqlite_runs_conn

    app = SqliteApp(str(tmp_path / "grant.db"))
    try:
        seen: list[Any] = []
        task = _waits_until(app, "ev:x", time.time() + 3600.0, seen)
        app.work_batch()
        assert _state_of(app, task) == "waiting"
        app.conn.execute(
            "UPDATE tasks SET available_at=? WHERE task_id=?", (time.time() - 1.0, str(task))
        )

        assert read_sqlite_parked_conn(app.conn) == ()
        assert park_name(app.conn, task) is None, "a grant keyed on a park nobody can answer"
        assert [r.waiting_on for r in read_sqlite_runs_conn(app.conn)] == ["ev:x"]
    finally:
        app.close()
