"""The generic task/park reader on both engines.

`effective.parked.read_sqlite_parked*` (0↔1) and `effective.parked.read_absurd_parked` (0↔N).
What the two engines agree on (the reader finds the park, reports the wake registration
verbatim, and drops it on resume) is pinned CROSS-ENGINE in `test_conformance.py`. What lives
here is per-engine, because the record genuinely differs on two fields and forcing them into
conformance would assert a parity that does not exist:

| | SQLite | Absurd |
|---|---|---|
| `parked_since` | `None`: the `tasks` table has no such column | the run's `started_at` |
| `task_id` natively | a `str` | a `uuid`, normalized to `str` by the reader |

`state` is normalized to `graphview.PARKED` by both readers, so it is pinned in conformance: a
field that cannot be compared cross-engine is a field the record should not carry. Each engine's
own word (`'waiting'`, `'sleeping'`) is pinned below, at the SQL boundary where it belongs.

The SQLite cases are infra-free; the Absurd cases need `just pgt-up` and auto-skip without it.
"""

import time
import uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

import psycopg
import pytest
from _authority import AUTHORITY_PARKS, approve_park
from _durable import DSN, absurd, pg_ready, run_until_result

from effective.api import await_event, call_tool, gather, sleep_until
from effective.bridge_absurd import read_absurd_task
from effective.budget import budget_grant_name
from effective.checkpoints import keys
from effective.domain import CallTool, DomainOp
from effective.graphview import PARKED, fold_cycles, from_keys, to_mermaid
from effective.handlers.absurd import ConcurrentAbsurdCtx, DurableHandler
from effective.keys import Index, Key, Scope
from effective.parked import (
    ParkedTask,
    pending_key,
    read_absurd_parked,
    read_sqlite_parked,
    read_sqlite_parked_conn,
)
from effective.sqlite import SqliteApp, SqliteLedger

PG = pg_ready()


class _Tools:
    """One tool, one value — the park is what is under test, not the domain."""

    def run(self, op: DomainOp) -> object:
        assert isinstance(op, CallTool)
        return 10


# ── the workflows: one op, then a park of some shape ─────────────────────────


def _await_wf(event: str | Key):
    """`Key.parse` at the call sites below, not a bare string: these parks are the SUBSTRATE's
    (`approve;`, `budget-grant:`, `govern:`), stood up here to exercise the reader across every
    authorization namespace. Author text may not name them — that is `event_name`'s refusal, and
    this reader's whole job is to see the parks that refusal exists to protect."""

    def factory(_run_id: str):
        def workflow():
            yield from call_tool("a", {}, int)
            return (yield from await_event(event, dict))

        return workflow

    return factory


def _sleep_wf(when: datetime):
    def factory(_run_id: str):
        def workflow():
            yield from call_tool("a", {}, int)
            yield from sleep_until(when)
            return "woke"

        return workflow

    return factory


def _two_parked_branches_wf(run_id: str):
    """Three branches, two of them parked on DIFFERENT events (indices 0 and 2) — the L-1 shape."""

    def await_branch(i: int):
        def branch():
            return (yield from await_event(f"ev{i}:{run_id}", dict))

        return branch

    def tool_branch():
        def branch():
            return (yield from call_tool("a", {}, int))

        return branch

    def workflow():
        return (yield from gather([await_branch(0), tool_branch(), await_branch(2)]))

    return workflow


# ── the 0↔1 engine ───────────────────────────────────────────────────────────


def _sqlite_run(app: SqliteApp, name: str, factory, run_id: str) -> UUID:
    @app.register_task(name)
    def task(params, ctx):
        rid = params["run_id"]
        ledger = SqliteLedger(app.conn, rid, app.write_lock)
        return DurableHandler(ctx, _Tools(), ledger=ledger).run(factory(rid))

    task_id = app.spawn(name, {"run_id": run_id})
    app.run_until_result(task_id)
    return task_id


def test_sqlite_reader_finds_the_park_and_reports_the_registration_verbatim():
    """The whole record, on the engine that has the fewest columns to give it."""
    app = SqliteApp(":memory:")
    park = approve_park(Key.parse("ledger;processed:m1"))
    task_id = _sqlite_run(app, "gated", _await_wf(park), "r1")

    (park,) = read_sqlite_parked_conn(app.conn)
    assert park == ParkedTask(
        task_id=task_id,
        task_name="gated",
        wake_event="approve;ledger;processed:m1",
        params={"run_id": "r1"},
        state=PARKED,  # normalized; this engine's own word is 'waiting'
        parked_since=None,  # ...and this engine records none at all
    )
    # the engine's own vocabulary, pinned where it actually lives — the row, not the record
    assert app.conn.execute("SELECT state FROM tasks").fetchone() == ("waiting",)
    app.close()


def test_sqlite_reader_lists_every_namespace_not_just_approve():
    """PARAMETERIZED over the minters, so the coverage is the table rather than three literals.

    A sign-off reader that filters `LIKE 'approve;%'` misses most parks; this reader must not.
    Both the park and the expectation come from `AUTHORITY_PARKS`, so a new namespace is one row
    and no hand-spelled park can agree with itself while the real minter moves on.
    """
    app = SqliteApp(":memory:")
    expected = []
    for i, (namespace, mint) in enumerate(AUTHORITY_PARKS):
        park = mint(f"r{i}", generation=0)
        _sqlite_run(app, namespace, _await_wf(park), f"r{i}")
        expected.append(park.stored())

    assert [p.wake_event for p in read_sqlite_parked_conn(app.conn)] == expected
    app.close()


def test_sqlite_reader_leaves_the_park_when_the_run_resumes():
    app = SqliteApp(":memory:")
    task_id = _sqlite_run(app, "gated", _await_wf("review:r1"), "r1")
    assert len(read_sqlite_parked_conn(app.conn)) == 1

    app.emit_event("review:r1", {"decision": "approve"})
    snap = app.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    assert read_sqlite_parked_conn(app.conn) == ()
    app.close()


def test_sqlite_reader_excludes_a_timed_park():
    """A `sleep_until` is `state='sleeping'` with no `waiting_event`. Nobody can act on it, and
    once its deadline passes it is *claimable* rather than parked — so it is out of the
    relation."""
    app = SqliteApp(":memory:")
    _sqlite_run(app, "sleeper", _sleep_wf(datetime.now(UTC) + timedelta(hours=1)), "r1")

    assert app.conn.execute("SELECT state, waiting_event FROM tasks").fetchone() == (
        "sleeping",
        None,
    )
    assert read_sqlite_parked_conn(app.conn) == ()
    app.close()


def test_sqlite_reader_reports_one_branch_of_a_parked_gather():
    """A gather with two parked branches registers only the LOWEST index's event (`_join`'s
    deterministic re-arm), so the reader reports 2 parks as 1, and answering that one produces a
    *new* park instead of finishing. This is substrate-correct, because wakes are serialized: the
    task waits on the lowest parked index. A page built on this reader has to say so."""
    app = SqliteApp(":memory:")
    task_id = _sqlite_run(app, "multi", _two_parked_branches_wf, "r1")

    (park,) = read_sqlite_parked_conn(app.conn)
    assert park.wake_event == "gather:0,0;ev0:r1"  # branch 2's park is not listed

    app.emit_event("gather:0,0;ev0:r1", {"n": 0})
    app.run_until_result(task_id)
    (park,) = read_sqlite_parked_conn(app.conn)
    assert park.wake_event == "gather:0,2;ev2:r1"  # a NEW park, not a completion
    app.close()


def test_sqlite_path_reader_is_the_web_shape_and_opens_read_only(tmp_path):
    """A second process holding only the file. It must not construct a
    `SqliteApp` to look: that constructor runs `executescript(_SCHEMA)`, i.e. DDL against somebody
    else's store to answer a GET. The read-only open proves the reader issues none."""
    path = tmp_path / "engine.db"
    app = SqliteApp(str(path))
    _sqlite_run(app, "gated", _await_wf("review:r1"), "r1")

    (park,) = read_sqlite_parked(path)  # a fresh connection, mode=ro
    assert (park.task_name, park.wake_event) == ("gated", "review:r1")
    assert read_sqlite_parked(str(path)) == read_sqlite_parked(path)  # str or Path
    app.close()


def test_sqlite_reader_on_an_engine_with_nothing_parked(tmp_path):
    app = SqliteApp(str(tmp_path / "engine.db"))
    assert read_sqlite_parked_conn(app.conn) == ()
    assert read_sqlite_parked(tmp_path / "engine.db") == ()
    app.close()


# ── the 0↔N engine (needs `just pgt-up`) ─────────────────────────────────────

pg_only = pytest.mark.skipif(not PG, reason="needs Postgres/Absurd (just pgt-up)")


@pytest.fixture
def conn():
    connection = psycopg.connect(DSN, autocommit=True)
    yield connection
    connection.close()


def _drain_until_sleeping(app, conn, task_id: UUID) -> None:
    """Drain until THIS task is `sleeping`, on a deadline rather than a fixed retry count.

    The `default` queue is SHARED across the durable lane, so `work_batch` may spend claims on
    other tests' tasks before it reaches ours (the justfile's one-queue note); and a real park
    involves two checkpoint round-trips, so a fixed count flakes under suite load."""
    t_tbl = "t_default"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row = conn.execute(
            t"SELECT state FROM absurd.{t_tbl:i} WHERE task_id = {task_id}::uuid"
        ).fetchone()
        if row is not None and row[0] == "sleeping":
            return
        app.work_batch()
    raise AssertionError(f"task {task_id} never parked within 30s")


def _absurd_run(app, conn, name: str, factory, run_id: str) -> UUID:
    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx):
        return DurableHandler(ConcurrentAbsurdCtx(ctx), _Tools()).run(factory(params["run_id"]))

    # `str(...)`: the SDK's `spawn` hands back a `UUID`, where `SqliteApp.spawn` hands back a
    # `str`. That asymmetry is exactly what `ParkedTask.task_id` normalizes away, so a caller
    # correlating a spawn with a park has to do the same — comparing the two raw is a silent
    # no-match, which is how this test failed on its first real-engine run.
    # No `str(...)`: the SDK returns `row["task_id"]` off a `uuid` column, so psycopg has
    # already handed back a `UUID` — the annotation on its `SpawnResult` TypedDict says `str`
    # and validates nothing. Stringifying here was throwing away the type the driver produced.
    task_id = app.spawn(name, {"run_id": run_id})["task_id"]
    _drain_until_sleeping(app, conn, task_id)
    return task_id


@pg_only
def test_absurd_reader_reports_a_park_timestamp_and_a_uuid_task_id(conn):
    """The one field conformance cannot compare: `parked_since` is a real timestamp where SQLite
    has none. `task_id` is comparable: SQLite mints `uuid7`, so both engines carry a `UUID` and
    the reader passes psycopg's through unconverted. `state` is comparable too: it is normalized
    to `PARKED` on both engines and asserted in conformance; this engine's own word
    (`'sleeping'`, for a timed park too) is pinned at the row below."""
    run_id = f"pk{uuid.uuid4().hex[:8]}"
    app = absurd()
    before = datetime.now(UTC) - timedelta(minutes=5)
    task_id = _absurd_run(
        app, conn, f"parked-{run_id}", _await_wf(approve_park(Key.parse(run_id))), run_id
    )

    parked = [p for p in read_absurd_parked(conn) if p.task_id == task_id]
    assert len(parked) == 1, parked  # one row per parked TASK, not per sleeping attempt
    (park,) = parked
    assert isinstance(park.task_id, UUID)  # the column is a uuid, and the record keeps it one
    assert park.task_name == f"parked-{run_id}"
    assert park.wake_event == f"approve;{run_id}"
    assert park.params == {"run_id": run_id}
    assert park.state == PARKED  # normalized; the row below still says 'sleeping'
    assert conn.execute(
        t"SELECT r.state FROM absurd.r_default r "
        t"JOIN absurd.t_default t ON t.last_attempt_run = r.run_id "
        t"WHERE t.task_id = {task_id}::uuid"
    ).fetchone() == ("sleeping",)
    assert park.parked_since is not None
    assert park.parked_since >= before
    app.close()


@pg_only
def test_absurd_reader_is_unfiltered_across_authorization_namespaces(conn):
    """A reader filtering `LIKE 'approve;%'` would see only the first of these."""
    run_id = f"pk{uuid.uuid4().hex[:8]}"
    app = absurd()
    approve = _absurd_run(
        app, conn, f"a-{run_id}", _await_wf(approve_park(Key.parse(run_id))), f"{run_id}a"
    )
    grant_event = f"budget-grant:{run_id},0"
    granted = _absurd_run(
        app, conn, f"g-{run_id}", _await_wf(budget_grant_name(run_id, 0)), f"{run_id}g"
    )

    found = {p.task_id: p.wake_event for p in read_absurd_parked(conn)}
    assert found.get(approve) == f"approve;{run_id}"
    assert found.get(granted) == grant_event
    app.close()


@pg_only
def test_the_pending_node_on_a_real_absurd_park_and_what_resume_does(conn):
    """The pending-node bridge end to end on the 0↔N engine, and the resume transition measured
    rather than assumed, since the node changes identity across its own lifecycle. The SQLite
    half is
    `test_graphview.py::test_a_parked_run_projects_a_pending_node_end_to_end_on_the_embedded_engine`.

    **While parked** the two engines agree completely: no checkpoint exists for the await on
    either, the park reader is the only witness, and `pending_key` synthesizes the same
    `event;{wake_event}` node from it.

    **On resume they do not, and this is the asymmetry to know about.** Absurd's SDK freezes the
    delivered payload as a `$awaitEvent:{name}` checkpoint, so a row *does* appear here where
    SQLite grows none. It does not restore the node, for two independent reasons pinned below:
    it is a different key (`$awaitEvent:review:x` vs the pending `event;review:x`, so a
    before/after alignment sees two alphabets), and `ENGINE_INTERNAL` filters it out of the
    default view anyway. So on both engines the user-visible answer is the same — **the pending
    node disappears and no committed counterpart replaces it** — and the evidence the answer
    landed is the ops that ran after it."""
    run_id = f"pk{uuid.uuid4().hex[:8]}"
    event = approve_park(Key.parse(run_id))
    name = event.stored()  # the text form, for the projection assertions below
    app = absurd()
    task_id = _absurd_run(app, conn, f"pending-{run_id}", _await_wf(event), run_id)

    parked = [p for p in read_absurd_parked(conn) if p.task_id == task_id]
    assert len(parked) == 1, parked
    (park,) = parked
    assert (
        pending_key(park).stored() == f"event;{name}"
    )  # the same spelling the SQLite half composes

    recorded = list(keys(read_absurd_task(conn, task_id)))
    assert recorded == ["step;tool:a"]  # the tool step; the await is in no bookkeeper
    graph = from_keys(run_id, recorded, pending=pending_key(park))
    assert [(n.key, n.kind, n.state) for n in graph.nodes] == [
        ("step;tool:a", "step", "committed"),
        (f"event;{name}", "await", PARKED),
    ]
    assert [(e.src, e.dst) for e in graph.edges] == [("step;tool:a", f"event;{name}")]
    assert f'{{{{"event;{name}<br/>(parked)"}}}}' in to_mermaid(graph)  # the hexagon, labelled
    assert fold_cycles(graph, drop=(Index,)).nodes[-1].state == PARKED

    # --- the resume transition ---------------------------------------------------------------
    app.emit_event(event, {"ok": True})  # Absurd events are by name
    snap = run_until_result(app, task_id)
    assert snap is not None
    assert snap.state == "completed", snap

    assert [p for p in read_absurd_parked(conn) if p.task_id == task_id] == []
    view = list(keys(read_absurd_task(conn, task_id)))
    unfiltered = list(keys(read_absurd_task(conn, task_id, exclude=())))
    assert view == ["step;tool:a"]  # the default view: the await left nothing behind
    assert unfiltered == [
        "step;tool:a",
        f"$awaitEvent:{name}",
    ]  # ...and this engine DID write a row
    assert f"$awaitEvent:{event}" != pending_key(park)  # a different key, not the same node
    assert "{{" not in to_mermaid(from_keys(run_id, view))  # so the hexagon is simply gone
    app.close()


@pg_only
def test_absurd_reader_excludes_a_timed_park(conn):
    """`r.wake_event IS NOT NULL` is what separates the two parks on an engine that spells both
    `'sleeping'` — the distinction SQLite makes with two state words."""
    run_id = f"pk{uuid.uuid4().hex[:8]}"
    app = absurd()
    wake = datetime.now(UTC) + timedelta(hours=1)
    task_id = _absurd_run(app, conn, f"sleeper-{run_id}", _sleep_wf(wake), run_id)

    assert not [p for p in read_absurd_parked(conn) if p.task_id == task_id]
    app.close()


def test_the_authority_park_table_covers_every_reserved_namespace_or_says_why():
    """A parameterization is only as good as its table, so the table is pinned against the source.

    `AUTHORITY_PARKS` drives the every-namespace tests here and in `test_dashboard`. A namespace
    added to `RESERVED_AUTHORITY_TAGS` without a row would silently reduce their coverage while
    every existing cell stays green: an instrument that cannot fail, in the parameterization
    rather than the assertions.

    The three exclusions are stated WITH their reason rather than left as a gap.
    """
    from effective.keys import RESERVED_AUTHORITY_TAGS

    NOT_A_PARK = {
        "gate-state": "a per-run STATE cell, not a question anyone parks on",
        "fork": "an event-world relocation — a fork child's park is the RENAMED inner name",
        "hyp": "a lineage-scoped ledger id, never an await address",
    }
    covered = {namespace for namespace, _ in AUTHORITY_PARKS}
    assert covered | set(NOT_A_PARK) == set(RESERVED_AUTHORITY_TAGS), (
        covered,
        set(RESERVED_AUTHORITY_TAGS) - covered - set(NOT_A_PARK),
    )
    assert not (covered & set(NOT_A_PARK)), "a namespace cannot be both parked-on and excluded"


def test_the_census_approve_minter_agrees_with_production():
    """`_authority.approve_park` holds a COPY of `permission.approval_name`'s template — so pin it.

    The copy exists because production derives its op key from an op's PLACEMENT while six callers
    here hold an arbitrary `Key`. Duplication that cannot be removed can still be stopped from
    drifting, and this is the cheap form: compose both ways and assert equality. Over several
    generations, not just 0 — the coordinate is omitted at its default, so a copy that had never
    heard of `generation` would agree at 0 and disagree everywhere else. Checking only the default
    is how a pin like this passes while being worthless.
    """
    from effective.domain import CallTool
    from effective.handlers.base import placed_key
    from effective.ops import CHAIN_GENERATION, Step
    from effective.permission import approval_name

    op = Step(name="tool:x", op=CallTool(name="x", args={}, result_schema=str))
    for generation in (0, 1, 7):
        token = CHAIN_GENERATION.set(generation)
        try:
            assert approve_park(placed_key(op), generation=generation) == approval_name(op), (
                generation
            )
        finally:
            CHAIN_GENERATION.reset(token)


@pytest.mark.parametrize(
    ("namespace", "mint"), AUTHORITY_PARKS, ids=lambda v: getattr(v, "__name__", v)
)
def test_a_settlement_namespace_separates_two_GENERATIONS_of_one_run(namespace, mint):
    """Every settlement namespace's park name separates two generations of one run.

    A census with one axis, namespace, answers "is there a park here?" and never "can this park
    tell two askers apart?". This row is the second axis.

    **A `respawn` chain keeps `run_id` stable and each generation is a fresh task**, so every
    within-run coordinate restarts — the handler's occurrence counter, the gate's pass, the
    checkpoint ordinal. Nothing below the name can separate generations, which is why the name
    must, and why a `Scope.SETTLEMENT` promise ("one answer settles one occurrence") is a claim
    about the generation coordinate whether or not its namespace has one.

    **Which namespaces owe this is DERIVED, not listed.** The scope is read off the minted key, so
    it comes from the `AuthorityTag` declaration rather than a curated set here that would sit one
    edit away from disagreeing with it. An `ACCRUAL` namespace is exempt by construction: a grant
    that raises the RUN's ceiling is meant to outlive the op, and a generation coordinate would
    re-ask for it every generation.
    """
    first, second = mint("r-gen", generation=0), mint("r-gen", generation=1)
    if first.scope is not Scope.SETTLEMENT:
        pytest.skip(f"{namespace} is {first.scope} — an accrual outlives the op by design")
    assert first != second, (
        f"{namespace} composes one name for two generations of one run: {first.stored()} — "
        "a single answer settles both, which is the $5-approves-$5,000,000 shape across tasks"
    )
