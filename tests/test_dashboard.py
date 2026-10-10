"""The local dashboard: three routes over one `SqliteApp`.

Everything except the drawn SVG is infra-free: the pages are pure functions of what the readers
returned, and `TestClient` drives them in-process. The one case that needs the pinned, sandboxed
elkjs container is the one that asserts on the SVG itself, gated exactly like
`test_graphlayout_elkjs.py`: the instrument is a container, and its absence is a skip rather than
a silent fallback.

What is pinned here, and why each one:

- the fleet list shows a parked run, and `parked_since` renders its ABSENCE (SQLite has no such
  column; inventing "now" would be a fact the engine never recorded);
- the run page carries the answer form on the RAW registration, so the emit is the name the run
  registered rather than one the page rebuilt;
- POSTing an answer resumes the run IN PROCESS and the park leaves the list (the drain half of
  the round trip: an emit alone does nothing);
- a parked gather says on screen that it is one branch of N;
- and, with the container, the parked node is a `<polygon>` hexagon carrying
  `data-node-id="event;{wake_event}"`, the handle a click reports.
"""

import json
from itertools import count
from pathlib import Path
from uuid import UUID, uuid7

import pytest
from _authority import AUTHORITY_PARKS, approve_park
from fastapi.testclient import TestClient

from effective.api import append_ledger, await_event, call_tool, gather
from effective.dashboard import dashboard
from effective.domain import CallTool, DomainOp
from effective.engines.sqlite import SqliteApp, SqliteLedger
from effective.graphlayout import elkjs
from effective.graphlayout import elkjs as elk_engine
from effective.handlers.durable import DurableHandler
from effective.keys import Key
from effective.layers import compose_domain
from effective.ops import LedgerRow
from effective.parked import read_sqlite_parked_conn
from effective.telemetry import otlp_jsonl_sink, traced


class _Tools:
    def run(self, op: DomainOp) -> object:
        assert isinstance(op, CallTool)
        return 10


def _await_wf(event: str | Key):
    """`Key.parse` at the substrate-namespace call sites: these parks belong to the
    approval/grant/gate machinery, stood up here so the fleet page can be asked to show
    every authorization namespace rather than just `approve;`."""

    def factory(run_id: str):
        def workflow():
            yield from call_tool("a", {}, int)
            # one ledger op too, so the graph has more than a single box to lay out
            yield from append_ledger(
                LedgerRow(event_id=Key.parse(f"{run_id}:submitted"), kind="submitted")
            )
            return (yield from await_event(event, dict))

        return workflow

    return factory


def _two_parked_branches_wf(run_id: str):
    """Three branches, two parked on DIFFERENT events (indices 0 and 2) — the L-1 shape."""

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


def _run(app: SqliteApp, name: str, factory, run_id: str) -> UUID:
    @app.register_task(name)
    def task(params, ctx):
        rid = params["run_id"]
        ledger = SqliteLedger(app.conn, rid, app.write_lock)
        return DurableHandler(ctx, _Tools(), ledger=ledger).run(factory(rid))

    task_id = app.spawn(name, {"run_id": run_id})
    app.run_until_result(task_id)
    return task_id


@pytest.fixture
def engine():
    app = SqliteApp(":memory:")
    yield app
    app.close()


@pytest.fixture
def client(engine):
    with TestClient(dashboard(engine)) as test_client:
        yield test_client


# ── GET / — the fleet ────────────────────────────────────────────────────────


def test_the_fleet_page_lists_a_parked_run_and_links_to_it(engine, client):
    park = approve_park(Key.parse("ledger;processed:m1"))
    task_id = _run(engine, "gated", _await_wf(park), "r1")

    page = client.get("/")
    assert page.status_code == 200
    assert "approve;ledger;processed:m1" in page.text
    assert f'href="/tasks/{task_id}"' in page.text


def test_the_fleet_page_renders_the_ABSENCE_of_a_park_timestamp(engine, client):
    """`parked_since` is `None` on SQLite — the `tasks` table has no such column. The column stays
    and says so; substituting "now" would report a fact the engine never recorded."""
    _run(engine, "gated", _await_wf("review:r1"), "r1")

    page = client.get("/")
    assert "not recorded by this engine" in page.text


def test_the_fleet_page_is_empty_when_nothing_is_parked(engine, client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Nothing is parked." in page.text


def test_every_authorization_namespace_shows_not_just_approve(engine, client):
    """The generalization past a sign-off reader's `LIKE 'approve;%'`, reaching the page.

    Parameterized over `AUTHORITY_PARKS` so the page is asserted against what each namespace's
    MINTER produces, not against literals that drift away from it.
    """
    expected = []
    for i, (namespace, mint) in enumerate(AUTHORITY_PARKS):
        park = mint(f"r{i}", generation=0)
        _run(engine, namespace, _await_wf(park), f"r{i}")
        expected.append(park.stored())

    text = client.get("/").text
    for name in expected:
        assert name in text, name


# ── GET /tasks/{task_id} — one run ───────────────────────────────────────────


def test_the_run_page_offers_the_form_on_the_RAW_registration(engine, client):
    """The name the page posts against is the one the engine stored, not a reconstruction."""
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")

    page = client.get(f"/tasks/{task_id}")
    assert page.status_code == 200
    assert f'action="/tasks/{task_id}/answer"' in page.text
    assert "review:r1" in page.text


def test_the_run_page_of_a_finished_run_has_no_form(engine, client):
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")
    engine.emit_event("review:r1", {"decision": "approve"})
    engine.run_until_result(task_id)

    page = client.get(f"/tasks/{task_id}")
    assert "completed" in page.text
    assert "nothing to answer" in page.text
    assert "<form" not in page.text


def test_an_unknown_task_is_a_404_and_a_malformed_one_never_reaches_the_handler(client):
    """Two different wrongs, and the route now tells them apart.

    A well-formed id for a task this engine does not hold is a **404** — the handler ran, looked,
    and found nothing. A path segment that is not a task id at all is a **422**, refused by
    FastAPI before any handler runs, because the route declares `task_id: UUID` and `UUID(text)`
    is a validating PARSE. That is the read-boundary parse arriving for free at the HTTP edge:
    while the id was a composed string, `/tasks/nope` was indistinguishable from a real lookup
    miss and this test asserted 404 for both."""
    assert client.get(f"/tasks/{uuid7()}").status_code == 404
    assert client.get("/tasks/nope").status_code == 422


def test_a_parked_gather_says_on_screen_that_it_is_one_branch_of_N(engine, client):
    """L-1, surfaced rather than fixed. The reader can only see the lowest-indexed parked branch,
    so answering it produces a NEW park instead of a completion — which reads as a bug unless the
    page says otherwise."""
    task_id = _run(engine, "multi", _two_parked_branches_wf, "r1")
    (park,) = read_sqlite_parked_conn(engine.conn)
    assert park.wake_event == "gather:0,0;ev0:r1"  # branch 2's park is invisible

    page = client.get(f"/tasks/{task_id}")
    assert "one branch of a gather" in page.text

    # ...and the caveat is not printed for an ordinary park
    plain = _run(engine, "gated", _await_wf("review:r2"), "r2")
    assert "one branch of a gather" not in client.get(f"/tasks/{plain}").text


def _clean_gather_wf(run_id: str):
    """Two branches with hand-disambiguated ids — the shape every in-branch append in this tree
    already uses, and the control for the alarm's absence."""

    def branch(tool: str, kind: str):
        def thunk():
            value = yield from call_tool(tool, {}, int)
            yield from append_ledger(LedgerRow(event_id=Key.parse(f"{run_id}:{kind}"), kind=kind))
            return value

        return thunk

    def workflow():
        return (yield from gather([branch("a", "ka"), branch("b", "kb")]))

    return workflow


def test_the_run_page_reports_when_the_tape_and_the_record_disagree():
    """The page RENDERS a disagreement it is handed; producing one is the store's business.

    Driven through `run_page` with a synthesized `Collision` rather than by running a colliding
    workflow, and the reason is the point rather than a convenience: once the store REFUSES a
    placed-writer collision, no workflow can reach this state on a live engine at all. A test
    that manufactured one by running it would have to be deleted the day the refusal landed —
    and, worse, would have been asserting that the substrate still loses rows.

    What remains true and worth pinning is the read-path contract: given a collision, the page
    says so and NAMES the writers, because a reader's next move is to look those keys up. The
    detector that finds them is pinned over real keys in `test_graphview.py`."""
    from effective.dashboard import run_page
    from effective.engines import TaskSnapshot, TaskState
    from effective.graphview import Collision

    page = run_page(
        uuid7(),
        TaskSnapshot(state=TaskState.COMPLETED, result=None, failure=None),
        None,
        None,
        (Collision("r1:done", ("gather:0,0;ledger;r1:done", "gather:0,1;ledger;r1:done")),),
    )
    assert "tape and the canonical record disagree" in page
    # the culprits are NAMED, not just counted — the drill-down the fold promises
    assert "gather:0,0;ledger;r1:done" in page
    assert "gather:0,1;ledger;r1:done" in page


def test_the_run_page_is_silent_when_the_two_bookkeepers_agree(engine, client):
    """The discrimination. Same shape, same frames, distinct authored ids — no alarm."""
    task_id = _run(engine, "clean", _clean_gather_wf, "r2")

    # Anti-vacuity: this test asserts an ABSENCE, so it must first prove the run happened and
    # that both branches really did reach the canonical record.
    rows = sorted(
        event_id
        for (event_id,) in engine.conn.execute(
            "SELECT event_id FROM ledger WHERE workflow_run_id=?", ("r2",)
        )
    )
    assert rows == ["r2:ka", "r2:kb"]

    page = client.get(f"/tasks/{task_id}")
    assert page.status_code == 200
    assert "tape and the canonical record disagree" not in page.text


# ── POST /tasks/{task_id}/answer — emit, then drain ──────────────────────────


def test_answering_resumes_the_run_and_the_park_leaves_the_list(engine, client):
    """The whole round trip, in one process. `follow_redirects=False` keeps this infra-free: the
    redirect target is the run page, which would want the layout container."""
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")
    assert len(read_sqlite_parked_conn(engine.conn)) == 1

    posted = client.post(
        f"/tasks/{task_id}/answer",
        data={"payload": '{"decision": "approve"}'},
        follow_redirects=False,
    )
    assert posted.status_code == 303
    assert posted.headers["location"] == f"/tasks/{task_id}"

    snapshot = engine.fetch_task_result(task_id)
    assert snapshot is not None
    assert snapshot.state == "completed", snapshot
    assert snapshot.result == {"decision": "approve"}
    assert read_sqlite_parked_conn(engine.conn) == ()
    assert "Nothing is parked." in client.get("/").text


def test_answering_a_gather_branch_reveals_the_next_park(engine, client):
    """The L-1 consequence the caveat warns about, pinned as behaviour."""
    task_id = _run(engine, "multi", _two_parked_branches_wf, "r1")

    client.post(f"/tasks/{task_id}/answer", data={"payload": "{}"}, follow_redirects=False)

    (park,) = read_sqlite_parked_conn(engine.conn)
    assert park.wake_event == "gather:0,2;ev2:r1"  # a new park, not a completion


def test_a_payload_that_is_not_json_is_refused_by_name(engine, client):
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")

    refused = client.post(
        f"/tasks/{task_id}/answer", data={"payload": "not json"}, follow_redirects=False
    )
    assert refused.status_code == 400
    assert "not json" in refused.text
    assert len(read_sqlite_parked_conn(engine.conn)) == 1  # still parked, nothing emitted


def test_answering_a_task_that_is_not_parked_is_a_404(engine, client):
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")
    engine.emit_event("review:r1", {})
    engine.run_until_result(task_id)

    posted = client.post(f"/tasks/{task_id}/answer", data={"payload": "{}"})
    assert posted.status_code == 404


# ── the drawn graph (needs `just elk-image`) ─────────────────────────────────

needs_elk = pytest.mark.skipif(
    not elk_engine.available(),
    reason="no pinned elkjs worker — run `just elk-image` (or set EFFECTIVE_ELK_HOST=1)",
)


@needs_elk
def test_the_run_page_renders_the_parked_node_as_a_hexagon_carrying_its_node_id(engine, client):
    """The whole read path, end to end: reader -> `pending_key` -> `from_keys(pending=…)` ->
    `prepare` -> ELK -> SVG, server-rendered and inlined, with no JavaScript anywhere.

    `data-node-id="event;{wake_event}"` is the pending node's spelling, and `<polygon>` is what
    makes it a hexagon, the one shape that means "the run can stop here"."""
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")

    page = client.get(f"/tasks/{task_id}")
    assert page.status_code == 200
    assert "<svg" in page.text
    assert 'data-node-id="event;review:r1"' in page.text
    assert "ev-g-node--parked" in page.text
    assert "<polygon" in page.text
    assert "<script" not in page.text
    elk_engine.shutdown()


@needs_elk
def test_a_finished_run_draws_no_pending_node(engine, client):
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")
    engine.emit_event("review:r1", {})
    engine.run_until_result(task_id)

    page = client.get(f"/tasks/{task_id}")
    assert "<svg" in page.text
    assert "data-node-id=" in page.text
    assert 'data-node-id="event;review:r1"' not in page.text
    elk_engine.shutdown()


@pytest.mark.skipif(elk_engine.available(), reason="the layout engine IS available here")
def test_without_a_layout_engine_the_page_names_the_command_that_builds_one(engine, client):
    """A missing graph with no culprit named is the dead end the DX doctrine treats as a defect."""
    task_id = _run(engine, "gated", _await_wf("review:r1"), "r1")

    page = client.get(f"/tasks/{task_id}")
    assert page.status_code == 200
    assert "just elk-image" in page.text


# ── the telemetry join, rendered ─────────────────────────────────────────────

_TELEMETRY_STEP = 0.25
"""Seconds per `clock()` tick. `traced` calls the clock twice per op — once before the yield and
once after — so each op's span is exactly one step, and the SVG's `{ns / 1e6:.0f}ms` is a fixed
string rather than whatever the machine happened to be doing."""


def _stepped_clock():
    ticks = count(0.0, _TELEMETRY_STEP)
    return lambda: next(ticks)


def _run_traced(app: SqliteApp, name: str, factory, run_id: str, sidecar: Path) -> UUID:
    """`_run`, with `traced` COMPOSED INTO THE DOMAIN and its spans written to `sidecar`.

    The domain seam, not `op_layers=`: a domain layer installed at the op seam would see `Step`
    and never the tool call it is meant to observe."""

    @app.register_task(name)
    def task(params, ctx):
        rid = params["run_id"]
        ledger = SqliteLedger(app.conn, rid, app.write_lock)
        layer = traced(otlp_jsonl_sink(sidecar), session_id=rid, clock=_stepped_clock())
        domain = compose_domain((layer,), _Tools())
        return DurableHandler(ctx, domain, ledger=ledger).run(factory(rid))

    task_id = app.spawn(name, {"run_id": run_id})
    app.run_until_result(task_id)
    return task_id


@pytest.mark.skipif(not elkjs.available(), reason="no pinned elkjs worker (run `just elk-image`)")
def test_the_run_page_reports_a_nodes_duration_when_a_sidecar_is_given(engine, tmp_path):
    """End to end: a span written by a worker reaches the node it observed, by key.

    Nothing in this path correlates. The sidecar row carries `effective.key`, the projection is
    built from the checkpoint keys, and the two meet because they are the SAME address: the key is
    the join column."""
    sidecar = tmp_path / "spans.jsonl"
    park = approve_park(Key.parse("ledger;processed:m1"))
    task_id = _run_traced(engine, "traced", _await_wf(park), "r-tel", sidecar)

    with TestClient(dashboard(engine, sidecar)) as client:
        page = client.get(f"/tasks/{task_id}").text

    assert f"{int(_TELEMETRY_STEP * 1000)}ms" in page  # the tool node's measured wall-clock


@pytest.mark.skipif(not elkjs.available(), reason="no pinned elkjs worker (run `just elk-image`)")
def test_the_run_page_renders_unmeasured_without_a_sidecar(engine, tmp_path):
    """`spans=None` keeps today's behaviour exactly — the page still renders, and reports no
    duration, because a node with no span is the ordinary left-outer miss rather than an error.

    ANTI-VACUITY: the same run, the same page, and the string is present in the test above. If
    this one passed because the graph failed to render at all, that one would have failed too."""
    sidecar = tmp_path / "spans.jsonl"
    park = approve_park(Key.parse("ledger;processed:m1"))
    task_id = _run_traced(engine, "traced", _await_wf(park), "r-tel", sidecar)

    with TestClient(dashboard(engine)) as client:  # no sidecar handed over
        page = client.get(f"/tasks/{task_id}").text

    assert f"{int(_TELEMETRY_STEP * 1000)}ms" not in page
    assert "svg" in page  # the graph DID render; the absence above is the telemetry, not a failure


@pytest.mark.skipif(not elkjs.available(), reason="no pinned elkjs worker (run `just elk-image`)")
def test_a_sidecar_path_that_does_not_exist_yet_is_not_an_error(engine, tmp_path):
    """Telemetry is optional and a worker may not have written yet. A dashboard pointed at a file
    that has not appeared must render the run, not 500."""
    park = approve_park(Key.parse("ledger;processed:m1"))
    task_id = _run(engine, "gated", _await_wf(park), "r-missing")

    with TestClient(dashboard(engine, tmp_path / "never-written.jsonl")) as client:
        response = client.get(f"/tasks/{task_id}")

    assert response.status_code == 200


@pytest.mark.skipif(not elkjs.available(), reason="no pinned elkjs worker (run `just elk-image`)")
def test_a_sidecar_holding_two_runs_renders_unmeasured_rather_than_wrong(engine, tmp_path):
    """The refusal reaching the surface that cannot answer the question it asks.

    `sidecar_measurements` refuses an unscoped read of a multi-run file, because checkpoint keys
    are not task-scoped: two runs of one workflow address the same node by construction, so
    folding them puts another run's seconds on this run's figure. The dashboard has no way to name
    the run it wants — a span's `gen_ai.conversation.id` is whatever the worker passed to
    `traced`, not this `task_id` — so it takes the refusal as "no measurement available", which
    is what `None` has always meant here and what every node rendered before a producer existed.

    Pinned because both other answers are tempting and both are worse. Summing is the wrong number
    (measured elsewhere: a 250 ms run reporting 4250 ms). Letting the refusal escape 500s a page
    whose parked-run content is perfectly readable with no cost on it at all.

    ANTI-VACUITY: the duration string this asserts absent is the one
    `test_the_run_page_reports_a_nodes_duration_when_a_sidecar_is_given` asserts PRESENT, from the
    same helper and the same clock. If this passed because the graph never rendered, that one
    would fail."""
    sidecar = tmp_path / "spans.jsonl"
    first = _run_traced(
        engine, "mixed-a", _await_wf(approve_park(Key.parse("ledger;a:1"))), "r-a", sidecar
    )
    _run_traced(
        engine, "mixed-b", _await_wf(approve_park(Key.parse("ledger;b:1"))), "r-b", sidecar
    )

    with TestClient(dashboard(engine, sidecar)) as client:
        page = client.get(f"/tasks/{first}").text

    assert f"{int(_TELEMETRY_STEP * 1000)}ms" not in page


@pytest.mark.skipif(not elkjs.available(), reason="no pinned elkjs worker (run `just elk-image`)")
def test_a_sidecar_in_an_older_format_renders_unmeasured_rather_than_500ing(engine, tmp_path):
    """An older span file is refused by the reader, and the page treats the refusal as "no
    measurement available", as it does a file holding two runs."""
    sidecar = tmp_path / "spans.jsonl"
    sidecar.write_text(json.dumps({"spanId": "a", "attributes": {"effective.key": "x"}}) + "\n")
    task_id = _run_traced(
        engine, "older", _await_wf(approve_park(Key.parse("ledger;old:1"))), "r-old", sidecar
    )

    with TestClient(dashboard(engine, sidecar), raise_server_exceptions=False) as client:
        response = client.get(f"/tasks/{task_id}")

    assert response.status_code == 200
    assert f"{int(_TELEMETRY_STEP * 1000)}ms" not in response.text
