"""The local dashboard: see the parked fleet, see one run, answer its park.

**One FastAPI app, one process, one SQLite engine.** A drain is `work_batch` *in a process that
holds the task registry*: a page host without the registry would claim the answered task, burn an
attempt per click, and fail the run permanently. `SqliteApp` holds the engine, the registry and
the connection together, so `dashboard(engine)` takes exactly one argument and answering a park
resumes it in the same breath. `SqliteApp._claim` also refuses to claim a name it did not
register, so the engine closes the same hole; this app never opens a second drain path.

**Everything here is a composition of existing read functions**: `effective.parked`,
`effective.checkpoints`, `effective.graphview` and `effective.graphlayout`. The whole run view
is two lines:

    graph = prepare(from_keys(task_id, keys, pending=pending_key(park)))
    svg = to_svg(graph, layout(graph))

so a Shiny app, a card surface or an MCP-app resource reaches the same picture without going
through anything here.

Hosting it is the engine you already have, plus uvicorn. This module has no entry point, because
the task registry belongs to the application that defines the workflows::

    engine = SqliteApp("engine.db")
    engine.register_task("intake")(intake)          # your workflows
    uvicorn.run(dashboard(engine))                  # one process: page + engine + registry

Deliberately absent: any injected-drain or layout-engine seam, an Absurd/Postgres mode,
JavaScript of any kind, auth, polling. A page load is the refresh.

**Route vocabulary: `/tasks/{task_id}`.** The path segment is a *task* id, what both readers are
keyed by; `run_id` is an application convention living in `tasks.params`, with no column.
`ParkedTask` is named for the same reason: a `/runs/` URL would assert a key the substrate does
not have, and a pasted `run_id` would 404 with no explanation. The graph is still a *run* graph:
`RunGraph.run_id` is filled with the task id here.
"""

import json
from pathlib import Path
from string.templatelib import Template
from urllib.parse import parse_qs
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from tdom import Markup, html

from effective.checkpoints import keys, read_sqlite_conn
from effective.engines.sqlite import SqliteApp, TaskSnapshot
from effective.graphlayout import GRAPH_CSS, elkjs, prepare, to_svg
from effective.graphview import Collision, branch_path, from_keys, ledger_collisions
from effective.parked import ParkedTask, pending_key, read_sqlite_parked_conn
from effective.parked import answer as answer_park
from effective.telemetry import MixedSessions, NotASpanFile, sidecar_measurements

PAGE_CSS = """
body { margin: 0 auto; max-width: 62rem; padding: 2rem 1rem;
       font: 15px/1.55 ui-sans-serif, system-ui, sans-serif; color: #1e2a2e; }
a { color: #2a5d6e; }
h1 { font-size: 1.35rem; margin: 0 0 1.25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .6rem; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #dde3e3;
         vertical-align: top; }
th { font-weight: 600; color: #4b5a5f; font-size: .82rem; text-transform: uppercase;
     letter-spacing: .04em; }
code, .ev-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
.ev-absent { color: #8a9799; font-style: italic; }
.ev-note { background: #fdf5e6; border-left: 3px solid #c07408; padding: .7rem .9rem;
           margin: 1rem 0; }
.ev-empty { color: #4b5a5f; }
.ev-alarm { background: #fdeaea; border-left: 3px solid #b3261e; padding: .7rem .9rem;
            margin: 1rem 0; }
.ev-alarm-list { margin: -.6rem 0 1rem; padding-left: 2.2rem; }
form { margin: 1rem 0; }
textarea { width: 100%; min-height: 4.5rem; font-family: ui-monospace, monospace; font-size: .9em;
           padding: .5rem; border: 1px solid #b9c4c4; border-radius: 4px; }
button { margin-top: .6rem; padding: .45rem 1.1rem; font-size: .95rem; cursor: pointer; }
figure { margin: 1rem 0; overflow-x: auto; }
"""
"""Page chrome only. The graph's own styling is `GRAPH_CSS`, which the fragment expects the host
page to carry — the contract `effective.graphlayout.svg` states, honoured here rather than
re-implemented."""

DEFAULT_PAYLOAD = "{}"
"""What the answer form is pre-filled with. Answer payloads are free-form JSON: the await's own
`schema` type is not surfaced, so the form is not derived from it."""


# ── the two pages, as pure functions of what the readers returned ────────────


def _page(title: str, body: Template) -> str:
    """The shell. `body` is a `Template` (or a list of them) so it SPLICES; a `str` would be
    escaped as text, which is the property that keeps markup from being built by concatenation
    (`effective.graphlayout.svg`'s third rule, one grammar up)."""
    return html(t"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{Markup(PAGE_CSS)}</style>
<style>{Markup(GRAPH_CSS)}</style>
</head><body>{body}</body></html>""")


def _since(park: ParkedTask) -> Template:
    """`parked_since`, or the honest absence.

    SQLite's `tasks` table records no park timestamp at all (`ParkedTask.parked_since`), so this
    column is empty on every row this app serves today. Rendering "now", or dropping the column,
    would both invent a fact — the absence is what the engine knows."""
    if park.parked_since is None:
        return t'<span class="ev-absent">not recorded by this engine</span>'
    return t"<span>{park.parked_since.isoformat(sep=' ', timespec='seconds')}</span>"


def fleet_page(parked: tuple[ParkedTask, ...]) -> str:
    """The fleet list: every task waiting on an event, oldest first, each linking to its run."""
    if not parked:
        body = t'<h1>Parked runs</h1><p class="ev-empty">Nothing is parked.</p>'
        return _page("Parked runs", body)
    rows = [
        t"""<tr>
          <td><a href="/tasks/{park.task_id}"><code>{park.task_id}</code></a></td>
          <td><code>{park.task_name}</code></td>
          <td><code>{park.wake_event}</code></td>
          <td>{_since(park)}</td>
        </tr>"""
        for park in parked
    ]
    return _page(
        "Parked runs",
        t"""<h1>Parked runs <span class="ev-absent">({len(parked)})</span></h1>
        <table><thead><tr><th>task</th><th>workflow</th><th>waiting on</th>
        <th>parked since</th></tr></thead><tbody>{rows}</tbody></table>""",
    )


def _gather_caveat(park: ParkedTask) -> Template | str:
    """The one thing this page must SAY rather than imply.

    A gather with N parked branches registers exactly one wake event, the lowest branch index
    (`_join`'s deterministic re-arm), so the reader reports 1 park where N are pending, and a
    successful answer surfaces a *new* park instead of finishing the run. Wakes are serialized
    by design, and that looks exactly like a bug from the page, so the page says it. Detected
    from the branch coordinate the engines carry INSIDE the awaited name (`pending_key`'s note):
    a park inside a gather is `gather:{g},{i};…`."""
    if not branch_path(park.wake_event):
        return ""
    return t"""<p class="ev-note"><strong>This park is one branch of a gather.</strong>
      A gather registers a wake event for only its lowest-indexed parked branch, so sibling
      branches may also be waiting and are not listed anywhere — and answering this one may
      reveal the next branch's park rather than finish the run. The engine is behaving
      correctly; the reader can only see one at a time.</p>"""


def _writer_items(collision: Collision) -> list[Template]:
    """The culprit keys, one `<li>` each — a `Template` per writer, nested under its collision.

    Nesting rather than flattening because the DATA nests: a collision has writers. `tdom.html`
    splices a `Template`, a list of them, or a list of lists, and escapes every non-`Template`
    value it reaches — so the structure is free to mirror the shape of what it describes, and the
    escaping is a property of the composition rather than of remembering to call something.
    A key rendered here is author-controlled text and lands as text."""
    return [t"<li><code>{writer}</code></li>" for writer in collision.writers]


def _collision_alarm(collisions: tuple[Collision, ...]) -> Template | str:
    """The two bookkeepers disagreeing, said out loud.

    This page draws the CHECKPOINT bookkeeper, and for a placed-writer collision the tape is
    *correct* — two disjoint keys, faithfully rendered — while the canonical ledger holds one row.
    So the graph above is not wrong, and that is exactly the problem: it is confidently right about
    the bookkeeper that did not lose anything. Naming the culprits is the point — `writers` are the
    keys the reader can go look up, which is the drill-down the fold promises.

    Deliberately phrased as what was OBSERVED rather than as a verdict on the ledger: this reads
    one bookkeeper and infers about the other, and CLAUDE.md forbids deriving one from the other.
    A run whose store already refuses the collision cannot reach this state at all."""
    if not collisions:
        return ""
    rows = [
        t"""<li><code>{collision.event_id}</code> — written by {collision.count}
          placed ops:<ul>{_writer_items(collision)}</ul></li>"""
        for collision in collisions
    ]
    return t"""<p class="ev-alarm"><strong>The tape and the canonical record disagree.</strong>
      {len(collisions)} authored <code>event_id</code>(s) below were written by more than one
      placed op in this run. The ledger is <code>UNIQUE(event_id)</code>, so it holds ONE row for
      each — every later write was dropped by <code>ON CONFLICT DO NOTHING</code> and the run
      still reported success. The graph above draws the checkpoints, which are correct and
      complete; it is the canonical record that is short.</p>
    <ul class="ev-alarm-list">{rows}</ul>"""


def run_page(
    task_id: UUID,
    snapshot: TaskSnapshot,
    park: ParkedTask | None,
    svg: str | None,
    collisions: tuple[Collision, ...] = (),
) -> str:
    """One run: its graph, and — when it is parked — the form that answers the park.

    `svg` is `None` when no layout engine is reachable. The page then says which command builds
    it, because "the graph is missing" with no culprit named is the kind of dead end the DX
    doctrine treats as a defect."""
    graph = (
        t"<figure>{Markup(svg)}</figure>"
        if svg is not None
        else t"""<p class="ev-note">No layout engine on this machine, so the graph is not drawn.
          Build the pinned, sandboxed renderer with <code>just elk-image</code> (or set
          <code>EFFECTIVE_ELK_HOST=1</code> after <code>just elk-setup</code>).</p>"""
    )
    if park is None:
        action = t"""<p class="ev-empty">This run is <code>{snapshot.state}</code> — not waiting
          on anything, so there is nothing to answer.</p>"""
    else:
        action = t"""{_gather_caveat(park)}
        <h2>Answer the park</h2>
        <p>Waiting on <code>{park.wake_event}</code> — emitted verbatim, exactly as the run
          registered it.</p>
        <form method="post" action="/tasks/{park.task_id}/answer">
          <label for="payload">Payload (JSON)</label>
          <textarea id="payload" name="payload">{DEFAULT_PAYLOAD}</textarea>
          <button type="submit">Emit and drain</button>
        </form>"""
    return _page(
        f"Run {task_id}",
        t"""<p><a href="/">&larr; parked runs</a></p>
        <h1>Task <code>{task_id}</code></h1>
        <p>State: <code>{snapshot.state}</code>{
            t" — failed: {snapshot.failure}" if snapshot.failure else ""
        }</p>
        {_collision_alarm(collisions)}
        {graph}
        {action}""",
    )


# ── the app ──────────────────────────────────────────────────────────────────


def _parked(engine: SqliteApp) -> tuple[ParkedTask, ...]:
    """Every parked run, read under the engine's write lock.

    **The lock is the point.** These readers take a raw `sqlite3.Connection`, so unlike the
    engine's own methods they have no way to reach `write_lock` themselves — and both GET routes
    below are plain `def`, which means FastAPI runs them on the threadpool, concurrently with
    `POST /answer` draining through `run_in_threadpool`. Two threadpool threads, one connection.
    Measured at exactly that seam: 149 reader errors over 600 page-loads against a 400-task drain
    (`InterfaceError`, short row unpacks, a `None` where a name belongs). The drain itself was
    unharmed and every task completed, so the damage is a 500 on a page load rather than
    corrupted durable state — which is why this is a bug in the dashboard and not in the engine.

    A second READ-ONLY connection would be the better answer, and it is what `enable_wal` says
    WAL was turned on to make viable — but it needs a path to reopen, and this module is handed
    an already-constructed engine that may be `:memory:`. The lock works for both and costs a
    page load serialized against a drain, which a local ops page can afford.

    Never call this from inside the lock, and never wrap a call that takes it itself
    (`fetch_task_result`, `emit_event`, `run_until_result` all do): `write_lock` is a plain
    `threading.Lock`, so re-entering it deadlocks.
    """
    with engine.write_lock:
        return read_sqlite_parked_conn(engine.conn)


def _park_of(engine: SqliteApp, task_id: UUID) -> ParkedTask | None:
    return next((p for p in _parked(engine) if p.task_id == task_id), None)


def dashboard(engine: SqliteApp, spans: Path | None = None) -> FastAPI:
    """The app, over one already-constructed `SqliteApp`: engine, registry and connection.

    There is no drain parameter: the only correct drain is this engine's own `work_batch`, so a
    parameter for it would be a hook with exactly one legal argument. `spans` is a different
    shape. Telemetry is the THIRD bookkeeper and lives somewhere else: spans are high-volume and
    disposable, and their store is open (a JSONL sidecar, Postgres, or a collector; the join is
    a key either way). The engine cannot know which, because nothing writes spans into it, so
    the caller supplies the path.

    With `spans=None` every node renders unmeasured. Pass the span file `otlp_jsonl_sink` wrote
    and the run view gains a cost and a duration per node, read at request time, so a growing
    file is picked up without a restart."""
    app = FastAPI(title="Effective — parked runs")

    @app.get("/", response_class=HTMLResponse)
    def fleet() -> str:
        return fleet_page(_parked(engine))

    @app.get("/tasks/{task_id}", response_class=HTMLResponse)
    def run(task_id: UUID) -> str:
        if (snapshot := engine.fetch_task_result(task_id)) is None:
            raise HTTPException(404, f"no task {task_id!r} in this engine")
        park = _park_of(engine, task_id)
        # `exclude=()` names the VIEW projection: keep the engine's own suspension bookkeeping,
        # because the await is the node a reader is looking for. SQLite writes none of those
        # rows, so it is inert here; spelling it anyway keeps the choice visible for the engine
        # where it is not.
        # Under the lock for the same reason as `_parked`, and taken here rather than around
        # the whole route: `fetch_task_result` above locks internally, so a route-wide `with`
        # would re-enter a non-reentrant lock and deadlock.
        with engine.write_lock:
            record = read_sqlite_conn(engine.conn, task_id, exclude=())
        # `str(task_id)`: a NAMED exit, where the id stops being an identity and becomes a label.
        # `RunGraph.run_id` is drawn (the SVG's `data-run-id`) and serialized into the ELK
        # request, which is JSON — so this is the display/wire form, not the identity, and the
        # conversion belongs at the boundary rather than in the type. The `run_id`-holding-a-task
        # -id mismatch this line also carries is the open identity decision noted in the module
        # docstring, and is deliberately not settled here.
        # Re-read per request rather than at construction: the sidecar is appended to by a
        # worker, and a dashboard that cached it would show a run's cost frozen at boot.
        # A missing file is not an error — telemetry is optional, and a node with no span is
        # the normal left-outer miss.
        # A MULTI-RUN SIDECAR RENDERS UNMEASURED RATHER THAN WRONG, and rather than 500ing.
        # `sidecar_measurements` refuses an unscoped read of a file holding several runs, because
        # checkpoint keys are not task-scoped and folding them would put another run's seconds on
        # this run's node. The dashboard cannot name the run to want (a span's run id comes
        # from whatever the worker passed to `traced` and is not this `task_id`), so it takes the
        # refusal as "no measurement available", which is exactly what `None` already means here
        # and what every node rendered before a producer existed. Wrong numbers are the thing this
        # projection work exists to avoid; a blank is honest.
        telemetry = None
        if spans is not None and spans.exists():
            try:
                telemetry = sidecar_measurements(spans)
            except MixedSessions, NotASpanFile:
                telemetry = None
        run_graph = from_keys(
            str(task_id),
            keys(record),
            pending=pending_key(park) if park else None,
            telemetry=telemetry,
        )
        # Asked of the UNFOLDED graph, before `prepare` — a folded view has already discarded the
        # member keys the alarm names as culprits.
        collisions = ledger_collisions(run_graph)
        graph = prepare(run_graph)
        # A subprocess probe per page load, which a local dev page can afford and which keeps the
        # "no silent unsandboxed fallback" rule that `available()` exists to enforce.
        svg = to_svg(graph, elkjs.layout(graph)) if elkjs.available() else None
        return run_page(task_id, snapshot, park, svg, collisions)

    @app.post("/tasks/{task_id}/answer")
    async def answer(task_id: UUID, request: Request) -> RedirectResponse:
        if (park := _park_of(engine, task_id)) is None:
            raise HTTPException(404, f"task {task_id!r} is not parked on an event")
        # `parse_qs` rather than FastAPI's `Form(...)`: that one requires `python-multipart`,
        # which reaches this venv only as a transitive of `shiny` and is not a declared runtime
        # dependency here. A form body is `application/x-www-form-urlencoded` and the stdlib
        # parses it — no dependency question to answer.
        fields = parse_qs((await request.body()).decode())
        raw = (fields.get("payload") or ["{}"])[0]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"payload is not JSON ({exc}): {raw!r}") from exc

        # Emit on the RAW registration the reader returned — never on a reconstructed name;
        # then drain, in this process, which holds the registry. Without the drain the click has
        # no visible consequence: emitting an event does not resume a run.
        #
        # **Off the event-loop thread**, for correctness. A workflow is
        # colorless and the sync/async color lives in the HANDLER: `DurableHandler` runs a
        # concurrent `gather` through `asyncio.run` (`handlers/durable.py`), which is illegal on a
        # thread that already has a running loop. Draining directly inside this `async def` fails
        # every gather-bearing run with "asyncio.run() cannot be called from a running event
        # loop", silently to the caller, because `work_batch` catches it
        # and retries the task to death. The two GET routes are plain `def`, so FastAPI already
        # runs them in the threadpool; this is the same placement, spelled explicitly.
        def resume() -> None:
            # `answer(engine, park, …)` rather than `engine.emit_event(task_id, park.wake_event,
            # …)`: this route already HOLDS the `ParkedTask` the reader returned, so the verb that
            # takes one is the one to call. It also stops this line from spelling the addressed
            # engine's 3-argument signature, which the delivery-parity change removes.
            answer_park(engine, park, payload)
            engine.run_until_result(task_id)

        await run_in_threadpool(resume)
        return RedirectResponse(f"/tasks/{task_id}", status_code=303)

    return app


__all__ = ["DEFAULT_PAYLOAD", "PAGE_CSS", "dashboard", "fleet_page", "run_page"]
