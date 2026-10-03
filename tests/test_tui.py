"""The terminal surface (`src/tui`), driven rather than read.

Textual ships a pilot (`App.run_test()`), so these are real app runs against a real SQLite store:
mount, refresh on a worker, redraw. That matters more here than usual, because the two defects
this file was written after were both invisible to reading — a key binding swallowed by a focused
`Input`, and a ledger join on the wrong id — and both are the kind a screenshot would have caught
in a second.

**What is NOT tested here is shaping**, because none of it lives here. The tree's text comes from
`graphview.to_text`, the joins from `runview.run_view`, the containment from
`grammar.split_frames`; each has its own tests over pure values. If a case here starts asserting a
projection's arithmetic, that is the signal that logic has leaked out of `effective` and into a UI
package that is supposed to hold widgets.
"""

import asyncio
import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from uuid import UUID

import pytest
from _coder_script import FIXED, MODULE, TEST, Deployment, fixing
from _conformance import StubbornSuiteDomain, coding_machine_wf
from textual.widgets import Input, Static, Tree

from effective import runread
from effective.api import await_event, compose_key
from effective.cost import MeteredInterpreter, Usage
from effective.handlers.absurd import DurableHandler
from effective.handlers.base import op_key
from effective.keys import Name
from effective.ops import StoreArtifact
from effective.runs import RunState
from effective.runview import to_markdown
from effective.sqlite import SqliteApp, SqliteLedger
from tui.app import RunViewApp, _summary

pytestmark = pytest.mark.journey

COMMITTED = op_key(
    StoreArtifact({MODULE: FIXED, "test_mod.py": TEST}, "application/json")
).display()
"""The postamble's artifact node: the tree with the bug fixed, content-addressed."""


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A completed coder run on disk, the machine a user opens this view over.

    Spawned with a `run_id` in its params: the ledger is keyed by `workflow_run_id`, a convention
    living there rather than the engine's task id, so a fixture without it would render a tree and
    exercise none of the ledger join. The model is scripted and nothing is metered.
    """
    db = tmp_path / "task.db"
    app = SqliteApp(str(db))

    @app.register_task("coder")
    def task(params, ctx):
        return DurableHandler(
            ctx, Deployment(), ledger=SqliteLedger(app.conn, params["run_id"], app.write_lock)
        ).run(lambda: fixing(params["run_id"]))

    app.run_until_result(app.spawn("coder", {"run_id": "r-demo"}))
    app.close()
    return db


@pytest.fixture
def machine_store(tmp_path: Path) -> Path:
    """A coding machine run over five visits, DRAFT twice, which the coder's one visit cannot show:
    outer visits with different state names, and a state entered again."""
    db = tmp_path / "machine.db"
    app = SqliteApp(str(db))

    @app.register_task("coding")
    def task(params, ctx):
        return DurableHandler(
            ctx,
            StubbornSuiteDomain(),
            ledger=SqliteLedger(app.conn, params["run_id"], app.write_lock),
        ).run(lambda: coding_machine_wf(params["run_id"]))

    snapshot = app.run_until_result(app.spawn("coding", {"run_id": "r-machine"}))
    app.close()
    assert snapshot is not None
    assert snapshot.result["path"] == VISITS
    return db


VISITS = ["test", "draft", "draft", "finalize", "review"]


def test_each_visit_is_its_own_level_and_a_projection_folds_them(machine_store):
    """Outer visits stay apart in the unrolled tree, in run order, and fold to one position under
    `project`, where a state entered twice is drawn once."""

    async def drive():
        app = RunViewApp(machine_store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            labels = _labels(app.query_one("#tree", Tree).root)
            visits = [label for label in labels if label.startswith("d:")]
            states = [label for label in labels if label.startswith("  state:")]
            assert visits == ["d:0", "d:1", "d:2", "d:3", "d:4"]
            assert states == [
                "  state:test",
                "  state:draft",
                "  state:draft",
                "  state:finalize",
                "  state:review",
            ]

            await pilot.press("p")
            for _ in range(100):
                labels = _labels(app.query_one("#tree", Tree).root)
                if labels[0] == "d":
                    break
                await pilot.pause(0.05)
            assert labels[0] == "d"
            assert [label for label in labels if label.startswith("  state:")] == [
                "  state:test",
                "  state:draft",
                "  state:finalize",
                "  state:review",
            ]

    asyncio.run(drive())


def _leaves(node):
    """Every leaf under a tree node, depth-first — the ones carrying a `Node` in `data`."""
    for child in node.children:
        if child.children:
            yield from _leaves(child)
        else:
            yield child


def _labels(node, depth: int = 0) -> list[str]:
    out = []
    for child in node.children:
        out.append("  " * depth + str(child.label))
        out.extend(_labels(child, depth + 1))
    return out


async def _mounted(app: RunViewApp, pilot) -> None:
    """Wait for the first refresh worker to deliver. Polling, because it runs on a thread."""
    for _ in range(100):
        if app.view is not None:
            return
        await pilot.pause(0.05)
    raise AssertionError("the refresh worker never delivered a view")


def test_the_tree_draws_the_run_the_way_it_ran(store):
    """Mount, read, redraw — and the frames become tree levels.

    The assertion is the whole shape rather than a spot check, because the failure worth catching
    is a level in the wrong place: a frameless postamble hoisted above the visits, or a leaf
    ordered by label instead of by run position.
    """

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            assert _labels(app.query_one("#tree", Tree).root) == [
                "d:0",
                "  state:work",
                "    d:0",
                "      step;react:turn",
                "      step;tool:bash",
                "    d:1",
                "      step;react:turn",
                "      step;tool:read",
                "    d:2",
                "      step;react:turn",
                "      step;tool:edit",
                "    d:3",
                "      step;react:turn",
                "      step;tool:edit",
                "    d:4",
                "      step;react:turn",
                "      step;tool:edit",
                "    d:5",
                "      step;react:turn",
                "    step;tool:run_suite",
                COMMITTED,
                "step;tool:run_suite",
                "ledger;machine:r-demo;commit",
                "ledger;machine:r-demo",
            ]

    asyncio.run(drive())


def test_a_single_key_binding_is_not_swallowed_by_the_command_line(store):
    """The defect that reading could not see, pinned.

    Textual focuses the first focusable widget on mount, and with an `Input` in the layout that
    was the command line — so `p` typed a "p" and the projection never changed, while the
    `BINDINGS` were declared correctly and the `Footer` advertised them. Every static check passed
    and the app was wrong.

    Asserting the projection CHANGED is what makes this a test of focus rather than of the action:
    calling `action_cycle_projection` directly would pass with the bug fully present.
    """

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            assert app.projection == "unrolled"
            await pilot.press("p")
            for _ in range(100):
                if app.projection == "project":
                    break
                await pilot.pause(0.05)
            assert app.projection == "project", "the key never reached the binding"

            # And the redraw really applied the projection: six turns fold to one position.
            for _ in range(100):
                labels = _labels(app.query_one("#tree", Tree).root)
                if "    d" in labels:
                    break
                await pilot.pause(0.05)
            assert labels == [
                "d:0",
                "  state:work",
                "    d",
                "      step;react:turn  (x6)",
                "      step;tool:bash",
                "      step;tool:read",
                "      step;tool:edit  (x3)",
                "    step;tool:run_suite",
                COMMITTED,
                "step;tool:run_suite",
                "ledger;machine:r-demo;commit",
                "ledger;machine:r-demo",
            ]

    asyncio.run(drive())


def test_the_status_line_reports_cost_as_UNMEASURED_rather_than_zero(store):
    """A run nobody metered is not a free run.

    The same claim `Node.cost` makes, arriving on the one line a reader actually looks at. A
    status bar showing `$0.0000` for an unmetered run is the field's whole failure mode, one
    rendering step later.
    """

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            status = str(app.query_one("#status", Static).render())
            assert "cost unmeasured" in status
            assert "$" not in status
            # 16, derived rather than observed: six turns, five tool calls, the judge's suite, and
            # the postamble's artifact, suite and two rows.
            assert "16 ops" in status

    asyncio.run(drive())


def test_the_ledger_join_uses_the_RUN_id_not_the_task_id(store):
    """A task and a run are different things, and this join is where the difference bites.

    `workflow_run_id` is a convention living in the spawn params; `task_id` is the engine's
    identity. Joining on the wrong one raises nothing: it returns no rows, and the pane just looks
    like a run that changed no files while rendering a complete and plausible tree.
    """
    (run,) = runread.runs(store)
    task_id = run.task_id
    assert run.run_id == "r-demo"
    assert str(task_id) != "r-demo"

    # `closing`, not a bare `with`: a `sqlite3.Connection` context manager commits the
    # transaction and does NOT close the connection. This test leaked one under
    # `-W error::ResourceWarning` — the same misreading the module under test had.
    with closing(sqlite3.connect(store)) as conn:
        (payload,) = conn.execute("SELECT payload FROM ledger").fetchone()
    assert json.loads(payload)["files"] == ["mod.py", "test_mod.py"]
    assert runread.view(store, task_id).files == ("mod.py", "test_mod.py")


def test_the_surface_registers_no_task_so_it_cannot_claim_one(store):
    """The structural half of "a client is not a drain".

    A page host without a registry once emitted a park's answer and called `work_batch()` per
    click; the human approved and the run failed permanently while every per-task check passed.
    `SqliteApp._claim` refuses a name it did not register, so a surface that registers nothing
    cannot reach that failure however it is driven.

    Asserted over the module's SOURCE rather than over behaviour, because the property is an
    absence: no run of the app can demonstrate that it never registers a task, and a behavioural
    test would pass right up until someone added one.
    """
    source = (Path(__file__).parent.parent / "src" / "tui" / "app.py").read_text()
    assert "register_task" not in source
    assert "work_batch" not in source
    assert "run_until_result" not in source


def test_the_command_line_is_entered_with_a_colon_and_left_with_escape(store):
    """Two modes, one rule each — and both obvious arrangements are broken in one direction.

    Leave the `Input` focused and every single-key binding is swallowed as text. Focus the tree
    instead and the command line is unreachable: `Tab` lands on the parks table, so typing
    `open 2` fires the `p` binding and cycles the projection. Both were measured by driving the
    app under tmux, and neither is visible from reading the widget tree.

    A command also RETURNS to navigation, which is the third thing driving found: staying in the
    input makes the mode sticky, so a second `:` is typed as text rather than re-entering.
    """

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            tree, command = app.query_one("#tree", Tree), app.query_one("#input", Input)
            assert app.focused is tree

            await pilot.press("colon")
            assert app.focused is command, "`:` did not open the command line"

            # In command mode a binding key is TEXT, which is the whole point of having a mode.
            await pilot.press("p")
            assert command.value == "p"
            assert app.projection == "unrolled", "a binding fired while typing"

            await pilot.press("escape")
            assert app.focused is tree, "`escape` did not leave the command line"

            # And a submitted command hands navigation back rather than leaving the mode sticky.
            await pilot.press("colon")
            command.value = "help"
            await pilot.press("enter")
            assert app.focused is tree, "the command line kept focus after submitting"

    asyncio.run(drive())


@pytest.fixture
def failed_store(tmp_path: Path) -> Path:
    """A run whose workflow RAISED — the case that produced this file's newest two tests.

    It commits nothing, which is the trap: a run that died before its first checkpoint and a run
    that genuinely did nothing leave the same tape, and only the engine's task row separates them.
    """
    db = tmp_path / "failed.db"
    app = SqliteApp(str(db))

    @app.register_task("machine")
    def task(params, ctx):
        def boom():
            raise ValueError("the payload was not what the park awaited")
            yield  # pragma: no cover  -- a workflow is a generator

        return DurableHandler(
            ctx, MeteredInterpreter(llm=lambda _op: ({}, Usage()), tools=lambda _op: "ran")
        ).run(boom)

    app.spawn("machine", {"run_id": "r-dead"})
    for _ in range(6):
        if not app.work_batch():
            break
    app.close()
    return db


def test_a_FAILED_run_does_not_render_as_an_empty_one(failed_store):
    """The bug the first real drive found, and it was silent.

    A prose-machine run died on a schema mismatch and the surface drew an empty tree with
    `0 ops · 0 parked` — which is exactly what a run that did nothing looks like. Neither
    bookkeeper can tell them apart: a run that died before committing did nothing, truthfully. The
    driver had to open the database to learn the run was dead at all, which is the drop-out this
    pins against.
    """

    async def drive():
        app = RunViewApp(failed_store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            assert app.status is not None
            assert app.status.state is RunState.FAILED
            assert app.status.failure is not None
            assert "not what the park awaited" in app.status.failure
            rendered = str(app.query_one("#status", Static).render())
            assert rendered.startswith("failed  ·"), rendered
            # NOT `"[failed]" in ...`: a `Static.update` string is Textual content MARKUP, so the
            # bracketed form is consumed as a style tag and the word never reaches the text. That
            # is how the first version of this fix passed nothing and looked like a missing string.
            assert "[failed]" not in rendered

    asyncio.run(drive())


def test_the_status_line_names_the_run_state_on_a_live_run_too(store):
    """The complement, so the assertion above is not satisfiable by hard-coding one word."""

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            rendered = str(app.query_one("#status", Static).render())
            assert rendered.startswith("completed  ·"), rendered
            assert "failed" not in rendered

    asyncio.run(drive())


def test_the_app_renders_the_ledger_join_rather_than_only_testing_it(store):
    """`runread.view` did the join correctly and NOTHING called it.

    `refresh_view` built its own `run_view(graph, parked=…)` with no `ledger=`, so `files` was
    always empty and the "Changed:" line never rendered on any run — a join that existed, was
    tested, and reached no surface. The old test asserted `runread.view(...).files` directly, which
    is exactly the assertion that stays green while the app ignores the function.
    """

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            assert app.view is not None
            assert app.view.files == ("mod.py", "test_mod.py")
            assert app.view.changed == ("mod.py",)
            # and it REACHES a renderer: the markdown floor is what a report, a Shiny board and an
            # MCP host all take unchanged, so a join stopping at `RunView` reaches none of them.
            assert "**Changed:**" in to_markdown(app.view)
            assert _summary(app.view).endswith("**Changed:** `mod.py`")
            assert "test_mod.py" not in _summary(app.view)

    asyncio.run(drive())


def test_the_projection_is_applied_before_the_join_not_after(store):
    """`runread.view(project=…)` exists so the joins are computed from the projected graph.

    A caller that projected afterwards would join against the unrolled graph and throw the result
    away — the same class of quiet wrongness as joining on the task id.
    """
    from effective.graphview import project

    task_id = runread.runs(store)[0].task_id
    unrolled = runread.view(store, task_id)
    projected = runread.view(store, task_id, project=project)
    assert len(projected.nodes) < len(unrolled.nodes)
    assert projected.files == unrolled.files == ("mod.py", "test_mod.py")


def test_selecting_a_node_shows_what_that_op_RECORDED(store):
    """The drop-out that cost the most on the first drive.

    A gate ran, failed, bounced the machine back a state, and the only way to learn WHICH check
    failed was `sqlite3` against the store — while the answer sat in the checkpoint the tree was
    already drawing. Selecting a leaf now renders it.
    """

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            leaf = next(
                n
                for n in _leaves(app.query_one("#tree", Tree).root)
                if n.data is not None and "run_suite" in n.data.key
            )
            assert app.view is not None
            app.selected = leaf.data
            rendered = _summary(app.view, app.selected)
            assert leaf.data.key in rendered
            assert "**exit_code:** `0`" in rendered  # the suite's recorded result, from the tape
            assert "not recorded" not in rendered

    asyncio.run(drive())


def test_a_node_with_no_single_producer_says_so_rather_than_showing_nothing(store):
    """`not recorded` is not the same claim as an empty result.

    A node under `project` represents several ops and the pending node has no producer at all, so
    a lookup MISSES structurally rather than accidentally. Saying "not recorded" is the difference
    between "this op returned nothing" and "this label does not address one op".
    """
    from effective.graphview import Node

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            assert app.view is not None
            phantom = Node(key="step;tool:never_ran", kind="step")
            assert "not recorded" in _summary(app.view, phantom)

    asyncio.run(drive())


def test_the_results_join_is_narrowed_to_the_nodes_actually_drawn(store):
    """A pane holds one run's results, not the whole tape — and a projected view carries no rows
    nothing on screen can address."""
    from effective.graphview import project

    task_id = runread.runs(store)[0].task_id
    unrolled = runread.view(store, task_id)
    projected = runread.view(store, task_id, project=project)
    assert set(unrolled.results) <= {n.key for n in unrolled.nodes}
    assert set(projected.results) <= {n.key for n in projected.nodes}
    assert len(unrolled.results) > len(projected.results)


def test_a_REFUSED_command_keeps_what_was_typed(store):
    """The second drive's cost, pinned.

    A driver answering faster than the drain advances finds `parks` momentarily empty; the command
    is refused, and clearing the input first meant the payload it refused was already gone. Three
    answers were lost that way with only a toast to say so, and retyping a long JSON answer is the
    most expensive thing this surface can ask for.
    """

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            field = app.query_one("#input", Input)
            assert app.parks == ()  # a completed run parks on nothing

            field.value = 'answer {"verdict": "approved"}'
            await pilot.press(":")
            await field.action_submit()
            await pilot.pause()
            assert field.value == 'answer {"verdict": "approved"}'

            field.value = "help"
            await field.action_submit()
            await pilot.pause()
            assert field.value == ""  # a command that ACTED clears

    asyncio.run(drive())


def test_opening_another_run_cannot_answer_the_previous_ones_park(store, monkeypatch):
    """`answer` reads `self.parks[0]`, and between an `open` and the refresh that completes it
    those parks belong to the run just left. Refusing in that window is the same
    "refused rather than routed" rule the machine's own transition follows."""
    from effective.parked import ParkedTask

    async def drive():
        app = RunViewApp(store)
        delivered = _deliveries(app, monkeypatch)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            app.parks = (
                ParkedTask(
                    task_id=runread.runs(store)[0].task_id,
                    task_name="other",
                    wake_event="review:r-other",
                    params={},
                    state="parked",
                ),
            )
            assert app._open("0") is True
            assert app.parks == ()
            await until(pilot, lambda: app._refresh_seq in delivered)

    asyncio.run(drive())


def test_a_multi_line_result_is_readable_rather_than_escaped_onto_one_line(store):
    """The defect that defeated the feature it was pointed at, on its first use.

    The prose machine's READ state runs a caller query so a human can inventory against evidence.
    `json.dumps` escapes every newline, so the evidence arrived as one long line the pane
    truncated — the reader could see that a query had run and not what it found.
    """
    from tui.app import _render_record

    record = {"exit_code": 0, "output": "keys.py:160  class Key:\n  711 candidates\n  USES (287)"}
    rendered = _render_record(record)
    assert "\\n" not in rendered  # not escaped
    assert "711 candidates" in rendered
    assert "USES (287)" in rendered
    assert "**exit_code:** `0`" in rendered  # a short field stays on one line


def test_a_non_dict_result_still_renders(store):
    """A step may record a bare string or a list; the pane must not assume a mapping."""
    from tui.app import _render_record

    assert "ran" in _render_record("ran")
    assert "1" in _render_record([1, 2])


def _deliveries(app: RunViewApp, monkeypatch) -> list[int]:
    """The refresh number of every picture that reached `_render`, drawn or dropped."""
    delivered: list[int] = []
    render = app._render

    def recording(seq, *picture):
        delivered.append(seq)
        render(seq, *picture)

    monkeypatch.setattr(app, "_render", recording)
    return delivered


async def until(pilot, ready) -> None:
    """Yield to the app until `ready()` holds; a refresh lands on the app's own loop."""
    for _ in range(200):
        if ready():
            return
        await pilot.pause(0.01)
    raise AssertionError("the pane never reached the state the test waits for")


def _parked_store(db: Path, runs: list[str]) -> list[UUID]:
    """Runs that each park on their own review, in one store, oldest first."""
    app = SqliteApp(str(db))

    @app.register_task("review")
    def task(params, ctx):
        def wf():
            return (yield from await_event(compose_key(t"review:{Name(params['run_id'])}"), dict))

        return DurableHandler(
            ctx, MeteredInterpreter(llm=lambda _op: ({}, Usage()), tools=lambda _op: None)
        ).run(wf)

    tasks = [app.spawn("review", {"run_id": run}) for run in runs]
    for task_id in tasks:
        snapshot = app.run_until_result(task_id)
        assert snapshot is not None
        assert snapshot.state == "waiting", "the run parks"
    app.close()
    return tasks


def test_a_read_that_lands_late_does_not_replace_the_run_opened_after_it(tmp_path, monkeypatch):
    """A read of the first run is held until the second is open and drawn, then released. The pane
    keeps the second run, and an answer settles the second run's park."""
    db = tmp_path / "parks.db"
    first, second = _parked_store(db, ["r-first", "r-second"])
    held, release = threading.Event(), threading.Event()
    hold = {"armed": False}
    status = runread.status

    def held_status(path, task_id):
        found = status(path, task_id)
        if hold["armed"] and task_id == first:
            hold["armed"] = False
            held.set()
            assert release.wait(5)
        return found

    monkeypatch.setattr(runread, "status", held_status)

    async def drive():
        app = RunViewApp(db, first)
        delivered = _deliveries(app, monkeypatch)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            hold["armed"] = True
            app.refresh_view()
            stale = app._refresh_seq
            await until(pilot, held.is_set)

            opened = next(i for i, run in enumerate(runread.runs(db)) if run.task_id == second)
            assert app._open(str(opened))
            await until(pilot, lambda: app.parks and app.parks[0].task_id == second)
            release.set()
            await until(pilot, lambda: stale in delivered)
            assert (app.task_id, app.parks[0].task_id) == (second, second)

            assert app._command(["answer", "{}"])
            await until(pilot, lambda: runread.parks(db, second) == ())
            await until(pilot, lambda: app._refresh_seq in delivered)
            assert [p.task_id for p in runread.parks(db, first)] == [first]
            assert runread.parks(db, second) == ()

    try:
        asyncio.run(drive())
    finally:
        release.set()


def test_a_read_that_lands_during_shutdown_draws_nothing(tmp_path, monkeypatch):
    """A read is held until shutdown has stopped the app, then released before the screens close:
    its picture is dropped. Textual's shutdown clears `is_running` before it prunes the screens,
    so every delivery that could reach a widget being removed lands after the check."""
    db = tmp_path / "parks.db"
    (first,) = _parked_store(db, ["r-first"])
    held, release = threading.Event(), threading.Event()
    hold = {"armed": False}
    status = runread.status

    def held_status(path, task_id):
        found = status(path, task_id)
        if hold["armed"]:
            hold["armed"] = False
            held.set()
            assert release.wait(5)
        return found

    monkeypatch.setattr(runread, "status", held_status)

    async def drive():
        app = RunViewApp(db, first)
        delivered = _deliveries(app, monkeypatch)
        close_all, to_ui, landed = app._close_all, app._to_ui, threading.Event()

        def delivering(*args, **kwargs):
            try:
                to_ui(*args, **kwargs)
            finally:
                landed.set()

        async def closing():
            assert not app.is_running, "Textual stops the app before it closes the screens"
            release.set()
            assert await asyncio.to_thread(landed.wait, 5)
            await close_all()

        monkeypatch.setattr(app, "_to_ui", delivering)

        monkeypatch.setattr(app, "_close_all", closing)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            hold["armed"] = True
            app.refresh_view()
            await until(pilot, held.is_set)
        return app._refresh_seq, delivered

    seq, delivered = asyncio.run(drive())
    assert seq not in delivered


def test_an_answer_refuses_a_park_from_another_run(tmp_path):
    db = tmp_path / "parks.db"
    first, second = _parked_store(db, ["r-first", "r-second"])

    async def drive():
        app = RunViewApp(db, second)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            app.parks = tuple(runread.parks(db, first))
            assert app._command(["answer", "{}"]) is False
            assert [p.task_id for p in runread.parks(db, first)] == [first]

    asyncio.run(drive())


def test_a_refresh_is_refused_off_the_ui_thread(store):
    """The refresh number is read and written by `refresh_view`, so a worker asks the UI thread."""

    async def drive():
        app = RunViewApp(store)
        async with app.run_test() as pilot:
            await _mounted(app, pilot)
            raised: list[BaseException] = []

            def from_a_worker() -> None:
                try:
                    app.refresh_view()
                except RuntimeError as refused:
                    raised.append(refused)

            worker = threading.Thread(target=from_a_worker, daemon=True)
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive(), "a refresh from a worker blocked instead of refusing"
            assert [str(r) for r in raised] == [
                "refresh_view runs on the UI thread; use call_from_thread"
            ]

    asyncio.run(drive())


def test_a_callback_after_the_app_has_stopped_is_dropped(store):
    """An answer is written before its refresh is asked for, so a closed app drops the refresh."""
    app = RunViewApp(store)
    outcome: list[object] = []

    def from_a_worker() -> None:
        try:
            app._to_ui(outcome.append, "drawn")
        except RuntimeError as refused:
            outcome.append(refused)
        else:
            outcome.append("dropped")

    worker = threading.Thread(target=from_a_worker)
    worker.start()
    worker.join()
    assert outcome == ["dropped"]


def test_a_view_takes_each_nodes_cost_and_time_from_telemetry(store):
    """Cost and time come from the run's telemetry; a node nothing measured stays `None`."""
    (run,) = runread.runs(store)
    first, *rest = runread.view(store, run.task_id).nodes
    view = runread.view(store, run.task_id, telemetry={first.key: (0.25, 4_000)})
    measured, *unmeasured = view.nodes
    assert (view.cost, measured.cost, measured.duration_ns) == (0.25, 0.25, 4_000)
    assert [node.cost for node in unmeasured] == [None] * len(rest)
    assert runread.view(store, run.task_id).cost is None
