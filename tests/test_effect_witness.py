"""`scripts/effect_witness.py`'s own pins: the instrument must find a planted path before its zeros
mean anything. Each is a P0 prediction in the effect-witness preregistration report, or a finding
from a review of the instrument. `scripts/effect_witness_mutants.py` runs this file against named
mutants of the script; set `EFFECT_WITNESS_SOURCE` to a copy's path to load that copy instead."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if (_source := os.environ.get("EFFECT_WITNESS_SOURCE")) is not None:
    _spec = importlib.util.spec_from_file_location("scripts.effect_witness", _ROOT / _source)
    assert _spec is not None
    assert _spec.loader is not None
    sys.modules["scripts.effect_witness"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["scripts.effect_witness"])

from effective.api import append_ledger, await_event, call_tool, store_artifact  # noqa: E402
from effective.counterfactual import ForkLedger  # noqa: E402
from effective.govern import govern  # noqa: E402
from effective.handlers import absurd, recording  # noqa: E402
from effective.handlers.absurd import DurableHandler  # noqa: E402
from effective.handlers.recording import RecordingHandler  # noqa: E402
from effective.handlers.replay import ReplayHandler  # noqa: E402
from effective.keys import Segment, compose_key  # noqa: E402
from effective.layers import op_layer  # noqa: E402
from effective.ops import AppendLedgerRow, DomainOp, LedgerRow  # noqa: E402
from effective.permission import Allow, as_policy, cascade, rules  # noqa: E402
from effective.sqlite import SqliteApp, SqliteLedger, SqliteTaskContext  # noqa: E402
from scripts import effect_witness as ew  # noqa: E402

pytestmark = pytest.mark.adversarial

_THIS = Path(__file__).resolve().relative_to(_ROOT).as_posix()


def row(name: str, kind: str = "witnessed") -> LedgerRow:
    return LedgerRow(event_id=compose_key(t"witnessed:{Segment(name)}"), kind=kind)


def line_of(comment: str) -> int:
    """The line ending in `comment` in this file, so a pin survives an edit above it."""
    lines = Path(__file__).read_text().splitlines()
    return next(i for i, text in enumerate(lines, 1) if text.rstrip().endswith(comment))


def here(comment: str) -> tuple[str, int]:
    return (_THIS, line_of(comment))


def where(frame: dict) -> tuple[str, int]:
    return (frame["path"], frame["line"])


def allow_all(_op: Any) -> Allow:
    return Allow()


class Answers:
    """A domain that answers every op with "ok", optionally writing the ledger as it does."""

    def __init__(self, ledger: SqliteLedger | None = None) -> None:
        self.ledger = ledger

    def run(self, op: DomainOp[Any]) -> Any:
        if self.ledger is not None:
            self.ledger.append(row("from-the-tool"))  # tool-side-door
        return "ok"


class ListLedger:
    """A ledger no sensor knows the engine of."""

    def __init__(self) -> None:
        self.rows: list[LedgerRow] = []

    def append(self, entry: LedgerRow, *, writer: Any = None) -> None:
        self.rows.append(entry)


class Audited:
    """A ledger writer that delegates to the store as `ForkLedger` does, and carries no sensor."""

    def __init__(self, inner: SqliteLedger) -> None:
        self.inner = inner

    def append(self, entry: LedgerRow, *, writer: Any = None) -> None:
        self.inner.append(entry, writer=writer)


@pytest.fixture
def witness():
    probe = ew.Witness(out=None)
    probe.install()
    yield probe
    probe.uninstall()


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


def of(probe: ew.Witness, effect: str, *, written: bool = True) -> list[dict]:
    return [r for r in probe.rows if r["effect"] == effect and r["written"] is written]


def run_durable(app: SqliteApp, workflow, *, layers=(), domain=None, ledger=None, body=None):
    @app.register_task("planted")
    def task(params, ctx):
        if body is not None:
            return body(ctx)
        writer = ledger if ledger is not None else SqliteLedger(app.conn, "run", app.write_lock)
        handler = DurableHandler(ctx, domain or Answers(), ledger=writer, op_layers=list(layers))
        return handler.run(workflow)

    return app.spawn("planted", {})


# ---------------------------------------------------------------- position


def test_a_direct_append_outside_any_handler_sits_outside_the_op_stream(witness, app):
    """Reddens if the ledger sensor stops seeing a write no handler drove, as fork genesis does."""
    SqliteLedger(app.conn, "run-1", app.write_lock).append(row("direct"))  # direct-append

    (only,) = of(witness, "ledger")
    assert (only["position"], only["engine"], only["beneath"]) == ("outside", "sqlite", None)
    assert where(only["writer"]) == here("# direct-append")


def test_the_same_row_through_a_cascaded_durable_handler_sits_under_one_guard(witness, app):
    """Reddens if the op-stream sensor loses the drive, if the guards stop naming the cascade, or
    if a checkpoint's class stops being the head tag of its key."""
    gate = cascade([rules(allow_all)])
    app.run_until_result(run_durable(app, lambda: append_ledger(row("guarded")), layers=[gate]))

    (ledger,) = of(witness, "ledger")
    assert ledger["position"] == "stream, 1 guard"
    assert ledger["layers"] == ["effective.permission.cascade.<locals>.gate"]
    assert (ledger["writer"]["path"], ledger["writer"]["qualname"]) == (
        "src/effective/handlers/absurd.py",
        "DurableHandler._handle.<locals>.<lambda>",
    )
    (checkpoint,) = of(witness, "checkpoint")
    assert (checkpoint["position"], checkpoint["class"]) == ("stream, 1 guard", "ledger")


def test_two_cascades_on_one_op_are_two_guards(witness):
    """Reddens if the sensor stops counting guards across a stack, the redundant-guard row."""

    def workflow():
        yield from append_ledger(row("twice-guarded"))

    RecordingHandler(op_layers=[cascade([rules(allow_all)]), cascade([rules(allow_all)])]).run(
        workflow
    )

    (only,) = of(witness, "ledger")
    assert only["position"] == "stream, 2 or more"


def test_a_tool_writing_the_ledger_inside_a_guarded_step_is_outside_beneath_it(witness, app):
    """Reddens if a write is placed on the stream for happening inside a drive: the guard decided
    the tool call, and the tool wrote the row through its own door. The tool writes the handler's
    own ledger, so only the op tells the two apart."""
    ledger = SqliteLedger(app.conn, "run-6", app.write_lock)

    def workflow():
        return (yield from call_tool("t", {}, str))

    gate = cascade([rules(allow_all)])
    task = run_durable(app, workflow, layers=[gate], domain=Answers(ledger), ledger=ledger)
    app.run_until_result(task)

    (side_door,) = of(witness, "ledger")
    assert (side_door["position"], side_door["beneath"]) == ("outside", "Step")
    assert where(side_door["writer"]) == here("# tool-side-door")
    assert [r["position"] for r in of(witness, "checkpoint")] == ["stream, 1 guard"]


def test_a_checkpoint_in_another_tasks_store_during_a_step_is_outside_beneath_it(witness, app):
    """Reddens if a write counts as the drive's because the op could have made it: a `Step` writes
    checkpoints, and this one lands in a task the handler does not own."""
    other = SqliteApp(":memory:")

    @other.register_task("foreign")
    def foreign(params, ctx):
        return ctx.step(compose_key(t"foreign:{Segment('x')}"), lambda: 1)

    class RunsAnotherTask:
        def run(self, op: DomainOp[Any]) -> Any:
            other.run_until_result(other.spawn("foreign", {}))
            return "ok"

    def workflow():
        return (yield from call_tool("t", {}, str))

    try:
        app.run_until_result(run_durable(app, workflow, domain=RunsAnotherTask()))
    finally:
        other.close()

    by_identity = {r["identity"]: r for r in of(witness, "checkpoint")}
    assert (by_identity["foreign:x"]["position"], by_identity["foreign:x"]["beneath"]) == (
        "outside",
        "Step",
    )
    assert by_identity["step;tool:t"]["position"] == "stream, 0 guards"


def test_a_nested_handlers_layer_writing_the_outer_store_is_outside(witness, app):
    """Reddens if a write is credited to the outer drive after a nested handler has pushed a drive
    of its own: the nested layer runs, the outer base does not, and the outer handler owns the
    store, so only the drive stack tells the two apart."""
    outer_ctx: dict[str, Any] = {}

    @op_layer
    def writes_outer(op):
        outer_ctx["ctx"].step(compose_key(t"layer:{Segment('w')}"), lambda: 1)
        return (yield op)

    def nested():
        yield from append_ledger(row("nested"))

    class RunsANestedHandler:
        def run(self, op: DomainOp[Any]) -> Any:
            RecordingHandler(op_layers=[writes_outer]).run(nested)
            return "ok"

    def body(ctx):
        outer_ctx["ctx"] = ctx
        handler = DurableHandler(ctx, RunsANestedHandler(), ledger=None)
        return handler.run(lambda: call_tool("t", {}, str))

    app.run_until_result(run_durable(app, None, body=body))

    by_identity = {r["identity"]: r for r in of(witness, "checkpoint")}
    assert (by_identity["layer:w"]["position"], by_identity["layer:w"]["beneath"]) == (
        "outside",
        "AppendLedgerRow",
    )


def test_a_layer_injected_write_is_marked_and_carries_the_layers_line(witness):
    """Reddens if an op a layer yields borrows the site of the op it rode in on, or if the chain is
    matched by type: the injected op and the driven op are both `AppendLedgerRow`."""

    @op_layer
    def audit(op):
        if isinstance(op, AppendLedgerRow) and op.row.kind == "witnessed":
            yield AppendLedgerRow(row=row("audit", kind="audit"))  # site-injected
        return (yield op)

    def workflow():
        yield from append_ledger(row("audited"))  # site-audited

    RecordingHandler(op_layers=[audit]).run(workflow)

    by_identity = {r["identity"]: r for r in of(witness, "ledger")}
    assert by_identity["witnessed:audit"]["injected"] is True
    assert where(by_identity["witnessed:audit"]["site"]) == here("# site-injected")
    assert by_identity["witnessed:audited"]["injected"] is False
    assert where(by_identity["witnessed:audited"]["site"]) == here("# site-audited")


@pytest.mark.parametrize(
    ("guards", "expected"),
    [(None, "outside"), (0, "stream, 0 guards"), (1, "stream, 1 guard"), (2, "stream, 2 or more")],
)
def test_position_is_total_over_guard_counts(guards, expected):
    assert ew.position(guards) == expected


def test_the_declared_guards_are_the_names_the_gates_carry():
    """Reddens if a gate is renamed and the instrument silently counts it as no guard."""
    permission_gate = cascade([rules(allow_all)])
    governed = govern(as_policy([rules(allow_all)]), gate="g", run_id="run-4")
    assert {ew.layer_name(permission_gate), ew.layer_name(governed)} == ew.GUARDS


# ---------------------------------------------------------------- authored site


def test_two_yield_sites_are_two_authored_sites_and_a_helper_names_its_own_line(witness):
    """Reddens if the authored site stops skipping `effective/api.py`, or stops matching by op."""

    def via_helper(entry: LedgerRow):
        return (yield from append_ledger(entry))  # site-helper

    def workflow():
        yield from append_ledger(row("a"))  # site-a
        yield from via_helper(row("b"))

    RecordingHandler().run(workflow)

    sites = {r["identity"]: where(r["site"]) for r in of(witness, "ledger")}
    assert sites["witnessed:a"] == here("# site-a")
    assert sites["witnessed:b"] == here("# site-helper")


def test_an_undriven_op_does_not_lend_its_site_to_the_next_driven_one(witness):
    """Reddens if the chain is taken without matching the op: an await the recording core decides
    before the layers yields a chain nothing drives, and the next op would inherit its line."""

    def workflow():
        yield from await_event("approved", str)  # site-undriven
        yield from append_ledger(row("after"))  # site-driven

    RecordingHandler(responses={"approved": "yes"}).run(workflow)

    (only,) = of(witness, "ledger")
    assert where(only["site"]) == here("# site-driven")


def test_a_replayed_op_of_the_same_type_does_not_lend_its_site(witness):
    """Reddens if the chain is matched by type: a replay yields `AppendLedgerRow` from one line and
    drives nothing, and the next driven `AppendLedgerRow` would take that line."""

    def replayed():
        yield from append_ledger(row("replayed"))

    def fresh():
        yield from append_ledger(row("fresh"))  # site-fresh

    recorder = RecordingHandler()
    recorder.run(replayed)
    ReplayHandler(recorder.trace).run(replayed)
    RecordingHandler().run(fresh)

    sites = {r["identity"]: where(r["site"]) for r in of(witness, "ledger")}
    assert sites["witnessed:fresh"] == here("# site-fresh")


# ---------------------------------------------------------------- what was stored


def test_a_fork_ledger_over_a_sqlite_ledger_records_one_write_under_its_stored_identity(
    witness, app
):
    """Reddens if nested ledger sensors each record the one append they share, or if the row
    carries the id the caller passed instead of the rescoped one the store holds."""
    base = SqliteLedger(app.conn, "run-3", app.write_lock, hypothetical=True)
    ForkLedger(base, child_run_id=Segment("child-3")).append(row("forked"))

    (stored,) = [r[0] for r in app.conn.execute("SELECT event_id FROM ledger")]
    (only,) = of(witness, "ledger")
    assert (only["identity"], only["engine"]) == (stored, "sqlite")


def test_a_delegating_writer_is_skipped_to_the_line_that_called_it(witness, app):
    """Reddens if the writer site stops skipping delegates: `Audited.append` hands the row on, and
    the line that asked for the write is the caller's."""
    Audited(SqliteLedger(app.conn, "run-9", app.write_lock)).append(row("audited"))  # via-audited

    (only,) = of(witness, "ledger")
    assert only["frames"][0]["qualname"] == "Audited.append"
    assert where(only["writer"]) == here("# via-audited")


def test_an_idempotent_reappend_is_an_attempt_and_not_a_write(witness, app):
    """Reddens if the ledger sensor counts an append the store folded into an existing row."""
    ledger = SqliteLedger(app.conn, "run-7", app.write_lock)
    ledger.append(row("once"))
    ledger.append(row("once"))

    assert len(of(witness, "ledger")) == 1
    assert len(of(witness, "ledger", written=False)) == 1


def test_a_resumed_run_records_checkpoint_hits_as_attempts(witness, app):
    """Reddens if a replayed checkpoint, a hit, is recorded as a write."""

    def workflow():
        yield from call_tool("t", {}, str)
        yield from await_event("go", str)
        yield from append_ledger(row("resumed"))

    task = run_durable(app, workflow)
    app.run_until_result(task)
    app.emit_event("go", "yes")
    app.run_until_result(task)

    (stored,) = app.conn.execute("SELECT count(*) FROM checkpoints").fetchone()
    assert len(of(witness, "checkpoint")) == stored
    assert [r["identity"] for r in of(witness, "checkpoint", written=False)] == ["step;tool:t"]


def test_a_step_nested_in_a_step_on_one_store_is_two_writes(witness, app):
    """Reddens if two checkpoint writes on one task context fold into one row."""

    def body(ctx):
        inner = compose_key(t"inner:{Segment('x')}")
        return ctx.step(compose_key(t"outer:{Segment('x')}"), lambda: ctx.step(inner, lambda: 1))

    app.run_until_result(run_durable(app, None, body=body))

    (stored,) = app.conn.execute("SELECT count(*) FROM checkpoints").fetchone()
    assert sorted(r["identity"] for r in of(witness, "checkpoint")) == ["inner:x", "outer:x"]
    assert stored == 2


def test_an_artifact_through_the_recording_handler_is_a_write(witness):
    """Reddens if the `StoreArtifact` arm loses its sensor."""
    RecordingHandler().run(lambda: store_artifact("body", "text/plain"))

    (only,) = of(witness, "artifact")
    assert (only["engine"], only["position"]) == ("recording", "stream, 0 guards")


def test_a_ledger_of_an_unknown_engine_is_counted_and_never_joined(witness, app):
    """Reddens if a write outside the declared domain reaches Join A's table."""

    def workflow():
        return (yield from append_ledger(row("elsewhere")))

    app.run_until_result(run_durable(app, workflow, ledger=ListLedger()))

    (only,) = of(witness, "ledger")
    assert only["engine"] == "unscanned"
    report = ew.join_a(witness.rows)
    assert "1 unscanned" in report
    assert "ledger      stream, 0 guards            0" in report


@pytest.mark.parametrize(
    ("identity", "head"),
    [
        ("step;tool:a", "step"),
        ("gather:0,1;step;tool:a#2", "step"),
        ("ledger;witnessed:x", "ledger"),
        ("respawn:run,3", "respawn"),
    ],
    ids=["plain", "framed-with-occurrence", "ledger", "one-term"],
)
def test_a_checkpoint_class_is_the_head_tag_past_its_frames(identity, head):
    assert ew.identity_class(identity) == head


# ---------------------------------------------------------------- joins


def test_one_identity_from_two_authored_sites_is_a_join_b_group(witness):
    """Reddens if Join B stops pairing two sites that land on one durable identity."""

    def workflow():
        yield from append_ledger(row("same"))
        yield from append_ledger(row("same"))

    RecordingHandler().run(workflow)

    report = ew.join_b(witness.rows)
    assert "1 groups: 1 test" in report
    assert "ledger witnessed:same" in report


def test_a_side_door_and_its_guarded_twin_are_a_mixed_join_b_group(witness, app):
    """Reddens if Join B drops writes that are off the stream, which is where the CWE-424 half of
    a pair sits."""

    def workflow():
        yield from append_ledger(row("invoice-7"))

    RecordingHandler(op_layers=[cascade([rules(allow_all)])]).run(workflow)
    SqliteLedger(app.conn, "run-8", app.write_lock).append(row("invoice-7"))

    report = ew.join_b(witness.rows)
    assert "1 groups: 1 test, mixed" in report
    assert "[outside]" in report
    assert "[stream, 1 guard]" in report


def fake_row(effect: str, path: str, line: int, *, written: bool, position: str) -> dict:
    frame = {"path": path, "line": line, "qualname": "f"}
    return {
        "effect": effect,
        "engine": "sqlite",
        "written": written,
        "position": position,
        "frames": [frame],
    }


HIT = fake_row("checkpoint", "a.py", 1, written=False, position="outside")
IN_STREAM = fake_row("checkpoint", "a.py", 1, written=True, position="stream, 0 guards")
OUTSIDE = fake_row("checkpoint", "a.py", 1, written=True, position="outside")
LEDGER = fake_row("ledger", "a.py", 1, written=True, position="outside")


@pytest.mark.parametrize(
    ("rows", "verdict"),
    [
        ([], "unwitnessed"),
        ([HIT], "reached, never wrote"),
        ([IN_STREAM], "witnessed in stream"),
        ([OUTSIDE], "witnessed outside"),
        ([LEDGER], "unwitnessed"),
    ],
    ids=["nothing", "hits-only", "stream", "outside", "another-kind"],
)
def test_join_c_disposes_a_step_site_by_checkpoint_rows(rows, verdict):
    """Reddens if a site reached only by hits reads as unreached, or a ledger row witnesses a
    `step` site."""
    sites = ew.StaticSites(
        candidates=1, resolved=[ew.StaticSite("a.py", 1, "step")], unresolved=[]
    )
    assert f"[{verdict}] a.py:1 .step" in ew.join_c(rows, sites)


# ---------------------------------------------------------------- the instrument itself


def test_a_durable_run_writes_rows_that_read_back(app, tmp_path):
    """Reddens if a row carries a value JSON cannot hold (a checkpoint's scope is a UUID), or if
    the store append nested inside `_record_ledger` stops leaving its frames for Join C."""
    probe = ew.Witness(out=tmp_path / "rows.jsonl")
    probe.current = "planted"
    probe.install()
    try:
        app.run_until_result(run_durable(app, lambda: append_ledger(row("written"))))
    finally:
        probe.uninstall()
    probe.pytest_sessionfinish(None, 0)

    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert sorted(r["effect"] for r in rows) == ["checkpoint", "ledger"]
    (ledger,) = [r for r in rows if r["effect"] == "ledger"]
    assert "DurableHandler._record_ledger" in {f["qualname"] for f in ledger["frames"]}


def test_the_command_keeps_the_tests_in_the_process_holding_the_sensors(tmp_path):
    """Reddens if the command lets xdist move the tests into workers, where no sensor is installed
    and the report is empty. Runs the command itself, since an argument list proves nothing about
    what the repository's plugins accept."""
    out = tmp_path / "rows.jsonl"
    pin = "tests/test_effect_witness.py::test_an_artifact_through_the_recording_handler_is_a_write"
    # Run under xdist, this process is a worker, and a session that inherits `PYTEST_XDIST_WORKER`
    # takes itself for one and asks for a per-worker database. The command is a fresh session.
    inherited = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID")
    env = {k: v for k, v in os.environ.items() if k not in inherited}
    done = subprocess.run(
        [sys.executable, ew.__file__, "--out", str(out), "--", pin, "-n", "2"],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert any(json.loads(line)["effect"] == "artifact" for line in out.read_text().splitlines())


def test_an_empty_run_is_refused(tmp_path, capsys):
    """Reddens if the command prints an empty report where it should refuse."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert ew.main(["--report", "--out", str(empty)]) == 2
    assert "saw nothing" in capsys.readouterr().err


def test_a_report_refuses_pytest_arguments(tmp_path):
    with pytest.raises(SystemExit):
        ew.main(["--report", "--out", str(tmp_path / "rows.jsonl"), "tests/"])


def module_copy() -> Any:
    spec = importlib.util.spec_from_file_location("effect_witness_copy", Path(ew.__file__))
    assert spec is not None
    assert spec.loader is not None
    copy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = copy
    spec.loader.exec_module(copy)
    return copy


@pytest.mark.parametrize("modules", ["one module", "two copies"])
def test_a_second_install_does_not_blind_the_first(app, modules):
    """Reddens if an inner witness wraps the unpatched `drive_through` and not the one already
    installed: the witness's own pins, run under the witness, would read as side doors. Two copies
    are the script's `__main__` beside these pins; one module shares a drive stack, where the
    second witness's drive of the same op sits above the first's."""
    outer_module = ew if modules == "one module" else module_copy()
    outer, inner = outer_module.Witness(out=None), ew.Witness(out=None)
    outer.install()
    inner.install()
    try:
        gate = cascade([rules(allow_all)])
        app.run_until_result(run_durable(app, lambda: append_ledger(row("twice")), layers=[gate]))
    finally:
        inner.uninstall()
        outer.uninstall()

    assert [r["position"] for r in of(outer, "ledger")] == ["stream, 1 guard"]
    assert [r["position"] for r in of(inner, "ledger")] == ["stream, 1 guard"]


def test_uninstall_restores_every_seam_and_frees_the_monitoring_tool():
    seams = [
        (SqliteTaskContext, "step"),
        (SqliteLedger, "append"),
        (ForkLedger, "append"),
        (DurableHandler, "_record_ledger"),
        (RecordingHandler, "_interpret"),
        (recording, "drive_through"),
        (absurd, "drive_through"),
    ]
    originals = [vars(owner)[name] for owner, name in seams]
    probe = ew.Witness(out=None)
    probe.install()
    tool = probe.sites.tool if probe.sites else None
    assert all(vars(o)[n] is not v for (o, n), v in zip(seams, originals, strict=True))
    probe.uninstall()
    assert all(vars(o)[n] is v for (o, n), v in zip(seams, originals, strict=True))
    assert tool is not None
    assert sys.monitoring.get_tool(tool) is None
