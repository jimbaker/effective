"""A tool's "no" as a recorded value, and the dispositions it then reaches.

ROLE: journey, driven on SQLite because the defect only exists where a handler
actually interprets ops. `RecordingHandler` cans results and never calls a runner, so the whole
failure mode is invisible to it.

**The hazard.** Were `coding.runners`' static gate to refuse a bad edit by raising, the
handler-side exception would never cross the yield boundary, so the op would never complete and
the engine would retry a refusal that is deterministic by construction: `ToolError('lint [syntax]
invalid syntax')`, three attempts, three identical `write_file` calls, zero ledger rows.

**The contract.** One call, the refusal recorded, the loop routing around it to a different
edit, the run completing with its two ledger rows and the edit on the record.
"""

import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import TypeAdapter

from effective.coding.runners import CODING_TOOLS, UnknownTool, serve_tool
from effective.coding.specs import coding_bind, coding_read
from effective.coding.states import DraftVerdict, FinalizeVerdict, ReviewVerdict, State
from effective.coding.transition import transition
from effective.domain import AskLLM, CallTool, DomainOp, ToolRefused
from effective.handlers.absurd import DurableHandler
from effective.keys import Run
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import ParkReason
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import agent_worker, build_specs
from effective.machine.trampoline import run_machine
from effective.react import AssistantTurn, Tool, ToolLog, ToolRequest
from effective.sqlite import SqliteApp, SqliteLedger

pytestmark = pytest.mark.journey

MODULE = "mod.py"
GOOD = "def add(a, b):\n    return a + b\n"
BAD = "def add(a, b):\n    return a +\n"
"""Does not parse. The static gate refuses it — this is the ladder working, not failing."""
FIXED = "def add(a, b):\n    return a + b  # fixed\n"


def wrote(content: str) -> AssistantTurn:
    return AssistantTurn(
        thought="edit",
        tool=ToolRequest(name="write_file", args={"path": MODULE, "content": content}),
    )


ANSWERED = AssistantTurn(thought="done", answer="finished")


class Deployment:
    """Serves the coding table THE WAY A DEPLOYMENT SHOULD: through `serve_tool`, so a tool's
    refusal is an op result rather than an exception that never reaches the workflow, and over the
    tree each call carries, so it keeps no workspace between calls. `written` records the trees
    the edits returned, for the assertions."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[str] = []
        self.written: list[dict[str, str]] = []

    def run(self, op: DomainOp) -> Any:
        match op:
            case AskLLM():
                return self.turns.pop(0) if self.turns else ANSWERED
            case CallTool(name="run_suite"):
                self.calls.append("run_suite")
                return CommandRun(exit_code=0)
            case CallTool(name=name, args=args):
                self.calls.append(name)
                result = serve_tool(name, args)
                if isinstance(result, dict):
                    self.written.append(result)
                return result
        raise AssertionError(op)


class Served(Deployment):
    """Serves `run_suite` through `serve_tool` too, so the predicate materializes the tree it
    names."""

    def run(self, op: DomainOp) -> Any:
        match op:
            case CallTool(name="run_suite" as name, args=args):
                self.calls.append(name)
                return serve_tool(name, args)
        return super().run(op)


class RefusedSuite(Deployment):
    """Answers the predicate with a refusal, whatever tree it names."""

    def run(self, op: DomainOp) -> Any:
        match op:
            case CallTool(name="run_suite" as name):
                self.calls.append(name)
                return ToolRefused(refused=True, tool=name, diagnostic="the suite refused")
        return super().run(op)


def judging(verdict):
    def judge(_ctx: Ctx, _evidence: Evidence) -> Any:
        return verdict
        yield  # pragma: no cover  -- a `Judge` is a generator

    return judge


def passthrough(ctx: Ctx) -> Any:
    return Evidence(summary=ctx.state.value)
    yield  # pragma: no cover


def unreachable(ctx: Ctx) -> Any:
    raise AssertionError(f"{ctx.state.value} should not run")
    yield  # pragma: no cover


def specs(draft_verdict=DraftVerdict.GREEN, max_iters: int = 6):
    return build_specs(
        State,
        workers={
            State.DRAFT: agent_worker(
                prompt=lambda ctx: ctx.goal,
                tools=CODING_TOOLS,
                read=coding_read,
                bind=coding_bind,
                max_iters=max_iters,
            ),
            State.FINALIZE: passthrough,
            State.REVIEW: passthrough,
        },
        judges={
            State.DRAFT: judging(draft_verdict),
            State.FINALIZE: judging(FinalizeVerdict.STILL_GREEN_TIDY),
            State.REVIEW: judging(ReviewVerdict.APPROVED),
        },
        default_worker=unreachable,
        default_judge=judging(DraftVerdict.GREEN),
    )


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


def drive(app, deployment, *, budget=4, tree=None, **spec_kwargs):
    executions: list[int] = []

    @app.register_task("refusal")
    def task(params, ctx):
        executions.append(1)
        rid = params["run_id"]
        return DurableHandler(
            ctx, deployment, ledger=SqliteLedger(app.conn, rid, app.write_lock)
        ).run(
            lambda: run_machine(
                Run(rid),
                "fix the sign",
                specs(**spec_kwargs),
                transition,
                start=State.DRAFT,
                budget=budget,
                tree={MODULE: GOOD} if tree is None else tree,
            )
        )

    tid = app.spawn("refusal", {"run_id": "ref-1"})
    snap = app.run_until_result(tid)
    rows = [(k, json.loads(p)) for k, p in app.conn.execute("SELECT kind, payload FROM ledger")]
    return snap, len(executions), rows


# --- REWORK: the disposition the ladder is designed around ---------------------------------------


def test_a_refused_edit_becomes_an_observation_and_the_loop_tries_another(app):
    """The mainline. The model writes something that does not parse, the gate refuses it, and the
    refusal arrives as the next observation instead of as a dead task.

    The counts are the assertion. Before the fix: three attempts, the SAME edit three times, zero
    rows. Here: no retries, two DIFFERENT edits, and the second one on the record."""
    deployment = Deployment([wrote(BAD), wrote(FIXED), ANSWERED])
    snap, executions, rows = drive(app, deployment)

    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert executions == 1, "a deterministic refusal was retried"
    assert deployment.calls == ["write_file", "write_file", "run_suite"]
    assert [kind for kind, _ in rows] == ["machine-committed", "machine-finished"]
    assert deployment.written[-1][MODULE] == FIXED


def test_the_refusal_is_RECORDED_so_the_op_completes(app):
    """The property the whole design rests on, and the one that distinguishes it from widening the
    exception-delivery set: the op COMPLETES and its answer is checkpointed, so a resume re-serves
    it with nothing to re-derive and the exception-delivery set never grows."""
    drive(app, Deployment([wrote(BAD), wrote(FIXED), ANSWERED]))
    values = [
        json.loads(state)
        for (state,) in app.conn.execute(
            "SELECT state FROM checkpoints WHERE name LIKE '%tool:write_file%' ORDER BY name"
        )
    ]
    refusals = [v for v in values if isinstance(v, dict) and v.get("refused") is True]
    assert len(refusals) == 1, values
    assert refusals[0]["tool"] == "write_file"
    assert "syntax" in refusals[0]["diagnostic"]


@pytest.mark.parametrize(
    "path", ["/".join(["d"] * 2100) + "/x.py", "\udc80.py"], ids=["long-path", "lone-surrogate"]
)
def test_a_path_the_filesystem_cannot_write_is_refused_and_the_run_commits(app, path):
    """Reddens if a key the filesystem rejects is admitted by `write_file` or crashes its check:
    the postamble's suite then raises, the engine retries, and the run leaves no rows."""
    written = AssistantTurn(
        thought="edit", tool=ToolRequest(name="write_file", args={"path": path, "content": GOOD})
    )
    deployment = Served([written, ANSWERED])
    snap, executions, rows = drive(app, deployment)

    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert executions == 1
    assert deployment.calls == ["write_file", "run_suite"]
    assert [kind for kind, _ in rows] == ["machine-committed", "machine-finished"]


def test_a_predicate_that_refuses_the_tree_is_recorded_as_not_passing(app):
    """Reddens if the postamble cannot record a refused predicate: the run then fails with a
    validation error and no rows, where the record owes a commit and an outcome."""
    snap, executions, rows = drive(app, RefusedSuite([ANSWERED]))

    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert executions == 1
    assert [kind for kind, _ in rows] == ["machine-committed", "machine-finished"]
    outcome = dict(rows)["machine-finished"]
    assert outcome["passed"] is False
    assert outcome["measured"]["refused"] is True


def test_an_unsafe_seed_is_never_committed(app):
    """Reddens if a run whose model calls no tool commits a seed no tool could have written: the
    predicate is the one call that reads it, and it fails the task on its first attempt."""
    snap, executions, rows = drive(app, Served([ANSWERED]), tree={f"./{MODULE}": GOOD})

    assert snap is not None
    assert snap.state == "failed"
    assert "UnsafeTreePath" in (snap.failure or "")
    assert executions == 1
    assert rows == []


# --- GIVE UP: what a run that never recovers actually does ---------------------------------------


def test_edits_that_are_ALWAYS_refused_exhaust_and_still_commit(app):
    """The disposition a refused edit reaches when it never recovers — and it is EXHAUSTION, not a
    rejection, which is worth stating because the plan for this fix assumed three separate arms.

    DRAFT's fibre is `GREEN | STILL_RED | REGRESSED`; none of them parks. So a state that keeps
    failing takes its self-edge until the interpreter mints `Exhausted` and the transition parks.
    "Park because a human said no" is `route_plan`'s, from a rejected PLAN, and is not reachable
    from a refused edit at all. The postamble still runs, which is the whole point: the run is on
    the record with its failure, rather than gone."""
    deployment = Deployment([wrote(BAD)] * 20)
    snap, executions, rows = drive(
        app, deployment, budget=3, draft_verdict=DraftVerdict.STILL_RED, max_iters=2
    )

    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert executions == 1
    assert [kind for kind, _ in rows] == ["machine-committed", "machine-parked"]
    outcome = dict(rows)["machine-parked"]
    assert outcome["reason"] == ParkReason.EXHAUSTED.value
    assert snap.result["stopped"] != {}
    # nothing was ever written: every edit was refused, so the commit is the seed tree
    assert deployment.written == [], "every write was refused, so none landed"


# --- what is NOT a value -------------------------------------------------------------------------


def test_an_unserved_name_still_RAISES_rather_than_answering(app):
    """`UnknownTool` is a `KeyError`, not a `ToolError`, so it passes `serve_tool`'s clause
    untouched — by construction rather than by an exemption someone has to remember.

    An unserved name is a composition defect: the deployment does not have that tool, and
    answering with a diagnostic would let a run look busy while doing nothing."""
    with pytest.raises(UnknownTool):
        serve_tool("no_such_tool", {})


def test_a_refusal_is_not_folded_as_the_tools_declared_result():
    """A refused `write_file` is still an entry whose TOOL declares `dict[str, str]`. Folding on
    the declaration alone would hand the refusal back as the workspace — and the trampoline
    commits whatever `Evidence.tree` holds, so that is a corrupted commit, not a type error."""
    log = ToolLog()
    write: Tool[Any, Any] = CODING_TOOLS["write_file"]
    log.entries.append((write, {MODULE: FIXED}))
    log.entries.append((write, ToolRefused(refused=True, tool="write_file", diagnostic="nope")))

    assert log.last(dict[str, str]) == {MODULE: FIXED}
    assert log.names == ("write_file", "write_file"), "the refusal left the record of what ran"


def test_a_record_that_does_not_say_it_refused_is_never_read_as_a_refusal():
    """Reddens if a predicate record with a tool and a diagnostic validates as `ToolRefused`: the
    postamble reads `ToolRefused | R`, and would record a passing scan as a failed run."""

    @dataclass(frozen=True)
    class Scan:
        tool: str
        diagnostic: str

    record = TypeAdapter(ToolRefused | Scan).validate_python({"tool": "docs", "diagnostic": ""})

    assert record == Scan(tool="docs", diagnostic="")
