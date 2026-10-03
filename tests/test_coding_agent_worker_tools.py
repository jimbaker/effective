"""`agent_worker` composed with `effective.coding`'s OWN tool table.

ROLE: journey: one path end to end through both grains, on a durable engine. It runs on SQLite
rather than under `RecordingHandler` because `RecordingHandler` returns a canned response
**uncoerced**, so a test of this composition under it is blind to the defect. The durable path
validates a `CallTool` result against the op's `result_schema` (`handlers/absurd.py`'s `_load`),
which is where the two grains are actually forced to agree.

The defect has two halves and only one of them is loud:

- `default_act` pins `result_schema=ToolResult` — one field, `content: str` — while
  `runners._write_file` returns the RESULTING TREE as a `dict[str, str]`, deliberately, because a
  workflow may only learn about a handler-side change through a recorded op result. Composing the
  two raises.
- the default reader leaves `Evidence.tree` unset, so even where the first half is stepped
  around, `_visit` returns `None`, `carried` never advances, and a canonical state's edits never
  reach the commit. Silently — where the `measured=None` twin raises by name.

The tools here are REAL (`runners.run_tool` applies the edit and the static gate vets it); only
the model's choices and the success predicate are scripted, because the predicate shells out
to pytest, and a case that spawned a subprocess per checkpoint would be measuring the fixture.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypedDict, Unpack

import pytest

from effective.coding.runners import CODING_TOOLS, TOOLS, Workspace, run_tool
from effective.coding.specs import coding_bind, coding_read
from effective.coding.states import DraftVerdict, FinalizeVerdict, ReviewVerdict, State
from effective.coding.transition import transition
from effective.domain import AskLLM, CallTool, DomainOp
from effective.handlers.absurd import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Run, Segment
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import agent_worker, build_specs
from effective.machine.trampoline import Session, run_machine
from effective.react import (
    AssistantTurn,
    Tool,
    ToolLog,
    ToolRequest,
    Trajectory,
    run_agent,
    typed_act,
)
from effective.sqlite import SqliteApp

pytestmark = pytest.mark.journey

MODULE = "mod.py"
TARGET = "test_adds_two_numbers"
"""The test the fixture is trying to make pass — named once, and never spelled into a whole
node id, for the reason `test_coding_read_folds_the_measurement_a_mechanical_judge_needs` gives."""
BROKEN = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"

WROTE_THE_FIX = AssistantTurn(
    thought="the sign is wrong",
    tool=ToolRequest(name="write_file", args={"path": MODULE, "content": FIXED}),
)

ANSWERED = AssistantTurn(thought="done", answer="fixed the sign")


class ToolTableDomain:
    """The package's own table, served for real — and a scripted model above it.

    `run_suite` is canned for the reason the conformance fixture states: the real predicate
    shells out to pytest, and this case is about whether the two grains compose, not about how
    fast pytest is. Every other tool goes through `runners.run_tool`, so `write_file` really
    edits the tree it is handed and really passes the static gate on the way. The tree comes in
    the call, bound by `coding_bind`; the undeclared composition binds none, and acts on the
    seed."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[str] = []

    def run(self, op: DomainOp) -> Any:
        match op:
            case AskLLM():
                return self.turns.pop(0) if self.turns else ANSWERED
            case CallTool(name="run_suite", args=args):
                self.calls.append("run_suite")
                tree = args.get("tree", {MODULE: BROKEN})
                return CommandRun(exit_code=0 if tree[MODULE] == FIXED else 1)
            case CallTool(name=name, args=args):
                self.calls.append(name)
                tree = args.get("tree", {MODULE: BROKEN})
                return run_tool(Workspace(tree=tree), name, args)
            case _:
                raise AssertionError(f"unexpected op: {op!r}")


def judging(verdict):
    def judge(_ctx: Ctx, _evidence: Evidence):
        return verdict
        yield  # pragma: no cover  -- a `Judge` is a generator

    return judge


def unreachable_worker(ctx: Ctx) -> Any:
    raise AssertionError(f"{ctx.state.value} should not run in this walk")
    yield  # pragma: no cover


def unreachable_judge(_ctx: Ctx, _evidence: Evidence) -> Any:
    raise AssertionError("no judge should run for a deferred state")
    yield  # pragma: no cover


def specs(**worker_kwargs: Unpack[Composed]):
    """DRAFT does the editing through a ReAct loop; FINALIZE and REVIEW wave it through.

    The walk is DRAFT -> FINALIZE -> REVIEW -> Finish, which is the shortest path that passes
    through a state declared `canonical` and out the other side."""
    return build_specs(
        State,
        workers={
            State.DRAFT: agent_worker(prompt=lambda ctx: ctx.goal, **worker_kwargs),
            State.FINALIZE: passthrough,
            State.REVIEW: passthrough,
        },
        judges={
            State.DRAFT: judging(DraftVerdict.GREEN),
            State.FINALIZE: judging(FinalizeVerdict.STILL_GREEN_TIDY),
            State.REVIEW: judging(ReviewVerdict.APPROVED),
        },
        canonical=frozenset({State.DRAFT, State.FINALIZE}),
        default_worker=unreachable_worker,
        default_judge=unreachable_judge,
    )


def passthrough(ctx: Ctx) -> Any:
    return Evidence(summary=f"{ctx.state.value}: nothing to do")
    yield  # pragma: no cover


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


def drive(app, domain, **worker_kwargs: Unpack[Composed]):
    @app.register_task("machine")
    def task(params, ctx):
        return DurableHandler(ctx, domain).run(
            lambda: run_machine(
                Run(params["run_id"]),
                "fix the sign",
                specs(**worker_kwargs),
                transition,
                start=State.DRAFT,
                budget=6,
                tree={MODULE: BROKEN},
            )
        )

    return app.run_until_result(app.spawn("machine", {"run_id": "ft7"}))


def committed_trees(app) -> list[dict[str, str]]:
    """Every artifact the postamble stored, read back off the checkpoint store — the commit as
    the ENGINE holds it, not as the workflow claims it."""
    import json

    rows = app.conn.execute(
        "SELECT state FROM checkpoints WHERE name LIKE '%artifact:%' ORDER BY name"
    ).fetchall()
    return [json.loads(state) for (state,) in rows]


FINDING_7 = pytest.mark.xfail(
    strict=True,
    reason=(
        "`agent_worker` untold what a tool RETURNS pins `result_schema=ToolResult` for "
        "everything, and a canonical state's edits never reach the commit. STRICT, so a fix "
        "cannot land quietly green."
    ),
)


class Composed(TypedDict, total=False):
    """What composing with the package's own table looks like: the DECLARED side of the table
    (`CODING_TOOLS`) says what each name returns, and `coding_read` folds those results into the
    `Evidence` a judge and the commit both depend on.

    A `TypedDict` rather than a bare literal because a `**dict[str, Any]` splat launders both
    values past `ty`, as measured here: the wrong one of these passed to `decide_for` type-checked
    clean, in a file whose whole subject is a type mismatch nobody was asked about."""

    tools: Mapping[str, Tool[Any, Any]]
    read: Callable[[Trajectory, ToolLog], Evidence]
    bind: Callable[[Ctx, ToolLog], Mapping[str, Any]]

    # `total=False` because one test deliberately composes with NEITHER — that is the
    # undeclared path, and it has to stay expressible for the same reason it has to stay red.


WITH_THE_TABLE: Composed = {"tools": CODING_TOOLS, "read": coding_read, "bind": coding_bind}


def test_an_UNDECLARED_tool_returning_a_tree_is_refused(app):
    """An undeclared tool's result is validated as `ToolResult`, and that is NOT a defect.

    With nothing declared, the act seam names every result `ToolResult`, and a tool that hands
    back the workspace fails validation on the durable path. That is correct: the seam was told
    nothing, so it has nothing to believe but the default. The tests below tell it.

    It is the reason the `tools=` parameter exists, and a reader who deletes `tools=` should see
    this fail rather than see nothing.

    Asserted as a task SNAPSHOT rather than a raised exception, because that is what a deployment
    actually sees: the task retries to death and the run is gone."""
    domain = ToolTableDomain([WROTE_THE_FIX, ANSWERED])
    snap = drive(app, domain)
    assert snap is not None
    assert snap.state == "failed"
    assert "ToolResult" in (snap.failure or ""), snap.failure


def test_the_packages_own_table_composes_with_the_loop(app):
    """THE LOUD HALF. Told what `write_file` returns, the run completes."""
    domain = ToolTableDomain([WROTE_THE_FIX, ANSWERED])
    snap = drive(app, domain, **WITH_THE_TABLE)
    assert snap is not None
    assert snap.state == "completed", (
        f"the run did not complete ({snap.state}). Tools reached: {domain.calls}\n{snap.failure}"
    )


def test_a_canonical_states_edits_reach_the_commit(app):
    """THE SILENT HALF: a canonical state's edits reach the commit.

    DRAFT is declared canonical and its worker really edited the workspace. If the edit does not
    travel back through a recorded op result into `Evidence.tree`, the postamble commits the SEED
    tree and the run records work it did not keep, with nothing raising anywhere."""
    domain = ToolTableDomain([WROTE_THE_FIX, ANSWERED])
    snap = drive(app, domain, **WITH_THE_TABLE)
    assert snap is not None
    assert snap.state == "completed"
    assert "write_file" in domain.calls, "the loop never reached the edit"

    trees = committed_trees(app)
    assert trees, "the postamble stored no artifact at all"
    assert trees[-1] == {MODULE: FIXED}, (
        f"the commit carries {trees[-1]!r} — a canonical state's edits did not reach it"
    )


def test_the_walk_is_the_one_the_specs_declare(app):
    """A guard against the two tests above passing for the wrong reason. If the machine never got
    past DRAFT — or never entered it — neither assertion above would mean what it says, and both
    would still be satisfiable by a run that committed the seed tree and stopped.

    **Read off the returned `Session`, and the first draft of this read the TAPE instead — which
    was wrong for a reason worth keeping.** A `state:` frame reaches the checkpoint store only if
    an op is minted inside it, so FINALIZE and REVIEW, whose workers here yield nothing, leave no
    trace at all: the assertion saw `{draft}` and would have read as a broken walk. It is the
    same trap `test_coding_agent_worker.py` records one grain down — a scripted decider that
    returns without yielding produces no ops, and every assertion about the tape then holds
    vacuously. The tape records what RAN; the session records what was DECIDED, and the walk is
    a decision."""
    snap = drive(app, ToolTableDomain([WROTE_THE_FIX, ANSWERED]), **WITH_THE_TABLE)
    assert snap is not None
    assert snap.state == "completed"
    assert [turn["state"] for turn in snap.result["turns"]] == ["draft", "finalize", "review"]
    assert [turn["verdict"] for turn in snap.result["turns"]] == [
        "green",
        "still-green-tidy",
        "approved",
    ]


def test_the_declared_table_and_the_served_table_are_the_same_names():
    """`CODING_TOOLS` declares; `TOOLS` serves. Two tables over one set of names is exactly the
    drift this repo keeps re-learning about — a declaration that quietly covers eight of nine is
    worse than none, because it is read on trust.

    Set equality, not containment, and in both directions: a declared name nobody serves is a
    tool the model will be offered and cannot call, and a served name nobody declares silently
    falls back to `ToolResult` — which is the defect this whole file is about."""
    assert set(CODING_TOOLS) == set(TOOLS)
    assert all(name == tool.name for name, tool in CODING_TOOLS.items())


# --- the acceptance condition for folding inside `act` -------------------------------------------


def machine():
    """The same walk, as a plain program the recording/replay pair can drive."""
    return run_machine(
        Run("ft7"),
        "fix the sign",
        specs(**WITH_THE_TABLE),
        transition,
        start=State.DRAFT,
        budget=6,
        tree={MODULE: BROKEN},
    )


LOOP = {
    "d:0;react:turn": WROTE_THE_FIX,
    "d:1;react:turn": ANSWERED,
    "d:0;tool:write_file": {MODULE: FIXED},
}
"""The loop's answers, keyed under the `d:{i}` frame of each turn."""

RECORDED = {
    "d:0;state:draft;d:0;react:turn": WROTE_THE_FIX,
    "d:0;state:draft;d:1;react:turn": ANSWERED,
    "d:0;state:draft;d:0;tool:write_file": {MODULE: FIXED},
    "tool:run_suite": CommandRun(exit_code=0),
}
"""The same answers placed inside the walk's first visit, and the suite green wherever it runs."""


def test_the_tool_log_re_derives_on_replay():
    """THE ACCEPTANCE CONDITION for folding inside `act` rather than widening the loop.

    The `ToolLog` is workflow-local mutable state, the shape that breaks replay when a machine
    hashes a shared dict in its postamble: `ReplayHandler` sees a different artifact key. What
    differs here is where the values come from. Every entry in the log is a `step` RESULT, so a
    replay, which runs no tool at all, rebuilds the identical log and folds the identical
    `Evidence.tree`.

    Asserted on the ARTIFACT ID, deliberately. It is content-addressed over the committed tree,
    so it is the one value that cannot agree by accident: a replay that re-derived an empty or
    stale workspace would commit a different digest."""
    handler = RecordingHandler(responses=RECORDED)
    recorded = handler.run(machine)
    assert isinstance(recorded, Session)
    assert recorded.commitment is not None
    assert recorded.commitment.files == (MODULE,)

    replayed = ReplayHandler(handler.trace).run(machine)
    assert isinstance(replayed, Session)
    assert replayed.commitment is not None
    assert replayed.commitment.artifact_id == recorded.commitment.artifact_id
    assert replayed.path == recorded.path


def test_the_recorded_commit_is_the_EDITED_tree_not_the_seed():
    """The digest above is only evidence if it is a digest of the right thing. A run that
    committed the seed tree on both passes would satisfy the equality perfectly."""
    seed = RecordingHandler(responses=RECORDED)
    edited = seed.run(machine)
    assert isinstance(edited, Session)

    untouched = RecordingHandler(
        responses={**RECORDED, "d:0;state:draft;d:0;tool:write_file": {MODULE: BROKEN}}
    )
    unedited = untouched.run(machine)
    assert isinstance(unedited, Session)

    assert edited.commitment is not None
    assert unedited.commitment is not None
    assert edited.commitment.artifact_id != unedited.commitment.artifact_id


# --- the two folds, and the per-visit rule, at their own grain -----------------------------------
#
# Driven directly rather than through a walk. Both properties below are invisible to the machine
# tests above — a mutation battery found them surviving: dropping the `suite` fold reddened
# nothing, because every judge in this file is scripted; and building the `ToolLog` per FACTORY
# instead of per invocation reddened nothing, because no walk here re-enters a state.


def logged(*entries: tuple[str, object]) -> ToolLog:
    """A `ToolLog` as `typed_act` would have built it, without running a loop."""
    log = ToolLog()
    log.entries.extend((CODING_TOOLS[name], value) for name, value in entries)
    return log


EMPTY_TRAJECTORY = Trajectory(answer="done", steps=[], stop_reason="finish")


def test_coding_read_folds_the_measurement_a_mechanical_judge_needs():
    """`Evidence.measured` is what makes three of the machine's states need no model, and a worker
    that ran the predicate must hand it forward. `mechanical_judges` raises BY NAME on a `None`,
    so a dropped fold surfaces as a wiring error rather than as a wrong verdict — but only if
    something asserts the fold happens at all."""
    # Built by concatenation, not spelled whole, and that is the house form rather than a dodge:
    # a full node id in a literal reads as a CITATION to `--test-citations`, and this names a test
    # in a fixture's imaginary tree. `test_coding_verdicts.py` composes its fixtures the same way.
    measured = CommandRun(exit_code=1, failures=("tests/t_sign.py::" + TARGET,))
    evidence = coding_read(EMPTY_TRAJECTORY, logged(("run_suite", measured)))
    assert evidence.measured == measured


def test_coding_read_folds_by_the_DECLARED_type_not_the_runtime_shape():
    """`read_file` and `check` both return `str`; `write_file` and `structural_apply` both return
    a tree. A fold that sniffed the value would pick whichever ran last of a matching SHAPE — so
    the discriminator is what the tool DECLARED, which is the only thing that distinguishes a
    workspace from a diff."""
    log = logged(("write_file", {MODULE: FIXED}), ("read_file", FIXED), ("check", "clean"))
    evidence = coding_read(EMPTY_TRAJECTORY, log)
    assert evidence.tree == {MODULE: FIXED}, "a later `str` result displaced the tree"


def test_coding_read_takes_the_LAST_result_of_each_kind():
    """Two edits in one visit: the workspace is what the second one left, not the first."""
    log = logged(("write_file", {MODULE: BROKEN}), ("write_file", {MODULE: FIXED}))
    assert coding_read(EMPTY_TRAJECTORY, log).tree == {MODULE: FIXED}


def test_nothing_of_a_kind_folds_to_None_rather_than_to_empty():
    """`None` says "not measured"; an empty `CommandRun` would say "measured, and clean", and an
    empty tree would say "the workspace is empty" — which would commit a deletion."""
    evidence = coding_read(EMPTY_TRAJECTORY, logged(("read_file", BROKEN)))
    assert evidence.measured is None
    assert evidence.tree is None


def test_a_second_visit_does_not_inherit_the_first_visits_results():
    """THE PER-INVOCATION RULE, and a self-edge makes it the ordinary case rather than an exotic
    one: `DraftVerdict.STILL_RED` routes DRAFT to itself, so a state's worker closure is re-entered
    with the machine expecting a fresh answer.

    A `ToolLog` built beside the closure instead of inside it would carry visit 0's edit into
    visit 1's evidence — and the trampoline advances its workspace from exactly that field, so a
    visit that touched nothing would re-commit work it did not do."""
    worker = agent_worker(prompt=lambda ctx: ctx.goal, **WITH_THE_TABLE)
    ctx = Ctx(run_id=Segment("ft7"), goal="fix the sign", state=State.DRAFT, visit=0)

    edited = RecordingHandler(responses=LOOP).run(lambda: worker(ctx))
    assert isinstance(edited, Evidence)
    assert edited.tree == {MODULE: FIXED}

    # The second visit's loop answers without touching a tool — so it changed nothing, and must
    # say so. The trampoline keeps what it had when a worker returns `None`.
    quiet = RecordingHandler(responses={"d:0;react:turn": ANSWERED}).run(lambda: worker(ctx))
    assert isinstance(quiet, Evidence)
    assert quiet.tree is None, "visit 1 reported visit 0's workspace as its own"


def test_the_TYPED_act_seam_also_lands_a_key_per_turn():
    """At `typed_act` too, one tool written on two turns lands two keys.

    The `ToolLog` sees both writes, which is what makes the second edit the one that reaches the
    commit."""
    wrote_twice = {
        "d:0;react:turn": AssistantTurn(
            thought="first",
            tool=ToolRequest(name="write_file", args={"path": MODULE, "content": BROKEN}),
        ),
        "d:0;tool:write_file": {MODULE: BROKEN},
        "d:1;react:turn": AssistantTurn(
            thought="second",
            tool=ToolRequest(name="write_file", args={"path": MODULE, "content": FIXED}),
        ),
        "d:1;tool:write_file": {MODULE: FIXED},
        "d:2;react:turn": ANSWERED,
    }
    handler = RecordingHandler(responses=wrote_twice)
    log = ToolLog()
    act = typed_act(CODING_TOOLS, log)
    handler.run(lambda: run_agent("fix it", act=act))

    keys = [e.key.stored() for e in handler.trace if "tool:" in e.key.stored()]
    assert keys == ["d:0;step;tool:write_file", "d:1;step;tool:write_file"]
    assert len(set(keys)) == 2, "one tool called twice still composes one key"
    assert log.last(dict[str, str]) == {MODULE: FIXED}, "the fold kept the wrong write"
