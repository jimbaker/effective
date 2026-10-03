"""`agent_worker`: the two grains composed.

The package docstring's diagram says `[ worker -> judge -> transition ]` with the worker being
`run_agent` one grain down. These tests drive that composition.

ROLE: journey: one path end to end through both grains. What a pass proves is that the
loop's ops land inside the state's scope and that the trampoline threads the result; it does NOT
prove anything about a model's choices, which are scripted here on purpose.
"""

from typing import Any

import pytest

from effective.coding.states import ReviewVerdict, State
from effective.coding.transition import Finish, transition
from effective.handlers.recording import RecordingHandler
from effective.keys import Run, Segment
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import agent_worker, build_specs
from effective.machine.trampoline import Session, run_machine
from effective.react import AssistantTurn, ToolLog, Trajectory

pytestmark = pytest.mark.journey


ANSWERED = AssistantTurn(thought="done", answer="fixed the bug")

TURNS = {"react:turn": ANSWERED}
"""What the loop's DEFAULT `decide` asks for. Driving through the default rather than a scripted
`decide` is deliberate: a scripted decider that returns without yielding produces NO ops, and every
assertion about where the loop's keys land would then hold of an empty tape. Measured — the first
version of this file did exactly that and its scope assertion passed vacuously."""


def scripted_decide(_needs):
    """A `Decide` factory of the shape `agent_worker` expects — repertoire in, decider out.

    This is where a deployment would put a bracketed caller masked to `needs`; masking is not the
    substrate's business, which is exactly why the parameter is a factory rather than a catalog.

    It yields NO op, which is fine for the one thing it is used to prove (that `needs` arrives) and
    is exactly why it is not used to prove anything about the tape."""

    def decide(_messages, _tag):
        return ANSWERED
        yield  # pragma: no cover  -- a `Decide` is a generator

    return decide


def approving_judge(_ctx: Ctx, _evidence: Evidence) -> Any:
    return ReviewVerdict.APPROVED
    yield  # pragma: no cover


def unreachable_worker(ctx: Ctx) -> Any:
    raise AssertionError(f"{ctx.state.value} should not run")
    yield  # pragma: no cover


def unreachable_judge(_ctx: Ctx, _evidence: Evidence) -> Any:
    raise AssertionError("no judge should run here")
    yield  # pragma: no cover


def specs_with_agent_review(**kwargs) -> dict[State, Any]:
    return build_specs(
        State,
        workers={State.REVIEW: agent_worker(prompt=lambda ctx: ctx.goal, **kwargs)},
        judges={State.REVIEW: approving_judge},
        default_worker=unreachable_worker,
        default_judge=unreachable_judge,
    )


GREEN = CommandRun(exit_code=0)
"""The postamble's predicate answer. It runs on EVERY exit path, so every driver here owes it one
— which is the unconditional tail asserting itself on the test harness, exactly as designed."""


def drive(specs) -> tuple[Session, RecordingHandler]:
    handler = RecordingHandler(responses={"tool:run_suite": GREEN} | TURNS)
    out = handler.run(
        lambda: run_machine(
            Run("aw-2026"), "review it", specs, transition, start=State.REVIEW, budget=4
        )
    )
    assert isinstance(out, Session)
    return out, handler


def test_a_run_agent_worker_drives_a_state_to_a_verdict():
    """The composition, running: REVIEW's work is a ReAct loop and its answer reaches the judge."""
    session, _ = drive(specs_with_agent_review(decide_for=scripted_decide))
    assert session.stopped == Finish()
    assert session.path == (State.REVIEW,)
    assert session.turns[0].verdict == ReviewVerdict.APPROVED


def test_the_loops_ops_land_INSIDE_the_state_scope():
    """The property that makes nesting safe, and the one a flat composition would lose.

    `run_agent` mints `react:turn` per turn. Driven as a state's worker it must land under that
    state's frame — otherwise two states running the same loop would collide on the tape, which is
    the `d:`/`state:` lesson one grain down."""
    _, handler = drive(specs_with_agent_review())
    react_keys = [e.key.stored() for e in handler.trace if "react:" in e.key.stored()]
    assert react_keys, "the worker yielded no ReAct op — the loop did not run"
    for key in react_keys:
        assert key.startswith("d:0;state:review;"), key


def test_the_default_reader_supplies_no_measurement_and_says_so():
    """`_answer_only` deliberately carries neither `suite` nor `tree`.

    Both are fields a replay depends on being derived from RECORDED results, and the default
    reader cannot know which tool result held them. A mechanical state wired to the default is a
    WIRING error, and `mechanical_judges` names it rather than failing on `None.green` two frames
    away — asserted here because a default that quietly returns half an `Evidence` is the kind of
    thing that only bites in a deployment."""
    from effective.coding.verdicts import mechanical_judges

    evidence = Evidence(summary="answered", detail="")
    judge = mechanical_judges("test_add")[State.TEST]
    # A `Judge` takes the state's `Ctx` as well as its evidence, so every judgement in the
    # machine knows where it is. This one ignores it — that is what its `_ctx` says.
    ctx = Ctx(run_id=Segment("r1"), goal="g", state=State.TEST, visit=0)
    with pytest.raises(ValueError, match="returned no measurement"):
        next(iter(judge(ctx, evidence)))


def test_the_repertoire_reaches_the_decide_factory():
    """`needs` is a state's repertoire, and the factory is the only thing that consumes it.

    If it did not arrive, `StateSpec.needs` would be a field nothing reads."""
    seen: list[frozenset[str]] = []

    def recording_decide_for(needs):
        seen.append(needs)
        return scripted_decide(needs)

    specs = specs_with_agent_review(
        decide_for=recording_decide_for, needs=frozenset({"read_file", "run_suite"})
    )
    drive(specs)
    assert seen == [frozenset({"read_file", "run_suite"})]


def test_a_worker_with_no_decide_factory_still_runs():
    """The factory is optional: with none, `run_agent` takes its own default `decide`, which is
    the plain `react:turn` op stream. A substrate that REQUIRED a caller could not be driven by a
    recording handler at all."""
    _, handler = drive(specs_with_agent_review())
    assert any("react:" in e.key.stored() for e in handler.trace)


def test_the_trajectory_reader_is_what_carries_a_measurement_forward():
    """`read` is the seam a real deployment uses — the map from a finished `Trajectory` to
    `Evidence` is domain knowledge, so it is a parameter rather than a guess."""

    def read(trajectory: Trajectory, log: ToolLog) -> Evidence:
        return Evidence(summary=trajectory.answer, detail=f"read by the deployment {log.names}")

    _, handler = drive(specs_with_agent_review(read=read))
    assert any("react:" in e.key.stored() for e in handler.trace)


def test_build_specs_refuses_a_map_that_is_not_total() -> None:
    """The failure this module exists to move from run time to assembly time."""
    with pytest.raises(ValueError, match="must be total over `State`"):
        build_specs(State, workers={State.REVIEW: unreachable_worker}, judges={})
