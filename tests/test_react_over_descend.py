"""`run_agent` is a `descend` judge: a turn is a level under `d:{i}`, the final level runs the
nudged decide, and a grantor adds turns before it.

The depth test is a guard: the `for` loop passed it too. It fails for a loop that keeps a frame per
turn, as a `yield from` recursion does (`tests/test_descend_depth.py`)."""

import sys
from collections.abc import Iterator, Mapping
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op

from effective.api import Effect, qualified_event_name, scoped
from effective.budget import Grant, depth_grant_name
from effective.combinators import grant_cascade, human_grant
from effective.compose import ASK_HUMAN, make_act
from effective.domain import AskLLM, CallTool, DomainOp
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Index, Run, compose_key
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent

DEPTH = 2 * sys.getrecursionlimit()
SPIN = AssistantTurn(thought="spin", tool=ToolRequest(name="spin"))


class Answers(Mapping[str, Any]):
    """Every turn acts and every action answers. A `Mapping` the recorder reads as given, where it
    copies a `dict`."""

    def __getitem__(self, key: str) -> Any:
        return ToolResult(content="spun") if key.endswith("tool:spin") else SPIN

    def __contains__(self, key: object) -> bool:
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def walk(handler: RecordingHandler, max_iters: int) -> Trajectory:
    match handler.run(lambda: run_agent("spin", max_iters=max_iters)):
        case Suspended() as parked:
            raise AssertionError(f"no park was expected: {parked}")
        case trajectory:
            return trajectory


def test_run_agent_past_the_recursion_limit_on_the_recorder_and_replay():
    handler = RecordingHandler(Answers())
    trajectory = walk(handler, DEPTH)
    assert len(trajectory.steps) == DEPTH + 1
    assert trajectory.stop_reason == "max_iters"
    replayed = ReplayHandler(handler.trace).run(lambda: run_agent("spin", max_iters=DEPTH))
    assert replayed == trajectory


def test_zero_turns_is_one_final_turn():
    handler = RecordingHandler(Answers())
    trajectory = walk(handler, 0)
    assert [e.key.stored() for e in handler.trace] == ["d:0;step;react:final"]
    assert (len(trajectory.steps), trajectory.stop_reason) == (1, "max_iters")


class SpinDomain:
    """A model that always acts, and a tool that always answers. The final turn's action is
    never run: its thought is the answer."""

    def __init__(self) -> None:
        self.asks = 0
        self.tools = 0

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM():
                self.asks += 1
                return SPIN
            case CallTool(name="spin"):
                self.tools += 1
                return ToolResult(content="spun")
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def add_two_turns_once(depth: int) -> Effect[Grant]:
    return Grant(add_depth=2) if depth == 1 else Grant(stop=True)
    yield  # pragma: no cover  -- a `Grantor` is a generator


def granted(run_id: str) -> Effect[dict[str, Any]]:
    trajectory = yield from run_agent("spin", max_iters=1, grantor=add_two_turns_once)
    return {"steps": len(trajectory.steps), "stop": trajectory.stop_reason}


def test_a_grantor_adds_turns_on_both_engines(backend):
    name, run_id = compose_key(t"granted-turns:{Run(str(uuid4()))}").stored(), str(uuid4())
    domain = SpinDomain()
    backend.register(name, granted, domain, Fault(), [])
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    assert snap.state == "completed", snap
    assert snap.result == {"steps": 4, "stop": "max_iters"}
    assert (domain.asks, domain.tools) == (4, 3)


def granted_by_a_human(run_id: str) -> Effect[dict[str, Any]]:
    grantor = grant_cascade([human_grant(run_id)])
    trajectory = yield from run_agent("spin", max_iters=1, grantor=grantor)
    return {"steps": len(trajectory.steps), "stop": trajectory.stop_reason}


GRANT_CYCLE_OPS = {FaultPosition.BEFORE_OP: 16, FaultPosition.AFTER_THUNK: 5}
"""Where a crash can land in a run of a turn, a park, a turn, a park and the final turn, counted
across the run's incarnations: before any op, or after one of the five steps' thunks."""


def run_granted(backend, fault: Fault):
    name, run_id = compose_key(t"human-turns:{Run(str(uuid4()))}").stored(), str(uuid4())
    domain = SpinDomain()
    backend.register(name, granted_by_a_human, domain, fault, [])
    task_id = backend.spawn(name, run_id)
    depth = 1
    for grant in ({"add_depth": 1}, {"stop": True}):
        snap = backend.run_until_result(task_id)
        assert snap.state not in ("completed", "failed"), (depth, snap)
        park = depth_grant_name(run_id, generation=0, depth=depth)
        backend.emit_event(task_id, park.stored(), grant)
        depth += grant.get("add_depth", 0)
    return backend.run_until_result(task_id), domain


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_human_grant_adds_turns_across_a_crash_at_every_op_on_both_engines(backend, position):
    """A turn, a park for more turns, a grant, a turn, a second park, a stop, the final turn. A
    crash lands at every position the run has, and it converges to the same trajectory."""
    unarmed = Fault(position=position)
    snap, _ = run_granted(backend, unarmed)
    assert snap.state == "completed", snap
    assert unarmed.count == GRANT_CYCLE_OPS[position], (
        "the loop changed shape; re-derive the bound"
    )
    for k, fault in at_every_op(unarmed):
        snap, domain = run_granted(backend, fault)
        assert snap.state == "completed", (k, snap)
        assert snap.result == {"steps": 3, "stop": "max_iters"}, k
        if position is FaultPosition.BEFORE_OP:
            assert domain.tools == 2, (k, domain.tools)


class AskingDomain:
    """A model that asks a human until a tool result is in the transcript, then answers with it."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages) if any(m["role"] == "tool" for m in messages):
                return AssistantTurn(thought="heard", answer=messages[-1]["content"])
            case AskLLM():
                return AssistantTurn(thought="ask", tool=ToolRequest(name=ASK_HUMAN))
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def shipping(run_id: str) -> Effect[str]:
    trajectory = yield from scoped(
        compose_key(t"shipping:{Run(run_id)}"),
        lambda: run_agent("ship it?", act=make_act()),
    )
    return trajectory.answer


def test_a_human_ask_parks_inside_its_turns_frame_on_both_engines(backend):
    """`ask:{turn}` sits under the turn's `d:{i}` frame, so the emitter qualifies it by both the
    run's scope and the frame."""
    name, run_id = compose_key(t"shipping-task:{Run(str(uuid4()))}").stored(), f"r{uuid4().hex}"
    backend.register(name, shipping, AskingDomain(), Fault(), [])
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    assert snap.state not in ("completed", "failed"), snap
    park = qualified_event_name(
        compose_key(t"shipping:{Run(run_id)}"),
        compose_key(t"d:{Index(0)}"),
        name=compose_key(t"ask:{Index(0)}").stored(),
    )
    backend.emit_event(task_id, park.stored(), {"text": "ship it"})
    snap = backend.run_until_result(task_id)
    assert snap.state == "completed", snap
    assert snap.result == "ship it"


class Broken(Exception):
    """A defect in a `decide`, which no refusal covers."""


def broken_decide(messages, level):
    raise Broken("the decide is broken")
    yield  # pragma: no cover  -- a `Decide` is a generator


def catching(run_id: str) -> Effect[str]:
    try:
        yield from run_agent("go", decide=broken_decide)
    except Broken:
        return "caught"
    return "not raised"


def test_a_non_refusal_raised_inside_a_turn_ends_the_run_on_every_interpreter(backend):
    """A turn is a scoped body, and only a refusal crosses a scope back into the workflow, so the
    caller's `except` never sees the defect: the recorder raises it and each engine fails the task
    of it once."""
    with pytest.raises(Broken):
        RecordingHandler().run(lambda: catching("r"))
    name, run_id = compose_key(t"broken-turn:{Run(str(uuid4()))}").stored(), str(uuid4())
    backend.register(name, catching, SpinDomain(), Fault(), [])
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == Broken.__name__
