"""A ReAct loop's turns on both engines: what a refusal inside a turn reaches, and a run through an
interrupt win, two compactions and a human ask crashed at every op.

Adopted from the `react-over-descend` review's probes."""

from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op
from _gate import at_spend

from effective import Allow, Deny, cascade, rules
from effective.api import Effect, qualified_event_name, scoped
from effective.budget import MeasuredBudget
from effective.budget import as_policy as budget_policy
from effective.combinators import Level
from effective.compose import ASK_HUMAN, Interrupted, make_act, tool_interrupt
from effective.domain import INTERRUPT_TOOL, AskLLM, CallTool, DomainOp
from effective.govern import BudgetRefused, Proceed, Refused, govern, routable
from effective.keys import Index, Run, compose_key
from effective.ops import Step
from effective.react import (
    AssistantTurn,
    ToolRequest,
    ToolResult,
    Trajectory,
    TrajectorySummary,
    run_agent,
)

pytestmark = pytest.mark.conformance

DANGER = AssistantTurn(thought="risky", tool=ToolRequest(name="danger"))
SAFE = AssistantTurn(thought="safe", tool=ToolRequest(name="safe"))
DONE = AssistantTurn(thought="done", answer="finished")


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m["role"] == "tool"]


class RoutingDomain:
    """The model tries `danger`, then `safe`, then answers; each tool answers with its name."""

    def __init__(self) -> None:
        self.tools: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages):
                return [DANGER, SAFE, DONE][len(tool_results(messages))]
            case CallTool(name=name):
                self.tools.append(name)
                return ToolResult(content=f"{name} ran")
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def block_danger(op: Any) -> Allow | Deny:
    match op:
        case Step(op=CallTool(name="danger")):
            return Deny("danger tool is not permitted")
        case _:
            return Allow()


def block_second_ask(op: Any) -> Allow | Deny:
    match op:
        case Step(op=AskLLM(messages=messages)) if tool_results(messages):
            return Deny("no second turn")
        case _:
            return Allow()


def routed(run_id: str) -> Effect[dict[str, Any]]:
    try:
        trajectory = yield from run_agent("go", max_iters=4)
    except BudgetRefused as refused:
        return {"budget": refused.reason}
    except Refused as refused:
        return {"caught": routable(refused).reason}
    return {"answer": trajectory.answer, "obs": [s.observation for s in trajectory.steps]}


def refusing_decide(messages: list[dict[str, Any]], level: Level) -> Effect[AssistantTurn]:
    yield from ()
    if level.depth == 1:
        asked = Step("react:turn", AskLLM(messages=messages, response_schema=AssistantTurn))
        raise Refused(asked, "decide refused")
    return DANGER


def routed_past_a_refusing_decide(run_id: str) -> Effect[dict[str, Any]]:
    try:
        yield from run_agent("go", max_iters=4, decide=refusing_decide)
    except Refused as refused:
        return {"caught": routable(refused).reason}
    return {"not": "caught"}


def run_routed(backend, workflow, layers):
    run_id = str(uuid4())
    name = compose_key(t"routed-turns:{Run(run_id)}").stored()
    domain = RoutingDomain()
    backend.register(name, workflow, domain, Fault(), layers)
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    return snap, domain


def test_a_denied_tool_is_an_observation_on_both_engines(backend):
    snap, domain = run_routed(backend, routed, [cascade([rules(block_danger)])])
    assert snap.state == "completed", snap
    assert snap.result == {
        "answer": "finished",
        "obs": ["[denied] danger tool is not permitted", "safe ran", None],
    }
    assert domain.tools == ["safe"]


def test_a_denied_decide_reaches_the_caller_on_both_engines(backend):
    snap, domain = run_routed(backend, routed, [cascade([rules(block_second_ask)])])
    assert snap.state == "completed", snap
    assert snap.result == {"caught": "no second turn"}
    assert domain.tools == ["danger"]


def test_a_budget_refusal_reaches_the_caller_on_both_engines(backend):
    over = at_spend(
        0.01, budget_policy(MeasuredBudget(overall=0.005, run_id="r", on_exhaust="fail"))
    )

    def tools_only(op, state):
        match op:
            case Step(op=CallTool()):
                return over(op, state)
            case _:
                return Proceed()

    snap, domain = run_routed(backend, routed, [govern(tools_only, gate="spend", run_id="r")])
    assert snap.state == "completed", snap
    assert set(snap.result) == {"budget"}, snap.result
    assert domain.tools == []


def test_a_decide_that_raises_refused_reaches_the_caller_on_both_engines(backend):
    snap, _ = run_routed(backend, routed_past_a_refusing_decide, [])
    assert snap.state == "completed", snap
    assert snap.result == {"caught": "decide refused"}


SEARCH = AssistantTurn(thought="search", tool=ToolRequest(name="search", args={"q": "x"}))
ASK = AssistantTurn(thought="ask", tool=ToolRequest(name=ASK_HUMAN))
SHIPPED = AssistantTurn(thought="done", answer="shipped")


class MixedDomain:
    """The model searches, asks a human, and answers once the human says ship it. The interrupt
    wins after turn 0's decide; every other poll is quiet."""

    def __init__(self) -> None:
        self.asks = 0
        self.compactions = 0
        self.polls = 0
        self.tools = 0

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(response_schema=schema) if schema is TrajectorySummary:
                self.compactions += 1
                return TrajectorySummary(facts=["f"], tried=["t"])
            case AskLLM(messages=messages):
                self.asks += 1
                results = tool_results(messages)
                if any(m["content"] == "ship it" for m in results):
                    return SHIPPED
                return ASK if results else SEARCH
            case CallTool(name=name, args={"turn": 0, "phase": "post"}) if name == INTERRUPT_TOOL:
                self.polls += 1
                return Interrupted(redirect="stop")
            case CallTool(name=name) if name == INTERRUPT_TOOL:
                self.polls += 1
                return Interrupted()
            case CallTool(name="search"):
                self.tools += 1
                return ToolResult(content="found")
            case _:
                raise AssertionError(f"unexpected op {op!r}")

    def counts(self) -> tuple[int, int, int, int]:
        return self.asks, self.compactions, self.polls, self.tools


def mixed(run_id: str) -> Effect[dict[str, Any]]:
    trajectory: Trajectory = yield from scoped(
        compose_key(t"mix:{Run(run_id)}"),
        lambda: run_agent(
            "go",
            max_iters=4,
            act=make_act(),
            interrupt=tool_interrupt(),
            compact=lambda messages: len(messages) >= 4,
        ),
    )
    return {
        "answer": trajectory.answer,
        "stop": trajectory.stop_reason,
        "obs": [s.observation for s in trajectory.steps],
    }


def run_mixed(backend, fault: Fault):
    run_id = f"m{uuid4().hex}"
    name = compose_key(t"mixed-turns:{Run(run_id)}").stored()
    domain = MixedDomain()
    backend.register(name, mixed, domain, fault, [])
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    assert snap.state not in ("completed", "failed"), snap
    park = qualified_event_name(
        compose_key(t"mix:{Run(run_id)}"),
        compose_key(t"d:{Index(2)}"),
        name=compose_key(t"ask:{Index(2)}").stored(),
    )
    backend.emit_event(task_id, park.stored(), {"text": "ship it"})
    return backend.run_until_result(task_id), domain


MIXED = {
    "answer": "shipped",
    "stop": "finish",
    "obs": ["[interrupted] stop", "found", "ship it", None],
}

MIXED_OPS = {FaultPosition.BEFORE_OP: 28, FaultPosition.AFTER_THUNK: 15}
"""Where a crash can land in the mixed run, counted across its incarnations: before any op, or
after a step's thunk."""


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_an_interrupted_compacting_asking_loop_survives_a_crash_at_every_op(backend, position):
    unarmed = Fault(position=position)
    snap, golden = run_mixed(backend, unarmed)
    assert snap.state == "completed", snap
    assert snap.result == MIXED
    assert golden.counts() == (4, 2, 8, 1)
    assert unarmed.count == MIXED_OPS[position], "the loop changed shape; re-derive the bound"
    for k, fault in at_every_op(unarmed):
        snap, domain = run_mixed(backend, fault)
        assert snap.state == "completed", (k, snap)
        assert snap.result == MIXED, k
        if position is FaultPosition.BEFORE_OP:
            assert domain.counts() == golden.counts(), k
