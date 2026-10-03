"""A subagent is an opaque tool to the parent; its cost is peeked via telemetry.

The parent delegates via a single `call_tool` Step. The child runs a full nested
`run_agent` (its own turns + its own narrow tool subset) under its own metered +
traced interpreter. Two properties hold:

1. **Opaque + replay-isolated.** The parent records one `tool:delegate` op with the
   child's final answer; on replay it binds that checkpoint without re-running the child.
2. **Telemetry is the peek.** The cost layer meters only `AskLLM`, so the child's
   spend is invisible to the *parent's* meter; the child's spans (tagged `agent_name`,
   carrying `Usage`) are how the parent observes it.
"""

from agent.runtime import LocalCtx, make_tool_runner, subagent_runner
from effective.cost import MeteredInterpreter, Usage
from effective.domain import AskLLM
from effective.handlers.absurd import DurableHandler
from effective.react import AssistantTurn, ToolRequest, Trajectory, run_agent


def scripted(turns):
    """A deterministic `LLMCall`: return each `(AssistantTurn, Usage)` in order."""
    it = iter(turns)

    def call(op: AskLLM):
        return next(it)

    return call


def boom(op):
    raise AssertionError("the parent must not call a model in this scenario")


def test_subagent_is_opaque_and_peeked_via_telemetry():
    spans = []

    # the child's model: act once (double 21), then answer.
    sub_caller = scripted(
        [
            (
                AssistantTurn(thought="compute", tool=ToolRequest(name="double", args={"n": 21})),
                Usage(prompt_tokens=10, completion_tokens=5, cost=0.001),
            ),
            (
                AssistantTurn(thought="report", answer="the answer is 42"),
                Usage(prompt_tokens=12, completion_tokens=4, cost=0.002),
            ),
        ]
    )
    delegate = subagent_runner(
        sub_caller, {"double": lambda a: a["n"] * 2}, agent_name="math-sub", sink=spans.append
    )

    # the parent: a pure step that delegates once, then relays the child's answer.
    parent_turns = iter(
        [
            AssistantTurn(
                thought="delegate the math",
                tool=ToolRequest(name="delegate", args={"task": "double 21"}),
            ),
            AssistantTurn(thought="relay", answer="subagent said: the answer is 42"),
        ]
    )

    def decide(messages, tag):
        yield from ()
        return next(parent_turns)

    interp = MeteredInterpreter(
        llm=boom, tools=make_tool_runner({}, agents={"delegate": delegate})
    )
    handler = DurableHandler(ctx=LocalCtx(), domain=interp)
    traj = Trajectory.model_validate(handler.run(lambda: run_agent("solve it", decide=decide)))

    assert "42" in traj.answer
    # the parent paid nothing at the model layer — the child's spend is not on its meter
    assert interp.meter.cost == 0.0

    # ...but telemetry peeked the child: LLM + tool spans, tagged, carrying real usage
    sub = [s for s in spans if s.agent_name == "math-sub"]
    assert {s.kind for s in sub} == {"LLM", "TOOL"}
    sub_cost = sum(
        s.usage_attributes.get("effective.cost.usd", 0.0) for s in sub if s.kind == "LLM"
    )
    assert round(sub_cost, 3) == 0.003


def test_subagent_cost_can_be_folded_into_a_parent_meter():
    """`accrue` lets a bench total the real (paid-once) cost across parent + children,
    even though the parent's own meter never sees the child's `AskLLM`s."""
    total = Usage()

    def accrue(u: Usage) -> None:
        nonlocal total
        total = total + u

    sub_caller = scripted(
        [(AssistantTurn(thought="done", answer="hi"), Usage(prompt_tokens=7, cost=0.005))]
    )
    delegate = subagent_runner(sub_caller, {}, agent_name="sub", accrue=accrue)

    parent_turns = iter(
        [
            AssistantTurn(
                thought="go", tool=ToolRequest(name="delegate", args={"task": "say hi"})
            ),
            AssistantTurn(thought="done", answer="ok"),
        ]
    )

    def decide(messages, tag):
        yield from ()
        return next(parent_turns)

    interp = MeteredInterpreter(
        llm=boom, tools=make_tool_runner({}, agents={"delegate": delegate})
    )
    DurableHandler(ctx=LocalCtx(), domain=interp).run(lambda: run_agent("q", decide=decide))

    assert total.cost == 0.005
