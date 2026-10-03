"""Refused-as-observation routing.

A permission cascade blocks a tool; instead of aborting the run, the loop catches the
`Refused` (delivered *into* the workflow by the handler) and turns the denial into the
next observation, so the step routes around it. The denial is recorded, so the whole
trajectory, denied turn included, replays deterministically without re-running the
cascade.
"""

import pytest
from _gate import at_spend

from effective import Allow, Deny, RecordingHandler, ReplayHandler, cascade, rules
from effective.budget import MeasuredBudget
from effective.budget import as_policy as budget_policy
from effective.domain import CallTool
from effective.govern import BudgetRefused, Proceed, govern
from effective.ops import Step
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent


def block_danger(op):
    """Deny the `danger` tool; allow everything else."""
    if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name == "danger":
        return Deny("danger tool is not permitted")
    return Allow()


def make_decide():
    script = iter(
        [
            AssistantTurn(thought="try the risky path", tool=ToolRequest(name="danger", args={})),
            AssistantTurn(thought="rerouting", tool=ToolRequest(name="safe", args={})),
            AssistantTurn(thought="done", answer="used the safe path instead"),
        ]
    )

    def decide(messages, tag):
        yield from ()
        return next(script)

    return decide


def gate():
    return cascade([rules(block_danger)])


def test_denied_tool_becomes_an_observation_and_the_loop_reroutes():
    canned = {"d:1;tool:safe": ToolResult(content="safe result")}
    h = RecordingHandler(canned, op_layers=[gate()])
    traj = h.run(lambda: run_agent("solve it", decide=make_decide()))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "used the safe path instead"
    # the denied turn recorded the denial as its observation; the loop then used `safe`
    assert traj.steps[0].observation == "[denied] danger tool is not permitted"
    assert traj.steps[1].observation == "safe result"
    # the refusal is in the trace (as an error entry) followed by the rerouted tool
    assert [(e.key.stored(), e.error is not None) for e in h.trace] == [
        ("d:0;step;tool:danger", True),
        ("d:1;step;tool:safe", False),
    ]


def test_refused_trajectory_replays_deterministically():
    canned = {"d:1;tool:safe": ToolResult(content="safe result")}
    rec = RecordingHandler(canned, op_layers=[gate()])
    live = rec.run(lambda: run_agent("solve it", decide=make_decide()))

    # replay gets only the recorded trace — no cascade, no canned responses re-run
    replayed = ReplayHandler(rec.trace).run(lambda: run_agent("solve it", decide=make_decide()))

    assert replayed == live
    assert replayed.steps[0].observation == "[denied] danger tool is not permitted"


def test_a_budget_refusal_ends_the_loop_at_the_refused_tool():
    """A spend ceiling refuses every later tool the same way, so the loop stops at the first one
    and asks the model for no further turn."""
    budget = MeasuredBudget(overall=0.005, run_id="r", on_exhaust="fail")
    over = at_spend(0.010, budget_policy(budget))

    def tools_only(op, state):
        is_tool = isinstance(op, Step) and isinstance(op.op, CallTool)
        return over(op, state) if is_tool else Proceed()

    turns = []

    def decide(messages, tag):
        turns.append(tag)
        return (yield from make_decide()(messages, tag))

    h = RecordingHandler({}, op_layers=[govern(tools_only, gate="spend", run_id="r")])
    with pytest.raises(BudgetRefused):
        h.run(lambda: run_agent("solve it", decide=decide))
    assert len(turns) == 1
    assert [e.key.stored() for e in h.trace] == ["d:0;step;tool:danger"]
