"""The loop is a durable driver over a typed step — `decide`/`act` are seams.

`test_react` pins the *default* behavior (the canonical `react:turn` / `tool:<name>`
op stream). These tests pin the *parameterization*: an arbitrary reasoning step
(including a pure-Python policy with no LLM op — the combinator-RLM / DSPy-module
slot) and an arbitrary tool-dispatch both drop into the same loop, and the loop
still drives them to a `Trajectory`.
"""

from effective import RecordingHandler, call_tool
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent


def test_loop_drives_a_pure_python_step():
    """`decide` need not call a model: a pure policy is a valid RLM step. Only the
    *action* is a checkpointed op; the reasoning step yields nothing."""
    script = iter(
        [
            AssistantTurn(thought="use the tool", tool=ToolRequest(name="echo", args={"x": "hi"})),
            AssistantTurn(thought="done", answer="echoed: hi"),
        ]
    )

    def decide(messages, tag):
        yield from ()  # a pure step: re-derives on replay, yields no op
        return next(script)

    h = RecordingHandler({"d:0;tool:echo": ToolResult(content="hi")})
    traj = h.run(lambda: run_agent("q", decide=decide))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "echoed: hi"
    # the only recorded op is the action — the step left no trace of its own
    assert [e.key.stored() for e in h.trace] == ["d:0;step;tool:echo"]


def test_custom_act_rewrites_tool_dispatch():
    """`act` owns how an action runs — here it routes every tool through a renamed
    op, proving dispatch is a seam (the same hook a subagent / HITL act plugs into)."""
    script = iter(
        [
            AssistantTurn(thought="act", tool=ToolRequest(name="search", args={"q": "x"})),
            AssistantTurn(thought="done", answer="ok"),
        ]
    )

    def decide(messages, tag):
        yield from ()
        return next(script)

    def act(request, tag):
        # `wrapped-`, not `wrapped:`: `call_tool` splices the name into `tool:{Segment(name)}`,
        # one coordinate, and a `Segment` refuses an interior delimiter loudly rather than
        # encoding it. The rename is this test's own vocabulary; what it exercises — a custom
        # `act` rewriting tool dispatch — is unchanged.
        result = yield from call_tool(f"wrapped-{request.name}", request.args, ToolResult)
        return result

    h = RecordingHandler({"tool:wrapped-search": ToolResult(content="r")})
    traj = h.run(lambda: run_agent("q", decide=decide, act=act))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "ok"
    assert [e.key.stored() for e in h.trace] == ["d:0;step;tool:wrapped-search"]
