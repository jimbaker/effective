"""The loop is a durable driver over a typed step — `decide`/`act` are seams.

`test_react` pins the *default* behavior (the canonical `react:turn` / `tool:<name>`
op stream). These tests pin the *parameterization*: an arbitrary reasoning step
(including a pure-Python policy with no LLM op — the combinator-RLM / DSPy-module
slot) and an arbitrary tool-dispatch both drop into the same loop, and the loop
still drives them to a `Trajectory`.
"""

from _steps import called_step

from effective import RecordingHandler, call_tool
from effective.domain import CallTool
from effective.ops import Step
from effective.react import (
    AssistantTurn,
    ToolRequest,
    ToolResult,
    Trajectory,
    run_agent,
)


def test_loop_drives_a_pure_python_step():
    """`decide` need not call a model: a pure policy is a valid RLM step. Only the
    *action* is a checkpointed op; the reasoning step yields nothing."""
    script = iter(
        [
            AssistantTurn(thought="use the tool", tool=ToolRequest(name="echo", args={"x": "hi"})),
            AssistantTurn(thought="done", answer="echoed: hi"),
        ]
    )

    def decide(request):
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

    def decide(request):
        yield from ()
        return next(script)

    def act(request, tag):
        # One router tool answers every request, and the requested tool rides in its args.
        args = {"tool": request.name, **request.args}
        result = yield from call_tool("wrapped", args, ToolResult)
        return result

    h = RecordingHandler({"tool:wrapped": ToolResult(content="r")})
    traj = h.run(lambda: run_agent("q", decide=decide, act=act))

    assert isinstance(traj, Trajectory)
    assert traj.answer == "ok"
    assert [e.key.stored() for e in h.trace] == [called_step("wrapped", "d:0;")]
    match h.trace[0].op:
        case Step(op=CallTool(args=args)):
            assert args == {"tool": "search", "q": "x"}
        case unexpected:
            raise AssertionError(unexpected)
