"""The durable subagent — its own checkpoints; a parent crash resumes it (Deepen #3).

`spawn_subagent` runs a nested `run_agent` inline under the parent's handler, namespaced
by the child's name, so the child's steps are checkpointed in the parent's durable ctx.
We record a parent that delegates to a child (ask → tool → ask), then simulate a crash
after the child's *tool* step but before its final turn, and resume: only the child's
un-checkpointed tail re-runs — the already-checkpointed turn and tool replay, the tool
never re-executes. The child *resumed* rather than restarted.

Contrast with `effective.interpreters.tools.subagent_runner` (the opaque, restart-on-crash child):
there the whole child is one parent Step. Here the child trades trace-opacity for durability; its
cost lands on the parent meter directly (no telemetry-peek needed).
"""

import pytest

from effective import call_tool
from effective.compose import spawn_subagent
from effective.contexts import RecordingCtx, ResumeCtx
from effective.cost import MeteredInterpreter, Usage
from effective.handlers.durable import DurableHandler
from effective.interpreters.scripted import scripted_caller
from effective.interpreters.tools import make_tool_runner
from effective.keys import Key
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent

pytestmark = pytest.mark.spine

U = Usage(prompt_tokens=10, completion_tokens=4, cost=0.001)
ACT_DOUBLE = AssistantTurn(thought="compute", tool=ToolRequest(name="double", args={"n": 21}))
FINISH_42 = AssistantTurn(thought="done", answer="42")


def parent_decide():
    delegate = ToolRequest(name="delegate", args={"task": "x"})
    script = iter(
        [
            AssistantTurn(thought="delegate", tool=delegate),
            AssistantTurn(thought="relay", answer="child said 42"),
        ]
    )

    def decide(messages, tag):
        yield from ()
        return next(script)

    return decide


def parent_act(request, tag):
    if request.name == "delegate":
        result = yield from spawn_subagent("child", request.args["task"], max_iters=3)
        return result
    result = yield from call_tool(request.name, request.args, ToolResult)
    return result


def program():
    return run_agent("solve", decide=parent_decide(), act=parent_act)


def test_durable_subagent_resumes_the_child_after_a_parent_crash():
    # --- record the full run ---
    rec_ctx = RecordingCtx()
    rec_interp = MeteredInterpreter(
        llm=scripted_caller([(ACT_DOUBLE, U), (FINISH_42, U)]),
        tools=make_tool_runner({"double": lambda a: a["n"] * 2}),
    )
    rec = Trajectory.model_validate(DurableHandler(rec_ctx, rec_interp).run(program))

    assert rec.answer == "child said 42"
    # the child's steps are checkpointed under namespaced keys in the PARENT's ctx
    expected = {
        "d:0;sub:child;d:0;step;react:turn",
        "d:0;sub:child;d:0;step;tool:double",
        "d:0;sub:child;d:1;step;react:turn",
    }
    assert expected <= {k.stored() for k in rec_ctx.log}
    # inline => the child's cost lands on the parent meter (no telemetry-peek needed)
    assert rec_interp.meter.cost == 0.002

    # --- simulate a crash after the child's tool step, before its final turn ---
    # `Key.parse`, not the bare literal: the log is keyed by IDENTITY, so a `!=` against the
    # raw string
    # is True for EVERY key — the filter would drop nothing, the child would replay whole, and
    # the assertion below would fail with an empty `ran` rather than a wrong one. The silent
    # half of the opaque-`Key` trade, landing where a test can see it.
    crashed_at = Key.parse("d:0;sub:child;d:1;step;react:turn")
    partial = {k: v for k, v in rec_ctx.log.items() if k != crashed_at}

    def exploding_double(args):
        raise AssertionError("the child's double tool must not re-run on resume")

    resume_ctx = ResumeCtx(partial)
    resume_interp = MeteredInterpreter(
        llm=scripted_caller([(FINISH_42, U)]),  # only the child's final turn re-runs
        tools=make_tool_runner({"double": exploding_double}),
    )
    resumed = Trajectory.model_validate(DurableHandler(resume_ctx, resume_interp).run(program))

    assert resumed.answer == "child said 42"
    # only the un-checkpointed child step re-ran; the turn + tool before the crash replayed
    assert [k.stored() for k in resume_ctx.ran] == ["d:0;sub:child;d:1;step;react:turn"]
    # the resume paid for exactly one turn (the tail), not the whole child
    assert resume_interp.meter.cost == 0.001
