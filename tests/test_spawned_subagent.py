"""Deepen #4 — the spawned subagent: a child Absurd *task* (opaque AND durable).

The third corner of the subagent design. The parent's trace is just a `spawn`
`CallTool` (checkpointed → idempotent) + an `AwaitEvent` on the child's completion (which
*suspends*, releasing the worker — deadlock-free on one queue); the child runs as a
*separate* task with its own checkpoints and `emit_event`s its answer. Proven on the
Podman test Postgres:

- the child runs as its own task (a distinct `task_id`, its own terminal result) and the
  parent receives the answer through the await — opaque (the child's ops are not in the
  parent's run);
- a parent crash *at the await* does not re-spawn or restart the child — the spawn replays
  from its checkpoint, the child (already its own durable task) is untouched, and the
  parent resumes to the same answer.
"""

from uuid import UUID, uuid4

import pytest
from _durable import IMMEDIATE_RETRY, Fault, FaultCtx, absurd, pg_ready
from _spawning import absurd_spawner

from effective import call_tool
from effective.budget import BUDGET_DEPTH_PARAM
from effective.compose import spawn_subagent_task
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL
from effective.govern import Refused
from effective.handlers.durable import DurableHandler
from effective.interpreters.scripted import scripted_caller
from effective.interpreters.tools import make_tool_runner, run_subagent_as_task, spawn_tool
from effective.react import AssistantTurn, ToolRequest, ToolResult, run_agent

pytestmark = [
    pytest.mark.spine,
    pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)"),
]

U = Usage(prompt_tokens=8, completion_tokens=3, cost=0.001)
ACT_DOUBLE = AssistantTurn(thought="compute", tool=ToolRequest(name="double", args={"n": 21}))
FINISH_42 = AssistantTurn(thought="done", answer="42")


def _boom_llm(op):
    raise AssertionError("the parent makes no model calls in this scenario")


def _register_child(app, child_name: str) -> None:
    @app.register_task(child_name)
    def child(params, ctx):
        # the child's OWN domain (own model + tools); run_agent under its OWN ctx.
        domain = MeteredInterpreter(
            llm=scripted_caller([(ACT_DOUBLE, U), (FINISH_42, U)]),
            tools=make_tool_runner({"double": lambda a: a["n"] * 2}),
        )
        return run_subagent_as_task(params, ctx, domain=domain, max_iters=3)


def _register_parent(
    app,
    parent_name: str,
    child_name: str,
    spawner,
    fault: Fault | None = None,
    max_attempts: int | None = None,
):
    @app.register_task(parent_name, default_max_attempts=max_attempts)
    def parent(params, ctx):
        mid = params["mid"]

        def decide(messages, level):
            yield from ()
            if level.depth == 0:
                return AssistantTurn(thought="go", tool=ToolRequest(name="delegate", args={}))
            return AssistantTurn(thought="relay", answer=f"child said: {messages[-1]['content']}")

        def act(request, tag):
            if request.name == "delegate":
                try:
                    return (
                        yield from spawn_subagent_task(child_name, "double 21", correlation=mid)
                    )
                except Refused:
                    return ToolResult(content="ceiling: depth exhausted")
            result = yield from call_tool(request.name, request.args, ToolResult)
            return result

        domain = MeteredInterpreter(
            llm=_boom_llm, tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)})
        )
        run_ctx = FaultCtx(ctx, fault) if fault is not None else ctx
        handler = DurableHandler(run_ctx, domain, params=params)
        return handler.run(lambda: run_agent("go", decide=decide, act=act))


def test_spawned_child_is_its_own_task_and_the_parent_awaits_it():
    app, spawner_app = absurd(), absurd()
    try:
        mid = f"spawn-{uuid4().hex[:8]}"
        pname, cname = f"p-{mid}", f"c-{mid}"
        ids: list[str] = []
        _register_child(app, cname)
        _register_parent(app, pname, cname, absurd_spawner(spawner_app, IMMEDIATE_RETRY, ids))

        spawned = app.spawn(pname, {"mid": mid})
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["answer"] == "child said: 42"  # the parent relayed the child's answer

        # the child ran as its OWN task: a distinct, completed task with its own result
        child_id = ids[0]
        assert child_id != str(spawned)
        child_snap = app.fetch_task_result(UUID(child_id))
        assert child_snap is not None
        assert child_snap.state == "completed"
        assert child_snap.result["answer"] == "42"
    finally:
        app.close()
        spawner_app.close()


def test_parent_crash_at_the_await_does_not_respawn_or_restart_the_child():
    app, spawner_app = absurd(), absurd()
    try:
        mid = f"crash-{uuid4().hex[:8]}"
        pname, cname = f"p-{mid}", f"c-{mid}"
        ids: list[str] = []
        _register_child(app, cname)
        # crash before the parent's 2nd ctx op (the await) — after the spawn has committed.
        _register_parent(
            app, pname, cname, absurd_spawner(spawner_app, IMMEDIATE_RETRY, ids), fault=Fault(2)
        )

        spawned = app.spawn(pname, {"mid": mid})
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["answer"] == "child said: 42"
        # the spawn was checkpointed: across the crash + retry the child was spawned once
        assert len(set(ids)) == 1, ids
    finally:
        app.close()
        spawner_app.close()


@pytest.mark.parametrize(
    ("depth", "answered", "children"),
    [(0, "ceiling: depth exhausted", 0), (1, "42", 1)],
    ids=["depth-0-refuses-before-enqueuing", "depth-1-permits-one-level"],
)
def test_the_depth_ceiling_holds_on_the_engine(depth, answered, children):
    """The handler holds a spawn to its task's depth: a refused spawn reaches the parent as a
    `Refused` it can answer, and enqueues nothing."""
    app, spawner_app = absurd(), absurd()
    try:
        mid = f"depth-{uuid4().hex[:8]}"
        pname, cname = f"p-{mid}", f"c-{mid}"
        ids: list[str] = []
        _register_child(app, cname)
        # One attempt: a retry would hide a refusal path that failed the first time.
        spawner = absurd_spawner(spawner_app, IMMEDIATE_RETRY, ids)
        _register_parent(app, pname, cname, spawner, max_attempts=1)

        params = {"mid": mid, BUDGET_DEPTH_PARAM: depth}
        spawned = app.spawn(pname, params)
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert answered in snap.result["answer"]
        assert len(ids) == children
    finally:
        app.close()
        spawner_app.close()
