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

from uuid import uuid4

import pytest
from _durable import IMMEDIATE_RETRY, Fault, FaultCtx, absurd, pg_ready, run_until_result

from agent.bench import scripted_caller
from agent.compose import spawn_subagent_task
from agent.runtime import make_tool_runner, run_subagent_as_task, spawn_tool
from effective import call_tool
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL
from effective.handlers.absurd import DurableHandler
from effective.react import AssistantTurn, ToolRequest, ToolResult, run_agent

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")

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


def _register_parent(app, parent_name: str, child_name: str, spawner, fault: Fault | None = None):
    @app.register_task(parent_name)
    def parent(params, ctx):
        mid = params["mid"]

        def decide(messages, level):
            yield from ()
            if level.depth == 0:
                return AssistantTurn(thought="go", tool=ToolRequest(name="delegate", args={}))
            return AssistantTurn(thought="relay", answer=f"child said: {messages[-1]['content']}")

        def act(request, tag):
            if request.name == "delegate":
                result = yield from spawn_subagent_task(child_name, "double 21", correlation=mid)
                return result
            result = yield from call_tool(request.name, request.args, ToolResult)
            return result

        domain = MeteredInterpreter(
            llm=_boom_llm, tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)})
        )
        run_ctx = FaultCtx(ctx, fault) if fault is not None else ctx
        return DurableHandler(run_ctx, domain).run(lambda: run_agent("go", decide=decide, act=act))


def _make_spawner(spawner_app, seen_ids: list[str]):
    """Spawn the child on an *independent* connection (never the running task's ctx
    connection). Records each returned id so a test can assert no duplicate child."""

    def spawn(task_name, params, idempotency_key, queue, *, max_attempts=None):
        result = spawner_app.spawn(
            task_name,
            params,
            queue=queue,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_strategy=IMMEDIATE_RETRY,
        )
        task_id = str(result["task_id"])  # the SDK returns a UUID; the channel schema is str
        seen_ids.append(task_id)
        return task_id

    return spawn


def test_spawned_child_is_its_own_task_and_the_parent_awaits_it():
    app, spawner_app = absurd(), absurd()
    try:
        mid = f"spawn-{uuid4().hex[:8]}"
        pname, cname = f"p-{mid}", f"c-{mid}"
        ids: list[str] = []
        _register_child(app, cname)
        _register_parent(app, pname, cname, _make_spawner(spawner_app, ids))

        spawned = app.spawn(pname, {"mid": mid}, retry_strategy=IMMEDIATE_RETRY)
        snap = run_until_result(app, spawned["task_id"])

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["answer"] == "child said: 42"  # the parent relayed the child's answer

        # the child ran as its OWN task: a distinct, completed task with its own result
        child_id = ids[0]
        assert child_id != str(spawned["task_id"])
        child_snap = app.fetch_task_result(child_id)
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
        _register_parent(app, pname, cname, _make_spawner(spawner_app, ids), fault=Fault(2))

        spawned = app.spawn(pname, {"mid": mid}, retry_strategy=IMMEDIATE_RETRY)
        snap = run_until_result(app, spawned["task_id"])

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["answer"] == "child said: 42"
        # the spawn was checkpointed: across the crash + retry the child was spawned once
        assert len(set(ids)) == 1, ids
    finally:
        app.close()
        spawner_app.close()
