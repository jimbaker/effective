"""The depth ceiling as a spawned subagent meets it.

The handler holds every spawn to its task's depth and refuses one past it
(`test_spawn_depth_ceiling.py`); these pin what `spawn_subagent_task` sees of that on Postgres:
depth 0 refuses with no child enqueued, and depth 1 permits exactly one level.
"""

import pytest

from agent.compose import spawn_subagent_task
from effective import call_tool
from effective.budget import BUDGET_DEPTH_PARAM, Budget
from effective.domain import SPAWN_TOOL
from effective.govern import Refused


def test_from_spawn_params_roundtrip_and_unbounded_default():
    parent = Budget(depth=2)
    child_params = {"task": "t", BUDGET_DEPTH_PARAM: parent.descend_one().depth}
    assert Budget.from_spawn_params(child_params).depth == 1
    # Absent key → unbounded (a spawn that carried no budget).
    assert Budget.from_spawn_params({"task": "t"}).depth is None


# --- durable: the ceiling holds on the real engine ------------------------------------

from _durable import IMMEDIATE_RETRY, absurd, pg_ready, run_until_result  # noqa: E402

from agent.bench import scripted_caller  # noqa: E402


@pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")
class TestDurableDepthCeiling:
    """Model on test_spawned_subagent: a parent task that delegates via a spawn, enqueued with a
    depth. Depth 0 refuses (graceful, no child); depth 1 permits one."""

    @staticmethod
    def _register_child(app, child_name):
        from agent.runtime import make_tool_runner, run_subagent_as_task
        from effective.cost import MeteredInterpreter, Usage
        from effective.react import AssistantTurn, ToolRequest

        u = Usage(prompt_tokens=8, completion_tokens=3, cost=0.001)
        act_double = AssistantTurn(thought="c", tool=ToolRequest(name="double", args={"n": 21}))
        finish = AssistantTurn(thought="done", answer="42")

        @app.register_task(child_name)
        def child(params, ctx):
            domain = MeteredInterpreter(
                llm=scripted_caller([(act_double, u), (finish, u)]),
                tools=make_tool_runner({"double": lambda a: a["n"] * 2}),
            )
            return run_subagent_as_task(params, ctx, domain=domain, max_iters=3)

    @staticmethod
    def _register_parent(app, parent_name, child_name, spawner):
        from agent.runtime import make_tool_runner, spawn_tool
        from effective.cost import MeteredInterpreter
        from effective.handlers.absurd import DurableHandler
        from effective.react import AssistantTurn, ToolRequest, ToolResult, run_agent

        def _boom(op):
            raise AssertionError("the parent makes no model calls in this scenario")

        @app.register_task(parent_name, default_max_attempts=1)
        def parent(params, ctx):
            mid = params["mid"]

            def decide(messages, level):
                yield from ()
                if level.depth == 0:
                    return AssistantTurn(thought="go", tool=ToolRequest(name="delegate", args={}))
                seen_msg = messages[-1]["content"]
                return AssistantTurn(thought="relay", answer=f"parent saw: {seen_msg}")

            def act(request, tag):
                if request.name == "delegate":
                    try:
                        return (
                            yield from spawn_subagent_task(
                                child_name, "double 21", correlation=mid
                            )
                        )
                    except Refused:
                        return ToolResult(content="ceiling: depth exhausted")
                return (yield from call_tool(request.name, request.args, ToolResult))

            domain = MeteredInterpreter(
                llm=_boom, tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)})
            )
            return DurableHandler(ctx, domain, params=params).run(
                lambda: run_agent("go", decide=decide, act=act)
            )

    def _make_spawner(self, spawner_app, seen):
        def spawn(task_name, params, idempotency_key, queue, *, max_attempts=None):
            result = spawner_app.spawn(
                task_name,
                params,
                queue=queue,
                idempotency_key=idempotency_key,
                max_attempts=max_attempts,
                retry_strategy=IMMEDIATE_RETRY,
            )
            tid = str(result["task_id"])
            seen.append(tid)
            return tid

        return spawn

    def _run(self, depth):
        from uuid import uuid4

        app, spawner_app = absurd(), absurd()
        seen: list[str] = []
        try:
            mid = f"bud-{uuid4().hex[:8]}"
            pname, cname = f"p-{mid}", f"c-{mid}"
            self._register_child(app, cname)
            self._register_parent(app, pname, cname, self._make_spawner(spawner_app, seen))
            params = {"mid": mid, BUDGET_DEPTH_PARAM: depth}
            spawned = app.spawn(pname, params, retry_strategy=IMMEDIATE_RETRY)
            snap = run_until_result(app, spawned["task_id"])
            return snap, seen
        finally:
            app.close()
            spawner_app.close()

    def test_depth_zero_refuses_gracefully_and_enqueues_no_child(self):
        snap, seen = self._run(depth=0)
        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert "ceiling" in snap.result["answer"]
        assert seen == []  # the ceiling refused BEFORE any child task was enqueued

    def test_depth_one_permits_exactly_one_spawn(self):
        snap, seen = self._run(depth=1)
        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert "42" in snap.result["answer"]
        assert len(seen) == 1  # one level spawned; the child carried depth 0 onward
