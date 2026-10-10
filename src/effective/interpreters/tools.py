"""Running a tool call: plain tools, and the opaque subagent.

The loop yields ops, and this is where a `CallTool` runs:

| tool | runs as |
|---|---|
| plain | a `dict -> value` callable, its result wrapped in a `ToolResult` |
| subagent | a nested `run_agent` under its own metered, traced interpreter |

To the parent a subagent is one opaque `CallTool` step: the parent records only the child's final
answer and, on replay, is served it from the checkpoint without running the child again. The
child carries its narrow tool set on its own stable prefix, so the parent's context stays lean.

The cost layer meters only model calls, so a subagent's spend does not reach the parent's meter;
it reaches telemetry instead, as spans tagged with the child's `agent_name` and `Usage`. The
in-process contexts that run these without an engine are in `effective.contexts`.
"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from effective.contexts import LocalCtx
from effective.cost import LLMCall, MeteredInterpreter, ToolRunner, Usage
from effective.domain import SPAWN_TOOL, CallTool, SpawnArgs, SpawnResult
from effective.engines.absurd import _adapt_ctx
from effective.handlers.durable import DurableHandler
from effective.react import ToolResult, Trajectory, run_agent
from effective.spawning import answer_parent
from effective.telemetry import Sink, traced

type ToolFn = Callable[[dict[str, Any]], Any]
type AgentTool = Callable[[CallTool[Any]], Any]


class Spawner(Protocol):
    """Enqueues a child task and returns its id. On SQLite this shares the running task's
    connection, which `SqliteApp.spawn` locks."""

    def __call__(
        self,
        task_name: str,
        params: dict[str, Any],
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str: ...


def make_tool_runner(
    tools: Mapping[str, ToolFn], agents: Mapping[str, AgentTool] | None = None
) -> ToolRunner:
    """A `ToolRunner` over plain callables plus opaque `agents` (subagents).

    `agents` take precedence and receive the whole `CallTool` op (they need the args
    and may run a nested workflow); plain `tools` are pure `dict -> value` callables
    whose result is stringified into a `ToolResult`. An unknown name returns an error
    observation rather than raising, so the loop can route around it.
    """
    agents = agents or {}

    def run(op: CallTool[Any]) -> ToolResult:
        if (agent := agents.get(op.name)) is not None:
            return agent(op)
        if (fn := tools.get(op.name)) is None:
            return ToolResult(content=f"unknown tool: {op.name}")
        try:
            return ToolResult(content=str(fn(op.args)))
        except Exception as exc:
            return ToolResult(content=f"error: {exc}")

    return run


def subagent_runner(
    caller: LLMCall,
    tools: Mapping[str, ToolFn],
    *,
    agent_name: str = "subagent",
    sink: Sink | None = None,
    accrue: Callable[[Usage], None] | None = None,
    max_iters: int = 4,
    task_key: str = "task",
) -> AgentTool:
    """A tool whose execution runs a nested `run_agent`, opaque to the parent.

    The child runs under its own `MeteredInterpreter` (so it has an independent meter
    and an independent, narrow tool set) with an optional `traced` layer (the telemetry
    peek: spans tagged `agent_name`). `accrue`, if given, folds the child's total
    `Usage` into a parent-supplied meter so a bench can still total real cost. The
    child's final answer is returned as the parent's single `ToolResult`.
    """

    def run(op: CallTool[Any]) -> ToolResult:
        subtask = str(op.args.get(task_key, ""))
        layers = [traced(sink, agent_name=agent_name)] if sink is not None else []
        runner = make_tool_runner(tools)
        interp = MeteredInterpreter(llm=caller, tools=runner, domain_layers=layers)
        handler = DurableHandler(ctx=LocalCtx(), domain=interp)
        raw = handler.run(lambda: run_agent(subtask, max_iters=max_iters))
        traj = Trajectory.model_validate(raw)
        if accrue is not None:
            accrue(interp.meter)
        return ToolResult(content=traj.answer)

    return run


def spawn_tool(spawner: Spawner) -> AgentTool:
    """The handler-side `spawn` tool: enqueue a child task and return its id.

    Wrapped in a parent `Step`, so the child id is checkpointed and a parent replay returns it
    without re-spawning; the `idempotency_key` the handler names the spawn with covers a crash
    before that checkpoint commits. It answers only under `SPAWN_TOOL`, the name the handler holds
    to its depth ceiling.

    `spawner` enqueues on the running task's own connection. A workflow can reach this from a
    `gather` branch, whose thunk runs off-lock on its own thread, so `SqliteApp.spawn` takes the
    write lock; a thunk never holds it, so the lock is not re-entered."""

    def run(op: CallTool[Any]) -> SpawnResult:
        if op.name != SPAWN_TOOL:
            raise ValueError(f"the spawn capability answers only {SPAWN_TOOL!r}, not {op.name!r}")
        args = SpawnArgs.model_validate(op.args)
        if args.idempotency_key is None:
            raise ValueError(
                f"a spawn of {args.task_name!r} reached the tool unnamed by a handler"
            )
        task_id = spawner(
            args.task_name,
            args.params,
            args.idempotency_key,
            args.queue,
            max_attempts=args.max_attempts,
        )
        return SpawnResult(task_id=task_id)

    return run


def run_subagent_as_task(
    params: dict[str, Any], ctx: Any, *, domain: Any, max_iters: int = 4
) -> Any:
    """The child Absurd-task body: run `run_agent` under its **own** ctx, so the child has
    its own checkpoints, then emit the parent's completion event with the
    final answer. The emit is a checkpointed step, so a child retry re-emits at most once.

    Register it with the app, e.g. ``app.register_task(name)(partial(run_subagent_as_task,
    domain=...))``; the parent's `spawn_subagent_task` passes ``task`` and ``done_event``."""
    ctx = _adapt_ctx(ctx)  # the answer below reaches the SDK too, and it speaks text
    raw = DurableHandler(ctx, domain, params=params).run(
        lambda: run_agent(params["task"], max_iters=max_iters)
    )
    traj = Trajectory.model_validate(raw)
    answer_parent(ctx, params, {"answer": traj.answer})
    return raw
