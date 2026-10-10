"""Agent bench: record once (pay), replay free (verify) — the cost thesis, executable.

The project thesis applied to its own evaluation. A bench task runs twice:

1. **Record** — under a `RecordingCtx`, every checkpointed step runs for real and its
   result is logged. The `MeteredInterpreter` accrues the true cost (the model is paid
   exactly once). This is the only pass that touches a model.
2. **Replay** — under a `ReplayCtx`, every step returns its logged result *without
   running its thunk*. The replay interpreter is wired to **explode** if the model or a
   tool is ever called; that it never fires (`replay_is_free`) is the proof that replay
   costs nothing — and that an opaque subagent inside a recorded step does not re-run.

So the harness (loop logic, termination, observation threading, subagent boundary) is
regression-tested for free, and a Pareto sweep over seam-configs pays for each config's
trajectory once. `recorded.cache_hit_ratio` carries the KV-cache metric a live run
populates (here the scripted usages leave it 0).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.tasks import Task
from effective.contexts import RecordingCtx, ReplayCtx
from effective.cost import MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool
from effective.handlers.durable import DurableHandler
from effective.interpreters.scripted import scripted_caller
from effective.interpreters.tools import AgentTool, ToolFn, make_tool_runner
from effective.react import AssistantTurn, Trajectory, run_agent


@dataclass(frozen=True)
class BenchResult:
    name: str
    answer: str
    passed: bool
    recorded: Usage  # cost paid on the record pass (the model is paid exactly once)
    replay_llm_calls: int
    replay_tool_calls: int
    replay_matched: bool

    @property
    def replay_is_free(self) -> bool:
        """No model and no tool ran on replay — the harness was re-verified for free."""
        return self.replay_llm_calls == 0 and self.replay_tool_calls == 0


def record_then_replay(
    task: Task,
    caller_script: list[tuple[AssistantTurn, Usage]],
    *,
    tools: Mapping[str, ToolFn] | None = None,
    agents: Mapping[str, AgentTool] | None = None,
    max_iters: int = 5,
) -> BenchResult:
    """Run `task` once for real (recording + metering), then replay it with no I/O."""
    program = lambda: run_agent(task.prompt, max_iters=max_iters)  # noqa: E731

    rec_interp = MeteredInterpreter(
        llm=scripted_caller(caller_script), tools=make_tool_runner(tools or {}, agents)
    )
    rec_ctx = RecordingCtx()
    rec = Trajectory.model_validate(DurableHandler(rec_ctx, rec_interp).run(program))

    calls = {"llm": 0, "tool": 0}

    def boom_llm(op: AskLLM[Any]) -> Any:
        calls["llm"] += 1
        raise AssertionError("model called on replay — replay must be free")

    def boom_tool(op: CallTool[Any]) -> Any:
        calls["tool"] += 1
        raise AssertionError("tool called on replay — replay must be free")

    rep_interp = MeteredInterpreter(llm=boom_llm, tools=boom_tool)
    rep_raw = DurableHandler(ReplayCtx(rec_ctx.log), rep_interp).run(program)
    rep = Trajectory.model_validate(rep_raw)

    return BenchResult(
        name=task.name,
        answer=rec.answer,
        passed=bool(task.check(rec.answer)),
        recorded=rec_interp.meter,
        replay_llm_calls=calls["llm"],
        replay_tool_calls=calls["tool"],
        replay_matched=rec.model_dump() == rep.model_dump(),
    )
