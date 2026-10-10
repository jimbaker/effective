"""Quality scorer: run an agent (any LLMCall) over a task suite, score, aggregate.

Each task runs through the same durable stack as production — `run_agent` under
`MeteredInterpreter` + `DurableHandler` — so the meter (cost, tokens, latency)
accrues per task and the trajectory is replayable. `quality` is the pass-rate;
the aggregate `Usage` carries cost/latency for the Pareto axes. Caller-agnostic:
pass an OpenAI or llama.cpp `LLMCall`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agent.tasks import Task
from effective.contexts import LocalCtx
from effective.cost import CostBudget, MeteredInterpreter, Usage
from effective.handlers.durable import DurableHandler
from effective.interpreters.tools import make_tool_runner
from effective.react import Trajectory, run_agent


@dataclass(frozen=True)
class TaskResult:
    name: str
    passed: bool
    answer: str
    turns: int
    usage: Usage


@dataclass(frozen=True)
class EvalResult:
    results: list[TaskResult]

    @property
    def quality(self) -> float:
        """Pass-rate over the suite (the quality axis)."""
        if not self.results:
            return 0.0
        return round(sum(r.passed for r in self.results) / len(self.results), 4)

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for r in self.results:
            total = total + r.usage
        return total


def evaluate(
    caller: Callable[[Any], tuple[Any, Usage]],
    tasks: Sequence[Task],
    tools: dict[str, Callable[[dict[str, Any]], Any]],
    max_iters: int = 5,
    budget: CostBudget | None = None,
) -> EvalResult:
    """Run each task through the durable agent stack and score the final answer.

    A shared ``budget`` caps total spend across the suite (a task that trips it
    raises BudgetExceeded and scores as a fail).
    """
    tool_runner = make_tool_runner(tools)
    results: list[TaskResult] = []
    for task in tasks:
        interp = MeteredInterpreter(llm=caller, tools=tool_runner, budget=budget)
        handler = DurableHandler(ctx=LocalCtx(), domain=interp)
        try:
            raw = handler.run(lambda t=task: run_agent(t.prompt, max_iters=max_iters))
            traj = Trajectory.model_validate(raw)
            answer, turns = traj.answer, len(traj.steps)
        except Exception as exc:
            answer, turns = f"[error: {exc}]", 0
        results.append(
            TaskResult(
                name=task.name,
                passed=bool(task.check(answer)),
                answer=answer,
                turns=turns,
                usage=interp.meter,
            )
        )
    return EvalResult(results)
