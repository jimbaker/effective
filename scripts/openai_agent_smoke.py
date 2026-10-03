"""Live smoke: the ReAct agent against real OpenAI gpt-5-nano, hard-capped.

Drives `run_agent` through the OpenAI `LLMCall` (effective.interpreters.openai) wrapped in
`MeteredInterpreter` with a `CostBudget` ceiling — proving the whole stack
end-to-end: durable-workflow loop + metered AskLLM + a real provider, with the
budget refusing any AskLLM past the cap. Spend is reported regardless of outcome.

    uv run python scripts/openai_agent_smoke.py

Reads OPENAI_API_KEY from the environment, or from ./api.env if unset. The cap
below is the safety net; gpt-5-nano turns cost fractions of a cent.
"""

import os
from pathlib import Path

from openai import OpenAI

from effective.cost import BudgetExceeded, CostBudget, MeteredInterpreter
from effective.domain import CallTool
from effective.handlers.absurd import DurableHandler
from effective.interpreters.openai import OpenAITurnCaller
from effective.react import ToolResult, Trajectory, run_agent

BUDGET_USD = 0.05  # hard ceiling for the run (user limit: <= $1)
MODEL = "gpt-5-nano"
QUESTION = "Use the multiply tool to compute 19 times 23, then state the product as your answer."

SYSTEM_PROMPT = """\
You are a ReAct agent that answers by either calling a tool or giving a final answer.

Tools available:
  multiply(a, b) -> the product of two numbers.

Each turn, fill `thought` with brief reasoning. To use a tool, set `tool.name` to
the tool and `tool.arguments_json` to a JSON object string of its arguments
(e.g. {"a": 19, "b": 23}) and set `answer` to null. When you have the result,
set `answer` to the final text and set `tool` to null. Prefer the tool over
mental arithmetic."""


class LocalCtx:
    """Non-durable Absurd ctx: a step just runs its thunk inline."""

    def step(self, name, thunk):
        return thunk()

    def await_event(self, name):
        raise NotImplementedError

    def sleep_until(self, when, /, *, name=None):
        raise NotImplementedError


def tool_runner(op: CallTool) -> ToolResult:
    if op.name == "multiply":
        a = float(op.args["a"])
        b = float(op.args["b"])
        product = a * b
        value = int(product) if product.is_integer() else product
        return ToolResult(content=str(value))
    return ToolResult(content=f"unknown tool: {op.name}")


def _load_api_env() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    path = Path(__file__).resolve().parents[1] / "api.env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    _load_api_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (and ./api.env not found)")

    client = OpenAI()
    caller = OpenAITurnCaller(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        max_completion_tokens=2000,
        extra={"reasoning_effort": "minimal"},
    )
    budget = CostBudget(limit=BUDGET_USD)
    interp = MeteredInterpreter(llm=caller, tools=tool_runner, budget=budget)
    handler = DurableHandler(ctx=LocalCtx(), domain=interp)

    print(f"model={MODEL}  cap=${BUDGET_USD:.2f}")
    print(f"Q: {QUESTION}\n")
    try:
        result = handler.run(lambda: run_agent(QUESTION, max_iters=4))
        traj = Trajectory.model_validate(result)
        print(f"answer: {traj.answer}")
        print(f"stop_reason: {traj.stop_reason}")
        for i, s in enumerate(traj.steps):
            act = f"{s.tool.name}({s.tool.args})" if s.tool else "(final)"
            print(f"  step {i}: {act} -> {s.observation}")
    except BudgetExceeded as exc:
        print(f"ABORTED by budget: {exc}")
    finally:
        m = interp.meter
        print(
            f"\nspend: ${m.cost:.6f}  prompt={m.prompt_tokens} "
            f"completion={m.completion_tokens} cached={m.cache_read_input_tokens} "
            f"cache_hit_ratio={m.cache_hit_ratio}"
        )


if __name__ == "__main__":
    main()
