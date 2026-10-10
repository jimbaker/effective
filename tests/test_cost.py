"""The AskLLM cost/KV-cache handler layer (red-green spec).

Cross-cutting concerns are handler layers: token cost, cache-hit ratio, and
budget enforcement are *handler layers over AskLLM*, not workflow logic. The
workflow yields AskLLM and gets a typed result; usage accrues on the side.

`MeteredInterpreter` is a `DomainInterpreter` decorator: it folds `Usage` over
every AskLLM and passes CallTool through. Driving the `run_agent` prototype
through it (via DurableHandler + a trivial in-memory ctx) meters the whole
agent trajectory — the workflow never knows it is being measured.
"""

from types import SimpleNamespace

import pytest

from effective.cost import BudgetExceeded, CostBudget, MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool
from effective.handlers.durable import DurableHandler
from effective.keys import Key
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent

QUESTION = "when was acme founded?"


# --- Usage: a monoid -------------------------------------------------------


def test_usage_is_a_monoid_with_cache_ratio():
    a = Usage(prompt_tokens=100, completion_tokens=10, cache_read_input_tokens=80, cost=0.002)
    b = Usage(prompt_tokens=120, completion_tokens=12, cache_read_input_tokens=100, cost=0.0025)

    assert Usage() + a == a  # left identity
    assert a + Usage() == a  # right identity
    total = a + b
    assert total.prompt_tokens == 220
    assert total.completion_tokens == 22
    assert total.cache_read_input_tokens == 180
    assert total.cost == pytest.approx(0.0045)
    assert a.cache_hit_ratio == pytest.approx(0.8)
    assert Usage().cache_hit_ratio == 0.0  # no divide-by-zero on an empty meter


def test_usage_tracks_latency_and_throughput():
    # Dollars don't apply to a local model; latency and tokens/sec do. Durations
    # are additive (a monoid), so they fold over the op stream like tokens.
    a = Usage(completion_tokens=50, latency_s=2.0)
    b = Usage(completion_tokens=30, latency_s=1.0)
    total = a + b

    assert total.latency_s == 3.0
    assert total.completion_tokens == 80
    assert total.tokens_per_second == round(80 / 3.0, 2)
    assert Usage().tokens_per_second == 0.0  # no divide-by-zero on an empty meter


def test_usage_from_response_is_provider_duck_typed():
    # Shaped like a litellm ModelResponse: a `usage` mapping + `_hidden_params`.
    resp = SimpleNamespace(
        usage={"prompt_tokens": 140, "completion_tokens": 20, "cache_read_input_tokens": 120},
        _hidden_params={"response_cost": 0.003},
    )
    u = Usage.from_response(resp)
    assert u.prompt_tokens == 140
    assert u.completion_tokens == 20
    assert u.cache_read_input_tokens == 120
    assert u.cost == pytest.approx(0.003)


# --- CostBudget: a ceiling -------------------------------------------------


def test_cost_budget_flips_at_the_ceiling():
    budget = CostBudget(limit=0.005)
    assert not budget.exceeded()
    budget.add(Usage(cost=0.002))
    assert not budget.exceeded()
    budget.add(Usage(cost=0.0035))  # 0.0055 >= 0.005
    assert budget.exceeded()


# --- MeteredInterpreter over a full agent run ------------------------------


class _LocalCtx:
    """A non-durable Absurd ctx: a step just runs its thunk inline."""

    def step(self, name, thunk):
        return thunk()

    def await_event(self, name):
        raise NotImplementedError

    def sleep_until(self, when, *, name: Key | None = None):
        raise NotImplementedError


def _turns_with_usage() -> list[tuple[AssistantTurn, Usage]]:
    return [
        (
            AssistantTurn(thought="search", tool=ToolRequest(name="search", args={"q": "acme"})),
            Usage(prompt_tokens=100, completion_tokens=10, cache_read_input_tokens=80, cost=0.002),
        ),
        (
            AssistantTurn(thought="lookup", tool=ToolRequest(name="lookup", args={"id": 7})),
            Usage(
                prompt_tokens=120, completion_tokens=12, cache_read_input_tokens=100, cost=0.0025
            ),
        ),
        (
            AssistantTurn(thought="done", answer="Acme Corp was founded in 1999."),
            Usage(
                prompt_tokens=140, completion_tokens=20, cache_read_input_tokens=120, cost=0.003
            ),
        ),
    ]


def _tool_runner(op: CallTool) -> ToolResult:
    return {
        "search": ToolResult(content="found id=7"),
        "lookup": ToolResult(content="Acme Corp, founded 1999"),
    }[op.name]


def _seq_llm(turns: list[tuple[AssistantTurn, Usage]]):
    it = iter(turns)

    def call(_op: AskLLM) -> tuple[AssistantTurn, Usage]:
        return next(it)

    return call


def test_metered_interpreter_folds_cost_over_the_whole_trajectory():
    interp = MeteredInterpreter(llm=_seq_llm(_turns_with_usage()), tools=_tool_runner)
    handler = DurableHandler(ctx=_LocalCtx(), domain=interp)

    result = handler.run(lambda: run_agent(QUESTION))
    traj = Trajectory.model_validate(result)

    assert traj.answer == "Acme Corp was founded in 1999."
    # 3 AskLLM turns folded; the 2 CallTool ops add nothing to cost.
    assert interp.meter.cost == pytest.approx(0.0075)
    assert interp.meter.prompt_tokens == 360
    assert interp.meter.cache_read_input_tokens == 300
    assert interp.meter.cache_hit_ratio == round(300 / 360, 4)  # 4-dp display contract


def test_budget_ceiling_refuses_a_later_askllm_and_aborts():
    budget = CostBudget(limit=0.003)  # blown after the 2nd turn (0.0045)
    interp = MeteredInterpreter(
        llm=_seq_llm(_turns_with_usage()), tools=_tool_runner, budget=budget
    )
    handler = DurableHandler(ctx=_LocalCtx(), domain=interp)

    with pytest.raises(BudgetExceeded):
        handler.run(lambda: run_agent(QUESTION))

    # spent through turn 1 (0.002) + turn 2 (0.0025); the 3rd AskLLM was refused.
    assert interp.meter.cost == pytest.approx(0.0045)


def test_tool_results_carry_no_cost():
    interp = MeteredInterpreter(llm=_seq_llm(_turns_with_usage()), tools=_tool_runner)
    # a bare CallTool op never touches the meter
    out = interp.run(CallTool(name="search", args={"q": "acme"}, result_schema=ToolResult))
    assert isinstance(out, ToolResult)
    assert interp.meter == Usage()


def _accrue_from_threads(accrue, *, workers=8, per_worker=5000):
    """Hammer a lock-guarded accrual from many threads; the exact total proves no
    read-modify-write update was dropped."""
    import threading

    def worker():
        u = Usage(prompt_tokens=1, cost=0.001)
        for _ in range(per_worker):
            accrue(u)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return workers * per_worker


def test_meter_accrual_is_lost_update_free_under_threads():
    """The telemetry meter RMW is locked, so concurrent gather branches accruing at once
    cannot drop an update (unlocked, this loses updates → a count below the total)."""
    interp = MeteredInterpreter(llm=lambda _op: (None, Usage()), tools=lambda _op: None)
    total = _accrue_from_threads(interp._accrue)
    assert interp.meter.prompt_tokens == total
    assert interp.meter.cost == pytest.approx(total * 0.001)


def test_cost_budget_add_is_lost_update_free_under_threads():
    """`CostBudget.add` is locked for the same reason: the telemetry ceiling's total is exact
    under concurrent accrual, though it is not the enforcement bookkeeper."""
    budget = CostBudget(limit=1e9)
    total = _accrue_from_threads(budget.add)
    assert budget.spent.prompt_tokens == total
