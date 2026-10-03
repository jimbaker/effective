"""The quality scorer — runs an LLMCall over tasks and aggregates (red-green).

Fake (scripted) callers, so no model and no spend: the scorer drives the real
durable stack (run_agent + MeteredInterpreter + DurableHandler) with canned turns.
"""

from agent.eval import evaluate
from agent.tasks import TOOLS, Task, has_number
from effective.cost import Usage
from effective.react import AssistantTurn, ToolRequest


def _scripted(turns: list[AssistantTurn]):
    it = iter(turns)

    def call(_op):
        return next(it), Usage(prompt_tokens=10, completion_tokens=5, latency_s=0.1)

    return call


def test_evaluate_scores_a_passing_run_and_folds_usage():
    task = Task("mul", "compute 19 times 23", has_number(437))
    caller = _scripted(
        [
            AssistantTurn(
                thought="act", tool=ToolRequest(name="multiply", args={"a": 19, "b": 23})
            ),
            AssistantTurn(thought="done", answer="437"),
        ]
    )
    res = evaluate(caller, [task], TOOLS)

    assert res.quality == 1.0
    assert res.results[0].passed
    assert res.results[0].answer == "437"
    assert res.total_usage.completion_tokens == 10  # two model calls x 5
    assert res.total_usage.latency_s == 0.2


def test_evaluate_scores_a_failing_run():
    task = Task("mul", "compute 19 times 23", has_number(437))
    caller = _scripted([AssistantTurn(thought="guess", answer="it is 12")])
    res = evaluate(caller, [task], TOOLS)

    assert res.quality == 0.0
    assert not res.results[0].passed


def test_quality_is_the_pass_rate_across_tasks():
    t_ok = Task("ok", "compute 19 times 23", has_number(437))
    t_bad = Task("bad", "compute 19 times 23", has_number(999))
    caller = _scripted(
        [
            # task 1: correct
            AssistantTurn(thought="a", tool=ToolRequest(name="multiply", args={"a": 19, "b": 23})),
            AssistantTurn(thought="d", answer="437"),
            # task 2: wrong answer vs its checker (999)
            AssistantTurn(thought="d", answer="437"),
        ]
    )
    res = evaluate(caller, [t_ok, t_bad], TOOLS)
    assert res.quality == 0.5
