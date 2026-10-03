"""The agent bench: record once (pay), replay free (verify) — across three shapes.

Each scenario asserts the same three properties that make replay-based eval honest:
the recorded run *passes* and *cost something* (the model was paid once), the replay
is *free* (no model, no tool ran), and the replayed trajectory *matches* the recorded
one bit-for-bit. The third scenario puts a metered subagent inside a recorded step and
shows it does not re-run on replay (the opaque boundary holds under replay).
"""

from agent.bench import record_then_replay
from agent.runtime import subagent_runner
from agent.tasks import Task, has_number, has_text
from effective.cost import Usage
from effective.react import AssistantTurn, ToolRequest

U = Usage(prompt_tokens=20, completion_tokens=8, cost=0.0012)


def act(name, **args):
    return AssistantTurn(thought="act", tool=ToolRequest(name=name, args=args))


def finish(answer):
    return AssistantTurn(thought="done", answer=answer)


def test_plain_tool_task_records_and_replays_free():
    task = Task("multiply", "What is 19 times 23?", has_number(437))
    result = record_then_replay(
        task,
        [(act("multiply", a=19, b=23), U), (finish("the product is 437"), U)],
        tools={"multiply": lambda a: a["a"] * a["b"]},
    )
    assert result.passed
    assert result.recorded.cost > 0  # the model was paid — once
    assert result.replay_is_free  # ...and replay touched neither model nor tool
    assert result.replay_matched


def test_two_step_task_records_and_replays_free():
    task = Task("two_step", "Add 7 and 5, then multiply by 3.", has_number(36))
    result = record_then_replay(
        task,
        [
            (act("add", a=7, b=5), U),
            (act("multiply", a=12, b=3), U),
            (finish("the result is 36"), U),
        ],
        tools={"add": lambda a: a["a"] + a["b"], "multiply": lambda a: a["a"] * a["b"]},
    )
    assert result.passed
    assert result.recorded.completion_tokens == 24  # three metered turns
    assert result.replay_is_free
    assert result.replay_matched


def test_subagent_task_replays_without_rerunning_the_child():
    """The parent delegates to a metered subagent; on replay the child does not re-run
    (proven by replay_tool_calls == 0 — the parent's delegate step never fires)."""
    sub_calls = {"n": 0}

    def counting_sub(op):
        sub_calls["n"] += 1
        script = [
            (act("population", city="boulder"), U),
            (finish("Boulder has 108000 people"), U),
        ]
        return script[sub_calls["n"] - 1] if sub_calls["n"] <= 2 else (finish("?"), U)

    delegate = subagent_runner(
        counting_sub, {"population": lambda a: 108000}, agent_name="lookup-sub"
    )
    task = Task("delegated_lookup", "Look up Boulder's population.", has_text("108000"))

    result = record_then_replay(
        task,
        [(act("delegate", task="population of boulder"), U), (finish("Boulder: 108000"), U)],
        agents={"delegate": delegate},
    )
    assert result.passed
    assert result.replay_is_free
    assert result.replay_matched
    assert sub_calls["n"] == 2  # the child ran exactly twice (record only), never on replay
