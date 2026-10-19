"""Every measured row of the Rosetta table, pinned. A probe that changes fails here first."""

import probes
import pytest
from probes import Calls


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        (lambda: probes.effective_repair(0), Calls(1, 1, "Repair", 1)),
        (lambda: probes.effective_repair(1), Calls(2, 1, "GenerateAnswer", 2)),
        (lambda: probes.pydantic_ai_repair(), Calls(3, 1, "GenerateAnswer", 2)),
        (lambda: probes.dspy_repair("refine"), Calls(3, 2, "Prediction", 2)),
        (lambda: probes.dspy_repair("best_of_n"), Calls(2, 2, "Prediction", 2)),
        (lambda: probes.langgraph_repair(), Calls(2, 1, "GenerateAnswer", 2)),
    ],
    ids=[
        "effective",
        "effective-loop",
        "pydantic-ai",
        "dspy-refine",
        "dspy-best-of-n",
        "langgraph",
    ],
)
def test_a_bad_citation_then_a_good_one(probe, expected):
    assert probe() == expected


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        (lambda: probes.effective_crash(), Calls(0, 0, "completed", 0)),
        (lambda: probes.langgraph_crash("inside"), Calls(1, 0, "GenerateAnswer", 1)),
        (lambda: probes.langgraph_crash("after"), Calls(0, 0, "GenerateAnswer", 0)),
        (lambda: probes.langgraph_crash("inside", True), Calls(1, 0, "GenerateAnswer", 1)),
        (lambda: probes.langgraph_crash("after", True), Calls(0, 0, "GenerateAnswer", 0)),
        (lambda: probes.pydantic_ai_crash(), Calls(1, 0, "GenerateAnswer", 1)),
        (lambda: probes.dspy_crash(cache=False), Calls(1, 1, "Prediction", 1)),
        (lambda: probes.dspy_crash(cache=True), Calls(0, 1, "Prediction", 0)),
    ],
    ids=[
        "effective-sqlite",
        "langgraph-inside-node",
        "langgraph-after-node",
        "langgraph-functional-inside-task",
        "langgraph-functional-after-task",
        "pydantic-ai-step-persistence",
        "dspy-cache-off",
        "dspy-cache-on",
    ],
)
def test_the_resume_after_a_crash(probe, expected):
    assert probe() == expected


@pytest.mark.parametrize(
    ("probe", "side_effects"),
    [
        (probes.effective_approval, 1),
        (probes.langgraph_approval, 2),
        (probes.pydantic_ai_approval, 2),
    ],
    ids=["effective", "langgraph", "pydantic-ai"],
)
def test_the_side_effect_before_an_approval(probe, side_effects):
    assert probe() == side_effects


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        (probes.effective_fork, Calls(1, 0, "GenerateAnswer", 1)),
        (probes.langgraph_fork, Calls(1, 0, "GenerateAnswer", 1)),
        (probes.pydantic_ai_fork, Calls(1, 0, "GenerateAnswer", 1)),
        (probes.dspy_fork, Calls(1, 1, "Prediction", 1)),
    ],
    ids=["effective", "langgraph", "pydantic-ai", "dspy"],
)
def test_a_fork_with_the_search_changed(probe, expected):
    assert probe() == expected


@pytest.mark.parametrize(
    ("framework", "lines"),
    [("Effective", 37), ("DSPy", 23), ("Pydantic AI", 52), ("LangGraph", 82)],
)
def test_lines_of_code(framework, lines):
    assert probes.loc(probes.PORTS[framework]) == lines
