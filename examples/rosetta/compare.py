"""Run every probe and print the Rosetta table: one row per question, one column per framework.

Run it with `uv run --project examples/rosetta python examples/rosetta/compare.py`.
"""

import contextlib
import io
import logging
from importlib.metadata import version

import probes

FRAMEWORKS = ["Effective", "Pydantic AI", "DSPy", "LangGraph"]


def rows() -> list[tuple[str, dict[str, str]]]:
    return [
        (
            "1. bad citation, then good: calls to a valid answer",
            {
                "Effective": f"{probes.effective_repair(0).cell()}; "
                f"author's loop: {probes.effective_repair(1).cell()}",
                "Pydantic AI": probes.pydantic_ai_repair().cell(),
                "DSPy": f"Refine: {probes.dspy_repair('refine').cell()}; "
                f"BestOfN: {probes.dspy_repair('best_of_n').cell()}",
                "LangGraph": f"conditional edge: {probes.langgraph_repair().cell()}",
            },
        ),
        (
            "2. crash after the model answered: the resume's calls",
            {
                "Effective": f"SQLite: {probes.effective_crash().cell()}",
                "Pydantic AI": f"StepPersistence + SqliteStepStore: "
                f"{probes.pydantic_ai_crash().cell()}",
                "DSPy": f"rerun, cache off: {probes.dspy_crash(cache=False).cell()}; "
                f"cache on: {probes.dspy_crash(cache=True).cell()}",
                "LangGraph": f"StateGraph inside node: {probes.langgraph_crash('inside').cell()}; "
                f"after node: {probes.langgraph_crash('after').cell()}; "
                f"Functional inside task: {probes.langgraph_crash('inside', True).cell()}; "
                f"after task: {probes.langgraph_crash('after', True).cell()}",
            },
        ),
        (
            "3. approval wait: runs of the side effect before it",
            {
                "Effective": f"await_event: {probes.effective_approval()}",
                "Pydantic AI": f"deferred approval: {probes.pydantic_ai_approval()}",
                "DSPy": "none: no wait primitive",
                "LangGraph": f"interrupt in node: {probes.langgraph_approval()}",
            },
        ),
        (
            "4. fork with the search changed: the fork's calls",
            {
                "Effective": f"fork_at: {probes.effective_fork().cell()}",
                "Pydantic AI": f"snapshot + edited history: {probes.pydantic_ai_fork().cell()}",
                "DSPy": f"rerun: {probes.dspy_fork().cell()}",
                "LangGraph": f"update_state: {probes.langgraph_fork().cell()}",
            },
        ),
        (
            "5. lines of code in the port",
            {name: str(probes.loc(probes.PORTS[name])) for name in FRAMEWORKS},
        ),
    ]


def main() -> None:
    logging.disable(logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()):
        measured = rows()
    packages = ["dspy", "pydantic-ai-slim", "pydantic-ai-harness", "langgraph", "langchain-core"]
    print("versions:", ", ".join(f"{name} {version(name)}" for name in packages))
    print()
    print("| question | " + " | ".join(FRAMEWORKS) + " |")
    print("|---" * (len(FRAMEWORKS) + 1) + "|")
    for question, cells in measured:
        print(f"| {question} | " + " | ".join(cells[name] for name in FRAMEWORKS) + " |")


if __name__ == "__main__":
    main()
