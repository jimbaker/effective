"""A scripted model: answers each model call with the next of a fixed list of turns."""

from collections.abc import Callable
from typing import Any

from effective.cost import Usage
from effective.domain import AskLLM
from effective.react import AssistantTurn


def scripted_caller(turns: list[tuple[AssistantTurn, Usage]]) -> Callable[[AskLLM[Any]], Any]:
    """An `LLMCall` that returns each `(AssistantTurn, Usage)` in order: a run with no provider."""
    it = iter(turns)

    def call(op: AskLLM[Any]) -> Any:
        return next(it)

    return call
