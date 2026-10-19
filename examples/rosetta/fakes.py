"""The offline world every port runs against: a search tool and a scripted model, each counting.

The framework adapters at the bottom wrap one `Model` script so every port sees the same answers
in the same order; they are fakes, and the line counts in `probes.loc` leave this file out.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

PASSAGES = [
    "The Rosetta Stone was found in 1799 near the town of Rashid.",
    "Its decree is written in hieroglyphic, Demotic and ancient Greek.",
]
FORKED = ["The stone has been in the British Museum since 1802."]

GOOD = {"reasoning": "The first passage says so.", "answer": "1799", "citation": 0}
BAD = {"reasoning": "A guess.", "answer": "1799", "citation": 7}


class Crash(BaseException):
    """A simulated process death, which no framework's `except Exception` catches."""


@dataclass
class Search:
    """The search tool: returns `results` and counts its calls."""

    results: list[str] = field(default_factory=lambda: list(PASSAGES))
    calls: int = 0

    def __call__(self, query: str, k: int = 3) -> list[str]:
        self.calls += 1
        return self.results[:k]


@dataclass
class Model:
    """The model: answers from `script` in order, repeating the last, and counts its calls.

    `calls` counts every request a framework sends; `answers` counts the requests that asked for
    the answer, which differs where the model also chooses tools."""

    script: list[dict[str, Any]] = field(default_factory=lambda: [GOOD])
    calls: int = 0
    answers: int = 0

    def answer(self) -> dict[str, Any]:
        self.calls += 1
        self.answers += 1
        return self.script[min(self.answers - 1, len(self.script) - 1)]

    def other(self) -> None:
        self.calls += 1


@dataclass
class Notifier:
    """The counted side effect that runs just before an approval wait."""

    calls: int = 0

    def __call__(self) -> None:
        self.calls += 1


@dataclass
class World:
    search: Search = field(default_factory=Search)
    model: Model = field(default_factory=Model)
    notify: Notifier = field(default_factory=Notifier)


# --- Effective ------------------------------------------------------------------------------


def effective_domain(world: World) -> Any:
    from effective import MeteredInterpreter, Usage

    def llm(op: Any) -> tuple[dict[str, Any], Usage]:
        return world.model.answer(), Usage()

    def tools(op: Any) -> Any:
        match op.name:
            case "search":
                return world.search(**op.args)
            case "notify":
                world.notify()
                return "notified"
            case other:
                raise KeyError(other)

    return MeteredInterpreter(llm=llm, tools=tools)


# --- DSPy -----------------------------------------------------------------------------------


def dspy_lm(model: Model, *, cache: bool = False) -> Any:
    """A `DummyLM` whose answers come from `model`. `cache=True` restores the response cache a
    real `dspy.LM` has by default and `DummyLM` turns off."""
    from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
    from dspy.utils.dummies import DummyLM

    feedback = {"discussion": "The citation names no passage.", "advice": {}}

    class Engine(DummyEngine):
        def _complete_messages(self, messages: list[dict[str, Any]]) -> Any:
            if "advice" in messages[0]["content"]:
                model.other()
                self.owner.answers = iter([feedback])
            else:
                self.owner.answers = iter([model.answer()])
            return super()._complete_messages(messages)

    class Counted(DummyLM):
        def __init__(self) -> None:
            super().__init__([])
            self.cache = cache
            self._cache_responses = cache
            self._wire()

        def _wire(self) -> None:
            self._engine_spec = Engine(self)
            self._async_engine_spec = AsyncDummyEngine(self._engine_spec)

        def copy(self, **kwargs: Any) -> Any:
            copied = super().copy(**kwargs)
            copied._wire()
            return copied

    return Counted()


# --- Pydantic AI ----------------------------------------------------------------------------


def pydantic_ai_model(model: Model, *, search_args: dict[str, Any] | None = None) -> Any:
    """A `FunctionModel` that calls `search`, then `request_review` when the agent has it, then
    answers through the output tool."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        returned = {
            part.tool_name
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        tools = {tool.name for tool in info.function_tools}
        if "search" not in returned:
            model.other()
            return ModelResponse(parts=[ToolCallPart("search", search_args or {"query": "q"})])
        if "request_review" in tools and "request_review" not in returned:
            model.other()
            return ModelResponse(parts=[ToolCallPart("request_review", {})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, model.answer())])

    return FunctionModel(respond)


# --- LangGraph ------------------------------------------------------------------------------


def langchain_model(model: Model, crash_after: Callable[[], bool] = lambda: False) -> Any:
    """A chat model that answers every structured-output request from `model`.

    `crash_after()` is consulted after each answer; `True` raises `Crash` once the model has
    answered, which is a process dying inside the node that called it."""
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self.bind(tools=tools, **kwargs)

        def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any):
            args = model.answer()
            if crash_after():
                raise Crash("the process died after the model answered")
            call = {"name": "GenerateAnswer", "args": args, "id": f"call-{model.calls}"}
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))]
            )

    return Scripted()


def once() -> Callable[[], bool]:
    """`True` the first time it is asked, `False` after."""
    flags: Iterator[bool] = iter([True])
    return lambda: next(flags, False)
