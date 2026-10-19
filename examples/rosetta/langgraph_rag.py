"""The retrieval program in LangGraph: a state graph with search and respond nodes, and the same
program in the Functional API."""

from collections.abc import Callable
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel


class GenerateAnswer(BaseModel):
    reasoning: str
    answer: str
    citation: int


class State(TypedDict, total=False):
    question: str
    passages: list[str]
    response: GenerateAnswer
    refusal: str | None
    attempts: int


def prompt(question: str, passages: list[str], refusal: str | None = None) -> str:
    numbered = "\n".join(f"{i}: {passage}" for i, passage in enumerate(passages))
    refused = f"\nYour last answer was refused: {refusal}" if refusal else ""
    return (
        "Answer the question from the passages, citing one by its number.\n"
        f"{numbered}\nQuestion: {question}{refused}"
    )


def build(
    search: Callable[..., list[str]],
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    repairs: int = 0,
    notify: Callable[[], None] | None = None,
    k: int = 3,
) -> Any:
    """The graph: search, then respond. `repairs` routes a bad citation back to respond that
    many times; `notify` adds a review node that waits for a human between the two."""
    structured = model.with_structured_output(GenerateAnswer)

    def search_node(state: State) -> State:
        return {"passages": search(state["question"], k=k)}

    def respond_node(state: State) -> State:
        passages = state["passages"]
        response = structured.invoke(prompt(state["question"], passages, state.get("refusal")))
        cited = 0 <= response.citation < len(passages)
        refusal = None if cited else "cite a passage by its number"
        return {"response": response, "refusal": refusal, "attempts": state.get("attempts", 0) + 1}

    def review_node(state: State) -> State:
        assert notify is not None
        notify()
        if not interrupt({"passages": state["passages"]}):
            raise ValueError("the reviewer refused the passages")
        return {}

    def after_respond(state: State) -> str:
        return "respond" if state.get("refusal") and state["attempts"] <= repairs else END

    graph = StateGraph(State)
    graph.add_node("search", search_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "search")
    if notify is None:
        graph.add_edge("search", "respond")
    else:
        graph.add_node("review", review_node)
        graph.add_edge("search", "review")
        graph.add_edge("review", "respond")
    graph.add_conditional_edges("respond", after_respond, ["respond", END])
    return graph.compile(checkpointer=checkpointer)


def build_functional(
    search: Callable[..., list[str]],
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    k: int = 3,
) -> Any:
    """The same program as an entrypoint over two tasks."""
    structured = model.with_structured_output(GenerateAnswer)

    @task
    def retrieve(question: str) -> list[str]:
        return search(question, k=k)

    @task
    def respond(question: str, passages: list[str]) -> GenerateAnswer:
        return structured.invoke(prompt(question, passages))

    @entrypoint(checkpointer=checkpointer)
    def rag(question: str) -> GenerateAnswer:
        passages = retrieve(question).result()
        return respond(question, passages).result()

    return rag
