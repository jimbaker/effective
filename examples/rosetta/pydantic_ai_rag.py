"""The retrieval program in Pydantic AI: an agent with a search tool and a validated output."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    ModelRetry,
    RunContext,
)
from pydantic_ai.messages import ToolReturnPart


class GenerateAnswer(BaseModel):
    reasoning: str
    answer: str
    citation: int


@dataclass
class Deps:
    search: Callable[..., list[str]]
    notify: Callable[[], None] = lambda: None
    k: int = 3


agent = Agent(
    deps_type=Deps,
    output_type=GenerateAnswer,
    instructions="Search for passages, then answer the question citing one by its number.",
)


@agent.tool
def search(ctx: RunContext[Deps], query: str) -> list[str]:
    return ctx.deps.search(query, k=ctx.deps.k)


@agent.output_validator
def cites_a_passage(ctx: RunContext[Deps], output: GenerateAnswer) -> GenerateAnswer:
    """Refuse a citation outside the passages the last search returned, read from the history so
    a run continued from saved messages checks against the same passages."""
    passages = next(
        part.content
        for message in reversed(ctx.messages)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "search"
    )
    if not 0 <= output.citation < len(passages):
        raise ModelRetry("cite a passage by its number")
    return output


reviewed = Agent(
    deps_type=Deps,
    output_type=[GenerateAnswer, DeferredToolRequests],
    instructions="Search for passages, request a review, then answer citing a passage by number.",
)
reviewed.tool(search)
reviewed.output_validator(cites_a_passage)


@reviewed.tool
def request_review(ctx: RunContext[Deps]) -> str:
    ctx.deps.notify()
    if not ctx.tool_call_approved:
        raise ApprovalRequired()
    return "approved"
