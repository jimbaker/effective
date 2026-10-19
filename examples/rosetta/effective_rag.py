"""The retrieval program in Effective: a workflow yields a search and a model call as ops."""

from pydantic import BaseModel

from effective import Effect, Subject, ask_llm, await_event, call_tool, compose_key
from effective.channels import Field, Gated, Repair, render


class GenerateAnswer(BaseModel):
    reasoning: str
    answer: str
    citation: int


def ask(question: str, passages: list[str], repairs: int = 0) -> Effect[GenerateAnswer | Repair]:
    """Ask for an answer citing one of `passages`, re-prompting up to `repairs` times with the
    reason the last answer was refused."""
    context = "\n".join(passages)
    reasoning, answer = Field(str), Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), "cite a passage by its number")
    prompt = t"""Answer the question from the passages, citing one by its number.
{context}
Question: {question}
Reasoning: {reasoning}
Answer: {answer}
Citation: {citation}"""
    outcome: GenerateAnswer | Repair = Repair("not asked")
    for _ in range(repairs + 1):
        respond = render(prompt, output=GenerateAnswer)
        response = yield from ask_llm("respond", respond.messages, dict)
        match respond.resolve(response):
            case GenerateAnswer() as outcome:
                return outcome
            case Repair(reason=reason) as outcome:
                prompt = t"{prompt}\nYour last answer was refused: {reason}"
    return outcome


def rag(question: str, k: int = 3, repairs: int = 0) -> Effect[GenerateAnswer | Repair]:
    passages = yield from call_tool("search", {"query": question, "k": k}, list[str])
    return (yield from ask(question, passages, repairs))


def reviewed_rag(request_id: str, question: str, k: int = 3) -> Effect[GenerateAnswer | Repair]:
    """`rag` with a reviewer's approval between the search and the answer."""
    passages = yield from call_tool("search", {"query": question, "k": k}, list[str])
    yield from call_tool("notify", {"passages": passages}, str)
    review = compose_key(t"review:{Subject(request_id)}")
    if not (yield from await_event(review, bool)):
        return Repair("the reviewer refused the passages")
    return (yield from ask(question, passages))
