"""A DSPy module ported to Effective: search for passages, then answer from them with a citation.

Run it with `uv run python examples/port_dspy_module.py`. `wiki/concepts/rosetta.md` maps each
DSPy term to Effective's. The DSPy original, which this file does not import:

class GenerateAnswer(dspy.Signature):
    "Answer the question from the passages, citing one by its number."

    passages: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citation: int = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self, k: int = 3):
        self.k = k
        self.respond = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        passages = search(question, k=self.k)
        return self.respond(passages=passages, question=question)
"""

from pydantic import BaseModel

from effective import Effect, RecordingHandler, ReplayHandler, ask_llm, call_tool
from effective.channels import Field, Gated, Repair, render


class GenerateAnswer(BaseModel):
    reasoning: str
    answer: str
    citation: int


def rag(question: str, k: int = 3) -> Effect[GenerateAnswer | Repair]:
    passages = yield from call_tool("search", {"query": question, "k": k}, list[str])
    context = "\n".join(passages)
    reasoning, answer = Field(str), Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), "cite a passage by its number")
    prompt = t"""Answer the question from the passages, citing one by its number.
{context}
Question: {question}
Reasoning: {reasoning}
Answer: {answer}
Citation: {citation}"""
    respond = render(prompt, output=GenerateAnswer)
    response = yield from ask_llm("respond", respond.messages, dict)
    return respond.resolve(response)


PASSAGES = [
    "[0] The Coase theorem concerns bargaining over externalities.",
    "[1] Coase's 1937 essay asks why firms exist at all.",
]
QUESTION = "Which essay asks why firms exist?"


def responses(citation: int) -> dict[str, object]:
    """A search tool and a model, as the recording handler answers them."""
    reply = {"reasoning": "passage 1 names the essay", "answer": "The Nature of the Firm"}
    return {"tool:search": PASSAGES, "respond": reply | {"citation": citation}}


def main() -> None:
    recorder = RecordingHandler(responses(citation=1))
    answer = recorder.run(lambda: rag(QUESTION))
    print("forward:  ", repr(answer), "via", *(entry.key.stored() for entry in recorder.trace))

    replayed = ReplayHandler(recorder.trace).run(lambda: rag(QUESTION))
    print("replay:   ", replayed == answer, "with no search and no model")

    refused = RecordingHandler(responses(citation=7)).run(lambda: rag(QUESTION))
    print("guardrail:", refused)


if __name__ == "__main__":
    main()
