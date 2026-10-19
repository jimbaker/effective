"""The retrieval program in DSPy: a signature, a module, and chain of thought."""

from collections.abc import Callable

import dspy


class GenerateAnswer(dspy.Signature):
    """Answer the question from the passages, citing one by its number."""

    context: list[str] = dspy.InputField(desc="numbered from 0")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citation: int = dspy.OutputField(desc="the number of the passage the answer comes from")


class RAG(dspy.Module):
    def __init__(self, search: Callable[..., list[str]], k: int = 3) -> None:
        super().__init__()
        self.search = search
        self.k = k
        self.respond = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        context = self.search(question, k=self.k)
        prediction = self.respond(context=context, question=question)
        return dspy.Prediction(context=context, **prediction)


def cites_a_passage(args: dict, prediction: dspy.Prediction) -> float:
    return 1.0 if 0 <= prediction.citation < len(prediction.context) else 0.0


def refined(rag: RAG, attempts: int = 2) -> dspy.Refine:
    return dspy.Refine(module=rag, N=attempts, reward_fn=cites_a_passage, threshold=1.0)


def best_of(rag: RAG, attempts: int = 2) -> dspy.BestOfN:
    return dspy.BestOfN(module=rag, N=attempts, reward_fn=cites_a_passage, threshold=1.0)
