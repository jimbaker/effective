"""Serve the `Judge` op with TypeSafe's Jev, metered against a token budget.

Needs the `judge` extra. Answers are kept under a hash of the request and the model, with the
input tokens they first cost, so a stopped run loses nothing and a rerun spends nothing on what it
already asked; a kept answer reports no usage, because it cost nothing this time.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, assert_never

try:
    import typesafe_sdk as jev
except ImportError as missing:
    raise ImportError(
        "effective.interpreters.jev needs the judge extra: effective[judge]"
    ) from missing

from effective.cost import Usage
from effective.domain import (
    Answers,
    ChoiceAnswer,
    Judge,
    NoulAnswer,
    ScoreAnswer,
    WireChoice,
    WireNoul,
    WireQuestion,
    WireScore,
)
from effective.interpreters.aio import shared_loop
from effective.spend import TokenBudget

MODEL = "jev-latest"

INPUT_PER_MTOK = 0.042
"""Dollars per million input tokens; output is not billed. OrcaRouter's published figure, read
2026-09-22."""


def _question(wire: WireQuestion) -> jev.Choice | jev.Noul | jev.Score:
    instructions = wire.instructions or None
    match wire:
        case WireChoice(criteria=criteria):
            return jev.Choice(criteria=dict(criteria), instructions=instructions)
        case WireNoul(criteria=criteria):
            sides = jev.NoulCriteria(**criteria) if criteria else None
            return jev.Noul(instructions=instructions, criteria=sides)
        case WireScore(criteria=criteria):
            return jev.Score(criteria=list(criteria), instructions=instructions)
        case unreachable:
            assert_never(unreachable)


def _answer(
    answer: jev.ChoiceAnswer | jev.NoulAnswer | jev.ScoreAnswer,
) -> ChoiceAnswer | NoulAnswer | ScoreAnswer:
    match answer:
        case jev.ChoiceAnswer(choice=choice, confidence=confidence, probabilities=probabilities):
            return ChoiceAnswer(choice=choice, confidence=confidence, probabilities=probabilities)
        case jev.NoulAnswer(noul=p):
            return NoulAnswer(p=p)
        case jev.ScoreAnswer(score=score, confidence=confidence, probabilities=probabilities):
            levels = {str(level): p for level, p in probabilities.items()}
            return ScoreAnswer(score=score, confidence=confidence, probabilities=levels)
        case unreachable:
            assert_never(unreachable)


class JevClient(Protocol):
    """The one call Jev serves; `jev.TypeSafeClient` is the live one."""

    def system_one(
        self,
        *,
        state: dict[str, Any],
        questions: Mapping[str, jev.Choice | jev.Noul | jev.Score],
        model: str,
    ) -> jev.SystemOneResponse: ...


def _read(kept: Path) -> Answers | None:
    """A kept answer, or `None` when there is none or it cannot be read, which asks again."""
    try:
        return Answers.model_validate(json.loads(kept.read_text())["answers"])
    except OSError, ValueError, KeyError:
        return None


def _keep(kept: Path, record: dict[str, Any]) -> None:
    """Write whole or not at all: a reader never sees half a record, and each writer has a
    partial file of its own, so two callers keeping one answer both succeed."""
    kept.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=kept.parent, suffix=".partial")
    with os.fdopen(handle, "w") as out:
        json.dump(record, out)
    os.replace(partial, kept)


@dataclass
class Jev:
    """A `JudgeCall`: pass it as `MeteredInterpreter(judge=...)`."""

    client: JevClient
    budget: TokenBudget
    cache: Path | None = None
    model: str = MODEL

    def __call__(self, op: Judge[Any]) -> tuple[Answers, Usage]:
        kept = self._kept(op)
        if kept is not None and (answers := _read(kept)) is not None:
            return answers, Usage()
        answers, tokens = self._ask(op)
        if kept is not None:
            _keep(kept, {"answers": answers.model_dump(), "input_tokens": tokens})
        return answers, Usage(prompt_tokens=tokens, cost=tokens * INPUT_PER_MTOK / 1e6)

    def _kept(self, op: Judge[Any]) -> Path | None:
        if self.cache is None:
            return None
        questions = {name: q.model_dump() for name, q in op.questions.items()}
        asked = {"wire": {"state": op.state, "questions": questions}, "model": self.model}
        digest = hashlib.sha256(json.dumps(asked, sort_keys=True).encode()).hexdigest()
        return (self.cache / digest).with_suffix(".json")

    def _ask(self, op: Judge[Any]) -> tuple[Answers, int]:
        self.budget.check()
        questions = {name: _question(q) for name, q in op.questions.items()}
        response = self._system_one(op.state, questions)
        tokens = response.usage.input_tokens or 0
        self.budget.add(tokens)
        return Answers({name: _answer(a) for name, a in response.answers.items()}), tokens

    def _system_one(
        self, state: dict[str, Any], questions: dict[str, jev.Choice | jev.Noul | jev.Score]
    ) -> jev.SystemOneResponse:
        return self.client.system_one(state=state, questions=questions, model=self.model)


class AsyncJevClient(Protocol):
    """The one call Jev serves, awaited; `jev.AsyncTypeSafeClient` is the live one."""

    async def system_one(
        self,
        *,
        state: dict[str, Any],
        questions: Mapping[str, jev.Choice | jev.Noul | jev.Score],
        model: str,
    ) -> jev.SystemOneResponse: ...


@dataclass
class AsyncJev(Jev):
    """`Jev` over `jev.AsyncTypeSafeClient`, its call awaited on `shared_loop()`. The answers, the
    cache and the budget are `Jev`'s."""

    client: AsyncJevClient

    def _system_one(
        self, state: dict[str, Any], questions: dict[str, jev.Choice | jev.Noul | jev.Score]
    ) -> jev.SystemOneResponse:
        asked = self.client.system_one(state=state, questions=questions, model=self.model)
        return shared_loop().run(asked)
