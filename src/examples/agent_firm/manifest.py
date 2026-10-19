"""What a resolution is admitted with: its inputs, pinned in its spawn params, so every attempt and
every replay of the run reads what the first attempt read, and the head of the firm's knowledge a
driver admits it at."""

from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from effective.handlers.base import content_digest
from effective.keys import Key

from .ar import Round
from .cases import Case
from .knowledge import NO_RULES, Mismatched, Pinned, adoption_of

MANIFEST = "manifest"
"""The spawn param a run's manifest rides under."""


class Manifest(BaseModel):
    """A resolution's inputs, as its driver admitted them.

    | field       | what it pins                                                  |
    |-------------|---------------------------------------------------------------|
    | `case`      | the invoice, the payment and the documents, by content        |
    | `round`     | the books and the round the run settles in                    |
    | `critic`    | whether the firm engages its critic                           |
    | `knowledge` | the snapshot of the firm's rules it reads                     |
    | `arm`       | the experiment arm the run was assigned to                    |
    | `code`      | the name its driver gives the firm's code and prompts         |
    | `models`    | the model behind each role                                    |
    | `budget`    | the dollar cap on the run's spend; `None` for none            |"""

    model_config = ConfigDict(frozen=True)

    case: Case
    round: Round
    critic: bool
    knowledge: Pinned
    arm: str = "main"
    code: str
    models: Mapping[str, str]
    budget: Decimal | None = None

    @property
    def case_digest(self) -> str:
        return content_digest(self.case.model_dump(mode="json"))

    def params(self) -> dict[str, Any]:
        return {MANIFEST: self.model_dump(mode="json")}

    @classmethod
    def admitted(cls, params: Mapping[str, Any]) -> Manifest:
        """The manifest a task was spawned with."""
        return cls.model_validate(params[MANIFEST])


def head(read: Callable[[Key], Mapping[str, Any] | None], books: str) -> Pinned:
    """The last height claimed in `books`, folded up from the founding `NO_RULES` at height 0, each
    height's parent the snapshot below it."""
    pinned = Pinned(height=0, digest=NO_RULES.digest)
    while (row := read(adoption_of(books, pinned.height + 1))) is not None:
        if row["parent"] != pinned.digest:
            raise Mismatched(f"height {pinned.height + 1} adopts from {row['parent']}")
        pinned = Pinned(height=pinned.height + 1, digest=row["child"])
    return pinned
