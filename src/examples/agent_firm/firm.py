"""`firm`: settle a book's cases until a pass changes none of them.

A case's state is the ledger's: its rounds' claims and endings. Each pass advances every open case
by what its ledger says.

| the ledger says of the case      | the pass                                   | the case is     |
|----------------------------------|--------------------------------------------|-----------------|
| its latest round completed       | does nothing                               | closed          |
| a round claimed and not ended    | does nothing                               | open, in flight |
| its latest round ended unlanded  | continues the leg under the round's keys   | as the leg ends |
| no claim of its first open round | reads the statement, and resolves it there | as resolved     |

A round that ended compensated passes the case to the next round, so the pass resolves it there.
The statement says what the firm did not do: an invoice that owes nothing closes its case unasked.
A case the approver denied stays open, denied, and the pass asks nobody again.

`firm` runs passes until two in a row leave the same cases open, each in the same round for the
same reason, each pass a level of the substrate's `fixpoint` under the books' own scope, its cases
advanced concurrently. `Converged` with open cases is the firm's dissent: what it agreed, and what
keeps the rest open. `Unconverged` is the budget running out while a case still moved.
"""

from collections.abc import Sequence
from enum import StrEnum
from functools import partial
from typing import assert_never

from pydantic import BaseModel

from effective.api import Effect, gather, scoped
from effective.combinators import Settled, fixpoint
from effective.keys import Name, compose_key

from .ar import (
    INVARIANTS,
    Claimed,
    Done,
    Superseded,
    Unclaimed,
    Uncontinued,
    Unfinished,
    Unopened,
    resume,
    standing,
    statement,
)
from .cases import Case
from .knowledge import NO_RULES, Knowledge
from .resolve import Denied, Resolved, Stopped, resolve
from .saga import Compensated, Completed, Unlanded


class Last(StrEnum):
    """What keeps an open case open."""

    STOPPED = "stopped"
    """A model gave no usable answer."""
    DENIED = "denied"
    """The approver refused the credit."""
    IN_FLIGHT = "in-flight"
    """Another settlement holds a round that has not ended."""
    UNLANDED = "unlanded"
    """A retriable leg ran out; the next pass continues it, or the next run once the budget is
    spent."""
    COMPENSATED = "compensated"
    """The settlement was undone; the next pass resolves the case in the next round, or the next
    run once the budget is spent."""


class Open(BaseModel, frozen=True):
    """A case the firm has not settled, what it breaks, why it is open, and its round."""

    case: str
    broken: tuple[str, ...]
    last: Last
    round: int


class Closed(BaseModel, frozen=True):
    """A case the firm has settled, or whose invoice owes nothing, as of its round."""

    case: str
    round: int


PASSES = 4
"""How many passes the firm makes before it answers `Unconverged`."""


class Books(BaseModel, frozen=True):
    """The firm's books after `passes` passes: the cases still open, and what keeps each, `None`
    before the first pass."""

    passes: int
    open: dict[str, Open | None]


def firm(
    books: str,
    cases: Sequence[Case],
    *,
    critic: bool,
    knowledge: Knowledge = NO_RULES,
    budget: int = PASSES,
) -> Effect[Settled[Books]]:
    """Pass over `cases` in `books` until two passes leave the same cases open, each in the same
    round for the same reason, none of them one a pass can still continue."""
    named = {case.invoice.id: case for case in cases}

    def passed(before: Books) -> Effect[Books]:
        steps = [
            partial(advance, named[id], books, last, critic=critic, knowledge=knowledge)
            for id, last in before.open.items()
        ]
        return Books(passes=before.passes + 1, open=_still_open((yield from gather(steps))))

    def settled(before: Books, after: Books) -> bool:
        runnable = {Last.UNLANDED, Last.COMPENSATED}
        moved = before.open != after.open
        return not moved and not any(
            o is not None and o.last in runnable for o in after.open.values()
        )

    first = Books(passes=0, open={id: None for id in named})
    loop = partial(fixpoint, first, passed, budget=budget, converged=settled)
    return (yield from scoped(compose_key(t"firm:{Name(books)}"), loop))


def _still_open(advanced: Sequence[Open | Closed]) -> dict[str, Open | None]:
    still: dict[str, Open | None] = {}
    for case in advanced:
        match case:
            case Open():
                still[case.case] = case
            case Closed():
                pass
            case unreachable:
                assert_never(unreachable)
    return still


def advance(
    case: Case, books: str, last: Open | None, *, critic: bool, knowledge: Knowledge = NO_RULES
) -> Effect[Open | Closed]:
    """One pass over `case` in `books`, `last` being how the pass before left it."""
    broken = tuple(name for name, holds in INVARIANTS.items() if not holds(case))

    def opened(why: Last, number: int) -> Open:
        return Open(case=case.invoice.id, broken=broken, last=why, round=number)

    if last is not None and last.last is Last.DENIED:
        return last
    match (yield from standing(case, books)):
        case Done(round=round):
            return Closed(case=case.invoice.id, round=round.number)
        case Claimed(round=round):
            return opened(Last.IN_FLIGHT, round.number)
        case Unfinished(round=round):
            ended: Resolved | Uncontinued = yield from resume(case, round)()
        case Unclaimed(round=round):
            if (yield from statement(case.invoice.id)).owed == 0:
                return Closed(case=case.invoice.id, round=round.number)
            ended = yield from resolve(case, critic=critic, round=round, knowledge=knowledge)()
        case unreachable:
            assert_never(unreachable)
    match _left_open(ended):
        case None:
            return Closed(case=case.invoice.id, round=round.number)
        case why:
            return opened(why, round.number)


def _left_open(resolved: Resolved | Uncontinued) -> Last | None:
    """What keeps a case open once `resolved`, or `None` when it closed the case."""
    match resolved:
        case Completed() | Uncontinued(ending="completed"):
            return None
        case Compensated():
            return Last.COMPENSATED
        case Unlanded():
            return Last.UNLANDED
        case Superseded() | Unopened() | Uncontinued():
            return Last.IN_FLIGHT
        case Stopped():
            return Last.STOPPED
        case Denied():
            return Last.DENIED
        case unreachable:
            assert_never(unreachable)
