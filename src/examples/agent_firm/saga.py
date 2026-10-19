"""`saga`: forward effects, each compensable one undone by a later forward effect if the run fails.

| leg           | what it is                     | if it is rejected                      |
|---------------|--------------------------------|----------------------------------------|
| `COMPENSABLE` | reversible by a later entry    | the landed compensable legs are undone |
| `PIVOT`       | the last effect that can fail  | the landed compensable legs are undone |
| `RETRIABLE`   | an effect after the pivot      | asked again; nothing is undone         |

Every compensable leg precedes the pivot, and every retriable leg follows it; `saga` refuses any
other order before anything runs. An undo is itself a step with its own key, so a crash between an
effect and its undo replays to both, once each. A rejected undo fails the run, since nothing
compensates a compensation, and leaves the landed legs it had not reached for a person to undo.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from types import MappingProxyType
from typing import Any, assert_never

from pydantic import BaseModel

from effective.api import Effect, scoped
from effective.combinators import Converged, Grantor, Unconverged, fixpoint
from effective.govern import BudgetRefused
from effective.keys import Name, compose_key

RETRIES = 3
"""Times a retriable leg is asked, without a grant, before the saga stops with it unlanded."""


class Landed(BaseModel):
    value: Any = None


class Rejected(BaseModel):
    reason: str


type Outcome = Landed | Rejected


class LegKind(StrEnum):
    COMPENSABLE = "compensable"
    PIVOT = "pivot"
    RETRIABLE = "retriable"


@dataclass(frozen=True)
class Leg:
    name: str
    kind: LegKind
    act: Callable[[int, Mapping[str, Any]], Effect[Outcome]]
    """What the leg does on an attempt, counted from 0, given the values the legs before it landed
    with, by name. Only a retriable leg is asked again."""
    undo: Callable[[Mapping[str, Any]], Effect[Outcome]] | None = None
    """What reverses this leg, given the values the legs landed with, by name. A compensable leg
    has one."""
    grantor: Grantor | None = None
    """Who may grant a retriable leg attempts past `RETRIES`."""


class SagaOrder(ValueError):
    """A saga whose legs cannot be compensated in the order given."""


class Completed(BaseModel):
    landed: dict[str, Any]


class Compensated(BaseModel):
    rejected: str
    reason: str
    undone: list[str]


class Unlanded(BaseModel):
    """A retriable leg after the pivot that did not land in `RETRIES` asks."""

    leg: str
    reason: str
    landed: dict[str, Any]


type Ended = Completed | Compensated | Unlanded

_RANK = {LegKind.COMPENSABLE: 0, LegKind.PIVOT: 1, LegKind.RETRIABLE: 2}


def saga(legs: Sequence[Leg]) -> Callable[[], Effect[Ended]]:
    """The program that runs `legs` in order, refused here if their order cannot be compensated."""
    ranks = [_RANK[leg.kind] for leg in legs]
    if ranks != sorted(ranks) or ranks.count(_RANK[LegKind.PIVOT]) > 1:
        raise SagaOrder(f"compensable, then at most one pivot, then retriable: {legs_of(legs)}")
    if unpaired := [
        leg.name for leg in legs if (leg.kind is LegKind.COMPENSABLE) != bool(leg.undo)
    ]:
        raise SagaOrder(f"a compensable leg and only a compensable leg has an undo: {unpaired}")
    return lambda: _run(legs)


def legs_of(legs: Sequence[Leg]) -> list[tuple[str, LegKind]]:
    return [(leg.name, leg.kind) for leg in legs]


def _run(legs: Sequence[Leg]) -> Effect[Ended]:
    landed: dict[str, Any] = {}
    for leg in legs:
        retriable = leg.kind is LegKind.RETRIABLE
        seen = MappingProxyType(landed)
        match (yield from (_retried(leg, seen) if retriable else leg.act(0, seen))):
            case Landed(value=value):
                landed[leg.name] = value
            case Rejected(reason=reason) if retriable:
                return Unlanded(leg=leg.name, reason=reason, landed=landed)
            case Rejected(reason=reason):
                undone = yield from _compensate(legs, landed)
                return Compensated(rejected=leg.name, reason=reason, undone=undone)
    return Completed(landed=landed)


@dataclass(frozen=True)
class _Tried:
    """A retriable leg's attempts so far, and how the last ended: an outcome, or the budget's
    refusal of its spend, which the loop raises again once it ends."""

    attempts: int
    ended: Outcome | BudgetRefused | None = None


def _settled(_last: _Tried, tried: _Tried) -> bool:
    """Does the attempt that made `tried` end the loop: a landed leg, or a refused spend?"""
    match tried.ended:
        case Landed() | BudgetRefused():
            return True
        case Rejected() | None:
            return False
        case unreachable:
            assert_never(unreachable)


def _retried(leg: Leg, landed: Mapping[str, Any]) -> Effect[Outcome]:
    """Ask `leg` until it lands, `RETRIES` times and any more a grant allows, each attempt a
    level of its own under the leg's `retry` scope. A refused spend ends the loop and is raised
    again, since every later metered op is refused the same way."""

    def again(tried: _Tried) -> Effect[_Tried]:
        try:
            outcome = yield from leg.act(tried.attempts, landed)
        except BudgetRefused as refused:
            return _Tried(tried.attempts + 1, refused)
        return _Tried(tried.attempts + 1, outcome)

    loop = partial(
        fixpoint, _Tried(0), again, budget=RETRIES, converged=_settled, grantor=leg.grantor
    )
    match (yield from scoped(compose_key(t"retry:{Name(leg.name)}"), loop)):
        case Converged(value=_Tried(ended=ended)) | Unconverged(value=_Tried(ended=ended)):
            pass
        case unreachable:
            assert_never(unreachable)
    match ended:
        case BudgetRefused() as refused:
            raise refused
        case Landed() | Rejected() as outcome:
            return outcome
        case None:
            return Rejected(reason="not asked")
        case unreachable:
            assert_never(unreachable)


def _compensate(legs: Sequence[Leg], landed: dict[str, Any]) -> Effect[list[str]]:
    """Undo each landed compensable leg, latest first."""
    undone = []
    for leg in reversed(legs):
        if leg.undo is not None and leg.name in landed:
            reversal: Outcome = yield from leg.undo(MappingProxyType(landed))
            match reversal:
                case Landed():
                    undone.append(leg.name)
                case Rejected(reason=reason):
                    raise RuntimeError(f"the undo of {leg.name} was rejected: {reason}")
                case unreachable:
                    assert_never(unreachable)
    return undone
