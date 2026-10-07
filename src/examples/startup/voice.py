"""A customer-voice radar: read a ticket corpus larger than one prompt, settle its themes, and send
each theme to the team that owns its kind.

| stage    | shape      | what decides                                             |
|----------|------------|----------------------------------------------------------|
| read     | `recurse`  | a model per page of tickets; the counts merge in a tree  |
| settle   | `fixpoint` | the model merges near-duplicate themes until none remain |
| dispatch | `route`    | a `select` among the kinds the code names, one per theme |

Convergence is a test in code: the theme names stop changing. A model that keeps renaming themes
spends the budget, and the radar reports `settled=False`. An answer still refused after its
re-prompts fails the run with `Unanswered`, since a skipped page or merge would be counted wrong.
"""

from collections import Counter
from collections.abc import Sequence
from functools import partial
from typing import assert_never

from pydantic import BaseModel

from effective.api import Effect, call_tool, gather, select
from effective.channels import Field, Repair
from effective.combinators import Converged, Unconverged, fixpoint, recurse, route
from effective.judgment import NO_MATCH

from .asking import asked

KINDS = {
    "bug": "the product does something wrong",
    "capability": "the product cannot do something customers ask for",
    "onboarding": "new customers get stuck before their first success",
    "pricing": "customers misread what they pay or why",
}

PAGE = 3
"""Tickets one model call reads."""

ROUNDS = 3
"""Consolidation rounds before the radar reports the themes as they stand."""

type Themes = dict[str, int]
"""Each theme's name and the tickets that raise it."""


class Unanswered(Exception):
    """The model's answer was still refused after every re-prompt."""


class Read(BaseModel):
    themes: Themes


class Radar(BaseModel):
    themes: Themes
    settled: bool
    routed: dict[str, str]
    """Each theme and the board it was filed on."""


def radar(since: str) -> Effect[Radar]:
    tickets = yield from call_tool("tickets", {"since": since}, list[str])
    if not tickets:
        return Radar(themes={}, settled=True, routed={})
    counted = yield from recurse(tickets, pages, observe, merged)
    match (yield from fixpoint(counted, consolidate, budget=ROUNDS, converged=same_names)):
        case Converged(value=themes):
            settled = True
        case Unconverged(value=themes):
            settled = False
        case unreachable:
            assert_never(unreachable)
    owners = {kind: partial(file, kind) for kind in KINDS} | {NO_MATCH: partial(file, "triage")}
    sent = yield from gather([partial(route, name, kind_of, owners) for name in themes])
    return Radar(themes=themes, settled=settled, routed=dict(zip(themes, sent, strict=True)))


def pages(tickets: Sequence[str]) -> Effect[Sequence[Sequence[str]]]:
    yield from ()
    return [tickets[i : i + PAGE] for i in range(0, len(tickets), PAGE)]


def observe(page: Sequence[str]) -> Effect[Themes]:
    tickets = t""
    for ticket in page:
        tickets += t"- {ticket:data}\n"
    themes = Field(dict[str, int])
    template = t"""Support tickets:
{tickets}Name each problem these tickets raise, with how many tickets raise it: {themes}"""
    match (yield from asked("observe", template, Read)):
        case Repair(reason=reason):
            raise Unanswered(reason)
        case Read(themes=found):
            return found
        case unreachable:
            assert_never(unreachable)


def merged(counts: Sequence[Themes]) -> Effect[Themes]:
    yield from ()
    total: Counter[str] = Counter()
    for count in counts:
        total.update(count)
    return dict(total)


def consolidate(current: Themes) -> Effect[Themes]:
    listed = t""
    for name, tickets in current.items():
        listed += t"- {name:data}: {tickets}\n"
    themes = Field(dict[str, int])
    template = t"""Themes from support tickets, with their ticket counts:
{listed}Merge themes that name the same problem, adding their counts: {themes}"""
    match (yield from asked("consolidate", template, Read)):
        case Repair(reason=reason):
            raise Unanswered(reason)
        case Read(themes=found):
            return found
        case unreachable:
            assert_never(unreachable)


def same_names(before: Themes, after: Themes) -> bool:
    return before.keys() == after.keys()


def kind_of(theme: str) -> Effect[str]:
    chosen = yield from select("kind", t"{theme} Which kind of problem is this theme?", KINDS)
    return chosen.choice


def file(kind: str, theme: str) -> Effect[str]:
    return (yield from call_tool("file", {"board": kind, "theme": theme}, str))
