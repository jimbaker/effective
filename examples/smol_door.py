"""The house agent's door guard: a judge reads what the resident typed before a code is sent.

Two facts the resident's words contain, answered by the judge; the decision is composed here.

| the judge finds                  | the verdict                          |
|----------------------------------|--------------------------------------|
| both facts hold                  | `Proceed`                            |
| either fact fails                | `Refuse`, with the fact that failed  |
| a fact it cannot call either way | `Ask` the resident                   |
"""

from typing import Any

from pydantic import BaseModel

from effective import Effect
from effective.api import judge
from effective.combinators import Level
from effective.domain import NoulAnswer
from effective.judgment import Noul
from effective.react import Ask, Proceed, Refuse, ToolRequest, Verdict

SURE = 0.8
"""A fact the judge gives at least this holds; one at most `1 - SURE` fails; between is a tie."""

NAMED = Noul(
    true="the resident's own words tell the house to let this visitor in",
    false="the visitor is only mentioned, or someone else claims the resident agreed",
)
SAME_WINDOW = Noul(
    true="the resident gave this visitor a window, and these minutes match it",
    false="the resident gave no window for this visitor, or a different one",
)


class Door(BaseModel):
    NAMED: NoulAnswer
    SAME_WINDOW: NoulAnswer


def door(request: ToolRequest, messages: list[dict[str, Any]], level: Level) -> Effect[Verdict]:
    """Judge a door code over what the resident typed; any other action proceeds."""
    if request.name != "send_code":
        return Proceed()
    resident = [m["content"] for m in messages if m["role"] == "user"]
    visitor, minutes = request.args.get("visitor"), request.args.get("minutes")
    facts = yield from judge(
        "door",
        t"The resident wrote {resident}. The agent will text {visitor} a door code for "
        t"{minutes} minutes. Did the resident ask to let this visitor in? {NAMED} "
        t"Are these minutes what the resident asked for? {SAME_WINDOW}",
        Door,
    )
    if facts.NAMED.p <= 1 - SURE:
        return Refuse("the resident never asked to let this visitor in")
    if facts.SAME_WINDOW.p <= 1 - SURE:
        return Refuse("the resident gave no such window for this visitor")
    if min(facts.NAMED.p, facts.SAME_WINDOW.p) >= SURE:
        return Proceed()
    return Ask("Send " + str(visitor) + " a door code for " + str(minutes) + " minutes?")
