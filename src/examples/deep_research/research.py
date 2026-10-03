"""Deep research as a frontier search over leads, stopped by coverage rather than by a model.

A question comes with cells, the facts a satisfactory answer states. A lead is a search aimed at
one cell. Each step acts on the two leads worth most: it searches, fetches the top pages, and has a
model read each page for claims and further leads. A claim is kept only when its quote states the
value and appears on the page it cites, an elided quote as its parts in order. Jev judges how
likely each new lead is to settle its cell, and the merge folds the claims into the evidence.

| rule                 | holds when                                                          |
|----------------------|---------------------------------------------------------------------|
| a cell is settled    | one value is quoted by at least two hosts, and by more than any rival |
| a cell is contested  | it holds two values and is not settled; a refuting lead joins the queue |
| a cell is alone      | one value, quoted by too few hosts; a lead aimed away from them joins |
| the research is done | every cell is settled, or the steps run out                          |

The model sees only `brief`: the question, the cells, the lead's cell and one page. The run's
history stays on the tape, and the report is a projection of the evidence.
"""

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from string.templatelib import Template
from typing import Any, assert_never

from pydantic import BaseModel

from effective.api import Effect, ask_llm, call_tool, gather, judge
from effective.channels import Field, render
from effective.domain import NoulAnswer
from effective.interpreters.web import (
    FETCH_TOOL,
    SEARCH_TOOL,
    Fetched,
    Page,
    Searched,
    Unreached,
    Unreadable,
    host,
)
from effective.judgment import Noul
from effective.keys import Key, Name, Subject, compose_key
from effective.query import search_query
from effective.search import Round, Step, frontier

PAGES = 3
"""Pages read per lead, from the top of its search."""
PICKED = 2
"""Leads acted on per step."""
DIGEST = 16
"""Hex digits of a query's digest in its lead's key."""
HOSTS = 2
"""Hosts that must quote a value before its cell is settled."""
UNSCORED = 0.5
REFUTING = 1.0
"""The worth of a refuting lead: a contested cell is looked at before anything else."""
CORROBORATING = 0.9
"""The worth of a corroborating lead: a cell one host quotes needs another host next."""


class Claim(BaseModel):
    cell: str
    value: str
    quote: str
    """The page's own words that state the value."""


class Next(BaseModel):
    cell: str
    query: str


class Read(BaseModel):
    """What a model read on one page."""

    claims: list[Claim]
    leads: list[Next]


class Worth(BaseModel):
    settles: NoulAnswer


@dataclass(frozen=True)
class Lead:
    """A web search, rendered by `search_query`, and the cell it is for. One query aimed at two
    cells is two leads, read for two different cells; the search behind them is paid for once."""

    cell: str
    query: str

    @property
    def key(self) -> Key:
        """`lead:{cell},{digest}`, the digest of the query, so two leads share a key only when they
        are one lead."""
        digest = "sha256-" + hashlib.sha256(self.query.encode()).hexdigest()[:DIGEST]
        return compose_key(t"lead:{Name(self.cell)},{Subject(digest)}")


@dataclass(frozen=True)
class Cited:
    cell: str
    value: str
    url: str

    @property
    def host(self) -> str:
        return host(self.url)


@dataclass(frozen=True)
class Finding:
    claims: tuple[Cited, ...]
    leads: tuple[Lead, ...]


@dataclass(frozen=True)
class Evidence:
    question: str
    cells: Mapping[str, str]
    """Each cell's name, and what it asks."""
    cited: tuple[Cited, ...] = ()
    worth: Mapping[Lead, float] = field(default_factory=dict)
    """Each lead's worth: Jev's judgment, `REFUTING` or `CORROBORATING`."""
    tried: frozenset[Lead] = frozenset()
    """Every lead acted on or queued, so none is asked twice."""


type Pick = Callable[[Sequence[Lead], Evidence], Sequence[int]]


def research(
    question: str,
    cells: Mapping[str, str],
    *,
    steps: int = 6,
    pick: Pick | None = None,
    corroborate: bool = True,
    until_settled: bool = True,
) -> Effect[dict[str, Any]]:
    """Research `question` until every cell is settled or `steps` run out, and report each cell's
    value and the pages that state it, or `None` for a cell left unsettled.

    | knob                    | what it changes                                        |
    |-------------------------|--------------------------------------------------------|
    | `pick`                  | which leads a step acts on: `worth_most` by default    |
    | `corroborate=False`     | a cell one host quotes gets no search away from it     |
    | `until_settled=False`   | every lead is explored until `steps` run out           |"""
    for cell in cells:
        compose_key(t"cell:{Name(cell)}")
    seeds = [Lead(cell, search_query(t"{question} {asks}")) for cell, asks in cells.items()]
    start = Evidence(question, cells, tried=frozenset(seeds))
    evidence = yield from frontier(
        seeds,
        partial(act, question, cells),
        partial(into_evidence, corroborate=corroborate, until_settled=until_settled),
        initial=start,
        steps=steps,
        pick=pick or worth_most,
        key=lambda lead: lead.key,
        children=lambda finding: finding.leads,
        score=worth,
    )
    return report(evidence)


def act(question: str, cells: Mapping[str, str], lead: Lead) -> Effect[Finding]:
    searched = yield from call_tool(SEARCH_TOOL, {"queries": [lead.query]}, Searched)
    urls = list(dict.fromkeys(hit.url for hit in searched.hits))[:PAGES]
    fetching = [
        partial(call_tool, FETCH_TOOL, {"url": url, "main": True}, Fetched) for url in urls
    ]
    pages = [page for page in (yield from gather(fetching)) if _readable(page)]
    reads = yield from gather([partial(read, question, cells, lead, page) for page in pages])
    claims = tuple(claim for found in reads for claim in found.claims)
    leads = tuple(dict.fromkeys(lead for found in reads for lead in found.leads))
    return Finding(claims, leads)


def _readable(fetched: Fetched) -> bool:
    match fetched.outcome:
        case Page(text=text):
            return bool(text)
        case Unreadable() | Unreached():
            return False
        case unreachable:
            assert_never(unreachable)


def read(question: str, cells: Mapping[str, str], lead: Lead, fetched: Fetched) -> Effect[Finding]:
    match fetched.outcome:
        case Page() as page:
            pass
        case Unreadable() | Unreached():
            return Finding((), ())
        case unreachable:
            assert_never(unreachable)
    prompt = render(brief(question, cells, lead, fetched.url, page), output=Read)
    found = yield from ask_llm("read", prompt.messages, Read)
    text = _words(page.text)
    claims = tuple(
        Cited(claim.cell, claim.value, fetched.url)
        for claim in found.claims
        if claim.cell in cells and _quoted(claim, text)
    )
    leads = tuple(
        Lead(n.cell, n.query) for n in found.leads if n.cell in cells and n.query.strip()
    )
    return Finding(claims, leads)


def brief(question: str, cells: Mapping[str, str], lead: Lead, url: str, page: Page) -> Template:
    """What the model sees of the research: the question, the cells, the cell this lead is for,
    and one page, fenced as data because it comes from the open web. Never the evidence so far."""
    asked = t""
    for name, asks in cells.items():
        asked += t"- {name}: {asks}\n"
    focus, text = lead.cell, page.text
    claims, leads = Field(list[Claim]), Field(list[Next])
    return t"""Research question: {question}
The cells an answer must state:
{asked}This page was found while looking for the cell `{focus}`.

The page's address and text:
{url:data}
{text:data}

Report every cell this page states a value for. Quote the page's own words for each value, as one
passage exactly as it appears, with ... where you skip anything between: {claims}
Suggest web searches that could state a cell this page leaves open, or contradict one it states:
{leads}"""


def worth(lead: Lead) -> Effect[float]:
    query, cell = lead.query, lead.cell
    settles = Noul(true="its top results state the value", false="they do not")
    judged = yield from judge(
        "worth",
        t"""Research needs the value of {cell}, and proposes a web search for {query}.
Will that search's top results state the value of the cell? {settles}""",
        Worth,
    )
    return judged.settles.p


def worth_most(queue: Sequence[Lead], evidence: Evidence) -> Sequence[int]:
    ranked = sorted(range(len(queue)), key=lambda i: -evidence.worth.get(queue[i], UNSCORED))
    return ranked[:PICKED]


def in_order(queue: Sequence[Lead], evidence: Evidence) -> Sequence[int]:
    return range(min(PICKED, len(queue)))


def into_evidence(
    step: Round[Lead, Finding, Evidence], *, corroborate: bool = True, until_settled: bool = True
) -> Step[Lead, Evidence]:
    before = step.value
    cited = before.cited + tuple(c for _, finding in step.expanded for c in finding.claims)
    worth = {**before.worth, **dict(step.scored)}
    evidence = replace(before, cited=cited, worth=worth)
    question, asks = before.question, before.cells
    refuting = [
        Lead(cell, search_query(t"{question} {asks[cell]} {values:either}"))
        for cell, values in contested(evidence).items()
    ]
    corroborating = [
        Lead(cell, search_query(t"{question} {asks[cell]} {value:phrase} {sorted(hosts):exclude}"))
        for cell, (value, hosts) in alone(evidence).items()
        if corroborate
    ]
    worth |= dict.fromkeys(refuting, REFUTING)
    worth |= dict.fromkeys(corroborating, CORROBORATING)
    candidates = [*step.rest, *(lead for lead, _ in step.scored), *refuting, *corroborating]
    settled = settled_cells(evidence) if until_settled else set()
    queue: list[Lead] = []
    tried = set(before.tried)
    for lead in candidates:
        if lead.cell in settled or (lead in tried and lead not in step.rest):
            continue
        if lead not in queue:
            queue.append(lead)
        tried.add(lead)
    evidence = replace(evidence, worth=worth, tried=frozenset(tried))
    return Step(queue, evidence, done=until_settled and settled == set(before.cells))


def _hosts(evidence: Evidence, cell: str) -> dict[str, set[str]]:
    """Each value quoted for `cell`, and the hosts quoting it."""
    quoted: dict[str, set[str]] = defaultdict(set)
    for claim in evidence.cited:
        if claim.cell == cell:
            quoted[_value(claim.value)].add(claim.host)
    return quoted


def settled_cells(evidence: Evidence) -> set[str]:
    return {cell for cell in evidence.cells if _settled(_hosts(evidence, cell)) is not None}


def _settled(quoted: Mapping[str, set[str]]) -> str | None:
    ranked = sorted(quoted.items(), key=lambda vh: -len(vh[1]))
    if not ranked or len(ranked[0][1]) < HOSTS:
        return None
    if len(ranked) > 1 and len(ranked[1][1]) >= len(ranked[0][1]):
        return None
    return ranked[0][0]


def contested(evidence: Evidence) -> dict[str, list[str]]:
    """Each unsettled cell quoted with two values or more, and its values."""
    return {
        cell: sorted(quoted)
        for cell in evidence.cells
        if len(quoted := _hosts(evidence, cell)) > 1 and _settled(quoted) is None
    }


def alone(evidence: Evidence) -> dict[str, tuple[str, set[str]]]:
    """Each cell quoted with one value by fewer than `HOSTS` hosts, its value and those hosts."""
    return {
        cell: (value, hosts)
        for cell in evidence.cells
        if len(quoted := _hosts(evidence, cell)) == 1
        for value, hosts in quoted.items()
        if len(hosts) < HOSTS
    }


def report(evidence: Evidence) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for cell in evidence.cells:
        value = _settled(_hosts(evidence, cell))
        sources = sorted(
            {c.url for c in evidence.cited if c.cell == cell and _value(c.value) == value}
        )
        answer[cell] = None if value is None else {"value": value, "sources": sources}
    return answer


ELISION = re.compile(r"\.\.\.|\u2026")


def _quoted(claim: Claim, page: str) -> bool:
    """The quote states the claim's value, and each part of it, split at an elision, is on `page`
    after the part before it."""
    parts = [_words(part) for part in ELISION.split(claim.quote) if part.strip()]
    if not parts or _value(claim.value) not in " ".join(parts):
        return False
    at = 0
    for part in parts:
        if (found := page.find(part, at)) < 0:
            return False
        at = found + len(part)
    return True


CLOSING = re.compile(r" ([,.;:!?)\]])")
"""A space before closing punctuation, which extraction leaves where markup split the text."""


def _words(text: str) -> str:
    return CLOSING.sub(r"\1", " ".join(text.lower().split()))


def _value(text: str) -> str:
    return " ".join(text.lower().strip(" .").split())
