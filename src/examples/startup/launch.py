"""A launch-readiness check: run the independent checks at once, judge readiness, and hand only
the exception to a person.

| stage     | shape         | what decides                                           |
|-----------|---------------|--------------------------------------------------------|
| checks    | `gather`      | one tool call per check, each with its own op key      |
| readiness | `judge`       | one judgment over every report: ready, and the blocker |
| ready     | a tool call   | the release ships with no person in the loop           |
| blocked   | `await_event` | the blocking check's owner: ship anyway, or hold       |

`SHIP_AT` is the policy: the judgment answers a probability, and the threshold in code turns it
into a decision, so moving the bar is a one-line diff the tape can be replayed against. The
owner's ruling is named for the build and the blocker, so it settles that build's exception only.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

from pydantic import BaseModel

from effective.api import Effect, await_event, call_tool, gather, judge
from effective.domain import ChoiceAnswer, NoulAnswer
from effective.judgment import Choice, Noul
from effective.keys import Name, Subject, compose_key

CHECKS = {
    "security": "the security review of the release's changes",
    "performance": "the load test against the release candidate",
    "docs": "the public documentation for the launched feature",
    "flags": "the feature flags the launch turns on",
}

SHIP_AT = 0.8
"""The readiness probability at or above which the release ships unattended."""


@dataclass(frozen=True)
class Candidate:
    release: str
    build: str
    feature: str


class Readiness(BaseModel):
    ready: NoulAnswer
    blocker: ChoiceAnswer


class Ruling(BaseModel):
    ship: bool
    note: str


class Launch(BaseModel):
    release: str
    shipped: bool
    blocker: str | None
    ruling: Ruling | None


def launch(candidate: Candidate) -> Effect[Launch]:
    asking = {"release": candidate.release}
    reports = yield from gather([partial(call_tool, check, asking, str) for check in CHECKS])
    readiness = yield from assess(candidate, dict(zip(CHECKS, reports, strict=True)))
    if readiness.ready.p >= SHIP_AT:
        yield from call_tool("release", asking, str)
        return Launch(release=candidate.release, shipped=True, blocker=None, ruling=None)
    blocker = readiness.blocker.choice
    owner = compose_key(t"owner:{Subject(candidate.build)},{Name(blocker)}")
    ruling = yield from await_event(owner, Ruling)
    if ruling.ship:
        yield from call_tool("release", asking, str)
    return Launch(release=candidate.release, shipped=ruling.ship, blocker=blocker, ruling=ruling)


def assess(candidate: Candidate, reports: Mapping[str, str]) -> Effect[Readiness]:
    feature = candidate.feature
    ready = Noul(true="every check passed or found only what the launch can carry")
    blocker = Choice(CHECKS)
    return (
        yield from judge(
            "readiness",
            t"""{feature} {reports} Is the release ready to ship? {ready}
Which check stands most in the way of the launch? {blocker}""",
            Readiness,
        )
    )
