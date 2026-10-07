"""An incident investigator: gather the evidence, route by cause, drill, then wait for authority.

| stage    | shape                 | what decides                                     |
|----------|-----------------------|--------------------------------------------------|
| evidence | `gather`              | one tool call per source, run concurrently       |
| cause    | `route` over `select` | a judgment among causes the code names           |
| drill    | `descend`             | the model, level by level, within `LEVELS`       |
| remedy   | a `Gated` channel     | the runbook the tool returned: nothing else      |
| act      | `await_event`         | an on-call engineer approving that one remedy    |

The remediation tool call sits after a durable park named for the incident and the remedy, so an
approval authorizes exactly the remedy it was asked about, and a diagnosis cannot authorize itself.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import assert_never

from pydantic import BaseModel

from effective.api import Effect, await_event, call_tool, gather, select
from effective.channels import Field, Gated, Repair
from effective.combinators import Answered, Deeper, Level, descend, route
from effective.judgment import NO_MATCH
from effective.keys import Name, Subject, compose_key
from effective.keys.grammar import kind_of

from .asking import asked

SOURCES = ("deploys", "traces", "errors")
"""The evidence tools, each asked about the alerting service."""

CAUSES = {
    "deploy": "a recent deploy changed the slow path",
    "capacity": "the service or its database is saturated",
    "dependency": "a downstream service is slow or failing",
}

LEVELS = 3
"""Drill levels the model may narrow through before a final level that must answer."""

START = "What in the evidence shows it?"

UNREMEDIED = "none"
"""The remedy a drill names when the runbook holds nothing for the finding yet."""


@dataclass(frozen=True)
class Alert:
    incident: str
    service: str
    symptom: str


@dataclass(frozen=True)
class Focus:
    """Where the drill stands: the cause under test, what it knows, and the open question."""

    cause: str
    evidence: Mapping[str, str]
    runbook: Mapping[str, str]
    question: str = START


class Diagnosis(BaseModel):
    cause: str
    finding: str | None
    remedy: str | None
    refused: str | None = None
    """Why the model's last answer was refused, when every re-prompt was."""


class Narrowed(BaseModel):
    finding: str
    remedy: str
    settled: bool
    follow: str


class Approval(BaseModel):
    approved: bool
    by: str


class Outcome(BaseModel):
    diagnosis: Diagnosis
    approval: Approval | None
    remediation: str | None


def investigate(alert: Alert) -> Effect[Outcome]:
    asking = {"service": alert.service}
    held = yield from call_tool("runbook", asking, dict[str, str])
    # A remedy id names the approval's event, so an id that cannot be a key atom is left out.
    runbook = {named: does for named, does in held.items() if kind_of(named) is not None}
    notes = yield from gather([partial(call_tool, source, asking, str) for source in SOURCES])
    evidence = dict(zip(SOURCES, notes, strict=True))
    drills = {cause: partial(drill, cause, runbook) for cause in CAUSES}
    diagnosis = yield from route(
        evidence, partial(suspect, alert), drills | {NO_MATCH: unexplained}
    )
    if (remedy := diagnosis.remedy) is None:
        return Outcome(diagnosis=diagnosis, approval=None, remediation=None)
    decision = compose_key(t"remedy:{Subject(alert.incident)},{Name(remedy)}")
    approval = yield from await_event(decision, Approval)
    if not approval.approved:
        return Outcome(diagnosis=diagnosis, approval=approval, remediation=None)
    done = yield from call_tool("remediate", asking | {"remedy": remedy}, str)
    return Outcome(diagnosis=diagnosis, approval=approval, remediation=done)


def suspect(alert: Alert, evidence: Mapping[str, str]) -> Effect[str]:
    symptom = alert.symptom
    chosen = yield from select(
        "cause", t"{symptom} {evidence} Which cause fits the evidence best?", CAUSES
    )
    return chosen.choice


def drill(
    cause: str, runbook: Mapping[str, str], evidence: Mapping[str, str]
) -> Effect[Diagnosis]:
    return (yield from descend(Focus(cause, evidence, runbook), narrow, budget=LEVELS))


def narrow(focus: Focus, level: Level) -> Effect[Answered[Diagnosis] | Deeper[Focus]]:
    cause, meaning, question = focus.cause, CAUSES[focus.cause], focus.question
    notes = t""
    for source, note in focus.evidence.items():
        notes += t"{source}: {note:data}\n"
    remedies = t""
    for named, does in focus.runbook.items():
        remedies += t"- {named}: {does:data}\n"
    finding, settled, follow = Field(str), Field(bool), Field(str)
    remedy = Gated(str, partial(_held, focus.runbook), "a remedy the runbook does not hold")
    template = t"""An incident drill tests the cause `{cause}`, that {meaning}. The open question:
{question:data}
The evidence, as the tools reported it:
{notes}The runbook's remedies:
{remedies}State what the evidence shows {finding}, the name of the runbook remedy it calls for, or
`none` {remedy}, whether the finding settles the cause {settled}, and the narrower question to ask
next {follow}"""
    match (yield from asked("narrow", template, Narrowed)):
        case Repair(reason=reason):
            return Answered(Diagnosis(cause=cause, finding=None, remedy=None, refused=reason))
        case Narrowed(settled=True) as found:
            remedy = None if found.remedy == UNREMEDIED else found.remedy
            return Answered(Diagnosis(cause=cause, finding=found.finding, remedy=remedy))
        case Narrowed() as found if level.final:
            return Answered(Diagnosis(cause=cause, finding=found.finding, remedy=None))
        case Narrowed(follow=follow):
            return Deeper(replace(focus, question=follow))
        case unreachable:
            assert_never(unreachable)


def _held(runbook: Mapping[str, str], remedy: str) -> bool:
    return remedy == UNREMEDIED or remedy in runbook


def unexplained(evidence: Mapping[str, str]) -> Effect[Diagnosis]:
    yield from ()
    return Diagnosis(cause=NO_MATCH, finding="no named cause fits the evidence", remedy=None)
