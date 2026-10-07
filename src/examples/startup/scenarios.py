"""One scripted scenario per workflow: its inputs, the world it reaches, and who it waits on.

| scenario   | the world                                       | the people                  |
|------------|-------------------------------------------------|-----------------------------|
| `incident` | a deploy slowed checkout; the drill finds it    | on-call approves a rollback |
| `launch`   | the load test regressed; the other checks pass  | its owner holds the launch  |
| `voice`    | three pages of tickets, two names for one theme | nobody                      |
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from pydantic import BaseModel

from effective.api import Effect
from effective.domain import Judge

from . import incident, launch, voice
from .engine import World


@dataclass(frozen=True)
class Scenario:
    program: Callable[[], Effect[BaseModel]]
    world: Callable[[], World]
    deliver: Callable[[str], BaseModel] | None = None


def _choice(label: str) -> dict[str, Any]:
    return {"choice": label, "confidence": 0.9, "probabilities": {label: 0.9}}


ALERT = incident.Alert("inc-0412", "checkout-api", "checkout p95 latency doubled at 14:05")

INCIDENT_TOOLS = {
    "deploys": lambda args: "checkout-api v412 deployed 14:02; it changes the cart pricing path",
    "traces": lambda args: (
        "POST /checkout p95 2.1s from 1.0s since 14:05; time is in pricing.quote"
    ),
    "errors": lambda args: "no new error classes; timeouts in pricing.quote up 4x",
    "runbook": lambda args: {
        "rollback": "roll back the last deploy",
        "scale-out": "add instances behind the load balancer",
        "failover": "fail over to the standby dependency",
    },
    "remediate": lambda args: "rolled back to v411",
}


def _drill(prompt: str) -> dict[str, Any]:
    """The first level narrows to the endpoint; the second settles on the deploy."""
    if incident.START in prompt:
        return {
            "finding": "latency began three minutes after v412 shipped",
            "remedy": incident.UNREMEDIED,
            "settled": False,
            "follow": "Does the slow span sit in code v412 changed?",
        }
    return {
        "finding": "pricing.quote, which v412 changed, holds the added latency",
        "remedy": "rollback",
        "settled": True,
        "follow": "",
    }


def _cause(op: Judge[Any]) -> dict[str, Any]:
    return {"pick": _choice("deploy")}


CANDIDATE = launch.Candidate("v2.4", "rc3", "usage-based billing")

LAUNCH_TOOLS = {
    "security": lambda args: "passed: no findings above low",
    "performance": lambda args: "regressed: p99 checkout 840ms against a 500ms budget",
    "docs": lambda args: "passed: billing guide and API reference published to staging",
    "flags": lambda args: "passed: billing flag defaults off and ramps by cohort",
    "release": lambda args: "released",
}


def _readiness(op: Judge[Any]) -> dict[str, Any]:
    return {"ready": {"p": 0.35}, "blocker": _choice("performance")}


TICKETS = [
    "can't log in after the password reset",
    "login fails on Safari",
    "invoice shows the wrong seat count",
    "login fails with SSO",
    "invoice shows last month's price",
    "export to CSV is missing",
    "can't log in on mobile",
]

READS = {
    "Safari": {"login fails": 2, "invoice is wrong": 1},
    "SSO": {"login fails": 1, "invoice is wrong": 1, "no CSV export": 1},
    "mobile": {"cannot sign in": 1},
}
"""What the model reads off each page, found by a ticket only that page carries."""


def _voice(prompt: str) -> dict[str, Any]:
    """A page is read by a ticket it carries; a consolidation merges the two sign-in names."""
    if "Merge themes" in prompt:
        return {"themes": {"login fails": 4, "invoice is wrong": 2, "no CSV export": 1}}
    return {"themes": next(read for marker, read in READS.items() if marker in prompt)}


KIND_OF = {"login fails": "bug", "invoice is wrong": "pricing", "no CSV export": "capability"}


def _kind(op: Judge[Any]) -> dict[str, Any]:
    return {"pick": _choice(KIND_OF[op.state["theme"]])}


VOICE_TOOLS = {
    "tickets": lambda args: TICKETS,
    "file": lambda args: args["board"],
}

SCENARIOS = {
    "incident": Scenario(
        program=partial(incident.investigate, ALERT),
        world=lambda: World(llm=_drill, tools=INCIDENT_TOOLS, judge=_cause),
        deliver=lambda event: incident.Approval(approved=True, by="on-call"),
    ),
    "launch": Scenario(
        program=partial(launch.launch, CANDIDATE),
        world=lambda: World(llm=lambda prompt: {}, tools=LAUNCH_TOOLS, judge=_readiness),
        deliver=lambda event: launch.Ruling(ship=False, note="hold until p99 is under budget"),
    ),
    "voice": Scenario(
        program=partial(voice.radar, "2026-09-01"),
        world=lambda: World(llm=_voice, tools=VOICE_TOOLS, judge=_kind),
    ),
}
