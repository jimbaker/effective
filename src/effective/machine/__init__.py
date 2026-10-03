"""A state machine over the effect substrate — generic over the states it walks.

The line between this package and `effective.coding` is which side of the referee a thing
sits on. Here: the trampoline that walks, the budget that bounds it, the outcome
vocabulary it walks between, the spec a state declares, and the unconditional postamble that
commits whatever a run reached. There: the states themselves, the edge table between them, and
the verdicts a judge may return: an embodiment's design rather than a mechanism.

Nothing in this package names a state. `S` is whatever `StrEnum` an embodiment declares, and
`effective.coding` is the first of them rather than the only one — a machine over `just
docs-check`, or over any process with a measurable predicate and a feedback loop, instantiates
the same walk.
"""

from effective.machine.audit import GraphDefect, audit, audit_relation
from effective.machine.evidence import CommandRun, Commitment, Measured, Predicate
from effective.machine.outcomes import (
    Advance,
    Exhausted,
    Finish,
    Outcome,
    Park,
    ParkReason,
    VerdictOutOfDomain,
)
from effective.machine.spec import Ctx, Evidence, Judge, Report, Run, StateSpec, Worker
from effective.machine.specs import agent_worker, build_specs, fuse
from effective.machine.trampoline import (
    Session,
    Stop,
    Turn,
    appending_states,
    canonical_violations,
    commitment_postamble,
    run_machine,
    stop_record,
)

__all__ = [
    "Advance",
    "CommandRun",
    "Commitment",
    "Ctx",
    "Evidence",
    "Exhausted",
    "Finish",
    "GraphDefect",
    "Judge",
    "Measured",
    "Outcome",
    "Park",
    "ParkReason",
    "Predicate",
    "Report",
    "Run",
    "Session",
    "StateSpec",
    "Stop",
    "Turn",
    "VerdictOutOfDomain",
    "Worker",
    "agent_worker",
    "appending_states",
    "audit",
    "audit_relation",
    "build_specs",
    "canonical_violations",
    "commitment_postamble",
    "fuse",
    "run_machine",
    "stop_record",
]  # fmt: skip
