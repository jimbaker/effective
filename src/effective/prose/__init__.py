"""The de-essaying machine — `effective.machine`'s second embodiment, driven by a human.

`states` is the vocabulary, `transition` the edges, `runners` what each state does. A driver
assembles them:

    specs = build_prose_specs(subject)
    session = yield from run_machine(run_id, goal, specs, transition, start=State.SELECT)

Every state but `VERIFY` parks, so the machine advances only when somebody answers — through the
terminal run view (`src/tui`), which is what this embodiment was built to give that surface a real
subject to show.
"""

from effective.machine.spec import StateSpec
from effective.prose.runners import (
    CALLERS_TOOL,
    DESTINATIONS,
    VERIFY_TOOL,
    AnswerRefused,
    decode,
    park_name,
    park_run,
    read_run,
    verdict_for_verify,
    verify_run,
)
from effective.prose.states import VERDICTS, State, Verdict
from effective.prose.transition import ROUTES, transition


def _run_for(subject: str, state: State):
    """The two states that are not one park, and everything else.

    `VERIFY` derives its verdict from a measurement, and `READ` runs the caller query before it
    parks so the human inventories against evidence rather than memory. Both are mechanical work
    the machine owes the judgment, which is why they are ops on the tape and not something a
    driver runs beside the run."""
    match state:
        case State.VERIFY:
            return verify_run(subject)
        case State.READ:
            return read_run(subject)
        case _:
            return park_run(subject, state)


def build_prose_specs(subject: str) -> dict[State, StateSpec]:
    """A TOTAL spec map, built by iterating `State` rather than by listing what a caller
    remembered.

    `machine.build_specs` is the usual way and it does not fit: it assembles every state from a
    worker/judge PAIR through `specs.fuse`, and every state here declines that construction —
    seven are one park and the eighth is one measurement. Iterating the enum is the property that
    matters, and it is kept."""
    return {
        state: StateSpec(
            state=state,
            run=_run_for(subject, state),
            canonical=state is State.REVIEW,
        )
        for state in State
    }


__all__ = [
    "CALLERS_TOOL",
    "DESTINATIONS",
    "ROUTES",
    "VERDICTS",
    "VERIFY_TOOL",
    "AnswerRefused",
    "State",
    "Verdict",
    "build_prose_specs",
    "decode",
    "park_name",
    "park_run",
    "read_run",
    "transition",
    "verdict_for_verify",
    "verify_run",
]
