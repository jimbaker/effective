"""The de-essaying machine's edges: pure, total, and never reading a register.

One router per state, each closed with `assert_never`, exactly as `coding.transition` is — so a
dropped or added arm is a `ty` error naming the arm rather than a walk that dies on the one path
that reaches it.

**`transition(state, verdict)` has a finite domain because guards are precomputed into the
verdict.** "Did the rubric grow?" is not a flag this function inspects; it is `REVIEW`'s own
`RUBRIC_GREW` arm. A textbook EFSM with guards over registers has an unbounded domain and loses
totality checking outright, which is why a request for `transition(state, verdict, acc)` is
refused rather than weighed.

**Two cells reach `Finish`, and that differs from the coding machine on purpose.** There the rule
is one cell, because a second route to `Finish` is how a machine ships unreviewed work. Here
`(SELECT, EXHAUSTED_WORKLIST)` also finishes — and it cannot ship anything, because `SELECT` is
the start state and **no edge returns to it**, so that cell is reachable only on the first visit,
before any region was read let alone edited. `tests/test_prose_transition.py` pins both halves:
the two cells, and the absence of any edge into `SELECT`.
"""

from typing import assert_never

from effective.machine.outcomes import (
    Advance,
    Exhausted,
    Finish,
    Outcome,
    Park,
    ParkReason,
    VerdictOutOfDomain,
)
from effective.prose.states import (
    ClassifyVerdict,
    DraftVerdict,
    ReadVerdict,
    RelocateVerdict,
    ReviewVerdict,
    RubricVerdict,
    SelectVerdict,
    State,
    Verdict,
    VerifyVerdict,
)


def route_select(verdict: SelectVerdict) -> Outcome:
    match verdict:
        case SelectVerdict.CHOSE:
            return Advance(State.READ)
        case SelectVerdict.EXHAUSTED_WORKLIST:
            return Finish()  # nothing was read, so nothing shipped — see the module docstring
        case unreachable:
            assert_never(unreachable)


def route_read(verdict: ReadVerdict) -> Outcome:
    match verdict:
        case ReadVerdict.INVENTORIED:
            return Advance(State.CLASSIFY)
        case unreachable:
            assert_never(unreachable)


def route_classify(verdict: ClassifyVerdict) -> Outcome:
    match verdict:
        case ClassifyVerdict.CLASSIFIED:
            return Advance(State.RELOCATE)
        case ClassifyVerdict.NO_DESTINATION:
            return Advance(State.RUBRIC)  # a rubric gap is not local — amend, then re-classify
        case unreachable:
            assert_never(unreachable)


def route_relocate(verdict: RelocateVerdict) -> Outcome:
    match verdict:
        case RelocateVerdict.REHOMED:
            return Advance(State.DRAFT)
        case RelocateVerdict.DESTINATION_MISSING:
            # A self-edge, not a park: the prose has a destination and simply has not arrived
            # there yet. The budget bounds the retries, which is what a budget is for.
            return Advance(State.RELOCATE)
        case unreachable:
            assert_never(unreachable)


def route_draft(verdict: DraftVerdict) -> Outcome:
    match verdict:
        case DraftVerdict.REWRITTEN:
            return Advance(State.VERIFY)
        case unreachable:
            assert_never(unreachable)


def route_verify(verdict: VerifyVerdict) -> Outcome:
    match verdict:
        case VerifyVerdict.CLEAN:
            return Advance(State.REVIEW)
        case VerifyVerdict.MECHANICAL:
            return Advance(State.DRAFT)  # a line too long, a stray escape — reword
        case VerifyVerdict.NOT_PROSE_ONLY:
            return Advance(State.DRAFT)  # the skeleton moved: undo the code change, keep the prose
        case VerifyVerdict.BROKEN_ENV:
            # The predicate could not RUN, so it has said nothing about the prose. Re-drafting
            # would be answering an infrastructure failure with a rewording.
            return Park(State.VERIFY, ParkReason.BROKEN_ENV)
        case unreachable:
            assert_never(unreachable)


def route_review(verdict: ReviewVerdict) -> Outcome:
    match verdict:
        case ReviewVerdict.APPROVED:
            return Finish()
        case ReviewVerdict.REVISE:
            return Advance(State.DRAFT)  # shallow: the prose is wrong, the destinations are not
        case ReviewVerdict.RUBRIC_GREW:
            # THE DEEP BACK EDGE, and landing it on CLASSIFY rather than DRAFT is the whole
            # nuance. A new destination criterion changes where things GO, so re-wording is
            # insufficient — every unit has to be classified again under the amended rubric.
            return Advance(State.RUBRIC)
        case ReviewVerdict.LEAVE_IT_ALONE:
            return Park(State.REVIEW, ParkReason.REJECTED)
        case unreachable:
            assert_never(unreachable)


def route_rubric(verdict: RubricVerdict) -> Outcome:
    match verdict:
        case RubricVerdict.AMENDED:
            return Advance(State.CLASSIFY)
        case RubricVerdict.REFUSED:
            # No destination AND no new rule. Neither loop makes progress, so it is a human's.
            return Park(State.RUBRIC, ParkReason.REJECTED)
        case unreachable:
            assert_never(unreachable)


def _owned[V: Verdict](owner: State, state: State, verdict: V) -> V:
    """The dependent sum's projection: a verdict's type says which state may produce it, so one
    arriving under any OTHER state is one of the excluded cells.

    Generic, and it has to be: a plain `Verdict` return widens every call back to the union and
    each router then refuses its own argument. The type variable is what carries the narrowing
    through the cross-check."""
    if state is not owner:
        raise VerdictOutOfDomain(
            f"{verdict!r} belongs to {owner.value} but arrived in {state.value} — refused rather "
            f"than routed"
        )
    return verdict


def transition(state: State, verdict: Verdict | Exhausted) -> Outcome:
    """The machine's control flow as one pure total function.

    Dispatch is on the VERDICT's type, not on the state, for the reason `coding.transition` gives:
    a verdict's type already says which state produced it, so routing on the state would ask a
    `str`-shaped question the types answer better. `_owned` still cross-checks.

    `Exhausted` is an arm of this match rather than a guard above it — legal in every state, which
    is exactly why it is not a member of any state's enum, and an `isinstance` guard here would be
    dispatch over a closed union that `--totality src` correctly refuses."""
    match verdict:
        case Exhausted():
            return Park(verdict.state, ParkReason.EXHAUSTED)
        case SelectVerdict():
            return route_select(_owned(State.SELECT, state, verdict))
        case ReadVerdict():
            return route_read(_owned(State.READ, state, verdict))
        case ClassifyVerdict():
            return route_classify(_owned(State.CLASSIFY, state, verdict))
        case RelocateVerdict():
            return route_relocate(_owned(State.RELOCATE, state, verdict))
        case DraftVerdict():
            return route_draft(_owned(State.DRAFT, state, verdict))
        case VerifyVerdict():
            return route_verify(_owned(State.VERIFY, state, verdict))
        case ReviewVerdict():
            return route_review(_owned(State.REVIEW, state, verdict))
        case RubricVerdict():
            return route_rubric(_owned(State.RUBRIC, state, verdict))
        case unreachable:
            assert_never(unreachable)


ROUTES = {
    State.SELECT: route_select,
    State.READ: route_read,
    State.CLASSIFY: route_classify,
    State.RELOCATE: route_relocate,
    State.DRAFT: route_draft,
    State.VERIFY: route_verify,
    State.REVIEW: route_review,
    State.RUBRIC: route_rubric,
}
"""The routers as data, pinned against `VERDICTS` by the graph test: a state in one and not the
other is a defect."""

__all__ = ["ROUTES", "transition"]
