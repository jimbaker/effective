"""The transition: a PURE, TOTAL function from a state and a typed verdict to what happens next.

One router per state, each closed with `assert_never`. That closure is the gate — `ty` reddens a
dropped or added arm and names it (measured: dropping `route_review`'s `TEST_GAP` arm reports
`Literal[ReviewVerdict.TEST_GAP]`; dropping this module's `DraftVerdict` arm reports
`DraftVerdict`).

Be precise about the division of labour, because it is easy to get backwards and one half of it
was learned the hard way here:

- **`--totality src` does not see the enum matches.** Its census emits zero rows for a `match`
  over a `StrEnum`, so a green lint says nothing about whether these routers are total. The
  evidence for that is `uvx ty check` reddening under arm deletion.
- **It does, however, see an `isinstance` guard** — and it caught one in this very function. The
  first draft handled `Exhausted` with `if isinstance(verdict, Exhausted)` above the match, and
  the gate correctly called it *"dispatch over a closed union, not closed by assert_never"*. The
  fix was to make it an arm rather than to write an escape, which is also the better code.
- **Neither reaches a graph property.** That `Finish` has exactly one producer, or that no state
  is unleavable, is a fact about a relation's transitive closure, not about any file.
  `tests/test_coding_transition.py` owns that half.

**Why the routers take a concrete enum rather than `Verdict`.** `route_test(v: TestVerdict)` makes
the 150 excluded cells of the naive product unspellable at the call site. `transition` is the one
place a runtime check is needed, because it is the one place a verdict arrives whose static type
is the whole union — that is the dependent sum's projection, and `VerdictOutOfDomain` is what a
mistyped judge hits.

**Why the incumbent's defect cannot recur here.** `tests/_coding.py:155`'s `advance` took
`outcome: str` and tested it nominally (`outcome != "pass"`), so every unrecognised string took
the back edge into the one canonically-appending phase: a case-variant typo committed 99 ledger
rows in 400 ops and never terminated. Here a bad verdict is not a back edge, it is a refusal.
"""

from typing import assert_never

from effective.coding.states import (
    DraftVerdict,
    ExploreVerdict,
    FinalizeVerdict,
    PlanVerdict,
    ReplVerdict,
    ReviewVerdict,
    State,
    TestVerdict,
    Verdict,
)
from effective.machine.outcomes import (
    Advance,
    Exhausted,
    Finish,
    Outcome,
    Park,
    ParkReason,
    VerdictOutOfDomain,
)


def route_plan(verdict: PlanVerdict) -> Outcome:
    match verdict:
        case PlanVerdict.APPROVED:
            return Advance(State.EXPLORE)
        case PlanVerdict.REVISE:
            return Advance(State.PLAN)
        case PlanVerdict.REJECTED:
            # NOT `Finish`: a rejected plan is not finished work, and routing it to the same
            # outcome as an approved review would make "we shipped it" and "a human said no"
            # indistinguishable on the tape.
            return Park(State.PLAN, ParkReason.REJECTED)
        case unreachable:
            assert_never(unreachable)


def route_explore(verdict: ExploreVerdict) -> Outcome:
    match verdict:
        case ExploreVerdict.MORE_EXPLORATION:
            return Advance(State.EXPLORE)
        case ExploreVerdict.READY_FOR_REPL:
            return Advance(State.REPL)
        case unreachable:
            assert_never(unreachable)


def route_repl(verdict: ReplVerdict) -> Outcome:
    match verdict:
        case ReplVerdict.NEED_MORE_EVIDENCE:
            return Advance(State.EXPLORE)
        case ReplVerdict.HYPOTHESIS_REJECTED:
            return Advance(State.EXPLORE)
        case ReplVerdict.HYPOTHESIS_SUPPORTED:
            return Advance(State.TEST)
        case unreachable:
            assert_never(unreachable)


def route_test(verdict: TestVerdict) -> Outcome:
    match verdict:
        case TestVerdict.RED:
            return Advance(State.DRAFT)  # the test discriminates — go make it pass
        case TestVerdict.GREEN_ALREADY:
            return Advance(State.TEST)  # it proves nothing; write a better one
        case TestVerdict.RED_ELSEWHERE:
            return Advance(State.REPL)  # a regression, not this test's verdict
        case TestVerdict.BROKEN_ENV:
            return Advance(State.REPL)  # not a fact about the test at all
        case unreachable:
            assert_never(unreachable)


def route_draft(verdict: DraftVerdict) -> Outcome:
    match verdict:
        case DraftVerdict.GREEN:
            return Advance(State.FINALIZE)
        case DraftVerdict.STILL_RED:
            return Advance(State.DRAFT)
        case DraftVerdict.REGRESSED:
            return Advance(State.REPL)
        case unreachable:
            assert_never(unreachable)


def route_finalize(verdict: FinalizeVerdict) -> Outcome:
    match verdict:
        case FinalizeVerdict.STILL_GREEN_TIDY:
            return Advance(State.REVIEW)
        case FinalizeVerdict.STILL_GREEN_DEBT_REMAINS:
            return Advance(State.FINALIZE)
        case FinalizeVerdict.BROKE_IT:
            return Advance(State.DRAFT)
        case unreachable:
            assert_never(unreachable)


def route_review(verdict: ReviewVerdict) -> Outcome:
    match verdict:
        case ReviewVerdict.APPROVED:
            return Finish()
        case ReviewVerdict.MISSING_UNDERSTANDING:
            return Advance(State.EXPLORE)
        case ReviewVerdict.INVALID_ASSUMPTION:
            return Advance(State.EXPLORE)
        case ReviewVerdict.NEED_RUNTIME_EVIDENCE:
            return Advance(State.REPL)
        case ReviewVerdict.TEST_GAP:
            return Advance(State.TEST)
        case ReviewVerdict.IMPLEMENTATION_DEFECT:
            return Advance(State.DRAFT)
        case ReviewVerdict.STRUCTURAL_DEBT:
            return Advance(State.FINALIZE)
        case unreachable:
            assert_never(unreachable)


def _owned[V: Verdict](owner: State, state: State, verdict: V) -> V:
    """The dependent sum's projection: this verdict's type says which state may produce it, so
    a verdict arriving under any OTHER state is one of the excluded cells."""
    if state is not owner:
        raise VerdictOutOfDomain(
            f"{type(verdict).__name__}.{verdict.name} belongs to {owner.value!r}, but the "
            f"machine was in {state.value!r} — one of the cells the dependent sum excludes, "
            f"refused rather than routed"
        )
    return verdict


def transition(state: State, verdict: Verdict | Exhausted) -> Outcome:
    """The whole machine's control flow, as one pure total function.

    **Dispatch is on the VERDICT's type, not on the state**, and that is structural rather than
    nominal: a verdict's type already says which state produces it, so routing on the state and
    then checking the verdict would be asking a `str`-shaped question the types answer better.
    The state is still checked — `_owned` refuses a verdict that arrived under the wrong one —
    but it is a cross-check, not the dispatcher.

    `Exhausted` is one arm of the same match rather than a guard above it. It is legal in every
    state, which is exactly why it is not an arm of any state's enum; putting it here keeps the
    exhaustion path total without adding a row to seven tables, and keeps the whole function one
    `match` closed by `assert_never` — an `isinstance` guard above the match is a dispatch over a
    closed union wearing a boundary check's clothes, and `--totality src` says so."""
    match verdict:
        case Exhausted():
            return Park(verdict.state, ParkReason.EXHAUSTED)
        case PlanVerdict():
            return route_plan(_owned(State.PLAN, state, verdict))
        case ExploreVerdict():
            return route_explore(_owned(State.EXPLORE, state, verdict))
        case ReplVerdict():
            return route_repl(_owned(State.REPL, state, verdict))
        case TestVerdict():
            return route_test(_owned(State.TEST, state, verdict))
        case DraftVerdict():
            return route_draft(_owned(State.DRAFT, state, verdict))
        case FinalizeVerdict():
            return route_finalize(_owned(State.FINALIZE, state, verdict))
        case ReviewVerdict():
            return route_review(_owned(State.REVIEW, state, verdict))
        case unreachable:
            assert_never(unreachable)


ROUTES = {
    State.PLAN: route_plan,
    State.EXPLORE: route_explore,
    State.REPL: route_repl,
    State.TEST: route_test,
    State.DRAFT: route_draft,
    State.FINALIZE: route_finalize,
    State.REVIEW: route_review,
}
"""The routers as data, so a `StateSpec` REFERENCES its router rather than re-spelling the edge.
Pinned against `VERDICTS` by the graph gate: a state in one and not the other is a defect."""
