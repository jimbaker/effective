"""Verdicts computed from a suite run — the states where the referee is mechanical, not a model.

Three of the machine's states can reach a verdict with a pure `match` over a `CommandRun`, and that
is the single biggest cost lever in the design: most transitions cost no model call at all. It is
also the design's first agreement with the incumbent — *the model is not the referee* — reached
independently by both.

**Each function is total over its state's fibre**, so `ty` reddens a verdict enum that grows an arm
nothing computes. That matters more here than in `transition.py`: a new arm the judge can return
but nothing mechanical produces is exactly a cell that will be reached by a model and never by a
measurement.

**FINALIZE is the instructive one, and its impossible cell is NAMED rather than defaulted.** Its
table is a genuine product of a mechanical axis (did the suite stay green?) and a judged one (is
the structure paid off?) — and when the suite is red the structural judgment is *not reached at
all*, because a refactor that changed behaviour is a fact about the code and no opinion about its
shape is worth having yet. That is `unfold`'s refusal of a node that descends at its final level,
in another domain: not a default, a refusal.
"""

from typing import Any

from effective.api import Effect
from effective.coding.states import DraftVerdict, FinalizeVerdict, State, TestVerdict
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Evidence, Judge


class StructuralJudgementRequired(RuntimeError):
    """FINALIZE's suite is green, so the structural question is live and nobody answered it.

    Raised rather than defaulted, because both defaults are wrong in a way that hides: assuming
    "tidy" ships unpaid debt, and assuming "debt remains" loops FINALIZE forever. The caller has to
    obtain the judgment — that is the one place FINALIZE needs a model."""


def verdict_for_test(run: CommandRun, target: str) -> TestVerdict:
    """TEST wrote a test, so **RED is the pass condition** — a green suite proves nothing about
    the test just written, which is why `GREEN_ALREADY` re-enters TEST rather than advancing.

    Order matters: a collection error is checked first because it is a fact about the environment
    and says nothing about the test."""
    if run.collection_error is not None:
        return TestVerdict.BROKEN_ENV
    if run.failed(target):
        return TestVerdict.RED
    if run.failures:
        return TestVerdict.RED_ELSEWHERE
    return TestVerdict.GREEN_ALREADY


def verdict_for_draft(run: CommandRun, target: str) -> DraftVerdict:
    """DRAFT is making the target test pass without breaking anything else.

    A collection error reads as `STILL_RED`, not as its own arm: the draft does not import, so the
    target is not passing, and `STILL_RED` is the answer that keeps a state whose job is to edit
    editing. It would be wrong in TEST — there the environment is the finding — and the two states
    differing here is what per-state verdicts are for."""
    if run.collection_error is not None:
        return DraftVerdict.STILL_RED
    if run.green:
        return DraftVerdict.GREEN
    if run.failed(target):
        return DraftVerdict.STILL_RED
    if run.failures:
        return DraftVerdict.REGRESSED
    # Not green, nothing named as failing: the command failed for a reason the report does not
    # itemize (exit 5, a crash). Keep editing rather than claim a regression we cannot point at.
    return DraftVerdict.STILL_RED


def verdict_for_finalize(run: CommandRun, *, debt_remains: bool | None = None) -> FinalizeVerdict:
    """FINALIZE refactors, so the suite must stay green AND the structure must be paid off.

    `debt_remains` is the judged axis and is **not consulted** when the suite is red — see the
    module docstring. Passing it anyway is harmless; the point is that it is never *required*
    for the mechanical answer."""
    if not run.green:
        return FinalizeVerdict.BROKE_IT
    if debt_remains is None:
        raise StructuralJudgementRequired(
            "the suite is green, so FINALIZE's structural question is live and needs an answer"
        )
    return (
        FinalizeVerdict.STILL_GREEN_DEBT_REMAINS
        if debt_remains
        else FinalizeVerdict.STILL_GREEN_TIDY
    )


def mechanical_judges(
    target: str, *, debt_remains: bool | None = None
) -> dict[State, Judge[State, Any]]:
    """The three judges that need no model, wired to the states that own them.

    `target` is the test whose failure the walk is chasing; `debt_remains` is FINALIZE's one judged
    input, which it refuses to guess (`StructuralJudgementRequired`) — so a caller that has no
    structural answer passes nothing and gets the refusal, which is the designed behaviour rather
    than an omission.

    Returns a partial map on purpose: it answers "which judges are mechanical", and the states not
    named here are exactly the ones needing a model. `build_specs` is what makes the final map
    total.

    **`_ctx` is the declaration, not an apology.** A `Judge` takes the state's `Ctx` so that every
    judgment in the machine knows where it is; these three ignore it, and the underscore says so
    at the signature — this verdict is a pure function of what was measured, and could be replayed
    from the `CommandRun` alone. A judge that DOES read `ctx` is visibly a different animal."""

    def judge_test(_ctx: Ctx[State], evidence: Evidence) -> Effect[TestVerdict]:
        return verdict_for_test(_measured(evidence, State.TEST), target)
        yield  # pragma: no cover  -- a `Judge` is a generator

    def judge_draft(_ctx: Ctx[State], evidence: Evidence) -> Effect[DraftVerdict]:
        return verdict_for_draft(_measured(evidence, State.DRAFT), target)
        yield  # pragma: no cover

    def judge_finalize(_ctx: Ctx[State], evidence: Evidence) -> Effect[FinalizeVerdict]:
        return verdict_for_finalize(_measured(evidence, State.FINALIZE), debt_remains=debt_remains)
        yield  # pragma: no cover

    return {
        State.TEST: judge_test,
        State.DRAFT: judge_draft,
        State.FINALIZE: judge_finalize,
    }


def _measured(evidence: Evidence, state: State) -> CommandRun:
    """The measurement a mechanical judge is entitled to, or a refusal naming the state.

    `Evidence.measured` is optional because EXPLORE, REPL and REVIEW have nothing mechanical to
    read.
    A mechanical judge reaching a `None` is a WIRING error — a state declared mechanical whose
    worker never ran the predicate — and it says so, rather than failing later as an
    `AttributeError` on `None.green` two frames away."""
    if evidence.measured is None:
        raise ValueError(
            f"{state.value!r} has a mechanical judge but its worker returned no measurement — "
            f"a worker for a mechanical state must run the predicate and carry the `CommandRun` "
            f"forward in `Evidence.measured`"
        )
    return evidence.measured
