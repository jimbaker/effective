"""Every cell WALKED by the machine, and the invariants that must hold on any walk at all.

**Why this exists, measured rather than assumed.** `test_coding_transition.py` covers all 25 cells
of the dependent sum by *calling* `transition`, which proves things about a pure function and
nothing about the composition. *Probing a pure function proves things about the pure function*: a
livelock can be invisible to a correct probe of `advance` because the damage happens two call
frames away.

So this file drives the machine once per cell, and then drives it over adversarial walks that no
hand-written fixture would think to write.

**Roles.** The per-cell sweep is `spine`: it exercises the trampoline's dispatch on
every arm. The invariant walks are `adversarial`: they assert properties that must hold for EVERY
walk, so a defect only reachable on an unusual path has somewhere to show up.

The key-injectivity invariant is the one that most needs this. A single two-visit backedge pins
it once; here it holds across walks of a dozen visits with states revisited many times,
which is where a collision would actually be produced.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from effective.api import Effect, ask_llm
from effective.coding.states import (
    VERDICTS,
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
from effective.coding.transition import Advance, Finish, Park, transition
from effective.handlers.recording import RecordingHandler
from effective.keys import Run, Segment, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Exhausted
from effective.machine.spec import Ctx, Evidence, StateSpec
from effective.machine.specs import fuse
from effective.machine.trampoline import (
    Session,
    appending_states,
    canonical_violations,
    run_machine,
)
from effective.ops import LedgerRow

GREEN = CommandRun(exit_code=0)


def answers() -> dict[str, object]:
    return {"work": "ok", "tool:run_suite": GREEN}


DEFAULTS: dict[State, Verdict] = {
    State.PLAN: PlanVerdict.APPROVED,
    State.EXPLORE: ExploreVerdict.READY_FOR_REPL,
    State.REPL: ReplVerdict.HYPOTHESIS_SUPPORTED,
    State.TEST: TestVerdict.RED,
    State.DRAFT: DraftVerdict.GREEN,
    State.FINALIZE: FinalizeVerdict.STILL_GREEN_TIDY,
    State.REVIEW: ReviewVerdict.APPROVED,
}
"""One advancing arm per state — what a state says when the sweep is not testing it. Chosen so the
machine keeps moving, since a default that looped would make every walk about the default."""

CELLS: list[tuple[State, Verdict]] = [(s, v) for s, fibre in VERDICTS.items() for v in fibre]


def quiet_worker(ctx: Ctx) -> Effect[Evidence]:
    """Yields ONE op per visit, and that is load-bearing rather than decoration.

    An earlier version yielded nothing, which made invariant 2 (all op keys distinct) vacuous:
    with no worker ops the tape held only the postamble's handful of keys, so freezing the
    `visit:` counter at zero produced no collision and the whole sweep stayed green under that
    mutation. A walk that revisits a state has to MINT something at each visit for injectivity to
    be a claim about anything."""
    said = yield from ask_llm("work", f"{ctx.state.value}:{ctx.visit}", str)
    return Evidence(summary=said)


def appending_worker(state: State) -> Any:
    """A worker that reaches the canonical record from inside its state — the thing
    `StateSpec.canonical` declares and `canonical_violations` checks against the tape."""

    def run(ctx: Ctx) -> Effect[Evidence]:
        from effective.api import append_ledger

        yield from append_ledger(
            LedgerRow(
                # lint: terminal-hole — `ctx.visit` is an `int`, so it is already an atom
                event_id=compose_key(t"work:{Segment(state.value)},{ctx.visit}"),
                kind="work-committed",
            )
        )
        return Evidence(summary="appended")

    return run


def scripted(seq: Sequence[Verdict]) -> Any:
    """A judge reading a script, falling back to its state's default once the script runs out."""
    remaining = list(seq)

    def judge(_ctx: Ctx, _evidence: Evidence) -> Effect[Verdict]:
        return remaining.pop(0)
        yield  # pragma: no cover

    return judge


def specs_for(
    scripts: Mapping[State, Sequence[Verdict]],
    *,
    canonical: frozenset[State] = frozenset({State.DRAFT, State.FINALIZE}),
    workers: Mapping[State, Any] | None = None,
) -> Mapping[State, StateSpec]:
    workers = workers or {}
    return {
        state: StateSpec(
            state=state,
            run=fuse(
                workers.get(state, quiet_worker),
                scripted(list(scripts.get(state, ())) + [DEFAULTS[state]] * 40),
            ),
            canonical=state in canonical,
        )
        for state in State
    }


def drive(
    scripts: Mapping[State, Sequence[Verdict]],
    *,
    start: State = State.PLAN,
    budget: int = 12,
    canonical: frozenset[State] = frozenset({State.DRAFT, State.FINALIZE}),
    workers: Mapping[State, Any] | None = None,
) -> tuple[Session, RecordingHandler, Mapping[State, StateSpec]]:
    handler = RecordingHandler(responses=answers())
    specs = specs_for(scripts, canonical=canonical, workers=workers)
    out = handler.run(
        lambda: run_machine(Run("paths"), "goal", specs, transition, start=start, budget=budget)
    )
    assert isinstance(out, Session)
    return out, handler, specs


# --- every cell, walked by the machine rather than only by `transition` --------------------


@pytest.mark.parametrize(
    ("state", "verdict"),
    CELLS,
    ids=[f"{s.value}-{v.name}" for s, v in CELLS],
)
def test_every_cell_is_walked_and_lands_where_the_table_says(state: State, verdict: Verdict):
    """Drive the machine INTO the cell and check where it actually went.

    `transition` being right about an edge and the trampoline taking that edge are two different
    claims; this is the second one, for all 25 cells. Fourteen of them had never been walked."""
    session, _handler, _specs = drive({state: [verdict]}, start=state, budget=2)
    first = session.turns[0]
    assert first.state is state
    assert first.verdict == verdict

    expected = transition(state, verdict)
    assert first.outcome == expected
    match expected:
        case Advance(to):
            assert session.path[1] is to, "the machine did not go where the table sent it"
        case Finish():
            assert session.stopped == Finish()
            assert len(session.turns) == 1, "a finished run kept running"
        case Park():
            # `PLAN.REJECTED` is the one cell that parks WITHOUT the budget running out — a human
            # said no, which is not the same event as exhaustion and must not be routed to
            # `Finish`. This arm existed in the design and a comment here first claimed no cell
            # could reach it; the sweep is what said otherwise.
            assert session.stopped == expected
            assert len(session.turns) == 1, "a parked run kept running"


def test_the_sweep_covers_the_whole_dependent_sum():
    """Anti-vacuity for the parametrization itself: if `CELLS` were built wrong the sweep above
    could pass while covering a fraction of the domain."""
    assert len(CELLS) == 25
    assert {s for s, _ in CELLS} == set(State)
    for state, fibre in VERDICTS.items():
        assert {v for s, v in CELLS if s is state} == set(fibre)


# --- invariants that must hold on ANY walk -------------------------------------------------


WALKS: dict[str, dict[State, Sequence[Verdict]]] = {
    "self-loops-everywhere": {
        State.PLAN: [PlanVerdict.REVISE, PlanVerdict.REVISE],
        State.TEST: [TestVerdict.GREEN_ALREADY, TestVerdict.GREEN_ALREADY],
        State.DRAFT: [DraftVerdict.STILL_RED, DraftVerdict.STILL_RED],
        State.FINALIZE: [FinalizeVerdict.STILL_GREEN_DEBT_REMAINS],
        State.EXPLORE: [ExploreVerdict.MORE_EXPLORATION],
    },
    "review-sends-it-all-the-way-back": {
        State.REVIEW: [
            ReviewVerdict.MISSING_UNDERSTANDING,
            ReviewVerdict.NEED_RUNTIME_EVIDENCE,
            ReviewVerdict.STRUCTURAL_DEBT,
        ]
    },
    "regression-then-recovery": {
        State.DRAFT: [DraftVerdict.REGRESSED, DraftVerdict.STILL_RED],
        State.TEST: [TestVerdict.RED_ELSEWHERE, TestVerdict.BROKEN_ENV],
        State.REPL: [ReplVerdict.NEED_MORE_EVIDENCE, ReplVerdict.HYPOTHESIS_REJECTED],
    },
    "finalize-breaks-it": {
        State.FINALIZE: [FinalizeVerdict.BROKE_IT, FinalizeVerdict.BROKE_IT],
        State.DRAFT: [DraftVerdict.GREEN, DraftVerdict.GREEN],
    },
    "review-defects-in-turn": {
        State.REVIEW: [
            ReviewVerdict.IMPLEMENTATION_DEFECT,
            ReviewVerdict.TEST_GAP,
            ReviewVerdict.INVALID_ASSUMPTION,
        ]
    },
    "straight-through": {},
}


def walk_cases() -> Iterator[tuple[str, State, int]]:
    for name in WALKS:
        for start in (State.PLAN, State.TEST, State.REVIEW):
            yield name, start, 12


@pytest.mark.parametrize(
    ("name", "start", "budget"),
    list(walk_cases()),
    ids=[f"{n}-from-{s.value}" for n, s, _ in walk_cases()],
)
def test_every_walk_holds_the_machine_invariants(name: str, start: State, budget: int):
    """Six adversarial scripts x three start states. Whatever path results, all of this holds.

    These are the properties a defect on an unusual path would break, and none of them is
    reachable by reading the transition table — they are facts about a run."""
    session, handler, _specs = drive(WALKS[name], start=start, budget=budget)

    # 1. It terminates, within the budget the interpreter holds.
    assert 1 <= len(session.turns) <= budget + 1
    assert isinstance(session.stopped, Finish | Park)

    # 2. THE TAPE STAYS INJECTIVE. The visit counter was pinned on one two-visit backedge; these
    #    walks revisit states many times, which is where a collision would actually be produced.
    keys = [entry.key.stored() for entry in handler.trace]
    assert len(keys) == len(set(keys)), f"{name}: colliding op keys"

    # 3. The canonical record gets exactly the postamble's two rows — not one per visit. An
    #    append inside the loop is the incumbent's defect, and a long walk is where it shows.
    assert [row.kind for row in handler.ledger][:1] == ["machine-committed"]
    assert len(handler.ledger) == 2

    # 4. The path is a REAL walk: every step is the edge `transition` gives for that verdict.
    for previous, following in zip(session.turns, session.turns[1:], strict=False):
        assert transition(previous.state, previous.verdict) == previous.outcome
        match previous.outcome:
            case Advance(to):
                assert following.state is to
            case _:  # pragma: no cover -- a terminal outcome ends the loop, so it has no successor
                raise AssertionError("a terminal outcome had a successor turn")

    # 5. Only the LAST turn may be exhausted — the budget is not consulted early.
    exhausted = [i for i, t in enumerate(session.turns) if isinstance(t.verdict, Exhausted)]
    assert exhausted in ([], [len(session.turns) - 1])


def test_a_long_walk_really_does_revisit_states():
    """Anti-vacuity for the invariants above: if the walks terminated immediately they would hold
    trivially. At least one script must produce genuine repetition."""
    session, _handler, _specs = drive(WALKS["self-loops-everywhere"], start=State.PLAN, budget=12)
    counts = {state: session.path.count(state) for state in set(session.path)}
    assert max(counts.values()) >= 2, counts
    assert len(session.turns) >= 6


# --- the declaration, checked against the tape ---------------------------------------------


def test_a_state_that_appends_without_declaring_it_is_caught():
    """**`StateSpec.canonical` gets a consumer.** It was a documented field nothing read — the
    docstring claimed a state "declares which bookkeeper it reaches" while no code compared the
    declaration to what happened. Now the `state:` scope frame on a ledger key makes the
    comparison possible, and the mismatch is a finding rather than a silence."""
    session, handler, specs = drive(
        {},
        start=State.TEST,
        budget=4,
        canonical=frozenset(),
        workers={State.TEST: appending_worker(State.TEST)},
    )
    keys = [entry.key.stored() for entry in handler.trace]
    assert State.TEST in appending_states(State, keys), (
        "the worker's append was not visible on the tape"
    )
    violations = canonical_violations(State, specs, keys)
    assert State.TEST in violations
    assert session.commitment is not None  # the run still completed; this is an audit, not a gate


def test_a_canonical_violation_names_the_site_and_says_what_is_wrong():
    """A canonical violation carries an ADDRESS on the tape and a sentence naming the skipped
    declaration.

    A plain test rather than an `xfail` pin: the contract holds, and a strict pin that passes
    fails the run. Both call sites check `State.TEST in violations` or `== {}`, a claim about the
    mapping's keys; `ty` covers the type and nothing else covers the content. Replacing every
    `where` and `why` with the constant `"MUTANT"` leaves every other test green.

    Asserted as the property rather than an exact string, so the wording can change."""
    _session, handler, specs = drive(
        {},
        start=State.TEST,
        budget=4,
        canonical=frozenset(),
        workers={State.TEST: appending_worker(State.TEST)},
    )
    keys = [entry.key.stored() for entry in handler.trace]
    found = canonical_violations(State, specs, keys)[State.TEST]

    assert found, "anti-vacuity: the state really did append without declaring it"
    assert all(violation.where in keys for violation in found), (
        f"a violation must name a site the tape carries: {[v.where for v in found]}"
    )
    assert all("canonical" in violation.why for violation in found), (
        f"a violation must say which declaration was skipped: {[v.why for v in found]}"
    )


def test_a_declared_canonical_state_appending_is_no_violation():
    _session, handler, specs = drive(
        {},
        start=State.DRAFT,
        budget=4,
        canonical=frozenset({State.DRAFT}),
        workers={State.DRAFT: appending_worker(State.DRAFT)},
    )
    keys = [entry.key.stored() for entry in handler.trace]
    assert State.DRAFT in appending_states(State, keys)
    assert canonical_violations(State, specs, keys) == {}


def test_the_postambles_own_rows_belong_to_no_state():
    """The postamble is minted outside every state scope, so it appears under no state — correct,
    because those rows belong to the RUN. If they were attributed to whichever state happened to
    run last, every walk would look like a canonical violation."""
    _session, handler, specs = drive({}, start=State.REVIEW, budget=2)
    keys = [entry.key.stored() for entry in handler.trace]
    assert appending_states(State, keys) == {}
    assert canonical_violations(State, specs, keys) == {}
