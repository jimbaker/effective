"""The trampoline: the walk, the keys it mints, and the postamble that always runs.

**Role: journey**, modelled on `tests/test_coding_machine_example.py`: one scripted
run driven end to end, asserted over all three bookkeepers (the session, the tape, the ledger).
The whole file runs under `RecordingHandler` with canned answers: **no LLM, no I/O, no clock.**

The two properties this module exists to pin are the ones that are structural rather than
behavioural, and a behavioural test alone would miss both:

- **the postamble runs on every exit path** — including the paths a happy fixture never takes;
- **a backedge stays injective** — `TEST -> DRAFT -> TEST` must mint distinct keys, which it does
  only because of the `d:` counter.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic_core import to_jsonable_python

from effective.api import Effect, ask_llm
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
from effective.coding.transition import Finish, Park, ParkReason, transition
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Run
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Exhausted
from effective.machine.spec import Ctx, Evidence, Judge, StateSpec
from effective.machine.specs import fuse
from effective.machine.trampoline import Session, Turn, run_machine, stop_record
from effective.ops import CHAIN_GENERATION

RUN = Run("cm-2026")
GOAL = "convert the two-arm decisions"


def worker(ctx: Ctx) -> Effect[Evidence]:
    """One op per visit, named plainly — the scope frames carry the placement, not the name."""
    said = yield from ask_llm("work", f"{ctx.state.value}: {ctx.goal}", str)
    return Evidence(summary=said)


def judge_for[V: Verdict](script: Sequence[V]) -> Judge[State, V]:
    """A judge that reads its answers off a script, so a walk is a fixture rather than a model.

    `Sequence` at the boundary rather than `list`: `list` is invariant, so a `list[PlanVerdict]`
    is not a `list[Verdict]` and every caller would have to widen by hand. `ty` said so."""
    remaining = list(script)

    def judge(_ctx: Ctx[State], _evidence: Evidence) -> Effect[V]:
        return remaining.pop(0)
        yield  # pragma: no cover  -- makes this a generator; the return above always fires

    return judge


def specs(**scripts: Sequence[Verdict]) -> Mapping[State, StateSpec]:
    """A total spec map, built by iterating `State` — which is what makes it total, and is not
    what the type says.

    `Mapping[State, StateSpec]` admits a partial map perfectly happily; this comprehension is the
    thing keeping the promise, and `run_machine` refuses at entry for the cases where nobody wrote
    a comprehension. `DONE` being an outcome rather than a state is why the map CAN be total."""
    return {
        state: StateSpec(
            state=state,
            run=fuse(worker, judge_for(scripts.get(state.value, ()))),
            canonical=state in (State.DRAFT, State.FINALIZE),
        )
        for state in State
    }


CANONICAL_KINDS = ("machine-committed", "machine-finished", "machine-parked")

GREEN = CommandRun(exit_code=0)
"""The predicate's canned answer. The postamble runs it on EVERY exit path, so every fixture here
needs one — which is the unconditional tail showing up in the test surface rather than only in the
source, and is worth noticing rather than working around."""


def answers(**extra: object) -> dict[str, object]:
    """Canned responses for one machine run: the worker's op plus the postamble's predicate."""
    return {"work": "did it", "tool:run_suite": GREEN, **extra}


def drive(handler: RecordingHandler, **scripts: Sequence[Verdict]) -> Session:
    out = handler.run(
        lambda: run_machine(RUN, GOAL, specs(**scripts), transition, start=State.PLAN, budget=12)
    )
    assert isinstance(out, Session)
    return out


# --- totality of the spec map ------------------------------------------------------------------


def _plan_only() -> Mapping[State, StateSpec]:
    """A map with PLAN and nothing else, whose judge APPROVES — so the walk really does advance
    into a state that is not there. A judge with no scripted verdict would fail first, on its own
    empty script, and the test would be measuring the fixture."""
    return {
        State.PLAN: StateSpec(
            state=State.PLAN, run=fuse(worker, judge_for([PlanVerdict.APPROVED]))
        )
    }


def test_a_partial_spec_map_is_REFUSED_before_the_first_op():
    """`specs` must be TOTAL over `State`, and the type cannot say so: `Mapping[S, StateSpec]`
    is partial and `ty` is clean.

    Unchecked, the walk advances PLAN -> EXPLORE and dies on `KeyError(<State.EXPLORE:
    'explore'>)`, a bare enum member with no message, with **one op on the tape and zero ledger
    rows**: work recorded, nothing on the canonical record, and a diagnostic that does not say
    what is wrong.

    So `run_machine` refuses at ENTRY, before anything is recorded, naming what is missing, the
    same shape as the run-id check below. Asserting the empty trace is the half that matters:
    a late `KeyError` raises too."""
    handler = RecordingHandler(responses=answers())
    partial = _plan_only()

    with pytest.raises(ValueError, match="must be total"):
        handler.run(
            lambda: run_machine(RUN, GOAL, partial, transition, start=State.PLAN, budget=12)
        )

    assert handler.trace == [], "the refusal arrived after work was already recorded"
    assert handler.ledger == [], "a partial map reached the canonical record"


def test_the_refusal_names_EVERY_missing_state_not_the_first_one_reached():
    """A walk finds missing states one at a time, in whatever order it happens to take — which is
    why the old `KeyError` named `explore` and said nothing about the other five. Naming the set
    is what makes the message a fix-list rather than a first-symptom."""
    handler = RecordingHandler(responses=answers())
    partial = _plan_only()

    with pytest.raises(ValueError, match="must be total") as refused:
        handler.run(
            lambda: run_machine(RUN, GOAL, partial, transition, start=State.PLAN, budget=12)
        )

    named = str(refused.value)
    for missing in State:
        if missing is not State.PLAN:
            assert missing.value in named, f"{missing.value} missing from {named!r}"


def test_a_TOTAL_map_still_runs():
    """The green arm. Without it the two reds above are satisfied by a `run_machine` that refuses
    everything."""
    session = drive(RecordingHandler(responses=answers()), **happy_scripts())
    assert session.commitment is not None


# --- the walk ---------------------------------------------------------------------------------


def happy_scripts() -> dict[str, Sequence[Verdict]]:
    """PLAN -> EXPLORE -> REPL -> TEST -> DRAFT -> FINALIZE -> REVIEW -> Finish, with the two
    backedges the machine exists for: a red-elsewhere test and a review that finds a test gap."""
    return {
        "plan": [PlanVerdict.APPROVED],
        "explore": [ExploreVerdict.READY_FOR_REPL],
        "repl": [ReplVerdict.HYPOTHESIS_SUPPORTED],
        "test": [TestVerdict.RED, TestVerdict.RED],
        "draft": [DraftVerdict.GREEN, DraftVerdict.GREEN],
        "finalize": [FinalizeVerdict.STILL_GREEN_TIDY, FinalizeVerdict.STILL_GREEN_TIDY],
        "review": [ReviewVerdict.TEST_GAP, ReviewVerdict.APPROVED],
    }


def test_the_machine_walks_the_path_the_transition_table_implies():
    session = drive(RecordingHandler(responses=answers()), **happy_scripts())
    assert session.path == (
        State.PLAN,
        State.EXPLORE,
        State.REPL,
        State.TEST,
        State.DRAFT,
        State.FINALIZE,
        State.REVIEW,
        State.TEST,
        State.DRAFT,
        State.FINALIZE,
        State.REVIEW,
    )
    assert session.stopped == Finish()


def test_the_backedge_is_taken():
    """Anti-vacuity. A machine that never went round would satisfy every other test here, and
    the backedge is the whole reason this is a machine rather than a pipeline."""
    session = drive(RecordingHandler(responses=answers()), **happy_scripts())
    assert session.path.count(State.TEST) == 2
    assert session.path.index(State.REVIEW) < session.path.index(State.TEST, 4)


# --- the keys: the property the visit counter buys ---------------------------------------------


def test_a_revisited_state_mints_DISTINCT_keys():
    """The measured hazard, pinned at the TAPE, which is the only half this test can see.

    A repeated `scoped` frame gets no occurrence suffix, so under `state:` alone the two TEST
    visits mint one key and every position-keyed reader (`graphview`, the key registry) loses the
    second. The DURABLE path does not replay the first visit's answer into the second: SQLite and
    Absurd both apply the SDK's `name#N` duplicate-step rule, so a run completes correctly even
    with a frozen counter, and this test says nothing about durable replay."""
    handler = RecordingHandler(responses=answers())
    drive(handler, **happy_scripts())
    keys = [entry.key.stored() for entry in handler.trace]
    assert len(keys) == len(set(keys)), "a re-entered state collided — the visit counter is gone"
    test_keys = [k for k in keys if "state:test" in k]
    assert len(test_keys) == 2
    assert test_keys[0] != test_keys[1]


def test_every_op_names_its_visit_and_its_state():
    handler = RecordingHandler(responses=answers())
    drive(handler, **happy_scripts())
    worked = [entry.key.stored() for entry in handler.trace if "step:work" in entry.key.stored()]
    assert worked, "the worker minted nothing"
    for key in worked:
        assert key.startswith("d:"), key
        assert ";state:" in key, key


# --- the two bookkeepers ------------------------------------------------------------------------


def test_only_the_commitment_points_reach_the_canonical_record():
    """Carried from the incumbent, but driven by the DECLARATION (`StateSpec.canonical`) rather
    than a hard-coded set — which is what `canonical` is for."""
    handler = RecordingHandler(responses=answers())
    drive(handler, **happy_scripts())
    kinds = {row.kind for row in handler.ledger}
    assert kinds <= set(CANONICAL_KINDS)
    # Two rows per run and no more: what was committed, and what the predicate said about it.
    # A per-visit append would make a hundred-turn run a hundred permanent rows.
    assert len(handler.ledger) == 2


# --- the postamble runs on EVERY exit path ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "scripts", "expected"),
    [
        ("approved", happy_scripts(), Finish()),
        (
            "rejected",
            {"plan": [PlanVerdict.REJECTED]},
            Park(State.PLAN, ParkReason.REJECTED),
        ),
    ],
)
def test_the_postamble_runs_on_every_exit_path(name, scripts, expected):
    """The incumbent's unconditional-tail property, and why it matters:
    at small-model scale the stop is *usually* exhaustion or a refusal, not success, so a
    postamble that only ran on the happy path would usually not run."""
    handler = RecordingHandler(responses=answers())
    session = drive(handler, **scripts)
    assert session.stopped == expected
    assert len(handler.ledger) == 2, f"{name}: the canonical record was not written"
    assert session.commitment is not None, f"{name}: nothing was committed"
    assert {row.kind for row in handler.ledger} >= {"machine-committed"}


def test_an_unwrapped_run_id_is_refused_before_any_op():
    """The record's ADDRESS is decided before the record's CONTENT is produced.

    A `run_id` first validated INSIDE the postamble fails two committed ops past the point of no
    return: 13 ops on the tape, one stored artifact, **zero ledger rows**. The whole walk runs
    and only the address is missing, so what survives is the disposable bookkeeper and what is
    lost is the canonical one.

    The value here is well-formed on purpose. The seam takes a `Segment`, and `compose_key`
    refuses an unwrapped hole by TYPE before it ever inspects the content, so this asserts the
    runtime fence below the annotation, which is the fence that matters for a caller who builds a
    `run_id` from data and never met it. Every caller across a task boundary is one of those:
    spawn params arrive as `str`.

    Typed `Any` so `ty` does not refuse the call the test exists to make."""
    unwrapped: Any = "cm-2026"  # a perfectly good run id, just not promoted
    handler = RecordingHandler(responses=answers())
    with pytest.raises(ValueError, match="run_id"):
        handler.run(
            lambda: run_machine(
                unwrapped, GOAL, specs(**happy_scripts()), transition, start=State.PLAN, budget=12
            )
        )
    assert handler.trace == [], "an op was performed for a run that has nowhere to be recorded"
    assert handler.ledger == [], "a partial record is worse than none"
    assert handler.artifacts == {}, "the commit happened before the address was known to be good"


@pytest.mark.parametrize(
    "bad",
    ["run 2026", "r,commit", "7run", "a;b"],
    ids=["space", "arity-separator", "leading-digit", "frame-delimiter"],
)
def test_a_malformed_run_id_costs_no_op(bad: str):
    """The other half of the run-id check, and the reason it is a TYPE plus an ORDERING.

    Two of these are refused by `Segment`'s delimiter fence at construction and two by the atom
    rule at composition, a split worth knowing, because `Segment` guarantees only that a value
    carries no separator, and well-formedness is the grammar's question. This test deliberately
    does not care which layer answers. The property is that the answer arrives before the machine
    performs an op, and it does for both layers only because the address is composed at
    entry."""
    handler = RecordingHandler(responses=answers())
    with pytest.raises(ValueError, match=re.escape(repr(bad))):
        handler.run(
            lambda: run_machine(
                Run(bad),
                GOAL,
                specs(**happy_scripts()),
                transition,
                start=State.PLAN,
                budget=12,
            )
        )
    assert handler.trace == [], "an op was performed for a run that has nowhere to be recorded"
    assert handler.ledger == [], "a partial record is worse than none"
    assert handler.artifacts == {}, "the commit happened before the address was known to be good"


def _addresses(generation: int) -> list[str]:
    """The ledger ids one happy run mints, at a given chain generation.

    Sets `CHAIN_GENERATION` INSIDE the run, set-call-reset around the delegation, because that is
    exactly what `combinators.respawn` does: the same mechanism, in the same place. A test that
    supplied the generation by its own route would be measuring its own arithmetic, which is the
    defect the parity arc found in the authority census.

    Inside is not a detail: every handler calls `ops.enter_task_run()` at the top of `run`, which
    sets `CHAIN_GENERATION` to `None`, because a task run starts unchained unless a respawn says
    otherwise. Setting it around `handler.run(...)` measures nothing — the first thing the handler
    does is throw the value away."""

    def driver() -> Effect[Session]:
        token = CHAIN_GENERATION.set(generation)
        try:
            return (
                yield from run_machine(
                    RUN, GOAL, specs(**happy_scripts()), transition, start=State.PLAN, budget=12
                )
            )
        finally:
            CHAIN_GENERATION.reset(token)

    handler = RecordingHandler(responses=answers())
    handler.run(driver)
    return [row.event_id.stored() for row in handler.ledger]


def test_generation_zero_carries_no_coordinate():
    """An unchained run is generation 0, and generation 0 must be OMITTED from the address.

    Same rule as `Key.occurrence`'s `#N` and `govern:`/`approve:`'s own generation, and the
    reason a coordinate can be added to a live wire at all. Nothing already recorded is
    orphaned; only a chained run moves.

    Asserts the PROPERTY, not the spelling. The separator between the run id and the commit row is
    the key registry's choice rather than a wire commitment; a verifier that hardcoded it would
    have to follow the system it is supposed to be independent of."""
    assert all("generation=" not in address for address in _addresses(0))


def test_two_generations_of_one_run_id_get_distinct_addresses():
    """A machine attached inside a `respawn` chain must not overwrite its own past.

    Generations share a `run_id` by construction — that is what makes a chain one run — so an
    address built from `run_id` alone cannot separate them. The store will not save us: a
    different-task collision is ALLOWED, silently and deliberately (`ops.py:346-348`), because
    one message triaged in generation 0 and again in generation 3 is meant to be ONE row. That
    rule is right and is not what changes here; what changes is that the machine's rows say which
    generation they belong to, so they stop asking the rule to guess.

    This is the same fix, at the same grain, as the one the parity arc gave `govern:` and
    `approve:` — and it is why that arc had to land first."""
    zero, one = _addresses(0), _addresses(1)
    assert set(zero).isdisjoint(one), "a later generation reuses an earlier generation's address"
    assert all("generation=1" in address for address in one), "the coordinate reached neither row"
    assert len(one) == 2, "both rows, or the postamble stopped being unconditional"


def test_exhaustion_parks_and_still_commits():
    """The arm the incumbent had nowhere, DEMONSTRATED rather than designed. A machine that never
    terminates is the defect this package exists to remove: driven with a verdict that always
    loops, the budget runs out, the interpreter mints `Exhausted` without asking the judge, and
    the run parks with its record written."""
    looping = {"plan": [PlanVerdict.REVISE] * 40}
    handler = RecordingHandler(responses=answers())
    session = drive(handler, **looping)
    assert isinstance(session.stopped, Park)
    assert session.stopped.why is ParkReason.EXHAUSTED
    assert isinstance(session.turns[-1].verdict, Exhausted)
    assert len(session.turns) == 13, "the budget bounds the walk"
    assert len(handler.ledger) == 2
    assert handler.ledger[0].kind == "machine-committed"
    assert handler.ledger[1].kind == "machine-parked"
    assert handler.ledger[1].get("reason") == "exhausted"


def test_the_two_park_reasons_are_DISTINGUISHABLE_on_the_canonical_record():
    """The two park reasons, budget spent and a human's no, differ on the LEDGER as on the tape.

    A bare `machine-parked` for both is the collapse `route_plan` refuses one file over, where it
    declines to route a rejected plan to `Finish` because *"'we shipped it' and 'a human said
    no'"* must differ on the tape. The two would differ there and agree on the ledger, in the
    same run.

    Both reasons driven end to end through the real transition, because the two are minted at
    different sites — `route_plan` for a rejection, the exhaustion router for a budget — and a
    test that constructed `Park` by hand would prove the record can carry a reason without
    proving either one reaches it."""
    rejected = drive(RecordingHandler(responses=answers()), plan=[PlanVerdict.REJECTED])
    exhausted = drive(RecordingHandler(responses=answers()), plan=[PlanVerdict.REVISE] * 40)

    assert rejected.stopped == Park(State.PLAN, ParkReason.REJECTED)
    assert isinstance(exhausted.stopped, Park)
    assert exhausted.stopped.why is ParkReason.EXHAUSTED

    both = [stop_record(s.stopped) for s in (rejected, exhausted)]
    assert [r.kind for r in both] == ["machine-parked", "machine-parked"]
    assert [r.reason for r in both] == ["rejected", "exhausted"]


def test_a_finished_run_carries_the_reason_field_as_None_rather_than_omitting_it():
    """`None` says "this row records reasons, and no park reason applies"; an ABSENT field would
    say that or "written before the field existed", which is two things.

    The same distinction `Evidence.measured` draws one grain up — `None` is "not measured", an
    empty
    `CommandRun` is "measured, and clean" — and it is why the postamble passes `reason` on every
    row rather than only on a park."""
    handler = RecordingHandler(responses=answers())
    drive(handler, **happy_scripts())
    outcome = handler.ledger[-1]
    assert outcome.kind == "machine-finished"
    assert "reason" in to_jsonable_python(outcome)
    assert outcome.get("reason") is None


def test_the_judge_is_not_consulted_once_the_budget_is_spent():
    """The MECHANISM, not the outcome — and it is the whole reason `Exhausted` is minted by the
    interpreter. `combinators.unfold` refuses a node that descends past an exhausted budget rather
    than trusting it to stop; here the final visit skips the judge entirely, so a judge that
    never says "exhausted" still cannot loop forever.

    Counted directly rather than inferred from a shrinking fixture list: an earlier version read
    the leftover length of the caller's script, which stopped measuring anything the moment
    `judge_for` copied its argument. A count of calls says what the sentence says."""
    calls = 0

    def counting_judge(_ctx: Ctx, _evidence: Evidence) -> Effect[Verdict]:
        nonlocal calls
        calls += 1
        return PlanVerdict.REVISE
        yield  # pragma: no cover  -- generator, as `Judge` requires

    always_plan = {
        state: StateSpec(state=state, run=fuse(worker, counting_judge)) for state in State
    }
    handler = RecordingHandler(responses=answers())
    out = handler.run(
        lambda: run_machine(RUN, GOAL, always_plan, transition, start=State.PLAN, budget=12)
    )
    assert isinstance(out, Session)
    # 13 visits, and only the first 12 asked — the last minted `Exhausted` without consulting.
    assert len(out.turns) == 13
    assert calls == 12


# --- replay -------------------------------------------------------------------------------------


def test_replay_re_derives_every_transition():
    """The determinism claim: `transition` is pure over recorded verdicts, so a replay against
    the recorded tape walks the identical path with no worker consulted."""
    handler = RecordingHandler(responses=answers())
    recorded = drive(handler, **happy_scripts())
    replayed = ReplayHandler(handler.trace).run(
        lambda: run_machine(
            RUN, GOAL, specs(**happy_scripts()), transition, start=State.PLAN, budget=12
        )
    )
    assert isinstance(replayed, Session)
    assert replayed.path == recorded.path
    assert replayed.stopped == recorded.stopped
    assert [t.verdict for t in replayed.turns] == [t.verdict for t in recorded.turns]


def test_a_turn_records_what_ran_and_where_it_went():
    session = drive(RecordingHandler(responses=answers()), **happy_scripts())
    first = session.turns[0]
    assert isinstance(first, Turn)
    assert (first.visit, first.state, first.verdict) == (0, State.PLAN, PlanVerdict.APPROVED)
