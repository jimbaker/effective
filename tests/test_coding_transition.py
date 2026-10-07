"""The coding machine's transition: totality over the dependent sum, and the graph properties.

**Roles.** The cell tests are `unit`; the graph-liveness tests are `adversarial`, and
an adversarial test counts only if it could have failed, so every graph check takes the edge
relation as a PARAMETER and the mutation battery drives it with deliberately broken relations.
A check that only ever sees the real table proves the table, not the check.

**What gates what, stated because it is easy to get backwards.**

- `assert_never` in each `route_*` is what makes a dropped or added enum arm a **`ty` error**.
  `ty` narrows a `StrEnum` match to `Never` and names the missing arm.
- `--totality src` does not see the enum matches (its census emits zero rows for a `match` over
  a `StrEnum`), so a green lint is not evidence of totality here, which is why these tests exist.
  It *does* see an `isinstance` guard.
- Neither of those can see a graph property: that `Finish` has exactly one producer, or that no
  state is unleavable, is a fact about the transitive closure of a relation, not about any file.
  That is this module's whole job.

The relation is always built by CALLING `transition` over the enumerated domain, never by
re-spelling the edges — a test that restates the table passes when the table and the restatement
are wrong together.
"""

from collections.abc import Callable, Iterator

import pytest

from effective.coding.states import (
    VERDICTS,
    PlanVerdict,
    ReviewVerdict,
    State,
    TestVerdict,
    Verdict,
)
from effective.coding.transition import (
    ROUTES,
    Advance,
    Finish,
    Park,
    ParkReason,
    VerdictOutOfDomain,
    transition,
)
from effective.machine.audit import GraphDefect, Relation, audit, audit_relation, relation
from effective.machine.outcomes import Exhausted


def domain() -> Iterator[tuple[State, Verdict]]:
    """The dependent sum, enumerated: every state paired with every arm of ITS fibre."""
    for state, fibre in VERDICTS.items():
        for verdict in fibre:
            yield state, verdict


def excluded() -> Iterator[tuple[State, Verdict]]:
    """The COMPLEMENT — every (state, verdict) the product allows and the sum excludes.
    Enumerated as a complement rather than hand-listed, so it cannot fall out of step."""
    for state, fibre in VERDICTS.items():
        for other, foreign in VERDICTS.items():
            if other is state:
                continue
            for verdict in foreign:
                if not isinstance(verdict, fibre):
                    yield state, verdict


# --- the domain itself ------------------------------------------------------------------------


def test_the_domain_is_the_dependent_sum_not_the_product():
    """The count is COMPUTED, never quoted: 22 cells outside PLAN plus one per `PlanVerdict`,
    since PLAN is a state of the machine."""
    sum_cells = len(list(domain()))
    product_cells = len(State) * sum(len(f) for f in VERDICTS.values())
    assert sum_cells == 22 + len(PlanVerdict) == 25
    assert product_cells == 175
    assert len(list(excluded())) == product_cells - sum_cells == 150


def test_every_state_has_a_fibre_and_a_router():
    """`VERDICTS` and `ROUTES` are two hand-written tables over the same key set; a state in one
    and not the other is a state whose verdicts nothing routes, or a router nothing feeds."""
    assert set(VERDICTS) == set(ROUTES) == set(State)


def test_expect_agrees_with_the_declared_fibres():
    """`transition` spells each state's fibre a second time, in its `_expect` call. Nothing
    structural stops that drifting from `VERDICTS`, so this is the pin: every arm of the
    declared fibre must route without raising."""
    for state, verdict in domain():
        assert transition(state, verdict) is not None


@pytest.mark.parametrize(("state", "verdict"), list(excluded()))
def test_a_foreign_verdict_is_refused_not_routed(state: State, verdict: Verdict):
    """The 150-cells-of-the-product problem, and the incumbent's actual defect.

    `tests/_coding.py:155` took `outcome: str` and tested it nominally, so an unrecognised value
    took the back edge into the one canonically-appending phase — a case-variant typo committed
    99 ledger rows in 400 ops and never terminated. Here it raises."""
    with pytest.raises(VerdictOutOfDomain):
        transition(state, verdict)


# --- the graph properties: enrolled in the SHIPPED audit ---------------------------------------

FINISH_PRODUCERS = [(State.REVIEW, ReviewVerdict.APPROVED)]
"""Coding's declaration. `machine.audit` checks it for EQUALITY rather than counting producers —
"exactly one" is this table's stance and not a law, so it is stated here rather than assumed
there. A second route to Finish is how a machine ships unreviewed work."""


def test_the_real_table_holds_every_graph_property():
    """The four closure properties, over the relation built by calling the real function.

    The checks live in `effective.machine.audit` because every embodiment ships an edge table.
    What stays here is coding's DECLARATION (its start state and its finish producers), which is
    the half only this embodiment can supply."""
    audit(VERDICTS, transition, start=State.PLAN, finish_producers=FINISH_PRODUCERS)


# --- the mutation battery: each mutation must redden a NAMED property --------------------------


def mutate_unleavable_draft(rel: Relation) -> Relation:
    """DRAFT self-loops on every arm."""
    return {
        cell: (Advance(State.DRAFT) if cell[0] is State.DRAFT else out)
        for cell, out in rel.items()
    }


def mutate_orphan_finalize(rel: Relation) -> Relation:
    """Nothing routes INTO FINALIZE any more."""
    return {
        cell: (
            Advance(State.REVIEW) if isinstance(out, Advance) and out.to is State.FINALIZE else out
        )
        for cell, out in rel.items()
    }


def mutate_second_finish(rel: Relation) -> Relation:
    """TEST can finish too — the mutation that ships unreviewed work."""
    return {**rel, (State.TEST, TestVerdict.RED): Finish()}


def mutate_no_finish(rel: Relation) -> Relation:
    """REVIEW.APPROVED loops back instead of finishing — the livelock, at the graph grain."""
    return {**rel, (State.REVIEW, ReviewVerdict.APPROVED): Advance(State.REVIEW)}


MUTATIONS: dict[str, tuple[Callable[[Relation], Relation], str]] = {
    "draft-self-loops-only": (mutate_unleavable_draft, "no-state-is-unleavable"),
    "finalize-orphaned": (mutate_orphan_finalize, "every-state-reachable-from-start"),
    "test-can-also-finish": (mutate_second_finish, "finish-producers-match-the-declaration"),
    "review-never-finishes": (mutate_no_finish, "finish-reachable-from-everywhere"),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_each_mutation_reddens_its_named_property(name: str):
    """Anti-vacuity, and it asserts the NAMED property rather than merely that something raised.

    The distinction matters: `mutate_second_finish` also orphans the states downstream of TEST,
    so an audit that stops at the first failure reports a reachability defect and the
    finish-producer property could be doing nothing. The audit therefore reports EVERY violated
    property and this asserts the expected one is among them."""
    mutate, expected = MUTATIONS[name]
    rel = relation(VERDICTS, transition)
    with pytest.raises(GraphDefect) as caught:
        audit_relation(
            mutate(rel), set(State), start=State.PLAN, finish_producers=FINISH_PRODUCERS
        )
    assert expected in caught.value.properties, (
        f"{name} did not redden {expected}; it fired {sorted(caught.value.properties)}"
    )


# --- exhaustion -------------------------------------------------------------------------------


@pytest.mark.parametrize("state", list(State))
def test_exhaustion_parks_from_every_state(state: State):
    """The arm the incumbent had nowhere: at budget zero the interpreter mints `Exhausted` and
    the machine parks, in EVERY state. Total by construction — `transition` handles it before
    the state dispatch — which is why it needs no row in seven tables."""
    outcome = transition(state, Exhausted(state, level=0))
    assert outcome == Park(state, ParkReason.EXHAUSTED)


def test_exhaustion_is_unspellable_by_a_judge():
    """The mechanism, not the manners: `Exhausted` is not an arm of any verdict enum, so a
    judge typed `Effect[V]` cannot return it. A judge-supplied exhaustion would relocate the
    livelock rather than kill it — a judge that never says it loops forever."""
    for fibre in VERDICTS.values():
        assert Exhausted not in set(fibre)
        assert not any(isinstance(arm, Exhausted) for arm in fibre)


def test_a_rejected_plan_parks_rather_than_finishing():
    """A human saying no and a review saying yes must not be the same outcome on the tape."""
    assert transition(State.PLAN, PlanVerdict.REJECTED) == Park(State.PLAN, ParkReason.REJECTED)
