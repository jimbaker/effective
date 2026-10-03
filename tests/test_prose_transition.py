"""Graph properties of the de-essaying machine's edge relation.

`ty` closes each router — a dropped arm is a diagnostic naming the arm, measured on three of them.
What `ty` cannot see is a fact about the relation's transitive closure: that every state is
reachable, that none is a dead end, that `Finish` has the producers it should, and that the
ordering constraint the design calls load-bearing is actually forced by the edges rather than by
the driver's good manners. Those live here, and every one is computed by CALLING `transition`
rather than by re-spelling the table beside it.
"""

from collections import deque

import pytest

from effective.machine.outcomes import (
    Advance,
    Exhausted,
    Finish,
    Park,
    ParkReason,
    VerdictOutOfDomain,
)
from effective.prose.states import VERDICTS, ReviewVerdict, SelectVerdict, State
from effective.prose.transition import ROUTES, transition

DOMAIN = [(state, verdict) for state, fibre in VERDICTS.items() for verdict in fibre]


def edges() -> dict[State, set[State]]:
    """The Advance relation, discovered by calling `transition` over the whole domain."""
    out: dict[State, set[State]] = {state: set() for state in State}
    for state, verdict in DOMAIN:
        match transition(state, verdict):
            case Advance(to=to):
                out[state].add(to)
            case _:
                pass
    return out


def test_the_domain_is_the_dependent_sum_not_the_product():
    """Each state's own fibre, not the product with every verdict in the machine.

    The naive `State x Verdict` product is 8 x 18 = 144 cells, of which 126 exist only to be
    refused. Here they are unspellable at the call site instead: `route_review` does not accept a
    `SelectVerdict`, so 126 of the 144 cannot be written down.

    The 18 was DERIVED from the design's table before it was run, and the derivation said 19 —
    which is why it is asserted rather than described. Recompute, do not trust the sentence."""
    all_verdicts = {v for fibre in VERDICTS.values() for v in fibre}
    assert len(DOMAIN) == sum(len(fibre) for fibre in VERDICTS.values()) == 18, DOMAIN
    assert len(State) * len(all_verdicts) == 144
    assert len(DOMAIN) == 18 < 144


def test_routes_and_verdicts_name_the_same_states():
    """A state in one map and not the other is a defect neither map can see alone."""
    assert set(ROUTES) == set(VERDICTS) == set(State)


@pytest.mark.parametrize(("state", "verdict"), DOMAIN, ids=lambda v: getattr(v, "value", v))
def test_every_cell_routes(state, verdict):
    assert transition(state, verdict) is not None


def test_every_state_is_reachable_from_select():
    reached, queue = {State.SELECT}, deque([State.SELECT])
    graph = edges()
    while queue:
        for nxt in graph[queue.popleft()]:
            if nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    assert reached == set(State), set(State) - reached


def test_every_state_can_terminate():
    """No state is a pocket the machine cannot leave. Exhaustion is excluded from the search on
    purpose — a budget park would make this vacuously true of a genuine livelock."""
    graph = edges()
    terminal = {
        state for state, verdict in DOMAIN if isinstance(transition(state, verdict), Finish | Park)
    }
    frontier, queue = set(terminal), deque(terminal)
    while queue:
        current = queue.popleft()
        for state, outs in graph.items():
            if current in outs and state not in frontier:
                frontier.add(state)
                queue.append(state)
    assert frontier == set(State), set(State) - frontier


def test_finish_has_exactly_two_producers_and_one_of_them_cannot_have_shipped():
    """The coding machine allows one cell, because a second is how a machine ships unreviewed work.

    Here there are two, and the second is safe for a STRUCTURAL reason rather than a promise:
    `(SELECT, EXHAUSTED_WORKLIST)` finishes, but no edge in the whole relation returns to `SELECT`,
    so that cell is reachable only on the first visit — before anything was read, let alone
    edited."""
    finishers = {(s, v) for s, v in DOMAIN if isinstance(transition(s, v), Finish)}
    assert finishers == {
        (State.REVIEW, ReviewVerdict.APPROVED),
        (State.SELECT, SelectVerdict.EXHAUSTED_WORKLIST),
    }
    assert all(State.SELECT not in outs for outs in edges().values())


def test_relocate_before_draft_is_forced_by_the_edges():
    """The design calls this load-bearing: you cannot compress what you have not rehomed.

    A product-BFS over `(state, has_relocated?)` — if any path reaches DRAFT with the flag still
    false, the ordering is the driver's discipline rather than the machine's property. Including
    the deep back edge, which is where an ordering constraint would most plausibly leak."""
    graph = edges()
    start = (State.SELECT, False)
    seen, queue = {start}, deque([start])
    while queue:
        state, relocated = queue.popleft()
        for nxt in graph[state]:
            step = (nxt, relocated or nxt is State.RELOCATE)
            if step not in seen:
                seen.add(step)
                queue.append(step)
    assert (State.DRAFT, False) not in seen, "reached DRAFT without ever visiting RELOCATE"
    assert (State.DRAFT, True) in seen


def test_the_deep_back_edge_lands_on_classify_not_draft():
    """A REVIEW verdict that adds a DESTINATION criterion is not a rewording: where things go
    has changed, so every unit is classified again."""
    assert transition(State.REVIEW, ReviewVerdict.RUBRIC_GREW) == Advance(State.RUBRIC)
    from effective.prose.states import RubricVerdict

    assert transition(State.RUBRIC, RubricVerdict.AMENDED) == Advance(State.CLASSIFY)


def test_a_refusal_is_distinguishable_from_shipping_on_the_tape():
    """`REVIEW` can say *leave this file alone*, and that is a park, not a finish."""
    assert transition(State.REVIEW, ReviewVerdict.LEAVE_IT_ALONE) == Park(
        State.REVIEW, ParkReason.REJECTED
    )
    assert transition(State.REVIEW, ReviewVerdict.APPROVED) == Finish()


def test_a_broken_environment_parks_rather_than_prescribing_a_reword():
    """VERIFY's broken-environment arm. `just check` reddens when the tmpfs Postgres container
    vanishes, with no prose defect anywhere."""
    from effective.prose.states import VerifyVerdict

    assert transition(State.VERIFY, VerifyVerdict.BROKEN_ENV) == Park(
        State.VERIFY, ParkReason.BROKEN_ENV
    )
    assert transition(State.VERIFY, VerifyVerdict.MECHANICAL) == Advance(State.DRAFT)


def test_exhaustion_parks_in_the_state_it_happened_in():
    for state in State:
        assert transition(state, Exhausted(state, level=0)) == Park(state, ParkReason.EXHAUSTED)


def test_a_verdict_from_the_wrong_state_is_refused_not_routed():
    """One of the excluded cells, arriving at run time through a mistyped driver."""
    with pytest.raises(VerdictOutOfDomain):
        transition(State.DRAFT, ReviewVerdict.APPROVED)
