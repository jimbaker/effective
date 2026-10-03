"""The scripted path through `_mcts`: a bounded search, driven end to end.

ROLE: journey. Its subject is the one neither sibling reaches: a workflow whose
control flow is decided by *accumulated recorded values* rather than by a classifier or a budget.
`route` dispatches on one recorded classification and `descend` on a budget; UCT reads every
rollout so far, which is what makes replay's claim here non-trivial.

**Nothing here spells a key.** Expectations are computed from `_mcts.ROUTE`, so a fourth stop
changes no assertion.
"""

import _mcts as mcts
import pytest

from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler

pytestmark = pytest.mark.journey


def test_the_search_orders_the_route_by_what_it_measured():
    """The dispatcher covers every stop, cheapest first — and the order is derived from the case,
    so it is a claim about the search rather than a recording of it."""
    plan, _, _ = mcts.scripted_run()
    assert set(plan.order) == {stop.name for stop in mcts.ROUTE}
    assert plan.order == tuple(stop.name for stop in sorted(mcts.ROUTE, key=lambda s: s.estimate))


def test_the_search_REVISITS_and_that_is_what_makes_it_a_search():
    """Anti-vacuity, and the fixture's whole claim about itself. A search that tried each
    candidate once is a fan-out wearing a search's clothes: its siblings would then be in
    bijection with the iteration index, and every question this fixture exists to ask would have
    a different answer.

    Asserted on the TAPE, because the revisit is a property of what ran."""
    _, _, tape = mcts.scripted_run()
    rollouts = [node for key in tape for node in mcts.nodes_in(key)]
    assert len(rollouts) > len(set(rollouts)), "no candidate was rolled out twice"


def test_one_ledger_row_per_move_and_the_park_lands_where_the_estimate_is_poor():
    """The canonical record: one dispatch per stop at one program point. The escalation is
    DERIVED — a move parks exactly when the stop it settled on estimates past `ESCALATE_OVER` —
    so no fixture field can drift from the behaviour it describes."""
    plan, ledger, _ = mcts.scripted_run()
    assert len(ledger) == len(mcts.ROUTE)
    assert {row.kind for row in ledger} == {"dispatched"}
    assert [row.get("leg") for row in ledger] == list(range(len(mcts.ROUTE)))

    asked = {leg.stop for leg in plan.legs if leg.approved}
    assert asked == {stop.name for stop in mcts.ROUTE if stop.estimate > mcts.ESCALATE_OVER}


def test_every_rollout_names_exactly_one_candidate():
    """Structural isolation, the search's version: a rollout carries the scope of the candidate
    it was about and no other, so a value cannot be attributed to the wrong child.

    The count says how much each move's search looked at: every candidate in its first round, then
    one per later round, with one move per stop."""
    _, _, tape = mcts.scripted_run()
    names = {stop.name for stop in mcts.ROUTE}
    examined = 0
    for key in tape:
        if not (under := mcts.nodes_in(key)):
            continue
        examined += 1
        assert len(under) == 1, f"{key} sits under {len(under)} candidates"
        assert under[0] in names, key
    candidates_per_move = range(len(mcts.ROUTE), 0, -1)
    assert examined == sum(n + mcts.ITERATIONS - 1 for n in candidates_per_move)


def test_replay_re_derives_every_selection():
    """The determinism claim that is actually load-bearing here. UCT is pure over the rollouts
    recorded so far, so a replay that re-binds those values must make the identical sequence of
    picks — and any drift shows up as a key that cannot bind rather than as a different answer.

    Reddens when: the selection reads anything but recorded values — a clock, a sample, or a set
    iteration order."""
    handler = RecordingHandler(responses=mcts.Answers(mcts.ROUTE))
    outcome = handler.run(lambda: mcts.dispatch(mcts.ROUTE_ID))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(mcts.CLEARED)
    assert ReplayHandler(handler.trace).run(lambda: mcts.dispatch(mcts.ROUTE_ID)) == outcome
