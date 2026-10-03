"""The scripted path through `_cart` — a checkout end to end, across all three bookkeepers.

Role: **journey**. Its subject is the one the funnel's journey does not reach: the
CANONICAL RECORD. The funnel's tape is rich and its ledger is empty; this one's tape is plain and
its ledger has a row per line item, which is what makes it the fixture the projection work needs.

**Nothing here spells a key.** Expectations are derived two ways instead: the FOLD carries the
shape, and the properties carry what a fold necessarily drops — which line an op sat in, the park's
address, replay. Everything is computed from `_cart.CART`, so adding a line changes no expectation.
"""

import _cart as cart
import pytest
from _keymap import by_op, including

from effective.api import GatherBranch, qualified_event_name
from effective.graphview import fold_cycles, from_keys
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Segment, compose_key

pytestmark = pytest.mark.journey


def _program(tape: list[str]):
    """The tape folded onto the program. `item:` is this workflow's own unrolling scope (one lane
    per SKU running identical code), so the CALLER declares it."""
    return fold_cycles(from_keys(cart.ORDER_ID, tape), keymap=including("tests/_cart.py"))


def test_the_checkout_runs_and_the_totals_are_the_cart():
    """The spine, and the arithmetic the barrier owes: a line reserves what is on hand, and the
    order totals what was actually reserved rather than what was asked for."""
    order, _, _ = cart.scripted_run()
    assert order.total == sum(item.reserved * item.unit_price for item in cart.CART)
    assert [line.sku for line in order.lines] == [item.sku for item in cart.CART]
    assert order.substitutions == tuple(item.substitute for item in cart.CART if item.asks)


def test_a_proposed_swap_parks_in_the_line_that_proposed_it():
    """A substitution escalates IN THE LANE: the run parks on that branch's fully-qualified event,
    mid-fan-out, while the sibling lines' work holds.

    The expected address is COMPOSED by the same function an emitter would use, not spelled — so
    this compares identities rather than wire text."""
    handler = RecordingHandler(responses=cart.Answers(cart.CART))
    parked = handler.run(lambda: cart.checkout(cart.ORDER_ID))
    assert isinstance(parked, Suspended)

    index, item = next((i, it) for i, it in enumerate(cart.CART) if it.asks)
    assert parked.awaiting == qualified_event_name(
        GatherBranch(0, index),
        compose_key(t"item:{Segment(item.sku)}"),
        name=f"swap:{cart.ORDER_ID}",
    )
    # nothing reached the canonical record on the line that is still waiting
    assert not any(row.event_id.stored().endswith(item.sku) for row in handler.ledger)


def test_the_record_carries_one_row_per_line_plus_the_order():
    """The canonical record, which is this fixture's reason to exist. One `reserved:` row per
    line at ONE program point, and one `placed:` row at another — the shape no shipped workflow
    produces, since every `append_ledger` in `src/` sits outside a loop."""
    _, ledger, _ = cart.scripted_run()
    reserved = [row for row in ledger if row.kind == "reserved"]
    placed = [row for row in ledger if row.kind == "placed"]
    assert len(reserved) == len(cart.CART)
    assert len(placed) == 1
    assert {row.get("quantity") for row in reserved} == {item.reserved for item in cart.CART}
    # the model's proposal reached the record, on the line that asked and no other
    swapped = {row.event_id.stored(): row.get("substituted") for row in reserved}
    assert {sku for sku, sub in swapped.items() if sub} == {
        compose_key(t"reserved:{Segment(item.sku)}").stored() for item in cart.CART if item.asks
    }


def test_the_tape_folds_back_into_the_program_and_replays():
    """The run writes its own graph, and folding it recovers the checkout: the counts ARE the
    structure. The resumed tape REPLAYS, re-binding every op including the in-lane event."""
    order, _, tape = cart.scripted_run()
    program = _program(tape)
    counts = {node.key: node.count for node in program.nodes}
    n = len(cart.CART)

    assert counts["step;tool:cart"] == 1  # the profile, once
    assert counts["gather:*,*;item:*;step;tool:wanted"] == n  # one lane per line...
    assert counts["gather:*,*;item:*;step;tool:stock"] == n
    assert counts["gather:*,*;item:*;step;tool:price"] == n
    assert counts["gather:*,*;item:*;step:substitute"] == sum(item.short for item in cart.CART)
    assert by_op(program)[f"event;swap:{cart.ORDER_ID}"] == sum(item.asks for item in cart.CART)
    assert program.executions == len(tape)  # fewer nodes, never fewer facts

    handler = RecordingHandler(responses=cart.Answers(cart.CART))
    outcome = handler.run(lambda: cart.checkout(cart.ORDER_ID))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(cart.APPROVED)
    assert ReplayHandler(handler.trace).run(lambda: cart.checkout(cart.ORDER_ID)) == order


def test_lines_are_closed_book_by_construction():
    """Structural isolation: every op inside a lane's sub-tape carries ONLY that line's scope, so
    a line physically cannot read a sibling's stock.

    Every assertion sits inside a `continue` filter, so a filter matching nothing would pass by
    examining nothing. The trailing count says how much the loop actually looked at."""
    _, _, tape = cart.scripted_run()
    skus = {item.sku for item in cart.CART}
    examined = 0
    for key in tape:
        if not (lines := cart.items_in(key)):
            continue
        examined += 1
        assert len(lines) == 1, f"{key} sits in {len(lines)} lines"
        assert lines[0] in skus, key
    # three tool calls and a ledger row per line, plus a substitute ask and a park where they fell
    assert examined == 4 * len(cart.CART) + sum(item.short for item in cart.CART) + sum(
        item.asks for item in cart.CART
    ), tape


def test_the_scripted_cart_is_the_shape_the_journey_assumes():
    """Anti-vacuity: the assertions above are only meaningful if the cart exercises what they
    inspect. Named rather than counted, so a change says which property lapsed."""
    assert any(not item.short for item in cart.CART), "a line that is simply in stock"
    assert any(item.asks for item in cart.CART), "a line that proposes a swap and parks"
    assert any(item.short and not item.substitute for item in cart.CART), (
        "a line short with NO substitute — it reserves what there is and never asks"
    )
    assert 0 < sum(item.asks for item in cart.CART) < len(cart.CART), "some park, some do not"
