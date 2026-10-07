"""A shopping-cart checkout on the substrate — the fixture that exercises the LEDGER.

A shared fixture like `_funnel.py` and `_composition.py`, and deliberately *shallower* than the
funnel in combinators: the funnel is about depth (five combinators, two arms, four frames), and
duplicating that here would be waste. What this has and nothing else in the suite does is a
**canonical-record axis** — one program point that appends a ledger row once per line item, with
the item count discovered at runtime.

**Why that shape had to be built rather than found.** `grep -rn 'yield from append_ledger(' src/`
finds no ledger append inside a loop, and `agent/` never appends at all. So no shipped workflow
puts a varying authored `event_id` at one program point, and the projection question, a hundred
ledger rows drawn as a hundred boxes, had only a graph fixture behind it. A cart is the smallest
honest workload that produces it: the world changes once per line, a human can intervene per line,
and how many lines there are is data.

**profile** — the cart comes back from a tool. N is not written here.

**lines** — one lane per SKU under ONE `gather`, each scoped `item:{sku}`. Closed-book the same way
the funnel's lanes are: a lane's closure holds one SKU and the handler places its keys, so a lane
cannot name a sibling's reservation.

**the substitution** — when stock is short the model proposes a swap. This is the hybrid seam: a
deterministic check decides *whether* to ask, and the model decides *what* to offer.

**the shopper's word** — a proposed substitution parks the run IN THE LANE on that branch's
qualified event. Nobody's groceries get silently swapped, so approval is derived from the
substitution rather than declared: a line parks iff the model proposed something.

**the reservation** — the world change, and one ledger row per line at ONE program point. This is
the axis the projection has to collapse: five lines are five executions of one position, not five
positions.

**checkout** — a second ledger row at a DIFFERENT position, once, after the barrier. It is what
keeps the projection honest: a fold that collapsed everything would be as wrong as one that
collapsed nothing.

**Nothing here spells a key.** `Answers` reads the PLACEMENT off the key it is asked for, so a new
SKU or a new frame grammar rewrites nothing.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from effective.api import (
    Effect,
    append_ledger,
    ask_llm,
    await_event,
    call_tool,
    gather,
    scoped,
)
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Index, Segment, compose_key
from effective.keys.grammar import KeySyntaxError, ParsedKey, parse
from effective.ops import LedgerRow

ORDER_ID = "ord-8412"


@dataclass(frozen=True)
class Item:
    """One cart line, and the world's answers about it — the case, written before the run.

    `substitute` is what the model offers when stock is short. Empty means it found nothing, which
    is a real branch: the line reserves what there is and never asks the shopper."""

    sku: str
    wanted: int
    on_hand: int
    unit_price: int  # cents, so the totals are integers and the oracle is exact
    substitute: str = ""

    @property
    def short(self) -> bool:
        return self.wanted > self.on_hand

    @property
    def asks(self) -> bool:
        """Does this line park? Derived, never declared — a line asks the shopper exactly when
        the model has something to propose."""
        return self.short and bool(self.substitute)

    @property
    def reserved(self) -> int:
        return min(self.wanted, self.on_hand)


CART = (
    Item(sku="oat-milk", wanted=2, on_hand=5, unit_price=449),
    Item(sku="sourdough", wanted=1, on_hand=0, unit_price=625, substitute="rye-loaf"),
    Item(sku="coffee-beans", wanted=3, on_hand=1, unit_price=1899),
)
"""The scripted cart: one line in stock, one short with a substitute (it parks), one short with
none (it reserves what there is and does not ask). Three lines, three outcomes."""


@dataclass(frozen=True)
class Line:
    """What one lane carried back to the barrier."""

    sku: str
    reserved: int
    unit_price: int
    substituted: str = ""


@dataclass(frozen=True)
class Order:
    """The checkout's result."""

    order_id: str
    lines: tuple[Line, ...]
    total: int
    substitutions: tuple[str, ...]


def _lane(sku: str, order_id: str) -> Effect[Line]:
    """One line's lane: check stock, offer a swap if short, ask the shopper, reserve.

    Every name here is PLAIN — the lane runs inside `scoped(compose_key(t"item:{sku}"))` and the
    handler places the keys, so what a reader sees is what this line does."""
    wanted = yield from call_tool("wanted", {"sku": sku}, int)
    on_hand = yield from call_tool("stock", {"sku": sku}, int)
    price = yield from call_tool("price", {"sku": sku}, int)

    substituted = ""
    if wanted > on_hand:
        offer = yield from ask_llm("substitute", f"Nearest in-stock alternative to {sku}?", str)
        if offer:
            # Nobody's groceries get swapped without a word. The park sits INSIDE the lane, so
            # the sibling lines' work holds while this one waits.
            #
            # `swap:`, not `approve:` — the substrate REFUSES the latter, and the refusal is the
            # rule working rather than an inconvenience: `approve:` is an authority namespace the
            # substrate delivers into, and Absurd's events are global per queue first-emit-wins,
            # so an authored await by that name would consume an answer meant for a gate. The name
            # is scoped on the ORDER, which is this question's subject; run-scoping it would make
            # the workflow unforkable.
            answer = yield from await_event(f"swap:{order_id}", str)
            substituted = offer if answer.startswith("ok") else ""

    reserved = min(wanted, on_hand)
    # THE WORLD CHANGES HERE — one row per line, at one program point. The axis.
    yield from append_ledger(
        LedgerRow(
            event_id=compose_key(t"reserved:{Segment(sku)}"),
            kind="reserved",
            quantity=reserved,
            substituted=substituted,
        )
    )
    return Line(sku, reserved, price, substituted)


def checkout(order_id: str) -> Effect[Order]:
    # the width is data: the cart comes back from the store, not from this file
    skus = yield from call_tool("cart", {"order": order_id}, list[str])
    lines = yield from gather(
        [
            (lambda s=sku: scoped(compose_key(t"item:{Index(s)}"), lambda: _lane(s, order_id)))
            for sku in skus
        ]
    )
    # pure Python between yields — the barrier owes the ledger nothing until it has a total
    total = sum(line.reserved * line.unit_price for line in lines)
    substitutions = tuple(line.substituted for line in lines if line.substituted)
    # a SECOND ledger position, once. A projection that collapsed this into the per-line rows
    # would be as wrong as one that collapsed nothing.
    yield from append_ledger(
        LedgerRow(
            event_id=compose_key(t"placed:{Segment(order_id)}"),
            kind="placed",
            total=total,
        )
    )
    return Order(order_id, tuple(lines), total, substitutions)


# --- the scripted run: the answers, and the spine that drives them -------------

_FRAME_TAGS = ("gather", "item")
"""The frames this run's combinators mint. A frame is placed BY the handler, so a fixture that
wants to know where it is reads them back rather than writing them down."""


def _placed(key: str) -> tuple[dict[str, str], str]:
    """Split a placed key into `{frame tag: its first coordinate}` and the author's own op name.

    `gather:0,1;item:sourdough;step:stock` -> `({"gather": "0", "item": "sourdough"}, "stock")`.

    Total: text the parser refuses comes back as its own name, which the caller then fails to
    find rather than mis-answering."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return {}, key
    frames = {
        term.tag: term.coordinates[0].atoms[0].text
        for term in terms
        if term.tag in _FRAME_TAGS and term.coordinates
    }
    body = next((i for i, term in enumerate(terms) if term.tag not in _FRAME_TAGS), len(terms))
    return frames, ParsedKey(terms[body:]).render() if body < len(terms) else key


def items_in(key: str) -> tuple[str, ...]:
    """Every `item:` coordinate a placed key carries, read through the production parser.

    The sibling of `_funnel.lanes_in`, and for the same reason: `item:oat` is a prefix of
    `item:oat-milk`, so a substring test reports one line as reading another's stock."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return ()
    return tuple(
        term.coordinates[0].atoms[0].text
        for term in terms
        if term.tag == "item" and term.coordinates
    )


class Answers(Mapping[str, object]):
    """The canned side of the run, answered by PLACEMENT rather than by a spelled key.

    Note what is NOT here: `approve:{order_id}`. An unanswered await is what parks the run; the
    shopper's word arrives through `resume`."""

    def __init__(self, cart: tuple[Item, ...]) -> None:
        self._cart = cart
        self._by_sku = {item.sku: item for item in cart}

    def __iter__(self):
        """The names answerable WITHOUT a lane frame, so iterating and then indexing agrees.

        A lane's ops (`stock`, `price`, `wanted`, `substitute`) are deliberately absent: each
        needs an `item:` frame to say which line is asking."""
        return iter(("tool:cart",))

    def __len__(self) -> int:
        return 1

    def _in_lane(self, item: Item, name: str) -> object:
        """What a lane's op returns. `call_tool` names arrive tagged (`tool:stock`), `ask_llm`
        names bare — the two wrappers, as the handler placed them."""
        match name:
            case "tool:wanted":
                return item.wanted
            case "tool:stock":
                return item.on_hand
            case "tool:price":
                return item.unit_price
            case "substitute":
                return item.substitute
        raise KeyError(name)

    def __getitem__(self, key: str) -> object:
        frames, name = _placed(str(key))
        if name == "tool:cart":
            return [item.sku for item in self._cart]
        if (item := self._by_sku.get(frames.get("item", ""))) is not None:
            return self._in_lane(item, name)
        raise KeyError(key)


APPROVED = "ok — swap it"


def scripted_run() -> tuple[Order, list[LedgerRow], list[str]]:
    """Drive the scripted cart: park on the substitution, approve it, check out.

    Returns the order, the canonical rows it appended, and the tape — so a caller asserts over
    all three bookkeepers without re-driving. Every expectation is computed from `CART`."""
    handler = RecordingHandler(responses=Answers(CART))
    outcome = handler.run(lambda: checkout(ORDER_ID))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(APPROVED)
    assert isinstance(outcome, Order)
    return outcome, handler.ledger, [entry.key.stored() for entry in handler.trace]
