"""Monte-Carlo tree search on the substrate: the search fixture, and π's adversarial case.

A shared fixture like `_funnel.py` and `_cart.py`. It exists for two reasons that pull in
opposite directions, which is what makes it worth having.

**As a workload**, it is the ledger axis from the other side. The cart appends once per line
across a `gather`, a fan-out, so the branch coordinate is already an axis the substrate mints.
This appends once per MOVE in a sequential loop, where the only coordinate is the one the workflow
itself walks. Two shapes, one question: what does a projection do with a row per iteration.

**As an adversary**, it is the case folding cannot serve: folding collapses siblings, which is the
one thing a tree search cannot tolerate. A search's whole content is that it tried several
children and preferred one, so a view that merges them has thrown away the answer. Whether
`graphview.project` does that is measured in `test_projection.py` rather than argued here.

**The domain is route dispatch**, because a rollout has to mean something. A dispatcher has stops
to cover and picks them one at a time; for each choice it runs a bounded search, estimating each
candidate by simulating the rest of the route, and then DISPATCHES, which is a world change and a
ledger row. An estimate that comes back poor asks a human before the truck moves.

**search**: `effective.search.mcts`, one search per move. Selection reads only recorded values,
so replay re-derives the same choice.

**the rollout**: the recorded op, under `search:{k}` and `node:{stop}`, so the tape carries both
when a rollout happened and what it was about, the pair a projection has to tell apart.

**the dispatch** — the world change. One row per move at one program point, in a plain loop.

**Nothing here spells a key.** `Answers` reads the PLACEMENT off the key it is asked for.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

from effective.api import (
    Effect,
    append_ledger,
    ask_llm,
    await_event,
    call_tool,
    scoped,
)
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Segment, compose_key
from effective.keys.grammar import KeySyntaxError, parse
from effective.ops import LedgerRow
from effective.search import best_child, mcts

ROUTE_ID = "rt-2026"

ITERATIONS = 4
"""Rounds per move. The first rolls every candidate out and each later round rolls one out again,
so the search revisits a child, the property the projection question turns on."""

ESCALATE_OVER = 40
"""An estimate this poor asks the dispatcher before the truck moves. Derived, not declared: the
run parks exactly when the chosen stop's own estimate crosses it."""


@dataclass(frozen=True)
class Stop:
    """One destination, and what a rollout through it costs. The case, written before the run."""

    name: str
    estimate: int
    """What the simulator returns for a rollout that goes through this stop — lower is better."""


ROUTE = (
    Stop("depot", 12),
    Stop("northgate", 55),
    Stop("riverside", 20),
)
"""Three stops, and one of them (`northgate`) estimates past `ESCALATE_OVER`, so a move that
settles on it asks a human. Ordered worst-last so the search has something to prefer."""


@dataclass(frozen=True)
class Dispatch:
    """One move, once it has been committed to the world."""

    stop: str
    estimate: int
    approved: str = ""


@dataclass(frozen=True)
class Plan:
    """The dispatcher's finished route."""

    route_id: str
    legs: tuple[Dispatch, ...]

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(leg.stop for leg in self.legs)


def _candidates(candidates: tuple[str, ...], _root: str) -> Effect[dict[str, str]]:
    """The move's candidates, each a leaf of the search."""
    yield from ()
    return {stop: stop for stop in candidates}


def _rollout(stop: str) -> Effect[float]:
    """Simulate finishing the route through `stop`, scored in [0, 1] with higher better: the scale
    `uct`'s exploration constant assumes."""
    estimate = yield from ask_llm("estimate", f"Cost of completing the route via {stop}?", int)
    return 1 - estimate / 100


def _search(candidates: tuple[str, ...]) -> Effect[tuple[str, int]]:
    """A bounded search over `candidates`; returns the preferred stop and its mean estimate."""
    tree = yield from mcts(
        "", partial(_candidates, candidates), _rollout, rounds=ITERATIONS, depth=1
    )
    best = best_child(tree)
    assert best is not None, "a search with candidates rolls each of them out"
    return best, round((1 - tree.stats[(best,)].mean) * 100)


def _move(candidates: tuple[str, ...], route_id: str, leg: int) -> Effect[Dispatch]:
    """One move: search the candidates, ask if the answer is poor, dispatch.

    **The whole body runs inside the move's scope, the append included, and that is a rule rather
    than tidiness.** Measured on this fixture: with the append outside, the three
    `ledger;dispatched:{stop}` rows project to three nodes and no projection can merge them —
    there is no index anywhere in the key for the axis to be discovered from, only the authored
    stop name. Move it inside and the same three rows become one node with `count=3`.

    So a ledger append inside a loop belongs inside the loop's own scope. It is where a reader
    expects it, and it is what lets a view say *this happened three times* instead of drawing
    three boxes that share nothing a machine can see."""
    best, estimate = yield from _search(candidates)
    approved = ""
    if estimate > ESCALATE_OVER:
        # A poor estimate asks before the truck moves. The name is scoped on the ROUTE, which is
        # this question's subject; `approve:` is a substrate authority namespace and is refused
        # for an authored await.
        approved = yield from await_event(f"clear:{route_id}", str)
    # THE WORLD CHANGES HERE — one row per move, at one program point, in a plain loop.
    yield from append_ledger(
        LedgerRow(
            event_id=compose_key(t"dispatched:{Segment(best)}"),
            kind="dispatched",
            leg=leg,
            estimate=estimate,
        )
    )
    return Dispatch(best, estimate, approved)


def dispatch(route_id: str) -> Effect[Plan]:
    # the stops are data: what has to be covered comes back from the yard, not from this file
    stops = yield from call_tool("manifest", {"route": route_id}, list[str])
    remaining = list(stops)
    legs: list[Dispatch] = []
    while remaining:
        move = len(legs)
        leg = yield from scoped(
            # lint: terminal-hole — `move` is `len(legs)`, an `int`, so it is already an atom
            compose_key(t"move:{move}"),
            lambda r=tuple(remaining), n=move: _move(r, route_id, n),
        )
        legs.append(leg)
        remaining.remove(leg.stop)
    return Plan(route_id, tuple(legs))


# --- the scripted run: the answers, and the spine that drives them -------------


def nodes_in(key: str) -> tuple[str, ...]:
    """Every `node:` coordinate a placed key carries, read through the production parser.

    The sibling of `_funnel.lanes_in` and `_cart.items_in`. Matching text would be wrong here for
    the usual reason — one stop's name can prefix another's — and for a sharper one: this is the
    coordinate the projection question is ABOUT, so reading it any way but the parser's would put
    a second reader beside the one being studied."""
    try:
        terms = parse(key).terms
    except KeySyntaxError:
        return ()
    return tuple(
        term.coordinates[0].atoms[0].text
        for term in terms
        if term.tag == "node" and term.coordinates
    )


class Answers(Mapping[str, object]):
    """The canned side of the run, answered by PLACEMENT rather than by a spelled key.

    A rollout's value depends on the `node:` frame it sits under and on nothing else — which is
    the fixture's whole claim about determinism: the same candidate estimates the same however
    many times the search returns to it, so UCT's arithmetic is stable and replay re-derives the
    identical sequence of picks.

    Note what is NOT here: `clear:{route_id}`. An unanswered await parks the run."""

    def __init__(self, route: tuple[Stop, ...]) -> None:
        self._route = route
        self._by_name = {stop.name: stop for stop in route}

    def __iter__(self):
        """The names answerable WITHOUT a frame. `estimate` is deliberately absent: it needs a
        `node:` frame to say which candidate is being simulated."""
        return iter(("tool:manifest",))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        try:
            last = parse(str(key)).terms[-1]
        except KeySyntaxError:
            raise KeyError(key) from None
        match last.tag, [c.atoms[0].text for c in last.coordinates], nodes_in(str(key)):
            case "tool", ["manifest"], _:
                return [stop.name for stop in self._route]
            case "estimate", [], [stop] if stop in self._by_name:
                return self._by_name[stop].estimate
            case _:
                raise KeyError(key)


CLEARED = "cleared — send it"


def scripted_run() -> tuple[Plan, list[LedgerRow], list[str]]:
    """Drive the scripted route: search, park on the poor estimate, dispatch every stop.

    Returns the plan, the canonical rows, and the tape — so a caller asserts over all three
    bookkeepers without re-driving. Every expectation is computed from `ROUTE`."""
    handler = RecordingHandler(responses=Answers(ROUTE))
    outcome = handler.run(lambda: dispatch(ROUTE_ID))
    while isinstance(outcome, Suspended):
        outcome = outcome.resume(CLEARED)
    assert isinstance(outcome, Plan)
    return outcome, handler.ledger, [entry.key.stored() for entry in handler.trace]
