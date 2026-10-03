"""The graph properties an embodiment's edge table must hold — the obligation genericity creates.

While the machine had exactly one table, four checks hand-aimed at `State` were enough, and they
lived in that table's own test file. Once `run_machine` walks whatever states it is handed, every
embodiment ships an edge table and every one of them can ship it broken — so the checks travel
with the walk instead.

**They are properties of a RELATION'S CLOSURE, not of a file**, which is why no per-file lint can
compute them and why `ty` closing each router with `assert_never` does not cover them. `ty` proves
every cell of the dependent sum is answered; nothing in the type system proves the answers compose
into a graph you can leave, finish from, or reach.

Called from an embodiment's own suite (`machine.audit(...)`), not at run time: the relation is
total and static, so a runtime check would re-derive on every visit what one test settles once.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from enum import StrEnum

from effective.machine.outcomes import Advance, Finish, Outcome


class GraphDefect(AssertionError):
    """An edge table that type-checks and still cannot be walked.

    An `AssertionError` because that is what a suite expects to see, and named because "assert
    failed" in a promoted check tells an embodiment author nothing about which of four properties
    they broke.

    Carries EVERY violated property, not the first — `properties` is the mapping. Short-circuiting
    looked tidier and quietly weakened the anti-vacuity battery: a mutation aimed at one property
    can trip a different one on the way past (making a state finish also orphans what came after
    it), so a battery that only sees the first failure cannot tell whether the property it meant
    to test does any work at all."""

    def __init__(self, properties: Mapping[str, str]) -> None:
        self.properties = dict(properties)
        super().__init__(
            "; ".join(f"{name}: {why}" for name, why in sorted(self.properties.items()))
        )


type Relation[S: StrEnum, V: StrEnum] = Mapping[tuple[S, V], Outcome[S]]


def _domain[S: StrEnum, V: StrEnum](verdicts: Mapping[S, Iterable[V]]) -> Iterator[tuple[S, V]]:
    for state, fibre in verdicts.items():
        for verdict in fibre:
            yield state, verdict


def relation[S: StrEnum, V: StrEnum](
    verdicts: Mapping[S, Iterable[V]], transition: Callable[[S, V], Outcome[S]]
) -> Relation[S, V]:
    """The edge relation, built by CALLING the real function over the real domain.

    Built rather than declared, deliberately: a table an author writes down beside the routers is
    a second speller of the edge table, and the two would drift. This asks the function."""
    return {(state, verdict): transition(state, verdict) for state, verdict in _domain(verdicts)}


def _successors[S: StrEnum, V: StrEnum](
    rel: Relation[S, V], states: Iterable[S]
) -> dict[S, set[S]]:
    out: dict[S, set[S]] = {state: set() for state in states}
    for (state, _), outcome in rel.items():
        if isinstance(outcome, Advance):
            out[state].add(outcome.to)
    return out


def finish_cells[S: StrEnum, V: StrEnum](rel: Relation[S, V]) -> set[tuple[S, V]]:
    """Every cell that produces `Finish` — the ONE enumeration three properties are built on.

    Written once because it was written three times: `leavable`, `can_finish` and the declaration
    check each filtered `isinstance(outcome, Finish)` separately, and `--totality` refused it.
    Three spellings of one predicate is three places for it to drift."""
    return {cell for cell, outcome in rel.items() if isinstance(outcome, Finish)}


def leavable[S: StrEnum, V: StrEnum](rel: Relation[S, V]) -> set[S]:
    """States with at least one edge to a DIFFERENT state, or to `Finish`."""
    moves_on = {
        state
        for (state, _), outcome in rel.items()
        if isinstance(outcome, Advance) and outcome.to is not state
    }
    return moves_on | {state for state, _ in finish_cells(rel)}


def can_finish[S: StrEnum, V: StrEnum](rel: Relation[S, V], states: Iterable[S]) -> set[S]:
    """Every state from which `Finish` is reachable — the transitive closure, which is precisely
    the property no per-file lint can compute."""
    reaching = {state for state, _ in finish_cells(rel)}
    edges, changed = _successors(rel, states), True
    while changed:
        changed = False
        for state, nexts in edges.items():
            if state not in reaching and nexts & reaching:
                reaching.add(state)
                changed = True
    return reaching


def reachable_from[S: StrEnum, V: StrEnum](
    rel: Relation[S, V], start: S, states: Iterable[S]
) -> set[S]:
    edges, seen, stack = _successors(rel, states), {start}, [start]
    while stack:
        for nxt in edges[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def audit[S: StrEnum, V: StrEnum](
    verdicts: Mapping[S, Iterable[V]],
    transition: Callable[[S, V], Outcome[S]],
    *,
    start: S,
    finish_producers: Iterable[tuple[S, V]],
) -> None:
    """Hold the four graph properties, or raise `GraphDefect` naming which one broke.

    `finish_producers` is DECLARED and checked for equality, not counted. "Exactly one" is the
    coding machine's stance and not a law — a machine with two review paths may legitimately have
    two — while "at most one" would pass a table with none, which never finishes. Equality against
    a declaration is the same shape `StateSpec.canonical` has with `canonical_violations`: say what
    you intend, and let the tape or the graph disagree with you.
    """
    audit_relation(
        relation(verdicts, transition),
        set(verdicts),
        start=start,
        finish_producers=finish_producers,
    )


def audit_relation[S: StrEnum, V: StrEnum](
    rel: Relation[S, V],
    states: Iterable[S],
    *,
    start: S,
    finish_producers: Iterable[tuple[S, V]],
) -> None:
    """The same four properties over an INJECTED relation.

    Separate from `audit` because an embodiment's anti-vacuity battery has to mutate the relation
    and watch a named property redden — and it cannot do that through a function it does not
    control. Without this seam the checks could all be `assert True` and the suite would look
    exactly the same, which is the failure mode a green gate cannot show you.
    """
    known = set(states)
    defects: dict[str, str] = {}

    if stuck := known - leavable(rel):
        defects["no-state-is-unleavable"] = f"no edge out of {sorted(stuck)}"
    if trapped := known - can_finish(rel, known):
        defects["finish-reachable-from-everywhere"] = f"cannot reach Finish from {sorted(trapped)}"
    if orphaned := known - reachable_from(rel, start, known):
        defects["every-state-reachable-from-start"] = (
            f"unreachable from {start.value}: {sorted(orphaned)}"
        )

    produced = finish_cells(rel)
    declared = set(finish_producers)
    if produced != declared:
        defects["finish-producers-match-the-declaration"] = (
            f"undeclared={sorted(produced - declared)} "
            f"missing={sorted(declared - produced)} — a route to Finish nobody declared ships "
            f"unreviewed work, and a declared one that never fires is a dead claim"
        )

    if defects:
        raise GraphDefect(defects)


__all__ = [
    "GraphDefect",
    "Relation",
    "audit",
    "can_finish",
    "leavable",
    "reachable_from",
    "relation",
]
