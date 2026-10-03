"""A pruned search whose bound is threaded through the recursion's own state, so nothing cancels.

Each level probes its frontier in one gather. A probe answers a node's children, the score of one
leaf beneath it and an admissible bound on every leaf beneath it. The level folds the scores into
`best` and keeps the nodes whose bound beats it, and `(frontier, best)` is the next level's
context. Pruning is a property of a value the recursion carries, which a race's cancellation is
not, so this is an ordinary shape row.

Predicted from `SCORES` before the row ran:

| the claim                                   | the number                        |
|---------------------------------------------|-----------------------------------|
| both rows answer the best leaf              | `max(SCORES)`, 20                 |
| the pruned search probes                    | 9 nodes, pruning at depths 1 to 3 |
| the unbounded search probes                 | all 31                            |
| one crash or two, at any checkpoints        | converge to the same answer       |
| a forced order of a level's probes          | changes nothing                   |

What a broken search does instead, predicted the same way:

| the mutant                           | what it shows              |
|--------------------------------------|----------------------------|
| the bound read at the root alone     | 31 probes                  |
| `best` forgotten between levels      | 11 probes                  |
| the left or the right probe kept     | answers 17                 |

The special case `fix` exists to find here is a bound applied only at the root. In one spelling
it breaks `agree`, since the spellings then probe different nodes; in every one it leaves them
alike, and the probe counts are what redden.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

from _shapes import Shape, agree, independent, interleave, sweep, sweep_pairs
from test_shape_conformance import level_scope

from effective.api import Effect, call_tool, gather, scoped
from effective.combinators import Answered, Decision, Deeper, Level, fix, unfold
from effective.domain import CallTool, DomainOp
from effective.search import Round, Step, frontier

DEPTH = 4
SCORES = (2, 1, 3, 20, 10, 7, 11, 10, 17, 3, 6, 4, 12, 13, 2, 1)
"""A leaf's score, by its path read as a binary number. The best, 20, is at `0011`, a path that
turns both ways, so a search keeping one side of each level answers 17."""

type State = tuple[tuple[str, ...], int]
"""A level's frontier, in the order the level probes it, and the best score found above it."""


class Scores:
    """The domain: probes a node by its path, which is the branch taken at each depth."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="probe", args={"path": str(path)}):
                self.calls.append(path)
                below = [
                    s for i, s in enumerate(SCORES) if format(i, f"0{DEPTH}b").startswith(path)
                ]
                leftmost = path.ljust(DEPTH, "0")
                return {
                    "children": [path + "0", path + "1"] if len(path) < DEPTH else [],
                    "bound": max(below),
                    "score": SCORES[int(leftmost, 2)],
                }
        raise TypeError(f"the search answers probe, not {op!r}")


def narrow(state: State, *, pruned: bool) -> Effect[State]:
    """One level: probe the frontier, fold the scores into `best`, and keep what could beat it."""
    frontier, best = state
    probes = yield from gather([partial(call_tool, "probe", {"path": p}, dict) for p in frontier])
    best = max(best, *(probe["score"] for probe in probes))
    kept = [probe for probe in probes if probe["bound"] > best or not pruned]
    return tuple(child for probe in kept for child in probe["children"]), best


START: State = (("",), 0)


def by_unfold(pruned: bool) -> Callable[[str], Effect[int]]:
    def node(state: State, level: Level) -> Effect[Decision[State, int]]:
        frontier, best = yield from narrow(state, pruned=pruned)
        return Deeper((frontier, best)) if frontier else Answered(best)

    return lambda _run_id: unfold(START, node, budget=DEPTH)


def by_fix(pruned: bool) -> Callable[[str], Effect[int]]:
    def close(solve: Callable[[State, int], Effect[int]]) -> Callable[[State, int], Effect[int]]:
        def body(state: State, depth: int) -> Effect[int]:
            frontier, best = yield from scoped(
                level_scope(depth), partial(narrow, state, pruned=pruned)
            )
            return (yield from solve((frontier, best), depth + 1)) if frontier else best

        return body

    return lambda _run_id: fix(close)(START, 0)


def by_frontier(pruned: bool) -> Callable[[str], Effect[int]]:
    """The level as a `frontier` step: the probe is the expansion, and the merge folds `best`
    and keeps what could beat it. A level that probes leaves is the last, one past `DEPTH`."""

    def probe(path: str) -> Effect[dict[str, Any]]:
        return (yield from call_tool("probe", {"path": path}, dict))

    def merge(step: Round[str, dict[str, Any], int]) -> Step[str, int]:
        best = max(step.value, *(probe["score"] for _, probe in step.expanded))
        kept = [probe for _, probe in step.expanded if probe["bound"] > best or not pruned]
        return Step(tuple(child for probe in kept for child in probe["children"]), best)

    return lambda _run_id: frontier([""], probe, merge, initial=0, steps=DEPTH + 1)


def search(pruned: bool) -> Shape:
    return Shape(
        spellings={
            "unfold": by_unfold(pruned),
            "fix": by_fix(pruned),
            "frontier": by_frontier(pruned),
        },
        domain=Scores,
        answer=lambda: max(SCORES),
        calls=lambda scores: sorted(scores.calls),
    )


PRUNED, UNBOUNDED = search(pruned=True), search(pruned=False)


def probed(backend, shape: Shape) -> dict[str, list[str]]:
    return {name: sorted(o.domain.calls) for name, o in agree(backend, shape).items()}


def test_both_searches_reach_the_best_leaf(backend):
    """`agree` holds each row's spellings to `max(SCORES)`: pruning loses no optimum."""
    pruned, unbounded = probed(backend, PRUNED), probed(backend, UNBOUNDED)
    assert {len(calls) for calls in pruned.values()} == {9}
    assert {len(calls) for calls in unbounded.values()} == {2 ** (DEPTH + 1) - 1}


def test_the_bound_prunes_below_the_root(backend):
    """Every depth between the root and the leaves probes fewer nodes than the one above it
    split into, so the bound is read at each level rather than at the first."""
    calls = probed(backend, PRUNED)["unfold"]
    at = [sum(len(path) == depth for path in calls) for depth in range(DEPTH + 1)]
    assert at == [1, 2, 2, 2, 2]
    assert all(at[depth + 1] < 2 * at[depth] for depth in range(1, DEPTH))


def test_a_crash_after_every_step_converges(backend):
    sweep(backend, PRUNED, "fix")


def test_two_crashes_converge(backend):
    assert sweep_pairs(backend, PRUNED, "fix") > 0


def test_the_search_is_schedule_independent(backend):
    """Predicted before it ran: a level folds its scores only once its gather has returned them
    all, in branch order, so no order of the probes can change what it keeps."""
    seen = interleave(backend, PRUNED)
    assert independent(seen), seen
