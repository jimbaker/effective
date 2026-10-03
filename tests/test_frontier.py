"""`frontier` where `beam` does not reach it: a pick that leaves nodes queued, the empty cases, and
a pick that names no usable position.

The best-first rows search `test_pruned_search`'s tree with a queue that persists across steps: a
child is queued under its parent's bound, `pick` takes the `k` highest, and the merge drops what
cannot beat `best`. Measured:

| the claim                                  | the number                   |
|--------------------------------------------|------------------------------|
| best-first answers the best leaf           | `max(SCORES)`, 20            |
| probing one node per step                  | 7 probes                     |
| probing two nodes per step                 | 9 probes                     |
| a queue holding one node twice             | expands it twice             |
| a crash after every step, forced schedules | the same answer              |

A node's results are recorded under its key, so a `pick` that reorders the nodes between two
attempts of one run cannot hand one node another's recorded expansion.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from _conformance import Fault, FaultPosition
from _shapes import Shape, independent, interleave, run, sweep
from test_pruned_search import PRUNED, SCORES, Scores

from effective.api import Effect, call_tool
from effective.domain import CallTool, DomainOp
from effective.keys import Name, compose_key
from effective.search import Round, Step, frontier

TREE: dict[str, list[str]] = {"a": ["a0", "a1"], "a0": [], "a1": []}


class Tree:
    """The domain: `kids` answers a node's children and counts the asks."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="kids", args={"node": str(node)}):
                self.calls.append(node)
                return TREE[node]
        raise TypeError(f"the tree answers kids, not {op!r}")


def kids(node: str) -> Effect[list[str]]:
    return (yield from call_tool("kids", {"node": node}, list[str]))


def breadth(step: Round[str, list[str], list[str]]) -> Step[str, list[str]]:
    """The unpicked nodes stay, the children join, and the value lists what was expanded."""
    children = [child for _, found in step.expanded for child in found]
    return Step([*step.rest, *children], [*step.value, *(node for node, _ in step.expanded)])


def first(queue: Sequence[str], value: object) -> Sequence[int]:
    return [0]


def test_a_node_queued_twice_is_expanded_twice(backend):
    def program(_run_id: str) -> Effect[list[str]]:
        return (
            yield from frontier(
                ["a", "a"], kids, breadth, initial=[], steps=2, pick=first, key=lambda n: n
            )
        )

    outcome = run(backend, program, Tree())
    assert outcome.snap.result == ["a", "a"]


def test_an_empty_queue_answers_before_any_step(backend):
    rounds: list[Round[str, list[str], list[str]]] = []

    def spy(step: Round[str, list[str], list[str]]) -> Step[str, list[str]]:
        rounds.append(step)
        return breadth(step)

    def program(_run_id: str) -> Effect[list[str]]:
        return (yield from frontier([], kids, spy, initial=["initial"], steps=3))

    outcome = run(backend, program, Tree())
    assert outcome.snap.result == ["initial"]
    assert rounds == []
    assert backend.checkpoint_keys(outcome.task) == []


@pytest.mark.parametrize(
    "positions",
    [[], [1], [-1], [0, 0]],
    ids=["no position", "past the queue", "before the queue", "a position twice"],
)
def test_a_pick_naming_no_usable_position_fails_its_task(backend, positions):
    def pick(queue: Sequence[str], value: object) -> Sequence[int]:
        return positions

    def program(_run_id: str) -> Effect[list[str]]:
        return (
            yield from frontier(
                ["a"], kids, breadth, initial=[], steps=2, pick=pick, key=lambda n: n
            )
        )

    outcome = run(backend, program, Tree())
    assert outcome.snap.state == "failed"
    assert "UnusablePick" in str(outcome.snap.failure)
    assert backend.task_attempts(outcome.task) == 1
    assert outcome.domain.calls == []


def test_two_distinct_nodes_sharing_a_key_fail_their_task(backend):
    """Two nodes under one key would own one recorded result between them."""

    def program(_run_id: str) -> Effect[list[str]]:
        return (
            yield from frontier(
                ["a", "a0"], kids, breadth, initial=[], steps=1, key=lambda n: n[0]
            )
        )

    outcome = run(backend, program, Tree())
    assert outcome.snap.state == "failed"
    assert "UnusableNodeLabel" in str(outcome.snap.failure)
    assert backend.task_attempts(outcome.task) == 1
    assert outcome.domain.calls == []


@pytest.mark.parametrize(
    "unsafe",
    [
        {"pick": first},
        {"score": lambda node: iter(())},
        {"children": lambda expansion: expansion},
    ],
    ids=["a pick without a key", "a score without children", "children without a score"],
)
def test_a_frontier_it_cannot_record_soundly_fails_its_task_once(backend, unsafe):
    def program(_run_id: str) -> Effect[list[str]]:
        return (yield from frontier(["a"], kids, breadth, initial=[], steps=1, **unsafe))

    outcome = run(backend, program, Tree())
    assert outcome.snap.state == "failed"
    assert "UnsoundFrontier" in str(outcome.snap.failure)
    assert backend.task_attempts(outcome.task) == 1
    assert outcome.domain.calls == []


def item(node: str):
    return compose_key(t"item:{Name(node)}")


def test_a_node_named_by_a_key_is_recorded_under_that_key(backend):
    def program(_run_id: str) -> Effect[list[str]]:
        return (yield from frontier(["a"], kids, breadth, initial=[], steps=1, key=item))

    outcome = run(backend, program, Tree())
    assert outcome.snap.result == ["a"]
    assert [k for k in backend.checkpoint_keys(outcome.task) if "kids" in k] == [
        "d:0;gather:0,0;node;item:a;step;tool:kids"
    ]


def test_two_distinct_nodes_named_by_one_key_fail_their_task(backend):
    def program(_run_id: str) -> Effect[list[str]]:
        same = compose_key(t"item:{Name('same')}")
        return (
            yield from frontier(
                ["a", "a0"], kids, breadth, initial=[], steps=1, key=lambda n: same
            )
        )

    outcome = run(backend, program, Tree())
    assert outcome.snap.state == "failed"
    assert "UnusableNodeLabel" in str(outcome.snap.failure)
    assert backend.task_attempts(outcome.task) == 1


def test_a_str_name_and_a_key_name_never_share_a_path(backend):
    """A str is one term `node:{name}`, and a Key always runs under the bare term `node` first."""
    names = {"a": "a", "a0": compose_key(t"a")}

    def program(_run_id: str) -> Effect[list[str]]:
        return (
            yield from frontier(
                ["a", "a0"], kids, breadth, initial=[], steps=1, key=lambda n: names[n]
            )
        )

    outcome = run(backend, program, Tree())
    assert outcome.snap.state == "completed"
    paths = sorted(k for k in backend.checkpoint_keys(outcome.task) if "kids" in k)
    assert paths == [
        "d:0;gather:0,0;node:a;step;tool:kids",
        "d:0;gather:0,1;node;a;step;tool:kids",
    ]


def test_a_key_of_more_than_one_term_fails_its_task_once(backend):
    two = compose_key(t"item:{Name('a')};other:{Name('b')}")

    def program(_run_id: str) -> Effect[list[str]]:
        return (yield from frontier(["a"], kids, breadth, initial=[], steps=1, key=lambda n: two))

    outcome = run(backend, program, Tree())
    assert outcome.snap.state == "failed"
    assert "UnusableNodeLabel" in str(outcome.snap.failure)
    assert backend.task_attempts(outcome.task) == 1


type Bounds = tuple[int, dict[str, int]]
"""`best`, and each queued node's bound: its parent's, since its own arrives with its probe."""


def probe(path: str) -> Effect[dict[str, Any]]:
    return (yield from call_tool("probe", {"path": path}, dict))


def highest(k: int):
    def pick(queue: Sequence[str], value: Bounds) -> Sequence[int]:
        _, bounds = value
        ranked = sorted(range(len(queue)), key=lambda i: (-bounds.get(queue[i], 10**9), queue[i]))
        return ranked[:k]

    return pick


def prune(step: Round[str, dict[str, Any], Bounds]) -> Step[str, Bounds]:
    best, bounds = step.value
    best = max(best, *(probe["score"] for _, probe in step.expanded))
    bounds = dict(bounds)
    queue = list(step.rest)
    for _, probe in step.expanded:
        for child in probe["children"]:
            bounds[child] = probe["bound"]
            queue.append(child)
    return Step([node for node in queue if bounds[node] > best], (best, bounds))


def best_first(k: int) -> Shape:
    def program(_run_id: str) -> Effect[int]:
        best, _ = yield from frontier(
            [""], probe, prune, initial=(0, {}), steps=40, pick=highest(k), key=lambda p: f"p{p}"
        )
        return best

    return Shape(
        spellings={"frontier": program},
        domain=Scores,
        answer=lambda: max(SCORES),
        calls=lambda scores: sorted(scores.calls),
    )


@pytest.mark.parametrize(("k", "probes"), [(1, 7), (2, 9)])
def test_best_first_reaches_the_best_leaf(backend, k, probes):
    outcome = run(backend, best_first(k).spellings["frontier"], Scores())
    assert outcome.snap.result == max(SCORES)
    assert len(outcome.domain.calls) == probes


def test_a_crash_after_every_step_of_best_first_converges(backend):
    sweep(backend, best_first(1), "frontier")


def test_best_first_is_schedule_independent(backend):
    seen = interleave(backend, best_first(2))
    assert independent(seen), seen


def test_a_crash_after_every_step_of_the_pruned_frontier_converges(backend):
    sweep(backend, PRUNED, "frontier")


TREES: dict[str, list[str]] = {"a": ["a0", "a1"], "b": ["b0"], "a0": [], "a1": [], "b0": []}


class Trees(Tree):
    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="kids", args={"node": str(node)}):
                self.calls.append(node)
                return TREES[node]
        raise TypeError(f"the trees answer kids, not {op!r}")


def expanded(step: Round[str, list[str], list[Any]]) -> Step[str, list[Any]]:
    children = [child for _, found in step.expanded for child in found]
    return Step([*step.rest, *children], [*step.value, *map(list, step.expanded)])


def test_a_pick_reordered_between_attempts_replays_each_node_its_own_expansion(backend):
    """The first attempt picks `a`, records its expansion, and crashes before the second step;
    the second picks `b` first, as an upgraded `pick` would. `b` is expanded afresh rather than
    handed `a`'s children, and `a` is expanded again at its new place."""
    attempts: list[int] = []

    def program(_run_id: str) -> Effect[list[Any]]:
        attempts.append(len(attempts))
        favored = "a" if len(attempts) == 1 else "b"

        def pick(queue: Sequence[str], value: object) -> Sequence[int]:
            return [queue.index(favored)] if favored in queue else [0]

        return (
            yield from frontier(
                ["a", "b"], kids, expanded, initial=[], steps=2, pick=pick, key=lambda n: n
            )
        )

    fault = Fault(on_name="d:1;", position=FaultPosition.BEFORE_OP)
    outcome = run(backend, program, Trees(), fault=fault)
    assert outcome.snap.result == [["b", ["b0"]], ["a", ["a0", "a1"]]]
    assert outcome.domain.calls == ["a", "b", "a"]
    assert backend.task_attempts(outcome.task) == 2


@dataclass(frozen=True)
class Finding:
    """What acting on a lead found: its own text and the leads it opened."""

    lead: str
    leads: tuple[str, ...]


def act(lead: str) -> Effect[Finding]:
    found = yield from call_tool("kids", {"node": lead}, list[str])
    return Finding(lead, tuple(found))


def worth(lead: str) -> Effect[float]:
    return (yield from call_tool("worth", {"lead": lead}, float))


class Leads(Tree):
    """The tree, and a lead's worth: its length."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="worth", args={"lead": str(lead)}):
                self.calls.append(f"worth:{lead}")
                return float(len(lead))
        return super().run(op)


def keep_leads(step: Round[str, Finding, list[Any]]) -> Step[str, list[Any]]:
    return Step([lead for lead, _ in step.scored], [*step.value, *map(list, step.scored)])


def research(_run_id: str) -> Effect[list[Any]]:
    """An expansion that is not itself a sequence of nodes, the shape of a research finding."""
    return (
        yield from frontier(
            ["a"],
            act,
            keep_leads,
            initial=[],
            steps=1,
            key=lambda lead: lead,
            pick=first,
            children=lambda f: f.leads,
            score=worth,
        )
    )


SCORED = [["a0", 2.0], ["a1", 2.0]]
RESEARCH = Shape(spellings={"frontier": research}, domain=Leads, answer=lambda: SCORED)


def test_children_names_what_the_second_phase_scores(backend):
    outcome = run(backend, research, Leads())
    assert outcome.snap.result == SCORED
    assert outcome.domain.calls == ["a", "worth:a0", "worth:a1"]


def test_a_crash_after_every_step_of_the_scored_phase_converges(backend):
    sweep(backend, RESEARCH, "frontier")
