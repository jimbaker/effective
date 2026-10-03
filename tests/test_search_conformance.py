"""MCTS and beam search on both engines, each spelled by its `effective.search` function and by
open recursion closed with `fix`.

The `fix` spelling is the reference, as in `test_shape_conformance.py`. The function has to
reproduce its answer and its checkpoint names, and a crash after every step's effect converges to
the same answer and names.

The tree is binary over the labels `a` and `b`, `DEPTH` levels below the root, and a node is its
string of labels. A node's score is a fixed function of that string."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition
from _shapes import Shape, agree, governing, run, run_shape, sweep

from effective.api import Effect, ask_llm, call_tool, gather, scoped
from effective.budget import Grant, MeasuredBudget, round_grant_name
from effective.budget import as_policy as budget_policy
from effective.combinators import Answered, Branch, Decision, Deeper, Level, fix, unfold
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.domain import CallTool, DomainOp
from effective.govern import (
    BudgetRefused,
    Exceeded,
    GateState,
    Policy,
    Proceed,
    Refuse,
    Verdict,
)
from effective.keys import Index, Name, Run, compose_key
from effective.keys.grammar import parse
from effective.ops import Step, WorkflowOp
from effective.search import SearchTree, Stats, beam, mcts, uct

DEPTH = 3
LABELS = "ab"
ROUNDS = 5
WIDTH = 2


def score_of(node: str) -> float:
    number = int("1" + node.translate(str.maketrans(LABELS, "01")), 2)
    return (number * 7 % 11) / 10


def children_of(node: str) -> list[str]:
    return [node + label for label in LABELS] if len(node) < DEPTH else []


class Scores:
    """The domain: a node's children and its score."""

    def __init__(self, score_of: Callable[[str], float] = score_of) -> None:
        self.score_of = score_of
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        self.calls.append(getattr(op, "name", ""))
        match op:
            case CallTool(name="expand", args={"node": str(node)}):
                return {child[-1]: child for child in children_of(node)}
            case CallTool(name="children", args={"node": str(node)}):
                return children_of(node)
            case CallTool(name="score", args={"node": str(node)}):
                return self.score_of(node)
        raise TypeError(f"the tree answers expand, children and score, not {op!r}")


def expand(node: str) -> Effect[dict[str, str]]:
    return (yield from call_tool("expand", {"node": node}, dict[str, str]))


def children(node: str) -> Effect[list[str]]:
    return (yield from call_tool("children", {"node": node}, list[str]))


def score(node: str) -> Effect[float]:
    return (yield from call_tool("score", {"node": node}, float))


def summary(tree: SearchTree[str]) -> list[list[Any]]:
    return sorted(["".join(path), s.visits, s.total] for path, s in tree.stats.items())


# --- MCTS: the reference spelling --------------------------------------------------------------


@dataclass(frozen=True)
class At:
    path: tuple[str, ...]
    node: str
    fresh: bool


def rollout_at(at: At) -> Effect[tuple[list[tuple[tuple[str, ...], float]], dict]]:
    return [(at.path, (yield from score(at.node)))], {}


def reference_node(tree: SearchTree[str]) -> Callable[[At, Level], Effect[Decision[At, Any]]]:
    def decide(at: At, level: Level) -> Effect[Decision[At, Any]]:
        if at.fresh or level.final:
            return Answered((yield from rollout_at(at)))
        known = tree.children.get(at.path)
        if known is None:
            found = yield from expand(at.node)
            if not found:
                scores, _ = yield from rollout_at(at)
                return Answered((scores, {at.path: found}))
            fresh = [At((*at.path, label), child, True) for label, child in found.items()]
            return Branch(fresh, partial(reference_join, at.path, found))
        if not known:
            return Answered((yield from rollout_at(at)))
        labels = list(known)
        visits = [tree.stats.get((*at.path, label), Stats()) for label in labels]
        label = labels[uct(visits, tree.stats.get(at.path, Stats()).visits)]
        return Deeper(At((*at.path, label), known[label], False))

    def node(at: At, level: Level) -> Effect[Decision[At, Any]]:
        body = partial(decide, at, level)
        for label in reversed(at.path):
            body = partial(scoped, compose_key(t"node:{Name(label)}"), body)
        return (yield from body())

    return node


def reference_join(path: tuple[str, ...], found: Mapping[str, str], values: Sequence[Any]):
    yield from ()
    expanded = {path: found}
    for _, more in values:
        expanded |= more
    return [s for scores, _ in values for s in scores], expanded


def reference_backprop(tree: SearchTree[str], value: Any) -> SearchTree[str]:
    scores, expanded = value
    stats = dict(tree.stats)
    for path, s in scores:
        for n in range(len(path) + 1):
            prior = stats.get(path[:n], Stats())
            stats[path[:n]] = Stats(prior.visits + 1, prior.total + s)
    return SearchTree(stats, {**tree.children, **expanded})


def open_mcts(close_over: Callable[[SearchTree[str], int], Effect[list[list[Any]]]]):
    def body(tree: SearchTree[str], k: int) -> Effect[list[list[Any]]]:
        if k == ROUNDS:
            return summary(tree)
        grow = partial(unfold, At((), "", False), reference_node(tree), budget=DEPTH)
        value = yield from scoped(compose_key(t"search:{Index(k)}"), grow)
        return (yield from close_over(reference_backprop(tree, value), k + 1))

    return body


def mcts_by_function(_run_id: str) -> Effect[list[list[Any]]]:
    return summary((yield from mcts("", expand, score, rounds=ROUNDS, depth=DEPTH)))


# --- beam: the reference spelling --------------------------------------------------------------


def top(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    ranked = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
    return [scored[i] for i in ranked[:WIDTH]]


type Scoring = Callable[[str], Effect[float]]


def scored(scoring: Scoring, nodes: Sequence[str]) -> Effect[list[tuple[str, float]]]:
    values = yield from gather([partial(scoring, node) for node in nodes])
    return list(zip(nodes, values, strict=True))


def open_beam(
    close_over: Callable[[list[tuple[str, float]], int], Effect[list[Any]]],
    scoring: Scoring = score,
):
    def body(frontier: list[tuple[str, float]], depth: int) -> Effect[list[Any]]:
        def level() -> Effect[list[tuple[str, float]] | None]:
            if depth == DEPTH:
                return None
            expansions = yield from gather([partial(children, node) for node, _ in frontier])
            if not (found := [child for expansion in expansions for child in expansion]):
                return None
            return top((yield from scored(scoring, found)))

        # The reference spelling catches for itself rather than through `within_budget`, so a
        # defect in that helper reaches one spelling and shows as disagreement. A level with no
        # children and a level that cannot pay both leave the frontier they were handed.
        narrowed: list[tuple[str, float]] | None = None
        with suppress(BudgetRefused):  # 3.12 splits a group, so one branch's refusal matches
            narrowed = yield from scoped(compose_key(t"d:{Index(depth)}"), level)
        if narrowed is None:
            return [list(pair) for pair in frontier]
        return (yield from close_over(narrowed, depth + 1))

    return body


def beam_by_fix(_run_id: str) -> Effect[list[Any]]:
    return (yield from fix(open_beam)(top((yield from scored(score, [""]))), 0))


def beam_by_function(_run_id: str) -> Effect[list[Any]]:
    frontier = yield from beam([""], children, score, width=WIDTH, depth=DEPTH)
    return [list(pair) for pair in frontier]


def beam_by_hand(score_of: Callable[[str], float], levels: int = DEPTH) -> list[list[Any]]:
    """The frontier after `levels` levels, computed without effects."""
    frontier = top([("", score_of(""))])
    for _ in range(levels):
        found = [child for node, _ in frontier for child in children_of(node)]
        if not found:
            break
        frontier = top([(child, score_of(child)) for child in found])
    return [list(pair) for pair in frontier]


# --- beam under a gate ---------------------------------------------------------------------

COST = 0.001
"""What scoring one node costs. Children come from a free tool, so a level spends one ask per
candidate of the frontier it expands."""


def asked_score(node: str) -> Effect[float]:
    """Scoring as a metered ask, which is what a budget refuses; `score` is a free tool."""
    return (yield from ask_llm(f"score:{node or 'root'}", node, float))


def spending() -> MeteredInterpreter:
    return MeteredInterpreter(
        llm=lambda op: (score_of(op.messages), Usage(cost=COST)),
        tools=lambda op: children_of(op.args["node"]),
    )


def spending_beam_by_function(_run_id: str) -> Effect[list[Any]]:
    frontier = yield from beam([""], children, asked_score, width=WIDTH, depth=DEPTH)
    return [list(pair) for pair in frontier]


def spending_beam_by_fix(_run_id: str) -> Effect[list[Any]]:
    close = partial(open_beam, scoring=asked_score)
    return (yield from fix(close)(top((yield from scored(asked_score, [""]))), 0))


SPENDING: dict[str, Callable[[str], Effect[Any]]] = {
    "function": spending_beam_by_function,
    "fix": spending_beam_by_fix,
}


type PolicyFor = Callable[[str], Policy]


def ceiling(asks: int) -> PolicyFor:
    """A gate that refuses once the run has spent `asks` scores."""
    return lambda run_id: budget_policy(
        MeasuredBudget(overall=asks * COST, run_id=run_id, on_exhaust="fail")
    )


def refuses(step: str) -> PolicyFor:
    """A gate that refuses one named op, wherever the scheduler runs it.

    A ceiling cannot aim at one branch of a level: the meter folds at the gather barrier, so a
    level reads one value and is refused whole. This is how a level comes to score some of its
    candidates and not others."""

    def policy(op: WorkflowOp, state: GateState) -> Verdict:
        match op:
            case Step(name=name) if name == step:
                return Refuse((f"{name} refused",), Exceeded(spent=COST, ceiling=COST))
            case Step(op=CallTool(name="children", args={"node": str(node)})) if (
                step == f"expand:{node}"
            ):
                return Refuse((f"{step} refused",), Exceeded(spent=COST, ceiling=COST))
            case _:
                return Proceed()

    return lambda _run_id: policy


def denies(step: str) -> PolicyFor:
    """A gate that refuses one named op for a reason that is not the budget's."""

    def policy(op: WorkflowOp, state: GateState) -> Verdict:
        match op:
            case Step(name=name) if name == step:
                return Refuse((f"{name} denied",))
            case _:
                return Proceed()

    return lambda _run_id: policy


CEILINGS = {"before the second level": (3, 1), "before the third": (5, 2)}
"""A ceiling in scores, and the levels the search keeps: the root costs one score, then a level
costs one per candidate it expands. A level is refused whole, because the meter the gate reads
folds at the gather barrier."""

REFUSALS = {
    "one candidate of a level": ("score:ab", 1),
    "one expansion of a level": ("expand:a", 1),
    "a candidate of the last level": ("score:aab", 2),
    "a candidate of the first": ("score:a", 0),
}
"""A refused op, and the levels the search keeps. The levels score `a b`, then `ba bb aa ab`, then
`baa bab aaa aab`, so each row names one op of a level whose other branches go through."""


SEARCHES = {
    "mcts": Shape(
        spellings={
            "function": mcts_by_function,
            "fix": lambda _r: fix(open_mcts)(SearchTree(), 0),
        },
        domain=Scores,
    ),
    "beam": Shape(
        spellings={"function": beam_by_function, "fix": beam_by_fix},
        domain=Scores,
        answer=partial(beam_by_hand, score_of),
    ),
}
"""Each search, spelled by its function and by `fix`. MCTS has no hand-computed answer, so its
spellings are held to each other."""


def gated(policy_for: PolicyFor, answer: Callable[[], Any] | None = None) -> Shape:
    """Beam under a `govern` gate, which is where a refusal comes from, on a handler that meters
    its asks."""
    return Shape(
        spellings=SPENDING,
        domain=spending,
        answer=answer,
        layers=governing(policy_for),
        contract=Contract.V1,
    )


@pytest.mark.parametrize("search", list(SEARCHES))
def test_the_search_reproduces_its_fix_spelling(backend, search):
    agree(backend, SEARCHES[search])


TIES = {"distinct scores": score_of, "every score tied": lambda _node: 0.5}


@pytest.mark.parametrize("scoring", list(TIES.values()), ids=list(TIES))
def test_beam_keeps_the_best_by_hand(backend, scoring):
    outcome = run(backend, beam_by_function, Scores(scoring))
    assert outcome.snap.result == beam_by_hand(scoring)


@pytest.mark.parametrize("spelling", list(SPENDING))
@pytest.mark.parametrize(("asks", "levels"), list(CEILINGS.values()), ids=list(CEILINGS))
def test_a_level_that_cannot_pay_answers_with_the_frontier_before_it(
    backend, asks, levels, spelling
):
    """A level refused before it scores anything leaves the search with the frontier it was
    handed, and both spellings answer the same. The two ceilings answer different frontiers, so
    the row is about where the stop landed rather than that it happened."""
    snap = run_shape(backend, gated(ceiling(asks)), spelling).snap

    assert snap.state == "completed", snap
    assert snap.result == beam_by_hand(score_of, levels=levels)
    assert snap.result != beam_by_hand(score_of)


@pytest.mark.parametrize(("step", "levels"), list(REFUSALS.values()), ids=list(REFUSALS))
def test_a_refusal_inside_a_level_answers_with_the_frontier_before_it(backend, step, levels):
    """The case `beam`'s docstring is about: a level whose other branches went through scored
    only some of its candidates, and a subset of a level is not a frontier.

    The two spellings agree on the checkpoint names as well as the answer, so a stop that
    descended again instead of answering would show here even where the extra levels answer the
    same frontier."""
    agree(backend, gated(refuses(step), partial(beam_by_hand, score_of, levels=levels)))


def test_a_frontier_with_no_children_answers_before_the_depth_runs_out(backend):
    """The other way a level ends: `DEPTH + 2` levels are granted, and the tree runs out first."""

    def deeper(_run_id: str) -> Effect[list[Any]]:
        frontier = yield from beam([""], children, score, width=WIDTH, depth=DEPTH + 2)
        return [list(pair) for pair in frontier]

    outcome = run(backend, deeper, Scores())

    assert outcome.snap.state == "completed", outcome.snap
    assert outcome.snap.result == beam_by_hand(score_of)
    # the root's level expands the root; the `DEPTH` levels after it each expand `WIDTH`
    # candidates, and the last of those finds no children and stops
    assert outcome.domain.calls.count("children") == 1 + WIDTH * DEPTH


@pytest.mark.parametrize("spelling", list(SPENDING))
def test_a_refusal_scoring_the_roots_is_raised_on(backend, spelling):
    """There is no frontier yet, so there is nothing to answer with: scoring the roots sits
    outside the stop, and the refusal ends the run."""
    snap = run_shape(backend, gated(ceiling(0)), spelling).snap

    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "BudgetRefused"


@pytest.mark.parametrize("spelling", list(SPENDING))
def test_a_refusal_that_is_not_the_budgets_is_raised_on(backend, spelling):
    """The stop is the budget's alone. A gate that denies a candidate for its own reason ends the
    search, because a search that answered here would hide every other refusal a level meets."""
    snap = run_shape(backend, gated(denies("score:ab")), spelling).snap

    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "Refused"


@pytest.mark.parametrize("spelling", list(SPENDING))
def test_a_gate_that_refuses_nothing_changes_nothing(backend, spelling):
    """Anti-vacuity for the rows above: the same programs under a ceiling nothing reaches answer
    what the unmetered beam answers, so the stop is what those rows measure."""
    snap = run_shape(backend, gated(ceiling(100)), spelling).snap

    assert snap.state == "completed", snap
    assert snap.result == beam_by_hand(score_of)


GATES = {"a named op": refuses("score:ab"), "a ceiling": ceiling(5)}
"""The two gates a crash has to survive. The ceiling is the one that reads the meter, so it is
also the row that says a replay re-derives the spend rather than carrying it."""


@pytest.mark.parametrize("spelling", list(SPENDING))
@pytest.mark.parametrize("gate", list(GATES.values()), ids=list(GATES))
@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_crash_at_every_step_of_a_refused_beam_converges(backend, position, gate, spelling):
    """The durable half: a replay re-derives the frontier the refused level was handed, rather
    than reading it off a store, and the gate re-fires over replayed ops without moving it."""
    sweep(backend, gated(gate), spelling, position)


def test_mcts_counts_every_rollout_at_the_root(backend):
    outcome = run(backend, mcts_by_function, Scores())
    rollouts = [k for k in backend.checkpoint_keys(outcome.task) if k.endswith("tool:score")]
    (root,) = [row for row in outcome.snap.result if row[0] == ""]
    assert root[1] == len(rollouts)


@pytest.mark.parametrize("search", list(SEARCHES))
def test_a_crash_after_every_step_of_a_search_converges(backend, search):
    sweep(backend, SEARCHES[search], "function")


def test_a_round_grant_park_extends_the_search_until_a_stop(backend):
    name, run_id = compose_key(t"search:{Run(str(uuid4()))}").stored(), str(uuid4())

    def program(_rid: str) -> Effect[list[list[Any]]]:
        tree = yield from mcts("", expand, score, rounds=2, depth=DEPTH, run_id=run_id)
        return summary(tree)

    backend.register(name, program, Scores(), Fault(), [])
    task = backend.spawn(name, run_id)
    grants = {2: Grant(add_rounds=3), 5: Grant(stop=True)}
    for rounds, grant in grants.items():
        assert backend.run_until_result(task).state != "completed"
        (parked,) = backend.parked(task)
        wake = round_grant_name(run_id, generation=0, rounds=rounds).stored()
        assert str(parked.wake_event) == wake
        backend.emit_event(task, parked.wake_event, grant.model_dump())
    snap = backend.run_until_result(task)

    assert snap.state == "completed", snap
    heads = [parse(k).terms[0] for k in backend.checkpoint_keys(task)]
    assert {int(h.coordinates[0].atoms[0].text) for h in heads if h.tag == "search"} == set(
        range(5)
    )


def test_a_label_that_cannot_be_a_key_fails_its_task_once(backend):
    name, run_id = compose_key(t"search:{Run(str(uuid4()))}").stored(), str(uuid4())

    def badly_labelled(node: str) -> Effect[dict[str, str]]:
        yield from ()
        return {"a b": node + "a"} if node == "" else {}

    def program(_rid: str) -> Effect[list[list[Any]]]:
        return summary((yield from mcts("", badly_labelled, score, rounds=2, depth=DEPTH)))

    backend.register(name, program, Scores(), Fault(), [])
    task = backend.spawn(name, run_id, max_attempts=3)
    snap = backend.run_until_result(task)
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "UnusableNodeLabel"
    assert backend.task_attempts(task) == 1
