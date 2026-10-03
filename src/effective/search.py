"""Searches over the recursion combinators: Monte-Carlo tree search on `tree_search`, and a
frontier search as a `descend` judge, with beam search one policy of it.

A search's statistics are plain values folded from recorded results, so replay re-derives every
round's selection without storing the tree."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import assert_never

from effective.api import Effect, gather, scoped
from effective.combinators import (
    Answered,
    Branch,
    Decision,
    Deeper,
    Grantor,
    Level,
    descend,
    tree_search,
    within_budget,
)
from effective.keys import Key, Name, compose_key
from effective.ops import CompositionRefused

type Path = tuple[str, ...]
"""A node's position: the labels of the children taken from the root."""

EXPLORATION = math.sqrt(2)


@dataclass(frozen=True)
class Stats:
    """A node's rollouts: how many passed through it, and the sum of their scores."""

    visits: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.visits


@dataclass(frozen=True)
class SearchTree[C]:
    """What the rounds so far have learned: each visited node's `Stats`, and each expanded node's
    children by label. A node expanded to no children is terminal."""

    stats: Mapping[Path, Stats] = field(default_factory=dict)
    children: Mapping[Path, Mapping[str, C]] = field(default_factory=dict)


@dataclass(frozen=True)
class Rollouts[C]:
    """One round's value: every rollout's path and score, and the nodes the round expanded."""

    scores: tuple[tuple[Path, float], ...]
    expanded: Mapping[Path, Mapping[str, C]]


def uct(children: Sequence[Stats], parent_visits: int, c: float = EXPLORATION) -> int:
    """The index of the child to descend into: an unvisited child first, else the highest
    `mean + c * sqrt(ln(parent_visits) / visits)`, the earliest child on a tie. A parent counts
    every rollout through its children, so `parent_visits` is positive once any child is."""
    if unvisited := [i for i, s in enumerate(children) if s.visits == 0]:
        return unvisited[0]
    log_n = math.log(parent_visits)
    scores = [s.mean + c * math.sqrt(log_n / s.visits) for s in children]
    return max(range(len(children)), key=lambda i: (scores[i], -i))


def best_child[C](tree: SearchTree[C], path: Path = ()) -> str | None:
    """The label of `path`'s visited child with the highest mean, the earliest on a tie, or `None`
    when no child of `path` has been rolled out."""
    visited = [
        (label, stats)
        for label in tree.children.get(path, {})
        if (stats := tree.stats.get((*path, label))) is not None
    ]
    if not visited:
        return None
    order = {label: i for i, (label, _) in enumerate(visited)}
    return max(visited, key=lambda ls: (ls[1].mean, -order[ls[0]]))[0]


@dataclass(frozen=True)
class _At[C]:
    """Where a round stands: the node's path and context, and whether a `Branch` made it, in which
    case it is rolled out before it is expanded."""

    path: Path
    context: C
    fresh: bool


type Expand[C] = Callable[[C], Effect[Mapping[str, C]]]
type Rollout[C] = Callable[[C], Effect[float]]
type Select = Callable[[Sequence[Stats], int], int]


def mcts[C](
    root: C,
    expand: Expand[C],
    rollout: Rollout[C],
    *,
    rounds: int,
    depth: int,
    select: Select = uct,
    run_id: str | None = None,
    grantor: Grantor | None = None,
) -> Effect[SearchTree[C]]:
    """Search from `root` for `rounds` rounds of at most `depth` levels, and return the tree.

    Each round descends from the root by `select` over expanded nodes. At an unexpanded node it
    calls `expand`; the children are rolled out together as a `Branch`, and a node with no children
    is rolled out itself. A node's work runs under one `node:{label}` scope per label on its path,
    and every rollout's score counts toward each node on its path. `run_id` and `grantor` refill
    rounds, and a budget refusal ends the search with the tree so far, as in `tree_search`."""
    return (
        yield from tree_search(
            _At((), root, fresh=False),
            partial(_node_for, expand, rollout, select),
            _backprop,
            initial=SearchTree[C](),
            iterations=rounds,
            depth=depth,
            run_id=run_id,
            grantor=grantor,
        )
    )


def _node_for[C](
    expand: Expand[C], rollout: Rollout[C], select: Select, tree: SearchTree[C]
) -> Callable[[_At[C], Level], Effect[Decision[_At[C], Rollouts[C]]]]:
    def node(at: _At[C], level: Level) -> Effect[Decision[_At[C], Rollouts[C]]]:
        decide = partial(_decide, expand, rollout, select, tree, at, level)
        return (yield from _under_path(at.path, decide))

    return node


def _under_path[T](path: Path, body: Callable[[], Effect[T]]) -> Effect[T]:
    for label in reversed(path):
        body = partial(scoped, compose_key(t"node:{Name(label)}"), body)
    return (yield from body())


def _decide[C](
    expand: Expand[C],
    rollout: Rollout[C],
    select: Select,
    tree: SearchTree[C],
    at: _At[C],
    level: Level,
) -> Effect[Decision[_At[C], Rollouts[C]]]:
    if at.fresh or level.final:
        return Answered(Rollouts(((at.path, (yield from rollout(at.context))),), {}))
    match tree.children.get(at.path):
        case None:
            children = yield from expand(at.context)
            _refuse_unusable_labels(children)
            if not children:
                score = yield from rollout(at.context)
                return Answered(Rollouts(((at.path, score),), {at.path: children}))
            fresh = [_At((*at.path, label), child, True) for label, child in children.items()]
            return Branch(fresh, partial(_join, at.path, children))
        case terminal if not terminal:
            return Answered(Rollouts(((at.path, (yield from rollout(at.context))),), {}))
        case expanded:
            labels = list(expanded)
            visits = [tree.stats.get((*at.path, label), Stats()) for label in labels]
            parent = tree.stats.get(at.path, Stats()).visits
            label = labels[select(visits, parent)]
            return Deeper(_At((*at.path, label), expanded[label], fresh=False))


class UnusableNodeLabel(CompositionRefused):
    """A node's name cannot record its results apart: an `mcts` child label or a `frontier` key
    that cannot be a key coordinate, a `frontier` key of more than one term, or two distinct
    nodes in one step under one name. A retry names them the same, so its task fails once."""


def _refuse_unusable_labels(children: Mapping[str, object]) -> None:
    for label in children:
        try:
            compose_key(t"node:{Name(label)}")
        except ValueError as unusable:
            raise UnusableNodeLabel(f"node label {label!r} cannot be a key: {unusable}") from None


def _join[C](
    path: Path, children: Mapping[str, C], values: Sequence[Rollouts[C]]
) -> Effect[Rollouts[C]]:
    yield from ()
    scores = tuple(score for value in values for score in value.scores)
    expanded = {path: children}
    for value in values:
        expanded |= value.expanded
    return Rollouts(scores, expanded)


def _backprop[C](tree: SearchTree[C], value: Rollouts[C]) -> SearchTree[C]:
    stats = dict(tree.stats)
    for path, score in value.scores:
        for n in range(len(path) + 1):
            prior = stats.get(path[:n], Stats())
            stats[path[:n]] = Stats(prior.visits + 1, prior.total + score)
    return replace(tree, stats=stats, children={**tree.children, **value.expanded})


type Scored[C] = tuple[C, float]


@dataclass(frozen=True)
class Round[N, E, A]:
    """What one step learned: the value it was handed, the queued nodes it did not pick, each
    picked node with its expansion, and, when the search scores, every child with its score."""

    value: A
    rest: Sequence[N]
    expanded: Sequence[tuple[N, E]]
    scored: Sequence[Scored[N]] = ()


@dataclass(frozen=True)
class Step[N, A]:
    """What the next step starts from. An empty queue or `done` ends the search with `value`."""

    queue: Sequence[N]
    value: A
    done: bool = False


def everything[N, A](queue: Sequence[N], value: A) -> Sequence[int]:
    """Every position in the queue, in queue order."""
    return range(len(queue))


class UnsoundFrontier(CompositionRefused):
    """A frontier that could not keep each node's recorded results its own: a `pick` of its own
    without a `key`, or `score` and `children` without each other. Its task fails once."""


class UnusablePick(CompositionRefused):
    """`pick` named no position of a non-empty queue, one outside it, or one twice. `pick` is
    pure, so a retry picks the same and its task fails once."""


def frontier[N, E, A](
    seeds: Sequence[N],
    expand: Callable[[N], Effect[E]],
    merge: Callable[[Round[N, E, A]], Step[N, A]],
    *,
    initial: A,
    steps: int,
    pick: Callable[[Sequence[N], A], Sequence[int]] = everything,
    key: Callable[[N], str | Key] | None = None,
    children: Callable[[E], Sequence[N]] | None = None,
    score: Callable[[N], Effect[float]] | None = None,
    run_id: str | None = None,
    grantor: Grantor | None = None,
) -> Effect[A]:
    """Search from `seeds` for up to `steps` expansions, and return the value the last step left.

    A step picks positions in the queue, expands those nodes together, and, with `score`, scores
    together every child that `children` finds in the expansions, in the order they list them.
    `merge` turns the step's `Round` into the next `Step`; `pick` and `merge` are pure, so replay
    re-derives the queue and the value from recorded results.

    | search     | `pick`       | `children`    | `score`    | `merge` keeps                     |
    |------------|--------------|---------------|------------|-----------------------------------|
    | beam       | `everything` | the expansion | the child  | the top `width` children          |
    | pruned     | `everything` | none          | none       | children whose bound beats `best` |
    | best-first | the top k    | the new nodes | the node   | the rest and the new children     |

    `key` names a node, and each node's expansion and score are recorded under its name, so a node
    replays its own results whatever order a `pick` puts it in: a `str` is the coordinate of
    `node:{key}`, and a one-term `Key` is scoped under `node`. Two distinct nodes in one step may
    not share a name. Without `key` a result is recorded by its place in the step, which
    is sound only for `everything` while the queue's order is itself stable across a redeploy, so
    any other `pick` needs a `key`.

    A step whose expansions have no children skips scoring. A ``BudgetRefused`` inside a step ends
    the search with the value that step was handed, and the final step answers without expanding.
    An empty queue ends the search before a step runs.
    `run_id` and `grantor` refill steps as they refill `descend`'s levels."""
    if (score is None) != (children is None):
        raise UnsoundFrontier("frontier: score and children come together, or neither")
    if pick is not everything and key is None:
        raise UnsoundFrontier(
            "frontier: a pick that orders the nodes needs a key naming each node"
        )
    judge = partial(_frontier_step, _Policy(expand, merge, pick, key, children, score))
    start = Step(tuple(seeds), initial)
    return (yield from descend(start, judge, budget=steps, run_id=run_id, grantor=grantor))


@dataclass(frozen=True)
class _Policy[N, E, A]:
    expand: Callable[[N], Effect[E]]
    merge: Callable[[Round[N, E, A]], Step[N, A]]
    pick: Callable[[Sequence[N], A], Sequence[int]]
    key: Callable[[N], str | Key] | None
    children: Callable[[E], Sequence[N]] | None
    score: Callable[[N], Effect[float]] | None


def _frontier_step[N, E, A](
    policy: _Policy[N, E, A], step: Step[N, A], level: Level
) -> Effect[Answered[A] | Deeper[Step[N, A]]]:
    if level.final or not step.queue:
        return Answered(step.value)
    match (yield from within_budget(partial(_advance, policy, step))):
        case None:
            return Answered(step.value)
        case (following,) if following.done or not following.queue:
            return Answered(following.value)
        case (following,):
            return Deeper(following)


def _advance[N, E, A](policy: _Policy[N, E, A], step: Step[N, A]) -> Effect[Step[N, A]]:
    positions = list(policy.pick(step.queue, step.value))
    _refuse_unusable_pick(positions, len(step.queue))
    picked = [step.queue[i] for i in positions]
    chosen = set(positions)
    rest = [node for i, node in enumerate(step.queue) if i not in chosen]
    expansions = yield from gather(_each(policy.key, policy.expand, picked))
    expanded = list(zip(picked, expansions, strict=True))
    scored: list[Scored[N]] = []
    children, score = policy.children, policy.score
    if children is not None and score is not None and (found := _found(children, expansions)):
        scores = yield from gather(_each(policy.key, score, found))
        scored = list(zip(found, scores, strict=True))
    return policy.merge(Round(step.value, rest, expanded, scored))


def _found[N, E](children: Callable[[E], Sequence[N]], expansions: Sequence[E]) -> list[N]:
    return [child for expansion in expansions for child in children(expansion)]


def _each[N, T](
    key: Callable[[N], str | Key] | None, body: Callable[[N], Effect[T]], nodes: Sequence[N]
) -> list[Callable[[], Effect[T]]]:
    """One branch per node: under the node's name when there is a key, else named by position."""
    if key is None:
        return [partial(body, node) for node in nodes]
    return [
        _under(scopes, partial(body, node))
        for scopes, node in zip(_node_scopes(key, nodes), nodes, strict=True)
    ]


def _under[T](scopes: Sequence[Key], body: Callable[[], Effect[T]]) -> Callable[[], Effect[T]]:
    """`body` nested under each of `scopes`, the first outermost."""
    for scope in reversed(scopes):
        body = partial(scoped, scope, body)
    return body


def _node_scopes[N](key: Callable[[N], str | Key], nodes: Sequence[N]) -> list[tuple[Key, ...]]:
    scopes = [_node_scope(key(node)) for node in nodes]
    owner: dict[tuple[str, ...], N] = {}
    for path, node in zip(scopes, nodes, strict=True):
        name = tuple(scope.stored() for scope in path)
        if owner.setdefault(name, node) != node:
            raise UnusableNodeLabel(f"two nodes in one step share the name {';'.join(name)!r}")
    return scopes


NODE = compose_key(t"node")


def _node_scope(label: str | Key) -> tuple[Key, ...]:
    """The scopes a node's ops run under: `node:{label}` for a str, and `node` then the key for a
    `Key`, which must be one term, since a scope is one term."""
    match label:
        case Key() if len(label.terms()) == 1:
            return (NODE, label)
        case Key():
            raise UnusableNodeLabel(f"node key {label.stored()!r} is not one term")
        case str():
            _refuse_unusable_labels({label: None})
            return (compose_key(t"node:{Name(label)}"),)
        case unreachable:
            assert_never(unreachable)


def _refuse_unusable_pick(positions: Sequence[int], size: int) -> None:
    if not positions:
        raise UnusablePick(f"pick named no position of a queue of {size}")
    if len(set(positions)) < len(positions):
        raise UnusablePick(f"pick named a position twice: {positions}")
    if outside := [i for i in positions if not 0 <= i < size]:
        raise UnusablePick(f"pick named {outside}, outside a queue of {size}")


def beam[C](
    roots: Sequence[C],
    expand: Callable[[C], Effect[Sequence[C]]],
    score: Callable[[C], Effect[float]],
    *,
    width: int,
    depth: int,
) -> Effect[list[Scored[C]]]:
    """Expand the `width` best candidates for up to `depth` levels, and return the last frontier,
    best first: a `frontier` search that expands every queued candidate and keeps the `width`
    highest children, the earliest child on a tie. A level whose frontier has no children answers
    with that frontier.

    A ``BudgetRefused`` inside a level ends the search with the frontier that level was handed:
    a level that ran out mid-expansion has scored some of its children and not others, and a
    subset of a level is not a frontier. Scoring the roots is outside that stop, so a refusal
    there leaves no frontier to answer with and is raised on."""
    if width < 1:
        raise ValueError(f"beam width must be >= 1 (got {width})")
    top = _top((yield from _score_all(score, roots)), width)

    def keep(step: Round[C, Sequence[C], list[Scored[C]]]) -> Step[C, list[Scored[C]]]:
        return _keep_top(width, step)

    seeds = [candidate for candidate, _ in top]
    return (
        yield from frontier(
            seeds, expand, keep, initial=top, steps=depth, children=_itself, score=score
        )
    )


def _itself[C](expansion: Sequence[C]) -> Sequence[C]:
    return expansion


def _keep_top[C](
    width: int, step: Round[C, Sequence[C], list[Scored[C]]]
) -> Step[C, list[Scored[C]]]:
    if not step.scored:
        return Step((), step.value)
    top = _top(list(step.scored), width)
    return Step([candidate for candidate, _ in top], top)


def _score_all[C](
    score: Callable[[C], Effect[float]], candidates: Sequence[C]
) -> Effect[list[Scored[C]]]:
    scores = yield from gather([partial(score, candidate) for candidate in candidates])
    return list(zip(candidates, scores, strict=True))


def _top[C](scored: list[Scored[C]], width: int) -> list[Scored[C]]:
    ranked = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
    return [scored[i] for i in ranked[:width]]
