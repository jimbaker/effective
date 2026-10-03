"""RLM combinators ``recurse``, ``route``, ``descend`` and ``hoisted``, and the recursion shapes
``unfold``, ``tree_search``, ``fixpoint`` and ``fix``.

Thin sugar over ``gather`` + ``activate_skill`` + caller-supplied effects
(``run_agent``, ``run_code``, ``ask_llm`` leaves), with four correctness rules
**encoded structurally**:

1. ``recurse``: ``decompose`` runs (and checkpoints) *before* the gather, so the
   data-dependent branch count recovers from recorded state on replay; ``combine``
   is applied as a **balanced tree-fold of gathers**, because the combine step has
   its own context rot at large N. A flat reduce is not writable through this sugar.
2. ``route``: uniform dispatch over ``Effect[T]`` values (recurse deeper, run a
   pinned skill script, run improvised code, answer flat), with an unknown label a
   loud ``LookupError``. The classifier is a sealed op, so the taken path is
   recorded and replay re-dispatches identically.
3. ``descend``: adaptive depth as a *linear drill*: fan-out is ``recurse``'s job,
   and the drill answers one narrowing question at a time. Termination is the
   judge's confidence gate OR budget exhaustion as a **durable park** (the cost
   boundary is a permission boundary). A park also composes *inside* a gather
   branch; the drill is written as ``yield from`` self-recursion, and the
   gather-qualified grant-name contract that composition implies is pinned
   (``descend``'s docstring; ``effective.api.qualified_event_name``).
4. ``hoisted``: pin-hoisting. Skill activation is value-independent, so it happens
   ONCE, above whatever fan-out ``body`` performs, and the pins thread in
   explicitly; ``activate_skill`` inside a branch would re-activate per branch.

Callbacks name their ops **plainly**. The sugar places each one inside a
``scoped(...)`` (``rec:{i}``, ``fold:{level},{k}``, ``d:{depth}``) and the handler
applies the prefix, in the same position it applies a gather's ``gather:{g},{i};``.
So no callback receives, threads, or splices a namespace; a checkpoint key is never
hand-built, and no call site decides a delimiter.

The combinators add no replay machinery: durability (checkpoint keys,
crash-resume, cascade gating) is carried entirely by the ops underneath.
"""

import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, assert_never

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from effective.api import Effect, await_event, gather, respawn_generation, scoped
from effective.budget import (
    BUDGET_DEPTH_PARAM,
    BUDGET_GENERATIONS_PARAM,
    Budget,
    Grant,
    chain_grant_name,
    depth_grant_name,
    round_grant_name,
)
from effective.govern import BudgetRefused
from effective.keys import Index, Key, Name, compose_key
from effective.ops import (
    CARRY_PARAM,
    CHAIN_DEPTH,
    CHAIN_GENERATION,
    GENERATION_PARAM,
    CompositionRefused,
    current_generation,
    refuse_nested_respawn,
)
from effective.skills import Pin, activate_skill
from effective.spawning import join_answer, spawn_child


def hoisted[T](
    skills: Sequence[str],
    body: Callable[[Mapping[str, Pin]], Effect[T]],
) -> Effect[T]:
    """Activate each skill once — above ``body``'s fan-out — and thread the pins.

    Activation is value-independent (rule 4), so it belongs outside any
    ``gather``: ``hoisted(("slicing",), lambda pins: recurse(...))`` gives every
    branch the same recorded ``Pin`` (replay-exact scripts via
    ``pins[name].script(...)``) for the cost of one disclosure. Duplicate names
    activate once each in ``skills`` order.
    """
    pins: dict[str, Pin] = {}
    for name in skills:
        if name not in pins:
            pins[name] = yield from activate_skill(name)
    result = yield from body(pins)
    return result


def recurse[C, T](
    ctx: C,
    decompose: Callable[[C], Effect[Sequence[C]]],
    leaf: Callable[[C], Effect[T]],
    combine: Callable[[Sequence[T]], Effect[T]],
    *,
    fanin: int = 4,
) -> Effect[T]:
    """Map-reduce over an oversized context: split, process per chunk, tree-merge.

    ``decompose`` (a sealed op: an LM call, a pure slice, or a ``run_code``)
    yields the chunks — sequenced strictly BEFORE the fan-out, so the branch
    count is recovered from recorded state on replay (rule 1a; a crash mid
    fan-out resumes with the same branch structure). Each ``leaf`` runs as its
    own ``gather`` branch inside ``scoped(compose_key(t"rec:{Index(i)}"))`` — per-branch
    crash-resume for free: committed leaves are never re-paid. ``combine``
    merges ``fanin``-sized groups in a balanced tree-fold of gathers under
    ``fold:{level},{k}`` (rule 1b); singleton groups pass through
    without a combine call. The durable ``gather`` prefix (``gather:{g},{i};``)
    namespaces every branch structurally on top of that, so a leaf's key is
    ``gather:{g},{i};rec:{i};…`` — two frames, neither of them authored.

    A ``leaf``/``combine`` may park durably (a ``human`` cascade tier, an
    in-branch ``await_event``): the await-in-gather park parks the whole task on
    the branch's fully-qualified event. Emitters must use that qualified name, composed with
    ``effective.api.qualified_event_name``, and the run-id-in-the-name rule still
    applies (neither the gather prefix nor a scope run-scopes).
    """
    if fanin < 2:
        raise ValueError(f"recurse fanin must be >= 2 (got {fanin})")
    chunks = list((yield from decompose(ctx)))
    if not chunks:
        raise ValueError("recurse: decompose produced no chunks")
    values: list[T] = yield from gather(
        [
            lambda c=chunk, i=i: scoped(compose_key(t"rec:{Index(i)}"), lambda: leaf(c))
            for i, chunk in enumerate(chunks)
        ]
    )
    level = 0
    while len(values) > 1:
        groups = [values[k : k + fanin] for k in range(0, len(values), fanin)]
        merged: list[T] = yield from gather(
            [
                lambda g=tuple(group), lv=level, k=k: scoped(
                    compose_key(t"fold:{Index(lv)},{Index(k)}"), lambda: combine(g)
                )
                for k, group in enumerate(groups)
                if len(group) > 1
            ]
        )
        folded = iter(merged)
        values = [group[0] if len(group) == 1 else next(folded) for group in groups]
        level += 1
    return values[0]


def route[C, T](
    chunk: C,
    classifier: Callable[[C], Effect[str]],
    handlers: Mapping[str, Callable[[C], Effect[T]]],
) -> Effect[T]:
    """Uniform dispatch — the skills join (rule 2, the thing ``dspy.RLM``
    structurally cannot express: its tool set is frozen at construction).

    Inside ``route`` every handler is a value of one type, ``Effect[T]``:
    recurse deeper (a nested ``recurse`` / ``run_agent``), run a pinned skill
    script (``run_code(pin.script(...))`` — zero marginal exploration tokens),
    run improvised code (metered — no scope frame, so its segments key as
    ``step;code:seg,{n},{name}``), or answer flat (a plain
    ``ask_llm``). ``classifier`` is a sealed op returning a handler label, so
    which path each chunk took is checkpoint-recorded and replay re-dispatches
    identically; the improvised-vs-pinned rate per label is the promotion /
    Pareto signal. An unknown label fails loudly: misrouting is a dev error.
    """
    label = yield from classifier(chunk)
    handler = handlers.get(label)
    if handler is None:
        raise LookupError(
            f"route: classifier chose {label!r}, not a handler (has {sorted(handlers)})"
        )
    result = yield from handler(chunk)
    return result


@dataclass(frozen=True)
class Level:
    """Where a ``descend`` judge or an ``unfold`` node stands, handed to it each level.

    Both drivers run each level inside ``scoped(compose_key(t"d:{Index(depth)}"))``, so a judge
    names its ops plainly and the handler places them. ``model`` is the ``promote`` choice for
    this depth, the cheap-model-deep, expensive-model-shallow knob; it is empty when unpromoted.
    ``final`` means the budget and any grants are exhausted, so the judge must answer: the
    ``run_agent`` max-iters nudge, lifted to depth."""

    depth: int
    model: str
    final: bool


@dataclass(frozen=True)
class Answered[T]:
    """The judge's confidence gate passed: stop here with ``value``."""

    value: T


@dataclass(frozen=True)
class Deeper[C]:
    """The judge narrowed the context: descend into ``context``."""

    context: C


class DescendedPastBudget(CompositionRefused):
    """A judge or node at its final level answered with a descent instead of a value.

    The final level is the last the budget grants, and replay serves the same verdict to every
    retry, so the refusal is deterministic."""


# --- the grantor seam: who answers a budget park, and with how much more ---------------

type GrantTier = Callable[[int], Effect[Grant | None]]
type Grantor = Callable[[int], Effect[Grant]]


def _generation() -> int:
    """Which generation of a `respawn` chain this grant park belongs to.

    Read at BOTH answerers, `refill_levels`' built-in arm and `human_grant`'s tier, because
    "two answerers, one park" is the design: they compose the same name by calling the same
    function, and a coordinate only one of them applied would put them back into disagreement.

    The rule generalized past `descend` when `govern:` and `approve:` grew the same coordinate, so
    the body moved to `ops.current_generation()` and this name is the local spelling of it. Two
    answerers was already the reason not to inline the read; four is the reason it left this
    module."""
    return current_generation()


def human_grant(run_id: str, schema: type[Grant] = Grant) -> GrantTier:
    """A grant tier that parks for a human and returns their ``Grant`` — the
    ``cascade([human])`` answerer for a budget park, the ``Grant``-typed analog of
    the permission ``human`` tier (``effective.permission.human``).

    Parks on ``depth_grant_name(run_id, depth)`` — the *same* name ``descend``'s built-in arm
    composes, which is the point: two answerers, one park. Any enclosing scope or gather branch
    prefixes the name handler-side, invisible here; an emitter composes the qualified
    form with ``effective.api.qualified_event_name``. Always decisive (a human
    answers), so it never escalates."""

    def tier(depth: int) -> Effect[Grant | None]:
        grant = yield from await_event(
            depth_grant_name(run_id, depth=depth, generation=_generation()), schema
        )
        return grant

    return tier


def grant_cascade(tiers: list[GrantTier], *, default: Grant | None = None) -> Grantor:
    """Compose grant ``tiers`` — the first to return a ``Grant`` wins, ``None``
    escalates to the next, and all-escalate falls to ``default`` (``Grant()`` = add
    nothing = answer with what you have now).

    Rhymes with ``effective.permission.cascade``'s first-decisive-tier structure, but
    over ``Grant`` (a *quantity* — "how much more budget") rather than ``Verdict``
    (allow / deny an op): a budget park asks *how much more*, not *whether*. The shipped
    cascade is ``[human_grant(...)]``; a deterministic ``rules`` grant tier or a ``fork``
    grantor is another entry in the same list."""
    resolved = default if default is not None else Grant()

    def grantor(depth: int) -> Effect[Grant]:
        for tier in tiers:
            grant = yield from tier(depth)
            if grant is not None:
                return grant
        return resolved

    return grantor


def granted_levels(grant: Grant) -> int:
    """How many more levels a budget park's resolution authorizes; 0 = answer now.

    ``stop`` forces 0, so a human's "answer with what you have" is honored regardless of
    ``add_depth``."""
    return 0 if grant.stop else grant.add_depth


def refill_rounds(rounds: int, *, run_id: str | None, grantor: Grantor | None) -> Effect[int]:
    """After `rounds` rounds, how many more a search may run: the `grantor`'s `add_rounds`, the
    built-in park on `round_grant_name`'s, or 0 when neither is set. `stop` grants 0."""
    if grantor is not None:
        grant = yield from grantor(rounds)
    elif run_id is not None:
        grant = yield from await_event(
            round_grant_name(run_id, generation=_generation(), rounds=rounds), Grant
        )
    else:
        return 0
    return 0 if grant.stop else grant.add_rounds


def refill_levels(depth: int, *, run_id: str | None, grantor: Grantor | None) -> Effect[int]:
    """At an exhausted budget, how many more levels a run may take: the ``grantor``'s answer, the
    built-in park on ``depth_grant_name(run_id, depth)``'s, or 0 when neither is set.

    The park is placed by the frames around the call, and a caller makes that call outside the
    level's own scope, so the name does not move as a drill deepens."""
    if grantor is not None:
        grant = yield from grantor(depth)
    elif run_id is not None:
        grant = yield from await_event(
            depth_grant_name(run_id, depth=depth, generation=_generation()), Grant
        )
    else:
        return 0
    return granted_levels(grant)


def descend[C, T](
    ctx: C,
    judge: Callable[[C, Level], Effect[Answered[T] | Deeper[C]]],
    *,
    budget: int,
    promote: Callable[[int], str] | None = None,
    run_id: str | None = None,
    grantor: Grantor | None = None,
) -> Effect[T]:
    """Adaptive depth under governance (rule 3): a linear drill, level by level.

    At each level ``judge`` decides answer-here vs recurse-deeper on a narrowed
    context, inside ``scoped(compose_key(t"d:{Index(depth)}"))``. ``budget`` bounds the
    levels; on exhaustion, with ``run_id`` set, the descent **parks durably** on
    ``depth_grant_name(run_id, depth)`` — the cost boundary as a permission boundary: a human
    grants more levels or answers 0 to demand the best answer now. The park sits at the
    descent's own scope, NOT inside the level's, so its name does not move as the drill deepens.

    On the durable engine event names are GLOBAL, which is what ``run_id`` is for — the same
    rule as ``human()`` approval events. Every enclosing frame (a ``recurse`` leaf's gather
    branch, any surrounding ``scoped``) prefixes the name handler-side and is invisible at this
    call site; an emitter composes the qualified form with
    ``effective.api.qualified_event_name``. Without ``run_id`` (or after a 0-level grant) the
    judge gets one ``final=True`` level and must answer; descending past it is a loud dev error.

    ``grantor`` is the swappable-answerer generalization: instead of parking on the built-in
    name it takes a ``Grantor`` — e.g. ``grant_cascade([human_grant(run_id)])`` — that answers
    with a ``Grant`` whose ``add_depth`` refills the level budget (``add_depth=0`` / ``stop`` =
    answer now). Pass ``run_id`` OR ``grantor``, never both; ``human_grant`` composes the same
    park name, so the two paths agree by construction rather than by contract.

    The recurse-or-answer *marginal* question ("was this level worth its cost?") belongs to
    ``fork``; ``promote`` maps depth to a model tag the judge applies, each depth's model a
    metered knob on the Pareto surface.

    A judge is an ``unfold`` node that never branches, and ``descend`` is that ``unfold``.
    """
    return (
        yield from unfold(
            ctx, judge, budget=budget, promote=promote, run_id=run_id, grantor=grantor
        )
    )


# --- the recursion shapes: one node decision, grown by a driver or closed by `fix` ---------


@dataclass(frozen=True)
class Branch[C, T]:
    """The node split its context: unfold every child one level deeper, then ``join`` their
    values in child order."""

    children: Sequence[C]
    join: Callable[[Sequence[T]], Effect[T]]


type Decision[C, T] = Answered[T] | Deeper[C] | Branch[C, T]
type Node[C, T] = Callable[[C, Level], Effect[Decision[C, T]]]


@dataclass(frozen=True)
class InTask:
    """A ``Branch``'s children run as ``gather`` branches of the task that branched."""


CONTEXT_PARAM = "unfold_context"
DEPTH_PARAM = "unfold_depth"


@dataclass(frozen=True)
class AcrossTasks[C, T]:
    """A ``Branch``'s children run as tasks named ``task``, each its own durable run.

    The task's body runs ``unfold_task`` from its params under ``spawning.run_child``. A child's
    levels are its spawn depth, ``BUDGET_DEPTH_PARAM``, so a node at its final level is one whose
    spawns the handler would refuse, and a task whose own depth is below the ``unfold`` budget
    grows a shallower tree. ``context`` and ``result`` carry a context and a value across
    the task boundary as JSON."""

    task: str
    context: TypeAdapter[C]
    result: TypeAdapter[T]

    def params(self, context: C, *, depth: int, levels: int) -> dict[str, Any]:
        """What a task unfolding ``context`` at ``depth`` with ``levels`` to go starts with."""
        return {
            CONTEXT_PARAM: self.context.dump_python(context, mode="json"),
            DEPTH_PARAM: depth,
            BUDGET_DEPTH_PARAM: levels,
        }


type Crossing[C, T] = InTask | AcrossTasks[C, T]

IN_TASK = InTask()


def unfold[C, T](
    root: C,
    node: Node[C, T],
    *,
    budget: int,
    promote: Callable[[int], str] | None = None,
    run_id: str | None = None,
    grantor: Grantor | None = None,
    crossing: Crossing[C, T] = IN_TASK,
) -> Effect[T]:
    """Grow a recursion from ``root`` one node decision at a time, within ``budget`` levels.

    Each level runs ``node`` inside ``scoped(compose_key(t"d:{Index(depth)}"))``, handed a
    ``Level`` whose ``model`` is ``promote(depth)``. ``Answered`` returns its value; ``Deeper``
    trampolines to the next level, so a linear chain does not grow the stack; ``Branch`` grows
    each child one level deeper under ``rec:{i}``, then joins. ``crossing`` says where a child
    grows: as a ``gather`` branch of this task, or as a spawned task (``AcrossTasks``), all
    spawned before any is joined. A ``descend`` judge is a node that never branches.

    The budget counts levels along a path. When a path exhausts it, ``refill_levels`` asks
    ``grantor``, or parks on ``depth_grant_name(run_id, depth)``, outside the level's scope; with
    neither, or a grant of 0, the level is final, so a budget of 0 is one final level. A node that
    does not answer at a final level raises ``DescendedPastBudget``. Each child of a ``Branch``
    carries its own remainder, so children that exhaust together park once each; a caller collapses
    those decisions in a ``grantor`` tier. Event names are global on the durable engine, which is
    what ``run_id`` is for; pass ``run_id`` or ``grantor``, not both. A refill grants levels and no
    spawn depth, so neither crosses a task boundary.

    A chain of single children is ``Deeper``: each in-task ``Branch`` level holds a thread and a
    nested ``gather``, so a chain of them is bounded by the process's recursion limit. Two
    ``unfold``s in one scope both name their root level ``d:0``, so each takes its own
    ``scoped``."""
    if budget < 0:
        raise ValueError(f"unfold budget must be >= 0 (got {budget})")
    if run_id is not None and grantor is not None:
        raise ValueError("unfold: pass run_id OR grantor, not both")
    match crossing:
        case AcrossTasks() if run_id is not None or grantor is not None:
            raise ValueError("unfold: a refill grants no spawn depth, so it cannot cross tasks")
        case InTask() | AcrossTasks():
            pass
        case unreachable:
            assert_never(unreachable)
    grow = _grower(node, promote=promote, run_id=run_id, grantor=grantor, crossing=crossing)
    return (yield from grow(root, 0, budget))


def unfold_task[C, T](
    params: Mapping[str, Any],
    node: Node[C, T],
    *,
    crossing: AcrossTasks[C, T],
    promote: Callable[[int], str] | None = None,
) -> Effect[T]:
    """The ``unfold`` a task spawned by ``crossing`` runs, from the context, depth and levels its
    params carry (``AcrossTasks.params``). A root task may start one the same way."""
    levels = Budget.from_spawn_params(params).depth
    if levels is None:
        raise ValueError(f"unfold_task: the task's params carry no {BUDGET_DEPTH_PARAM}")
    context = crossing.context.validate_python(params[CONTEXT_PARAM])
    grow = _grower(node, promote=promote, run_id=None, grantor=None, crossing=crossing)
    return (yield from grow(context, int(params[DEPTH_PARAM]), levels))


def _grower[C, T](
    node: Node[C, T],
    *,
    promote: Callable[[int], str] | None,
    run_id: str | None,
    grantor: Grantor | None,
    crossing: Crossing[C, T],
) -> Callable[[C, int, int], Effect[T]]:
    """``unfold``'s driver: grow ``context`` from ``depth`` with ``remaining`` levels."""

    def grow(context: C, depth: int, remaining: int) -> Effect[T]:
        while True:
            if remaining == 0:
                remaining = yield from refill_levels(depth, run_id=run_id, grantor=grantor)
            final = remaining == 0
            model = promote(depth) if promote is not None else ""
            decision = yield from scoped(
                compose_key(t"d:{Index(depth)}"),
                partial(node, context, Level(depth, model, final)),
            )
            match decision:
                case Answered(value):
                    return value
                case Deeper(narrowed) if not final:
                    context, depth, remaining = narrowed, depth + 1, remaining - 1
                case Branch(children, join) if not final:
                    values = yield from _grow_children(
                        grow, crossing, children, depth + 1, remaining - 1
                    )
                    return (yield from join(values))
                case Deeper() | Branch():
                    raise DescendedPastBudget(
                        f"unfold: the node descended past an exhausted budget at depth {depth}"
                    )
                case unreachable:
                    assert_never(unreachable)

    return grow


def _grow_children[C, T](
    grow: Callable[[C, int, int], Effect[T]],
    crossing: Crossing[C, T],
    children: Sequence[C],
    depth: int,
    remaining: int,
) -> Effect[list[T]]:
    """Grow a ``Branch``'s children at ``depth`` under ``rec:{i}``, where ``crossing`` says."""
    match crossing:
        case InTask():
            return (
                yield from gather(
                    [
                        partial(
                            scoped,
                            compose_key(t"rec:{Index(i)}"),
                            partial(grow, child, depth, remaining),
                        )
                        for i, child in enumerate(children)
                    ]
                )
            )
        case AcrossTasks() as across:
            return (yield from _across_tasks(across, children, depth, remaining))
        case unreachable:
            assert_never(unreachable)


def _across_tasks[C, T](
    across: AcrossTasks[C, T], children: Sequence[C], depth: int, remaining: int
) -> Effect[list[T]]:
    """Spawn every child under ``rec:{i}``, then join them in order: a join in a gather branch is
    refused, and the children already run concurrently in their own tasks."""
    spawned = []
    for i, child in enumerate(children):
        params = across.params(child, depth=depth, levels=remaining)
        spawn = partial(spawn_child, across.task, "child", params)
        spawned.append((yield from scoped(compose_key(t"rec:{Index(i)}"), spawn)))
    values: list[T] = []
    for handle in spawned:
        values.append(across.result.validate_python((yield from join_answer(handle))))
    return values


def tree_search[S, C, T](
    root: C,
    node_for: Callable[[S], Node[C, T]],
    update: Callable[[S, T], S],
    *,
    initial: S,
    iterations: int,
    depth: int,
    run_id: str | None = None,
    grantor: Grantor | None = None,
) -> Effect[S]:
    """Search from ``root`` for ``iterations`` rounds, each an ``unfold`` guided by what the
    earlier rounds learned, and return the final state.

    When the granted rounds run out, ``refill_rounds`` asks ``grantor`` for ``add_rounds``, or
    parks on ``round_grant_name`` with ``run_id``, outside every round's scope; a second search's
    park in one run carries an occurrence (``#2``), which an emitter reads off the parked name. A
    ``BudgetRefused`` out of that ask, or out of a round, bare or among its branches' refusals,
    ends the search with the state before it; the round's other refusals are raised on.

    Round ``k`` runs ``unfold(root, node_for(state), budget=depth)`` inside
    ``scoped(compose_key(t"search:{Index(k)}"))``, then folds its value in with ``update``. The
    node selects by ``Deeper`` and expands by ``Branch``, and ``update`` is the backpropagation,
    a pure function of recorded values, so replay re-derives every round's state. Each round has
    its own scope, so its ops, ledger rows, parks and spawns never share a name with another round
    of the same search. Two ``tree_search``es in one scope both name their first round
    ``search:0``, so each takes its own ``scoped``."""
    if iterations < 1:
        raise ValueError(f"tree_search iterations must be >= 1 (got {iterations})")
    if run_id is not None and grantor is not None:
        raise ValueError("tree_search: pass run_id OR grantor, not both")
    state, rounds, granted = initial, 0, iterations
    while True:
        if rounds == granted:
            ask = partial(refill_rounds, rounds, run_id=run_id, grantor=grantor)
            match (yield from within_budget(ask)):
                case (0,) | None:
                    return state
                case (more,):
                    granted += more
                case unreachable:
                    assert_never(unreachable)
        grow = partial(unfold, root, node_for(state), budget=depth)
        match (
            yield from within_budget(partial(scoped, compose_key(t"search:{Index(rounds)}"), grow))
        ):
            case (value,):
                state, rounds = update(state, value), rounds + 1
            case None:
                return state


def within_budget[T](body: Callable[[], Effect[T]]) -> Effect[tuple[T] | None]:
    """``(value,)`` from ``body``, or ``None`` when a ``BudgetRefused`` stopped it.

    A tuple rather than the value, because a body that answers ``None`` is not a refusal. What a
    stopped body answers with is the caller's, which is why this returns the fact rather than a
    default."""
    try:
        return ((yield from body()),)
    except* BudgetRefused:
        pass
    return None


@dataclass(frozen=True)
class Converged[C]:
    """``converged`` accepted a step's result against its input; ``value`` is the result."""

    value: C


@dataclass(frozen=True)
class Unconverged[C]:
    """A budget stopped ``fixpoint`` before a step converged, holding the last value a step
    returned whole, or ``initial`` when none has.

    | cause        | what ran out               | the recourse         |
    |--------------|----------------------------|----------------------|
    | `"levels"`   | ``budget`` and every grant | more levels, granted |
    | `"governed"` | a governed op's spend      | the governor         |

    A step the governor stopped has run part of its ops, and those effects stand."""

    value: C
    cause: Literal["levels", "governed"]


type Settled[C] = Converged[C] | Unconverged[C]
"""What ``fixpoint`` answers: a value ``converged`` accepted, or the one a budget stopped at."""


@dataclass
class _GovernedGrant:
    """``fixpoint``'s ask for more levels, made under ``within_budget``: an ask refused for its
    spend answers a stop and is remembered, so the final level can say which budget ran out."""

    run_id: str | None
    grantor: Grantor | None
    refused: bool = False

    def __call__(self, depth: int) -> Effect[Grant]:
        ask = partial(refill_levels, depth, run_id=self.run_id, grantor=self.grantor)
        match (yield from within_budget(ask)):
            case None:
                self.refused = True
                return Grant(stop=True)
            case (levels,):
                return Grant(add_depth=levels)
            case unreachable:
                assert_never(unreachable)


def fixpoint[C](
    initial: C,
    step: Callable[[C], Effect[C]],
    *,
    budget: int,
    converged: Callable[[C, C], bool] = operator.eq,
    run_id: str | None = None,
    grantor: Grantor | None = None,
) -> Effect[Settled[C]]:
    """Apply ``step`` until ``converged(value, step(value))``, at most ``budget`` times without a
    grant.

    Each application is a ``descend`` level under ``d:{n}``, so the loop runs in constant stack
    and a grant extends it as it extends a descent. The ask for more levels is governed like a
    step: refused for its spend, it ends the loop ``"governed"``. A budget refusal inside a step
    is caught whole or as a member of a group; any other refusal in the group is raised. Two
    ``fixpoint`` calls in one scope both start at ``d:0``, so each takes its own ``scoped``.

    A step that runs a ``fixpoint`` gets its ``Unconverged`` back as a value, so a step returning
    the inner value unchecked can hand the outer loop the same value twice, which ``converged``
    reads as convergence."""
    if run_id is not None and grantor is not None:
        raise ValueError("fixpoint: pass run_id OR grantor, not both")
    ask = _GovernedGrant(run_id, grantor)

    def settle(result: Settled[C]) -> Answered[Settled[C]]:
        return Answered(result)

    def judge(value: C, level: Level) -> Effect[Answered[Settled[C]] | Deeper[C]]:
        if level.final:
            return settle(Unconverged(value, "governed" if ask.refused else "levels"))
        match (yield from within_budget(partial(step, value))):
            case None:
                return settle(Unconverged(value, "governed"))
            case (stepped,) if converged(value, stepped):
                return settle(Converged(stepped))
            case (stepped,):
                return Deeper(stepped)
            case unreachable:
                assert_never(unreachable)

    return (yield from descend(initial, judge, budget=budget, grantor=ask))


@dataclass(frozen=True)
class Hand[K: str, V]:
    """A role hands ``value`` to ``role`` and ends: ``mutual``'s tail transition."""

    role: K
    value: V


type Role[K: str, V, T] = Callable[[V, Level], Effect[Answered[T] | Hand[K, V]]]


class UnusableRoleName(CompositionRefused):
    """A ``mutual`` role's name cannot be a key coordinate. A retry names it the same, so its task
    fails once."""


def _role_scopes[K: str](names: Iterable[K]) -> dict[K, Key]:
    scopes: dict[K, Key] = {}
    for name in names:
        try:
            scopes[name] = compose_key(t"state:{Name(name)}")
        except ValueError as unusable:
            raise UnusableRoleName(f"mutual: role {name!r} cannot be a key: {unusable}") from None
    return scopes


def mutual[K: str, V, T](
    start: Hand[K, V],
    roles: Mapping[K, Role[K, V, T]],
    *,
    budget: int,
    run_id: str | None = None,
    grantor: Grantor | None = None,
) -> Effect[T]:
    """Roles that call each other in tail position: each runs, then answers or hands off.

    Each hop is a ``descend`` level, the role running under ``d:{n};state:{role}``, so the calls
    run in constant stack. ``budget`` bounds the hands: without a grant ``budget + 1`` roles run,
    the last at its final level. Two ``mutual`` calls in one scope both start at ``d:0``, so each
    takes its own ``scoped``.

    | when                                  | it raises               | before         |
    |---------------------------------------|-------------------------|----------------|
    | a role name cannot be a key           | ``UnusableRoleName``    | any role runs  |
    | a ``Hand`` names no role in ``roles`` | ``CompositionRefused``  | that role runs |
    | a role hands off at its final level   | ``DescendedPastBudget`` | another level  |
    | a role's op is refused                | the refusal             |                |

    With role names typed as a ``Literal``, spell the start ``Hand[K, V](...)`` so its key is not
    widened to ``str``."""
    table = dict(roles)
    scopes = _role_scopes(table)

    def known(hand: Hand[K, V]) -> Hand[K, V]:
        if hand.role not in table:
            raise CompositionRefused(
                f"mutual: {hand.role!r} is handed a value and is not a role; the roles are "
                f"{sorted(table)}"
            )
        return hand

    def judge(hand: Hand[K, V], level: Level) -> Effect[Answered[T] | Deeper[Hand[K, V]]]:
        role = partial(table[hand.role], hand.value, level)
        match (yield from scoped(scopes[hand.role], role)):
            case Answered() as answered:
                return answered
            case Hand() as handed:
                return Deeper(known(handed))
            case unreachable:
                assert_never(unreachable)

    return (yield from descend(known(start), judge, budget=budget, run_id=run_id, grantor=grantor))


def fix[**P, T](
    open_body: Callable[[Callable[P, Effect[T]]], Callable[P, Effect[T]]],
) -> Callable[P, Effect[T]]:
    """The fixpoint of an open recursion: ``fix(f)(*args)`` runs ``f(fix(f))(*args)``.

    ``f`` takes its recursive call as an argument, so a shape is written once and the recursion is
    supplied where it is closed. The untyped combinator this spells is Z, the call-by-value Y::

        Z = λf. (λx. f (λv. x x v)) (λx. f (λv. x x v))

    The closure ties the knot by name, which ``ty`` checks and the self-application cannot."""

    def recur(*args: P.args, **kwargs: P.kwargs) -> Effect[T]:
        return open_body(recur)(*args, **kwargs)

    return recur


# --- respawn: the OUTER loop ----------------------------------------------------------------
#
# The third trampoline, and the one that is not like the others. `recurse`, `route` and
# `descend` compose INSIDE a task; `respawn` cuts it. Its column in the composition table is
# full (anything can sit inside a generation) and its row is empty (it sits inside nothing),
# which is the structural signature of an outer loop, and why `Respawn` needs a Step-side
# guard of its own rather than inheriting the await-side one.


@dataclass(frozen=True)
class Done[T]:
    """Terminal: the loop answers with ``value``."""

    value: T


@dataclass(frozen=True)
class Again[S]:
    """Transition: this generation ends here; ``state`` is what survives into the next.

    The marker names the TRANSITION, not the mechanism — which is why it is `Again` and not
    `Respawn`. One mechanism, one word: `respawn` appears at the call site and as
    the op. This answers a different question, asked once per generation — *another generation,
    or finish?* — and it reads correctly at the idle exit, where a poller goes round on an empty
    inbox and nothing was respawned in any interesting sense."""

    state: S


@dataclass(frozen=True)
class Turn:
    """Where a `respawn` chain stands — handed to the step function each generation.

    `descend`'s `Level` protocol, minus the scope. **`Turn` carries no `scope`**, for three
    reasons, the last decisive:

    1. Checkpoints need no generation scope: a generation is a NEW TASK with a new store, so two
       generations cannot collide.
    2. A scope invites splicing into an op name with an f-string (`f"next:{turn.scope}{i}"`),
       an identity built outside the key composer.
    3. *The cycle view under `generations=20` equals the cycle view under `generations=None`.*
       A per-generation scope in the keys would make generations visible in the program's shape.

    `final` exists because the combinator holds no `T`: when the chain may not go round again,
    *something* must be asked for an answer one last time."""

    generation: int
    final: bool


@dataclass(frozen=True)
class Chain[S]:
    """The plumbing a generation boundary needs, named once instead of hidden in a call site.

    A spawn needs the workflow's OWN registered task name (the substrate re-spawns the same
    program), the generation ordinal, the carry, and the params to re-thread. v1's one-argument
    call site hid all three; the honest call site is three lines, and it is still good — the
    same move `Budget.from_spawn_params` already makes for depth."""

    task: str
    state: S
    run_id: str
    generation: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_params[T](
        cls,
        params: Mapping[str, Any],
        *,
        task: str,
        schema: type[T],
        initial: T,
        run_id: str | None = None,
    ) -> Chain[T]:
        """Rebuild the chain from THIS task's spawn params — generation 0 or the n-th.

        `schema` validates the carry back off the wire: it crossed as JSON, so this is the
        boundary where it becomes a value again (the same `TypeAdapter` discipline checkpoints
        use). At generation 0 there is no carry and `initial` is it."""
        carried = params.get(CARRY_PARAM)
        return Chain(
            task=task,
            state=initial if carried is None else TypeAdapter(schema).validate_python(carried),
            run_id=run_id if run_id is not None else str(params.get("run_id", "")),
            generation=int(params.get(GENERATION_PARAM, 0)),
            params={k: v for k, v in params.items() if k not in (GENERATION_PARAM, CARRY_PARAM)},
        )


def _refill_generations(budget: Budget, generation: int) -> Effect[int]:
    """At exhaustion, ask for more generations — `descend`'s `refill`, on the chain axis.

    The park name carries the GENERATION (`chain-grant:{run_id}:{generation}`), and that is not
    decoration: without it a chain granted `add_generations=10` once would be granted forever,
    because the base name would be constant and an Absurd event is a durable fact answered from
    cache (the engine semantics are pinned by `tests/test_respawn_hazards_pending.py`). The chain
    axis has its own tag, apart from the depth axis's (`budget.depth_grant_name`).

    No `run_id` means no one to ask, so the chain gets one final turn and must answer —
    the same forced-final `descend` gives a drill with no grantor."""
    if budget.run_id is None or budget.on_exhaust != "park":
        return 0
    grant = yield from await_event(chain_grant_name(budget.run_id, generation), Grant)
    return 0 if grant.stop else grant.add_generations


def respawn[S, T](
    step: Callable[[S, Turn], Effect[Again[S] | Done[T]]],
    chain: Chain[S],
    *,
    budget: Budget | None = None,
) -> Effect[T]:
    """Run a long-lived loop as a CHAIN of tasks, so replay history stays bounded.

    Each generation runs `step` once; `Again(state)` ends this task and continues in a fresh one
    carrying `state`, `Done(value)` ends the chain. The verb names the strategy and the function
    argument is how, exactly as `descend(ctx, judge)` is "descend, using `judge` per level".

    **The prologue re-executes, and that is not a gotcha — it is what an outer loop IS.** Code
    textually *before* this call runs live and un-checkpointed in EVERY generation, because
    respawn is the outer loop wearing an inner call's syntax: what looks like it precedes the
    loop is inside it. So a `call_tool` up there sends mail once per generation, and a ledger
    append survives only if it is subject-scoped. Put nothing before `respawn` that must happen
    once; post-`respawn` code runs only in the `Done` generation and is the right home for it.

    `budget.generations` bounds the chain and threads through spawn params exactly as
    `budget.depth` does — same shape, different axis. On exhaustion with `on_exhaust="park"` and
    a `run_id`, the chain **parks durably** and a human answers with `add_generations` or a
    `stop`: *"how long may this agent keep going?"* becomes the same governed question as *"how
    deep may it drill?"*. Without a grantor the step gets one `turn.final` and must answer.

    **The measured (dollar) accrual is carried invisibly**, by the handler, under reserved params
    keys — it is substrate bookkeeping, not the author's, exactly as checkpoint ids are. Two
    carries, one per bookkeeper: this one is explicit and typed, that one is automatic. What a
    chain has spent is a LEDGER question (the `respawned` row), never a params read.
    """
    # The DECLARED bound is the literal in the workflow; the REMAINDER comes from params, which
    # is what makes it decrement. A budget rebuilt from a literal every generation never
    # decrements (measured: 31 generations ran under a bound of 4).
    #
    # Read from `chain.params` rather than made the author's problem at the spawn site (which is
    # how `depth` does it, because a fork's caller chooses the child's depth). A chain re-enters
    # the SAME program, so the bound belongs in the program and the substrate threads it.
    remaining = None if budget is None else budget.generations
    if budget is not None and BUDGET_GENERATIONS_PARAM in chain.params:
        remaining = int(chain.params[BUDGET_GENERATIONS_PARAM])
    # `generations=N` means N generations RUN, not N extra ones — so the last of them is the
    # one holding `remaining == 1`, and that is where the chain asks for more before being told
    # to answer. Asking before the step runs (rather than after it returns `Again`) is what
    # gives `turn.final` a meaning: the combinator holds no `T`, so a chain that may not go
    # round again has to be asked for an answer while it can still give one.
    granted = 0
    if remaining is not None and remaining <= 1 and budget is not None:
        granted = yield from _refill_generations(budget, chain.generation)
        remaining += granted
    final = remaining is not None and remaining <= 1

    # `respawn ∘ respawn`, refused here rather than in a handler: an interpreter never sees "we
    # are inside a respawn" — this combinator is ordinary workflow code until its step returns
    # `Again`, so the handler observes exactly one `Respawn` op, the INNER one. The fact lives
    # where it is known. Reset is the handlers' job, at the task boundary (`ops.enter_task_run`);
    # a `finally` here does NOT run when a branch's refusal abandons this generator.
    if CHAIN_DEPTH.get():
        refuse_nested_respawn(chain.task)
    CHAIN_DEPTH.set(CHAIN_DEPTH.get() + 1)
    # The generation any grant park inside this step will name. Set here rather than passed,
    # so an author who writes `descend(...)` inside a chain cannot forget it — a forgotten
    # coordinate would alias two generations' grants, silently.
    generation_token = CHAIN_GENERATION.set(chain.generation)
    outcome = yield from step(chain.state, Turn(generation=chain.generation, final=final))
    CHAIN_GENERATION.reset(generation_token)
    CHAIN_DEPTH.set(CHAIN_DEPTH.get() - 1)
    match outcome:
        case Done(value):
            return value
        case Again(_) if final:
            raise ValueError(
                f"respawn: the step returned Again at generation {chain.generation}, which is "
                "final — the chain cannot go round again, so this turn had to answer with Done. "
                "Check `turn.final` before returning Again (the same loud error `descend` raises "
                "for Deeper past an exhausted budget)."
            )
        case Again(state):
            params = dict(chain.params)
            if budget is not None and budget.generations is not None:
                # The remainder rides the params, the way `spawn_fork` writes
                # `budget.descend_one().depth`. A grant refills `remaining` above, so what is
                # written here is what the NEXT generation may still spend.
                params[BUDGET_GENERATIONS_PARAM] = (remaining or 1) - 1
            return (
                yield from respawn_generation(
                    task=chain.task,
                    generation=chain.generation + 1,
                    state=to_jsonable_python(state),
                    params=params,
                    run_id=chain.run_id,
                    granted=granted,
                )
            )
