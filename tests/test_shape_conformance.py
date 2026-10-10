"""Three recursion shapes on both engines, each spelled by its combinator and by open recursion
closed with `fix`: divide and conquer and linear descent through `unfold`, and tree search
through `tree_search`, whose rounds must also keep their approvals and spawns apart.

The `fix` spelling is the reference: each shape written directly, its level's ops scoped under
`d:{depth}` and the recursive call outside that scope. `unfold` has to reproduce its answer, its
checkpoint names and its ledger rows. The durable claim is the crash cells: a crash after every
step's effect converges to the same answer and the same rows. The nested spelling, whose
recursion sits inside its level's scope, is its own row with its own names.

The tree is complete and binary, `DEPTH` levels below the root, and a node is its path of child
indices. A leaf answers its path and a join concatenates, so a halving answers its leaves in order.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any
from uuid import UUID, uuid4

import pytest
from _conformance import Approval, Fault, FaultPosition, private
from _shapes import Program, Shape, agree, ledger_ids, placed, run, sweep, sweep_pairs
from pydantic import TypeAdapter

from effective.api import Effect, append_ledger, call_tool, gather, scoped
from effective.budget import BUDGET_DEPTH_PARAM
from effective.combinators import (
    CONTEXT_PARAM,
    DEPTH_PARAM,
    AcrossTasks,
    Answered,
    Branch,
    Decision,
    Deeper,
    Level,
    fix,
    tree_search,
    unfold,
    unfold_task,
)
from effective.cost import MeteredInterpreter, Usage
from effective.domain import SPAWN_TOOL, CallTool, DomainOp, SpawnArgs, Spawned, SpawnResult
from effective.handlers.base import op_key
from effective.handlers.durable import spawn_done_name
from effective.interpreters.tools import make_tool_runner, spawn_tool
from effective.keys import Index, Key, Run, Segment, compose_key
from effective.ops import DONE_EVENT_PARAM, LedgerRow, Step, Writer
from effective.permission import APPROVE, Allow, Escalate, cascade, human, rules
from effective.spawning import join_answer, spawn_child

pytestmark = pytest.mark.conformance

DEPTH = 3
ROOT = ""
COUNTDOWN_FROM = 3


class Tree:
    """The domain: splits a path into its two children above `DEPTH`, answers a leaf's path, and
    scores a rollout as `int("1" + path, 2)`."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="split", args={"path": str(path)}):
                self.calls.append(op.name)
                return [path + "0", path + "1"] if len(path) < DEPTH else []
            case CallTool(name="leaf", args={"path": str(path)}):
                self.calls.append(op.name)
                return path
            case CallTool(name="join", args={"values": list(values)}):
                self.calls.append(op.name)
                return "".join(values)
            case CallTool(name="rollout", args={"path": str(path)}):
                self.calls.append(op.name)
                return int("1" + path, 2)
        raise TypeError(f"the tree answers split, leaf and join, not {op!r}")


def leaf_id(run_id: str, path: str) -> Key:
    """A leaf's ledger row: the run, and the node's number in heap order."""
    return compose_key(t"shape-leaf:{Run(run_id)},{Index(int('1' + path, 2))}")


def leaf(run_id: str, path: str) -> Effect[str]:
    value = yield from call_tool("leaf", {"path": path}, str)
    yield from append_ledger(LedgerRow(event_id=leaf_id(run_id, path), kind="leaf", path=value))
    return value


def joined(values: Sequence[str]) -> Effect[str]:
    return call_tool("join", {"values": list(values)}, str)


def level_scope(depth: int) -> Key:
    return compose_key(t"d:{Index(depth)}")


# --- halving: divide and conquer -------------------------------------------------------------


def halving(run_id: str) -> Callable[[str, Level], Effect[Decision[str, str]]]:
    def node(path: str, level: Level) -> Effect[Decision[str, str]]:
        if level.final:
            return Answered((yield from leaf(run_id, path)))
        children = yield from call_tool("split", {"path": path}, list)
        if not children:
            return Answered((yield from leaf(run_id, path)))
        return Branch(children, joined)

    return node


def open_halving(run_id: str, budget: int):
    def close(solve: Callable[[str, int], Effect[str]]) -> Callable[[str, int], Effect[str]]:
        def body(path: str, depth: int) -> Effect[str]:
            def level() -> Effect[tuple[str, list[str]]]:
                if depth == budget:
                    return (yield from leaf(run_id, path)), []
                children = yield from call_tool("split", {"path": path}, list)
                if not children:
                    return (yield from leaf(run_id, path)), []
                return "", children

            value, children = yield from scoped(level_scope(depth), level)
            if not children:
                return value
            values = yield from gather(
                [
                    partial(
                        scoped, compose_key(t"rec:{Index(i)}"), partial(solve, child, depth + 1)
                    )
                    for i, child in enumerate(children)
                ]
            )
            return (yield from joined(values))

        return body

    return close


# --- countdown: linear descent -----------------------------------------------------------------


def countdown(run_id: str) -> Callable[[int, Level], Effect[Decision[int, str]]]:
    def node(n: int, level: Level) -> Effect[Decision[int, str]]:
        # The path comes from the level, so a driver that misnumbers depth collides or writes the
        # wrong rows.
        value = yield from leaf(run_id, "0" * (COUNTDOWN_FROM - level.depth))
        return Answered(value) if n == 0 or level.final else Deeper(n - 1)

    return node


def open_countdown(run_id: str, budget: int):
    def close(solve: Callable[[int, int], Effect[str]]) -> Callable[[int, int], Effect[str]]:
        def body(n: int, depth: int) -> Effect[str]:
            value = yield from scoped(level_scope(depth), partial(leaf, run_id, "0" * n))
            if n == 0 or depth == budget:
                return value
            return (yield from solve(n - 1, depth + 1))

        return body

    return close


def nested_countdown(run_id: str, budget: int):
    def close(solve: Callable[[int, int], Effect[str]]) -> Callable[[int, int], Effect[str]]:
        def body(n: int, depth: int) -> Effect[str]:
            def level() -> Effect[str]:
                value = yield from leaf(run_id, "0" * n)
                if n == 0 or depth == budget:
                    return value
                return (yield from solve(n - 1, depth + 1))

            return (yield from scoped(level_scope(depth), level))

        return body

    return close


# --- the product --------------------------------------------------------------------------------

type Budgeted = Callable[[str, int], Effect[str]]

SPELLINGS: dict[str, dict[str, Budgeted]] = {
    "halving": {
        "unfold": lambda run_id, budget: unfold(ROOT, halving(run_id), budget=budget),
        "fix": lambda run_id, budget: fix(open_halving(run_id, budget))(ROOT, 0),
    },
    "countdown": {
        "unfold": lambda run_id, budget: unfold(COUNTDOWN_FROM, countdown(run_id), budget=budget),
        "fix": lambda run_id, budget: fix(open_countdown(run_id, budget))(COUNTDOWN_FROM, 0),
    },
}

BUDGETS = {"ample": 9, "exact": DEPTH, "short": DEPTH - 1}


def at(program: Budgeted, budget: int) -> Program:
    return lambda run_id: program(run_id, budget)


def leaves(shape: str, budget: int) -> list[str]:
    """The paths a run writes a ledger row for, derived from the tree and the budget."""
    match shape:
        case "halving":
            reach = min(budget, DEPTH)
            return [format(i, f"0{reach}b") if reach else "" for i in range(2**reach)]
        case "countdown":
            return [
                "0" * (COUNTDOWN_FROM - depth) for depth in range(min(budget, COUNTDOWN_FROM) + 1)
            ]
    raise ValueError(shape)


def answer(shape: str, budget: int) -> str:
    match shape:
        case "halving":
            return "".join(leaves(shape, budget))
        case "countdown":
            return leaves(shape, budget)[-1]
    raise ValueError(shape)


def expected_rows(run_id: str, shape: str, budget: int) -> list[str]:
    expected = sorted(leaf_id(run_id, path).stored() for path in leaves(shape, budget))
    assert len(set(expected)) == len(expected), "two leaves would share a ledger row"
    return expected


def row(shape: str, budget: int) -> Shape:
    """`shape` at `budget`, its answer and rows derived from the tree. A run checkpoints each
    leaf's op and its row, each split and each join, and nothing else."""
    return Shape(
        spellings={
            spelling: at(program, budget) for spelling, program in SPELLINGS[shape].items()
        },
        domain=Tree,
        answer=partial(answer, shape, budget),
        ledger_ids=partial(expected_rows, shape=shape, budget=budget),
        count=lambda base: (
            2 * len(leaves(shape, budget))
            + base.domain.calls.count("split")
            + base.domain.calls.count("join")
        ),
        calls=lambda tree: tree.calls.count("split"),
    )


@pytest.mark.parametrize("budget", list(BUDGETS.values()), ids=list(BUDGETS))
@pytest.mark.parametrize("shape", list(SPELLINGS))
def test_unfold_reproduces_the_fix_spelling(backend, shape, budget):
    agree(backend, row(shape, budget))


@pytest.mark.parametrize("spelling", ["fix", "unfold"])
@pytest.mark.parametrize("shape", list(SPELLINGS))
def test_a_crash_after_every_step_converges(backend, shape, spelling):
    sweep(backend, row(shape, BUDGETS["exact"]), spelling)


def test_the_nested_spelling_names_each_level_inside_its_parent(backend):
    budget = BUDGETS["exact"]
    flat = run(backend, at(SPELLINGS["countdown"]["fix"], budget), Tree())
    nested = run(
        backend, lambda rid: fix(nested_countdown(rid, budget))(COUNTDOWN_FROM, 0), Tree()
    )

    assert nested.snap.state == "completed", nested.snap
    assert nested.snap.result == flat.snap.result == answer("countdown", budget)
    assert ledger_ids(backend, nested.run_id) == expected_rows(nested.run_id, "countdown", budget)
    assert placed(backend, flat) == [
        "d:0;step;tool:leaf",
        "d:1;step;tool:leaf",
        "d:2;step;tool:leaf",
        "d:3;step;tool:leaf",
    ]
    assert placed(backend, nested) == [
        "d:0;d:1;d:2;d:3;step;tool:leaf",
        "d:0;d:1;d:2;step;tool:leaf",
        "d:0;d:1;step;tool:leaf",
        "d:0;step;tool:leaf",
    ]


def stubborn(run_id: str) -> Callable[[int, Level], Effect[Decision[int, str]]]:
    def node(n: int, level: Level) -> Effect[Decision[int, str]]:
        yield from call_tool("leaf", {"path": "0" * n}, str)
        return Deeper(n)

    return node


def test_a_node_past_its_budget_fails_its_task(backend):
    past = run(
        backend,
        lambda run_id: unfold(COUNTDOWN_FROM, stubborn(run_id), budget=BUDGETS["short"]),
        Tree(),
    )

    assert past.snap.state == "failed", past.snap
    assert backend.failure_kind(past.snap) == "DescendedPastBudget"
    assert backend.task_attempts(past.task) == 1
    assert past.domain.calls == ["leaf"] * (BUDGETS["short"] + 1)


# --- tree search: rounds of unfold, guided by what the earlier rounds learned ------------------

type Stats = dict[str, tuple[int, int]]
type SearchCtx = tuple[str, bool]
type Scored = tuple[str, int]

ROUNDS = 4
SEARCH_DEPTH = DEPTH
# Derived by hand from the design: a rollout scores `int("1" + path, 2)`; each round selects the
# child with fewest visits (the first on a tie), expands the first unexpanded node on that path
# by rolling out its children, and credits the node with their best score.
SEARCH_ROLLOUTS = ["0", "1", "00", "01", "10", "11", "000", "001"]
SEARCH_ANSWER = [["", 1, 3], ["0", 1, 5], ["00", 1, 9], ["1", 1, 7]]


def rollout(run_id: str, path: str) -> Effect[Scored]:
    score = yield from call_tool("rollout", {"path": path}, int)
    yield from append_ledger(
        LedgerRow(event_id=leaf_id(run_id, path), kind="rollout", score=score)
    )
    return path, score


def best_of(parent: str, values: Sequence[Scored]) -> Effect[Scored]:
    yield from ()
    return parent, max(score for _, score in values)


def searcher(
    run_id: str, stats: Stats
) -> Callable[[SearchCtx, Level], Effect[Decision[SearchCtx, Scored]]]:
    def node(ctx: SearchCtx, level: Level) -> Effect[Decision[SearchCtx, Scored]]:
        path, evaluate = ctx
        if evaluate or level.final:
            return Answered((yield from rollout(run_id, path)))
        if path not in stats:
            children = yield from call_tool("split", {"path": path}, list)
            if not children:
                return Answered((yield from rollout(run_id, path)))
            return Branch([(child, True) for child in children], partial(best_of, path))
        child = min((path + "0", path + "1"), key=lambda c: stats.get(c, (0, 0))[0])
        return Deeper((child, False))

    return node


def backprop(stats: Stats, value: Scored) -> Stats:
    path, score = value
    visits, total = stats.get(path, (0, 0))
    return {**stats, path: (visits + 1, total + score)}


def summary(stats: Stats) -> list[list[Any]]:
    return sorted([path, visits, total] for path, (visits, total) in stats.items())


def search_by_combinator(run_id: str, depth: int) -> Effect[list[list[Any]]]:
    stats = yield from tree_search(
        ("", False),
        partial(searcher, run_id),
        backprop,
        initial={},
        iterations=ROUNDS,
        depth=depth,
    )
    return summary(stats)


def open_search(run_id: str, depth: int):
    def close(again: Callable[[Stats, int], Effect[list[list[Any]]]]):
        def body(stats: Stats, k: int) -> Effect[list[list[Any]]]:
            if k == ROUNDS:
                return summary(stats)
            value = yield from scoped(
                compose_key(t"search:{Index(k)}"),
                partial(unfold, ("", False), searcher(run_id, stats), budget=depth),
            )
            return (yield from again(backprop(stats, value), k + 1))

        return body

    return close


def search_rows(run_id: str) -> list[str]:
    return sorted(leaf_id(run_id, path).stored() for path in SEARCH_ROLLOUTS)


SEARCH = Shape(
    spellings={
        "tree_search": partial(search_by_combinator, depth=SEARCH_DEPTH),
        "fix": lambda run_id: fix(open_search(run_id, SEARCH_DEPTH))({}, 0),
    },
    domain=Tree,
    answer=lambda: SEARCH_ANSWER,
    ledger_ids=search_rows,
    count=lambda _base: 2 * len(SEARCH_ROLLOUTS) + ROUNDS,
)
"""A search checkpoints each rollout and its row, and one join per round."""


def test_tree_search_reproduces_its_fix_spelling(backend):
    agree(backend, SEARCH)


@pytest.mark.parametrize("spelling", list(SEARCH.spellings))
def test_a_crash_after_every_step_of_a_tree_search_converges(backend, spelling):
    sweep(backend, SEARCH, spelling)


def test_two_crashes_in_a_tree_search_converge(backend):
    """A row whose rollouts run in gathered branches, beside the watched descent's single
    thread."""
    first, *_ = SEARCH.spellings
    assert sweep_pairs(backend, SEARCH, first) > 0


# --- each round of a search keeps its approvals and spawns apart --------------------------------

ONE_ROUND_SCOPE = compose_key(t"round")


def acting(tool: str, args: dict[str, Any], schema: type) -> Callable[[str, Level], Effect[Any]]:
    """A node that fans out to one child which calls `tool` once: a gather in every round."""

    def node(ctx: str, level: Level) -> Effect[Decision[str, Any]]:
        if level.final:
            return Answered((yield from call_tool(tool, args, schema)))
        yield from ()
        return Branch([ctx], lambda values: first_of(values))

    return node


def first_of(values: Sequence[Any]) -> Effect[Any]:
    yield from ()
    return values[0]


def two_rounds(
    node: Callable[[str, Level], Effect[Any]], scoping: str
) -> Callable[[str], Effect[Any]]:
    """Two rounds of one search, through `tree_search` or as a hand loop that enters one unindexed
    scope per round, the shape a scope entered twice once aliased."""

    def by_tree_search(_run_id: str):
        return (
            yield from tree_search(
                "root",
                lambda _state: node,
                lambda state, value: [*state, value],
                initial=[],
                iterations=2,
                depth=1,
            )
        )

    def by_one_scope(_run_id: str):
        values = []
        for _ in range(2):
            values.append(
                (yield from scoped(ONE_ROUND_SCOPE, partial(unfold, "root", node, budget=1)))
            )
        return values

    return {"tree_search": by_tree_search, "one scope per round": by_one_scope}[scoping]


SCOPINGS = ["tree_search", "one scope per round"]


def _gate_act(op: Any) -> Any:
    if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name == "act":
        return Escalate("needs a ruling")
    return Allow()


@pytest.mark.parametrize("scoping", SCOPINGS)
def test_each_round_asks_for_its_own_approval(backend, scoping):
    run_id, task, tree = str(uuid4()), private("rounds"), Counts()
    gate = cascade(
        [
            rules(_gate_act),
            human(
                Approval,
                event_name=lambda op: compose_key(
                    # lint: terminal-hole: `op_key` returns a `Key`, spliced by induction.
                    t"{APPROVE}:{Segment(run_id)};{op_key(op):domain=identity}"
                ),
            ),
        ]
    )
    backend.register(task, two_rounds(acting("act", {}, int), scoping), tree, Fault(), (gate,))
    task_id = backend.spawn(task, run_id)
    backend.run_until_result(task_id)

    (first,) = backend.parked(task_id)
    backend.emit_event(task_id, first.wake_event, {"decision": "approve"})
    assert backend.run_until_result(task_id).state != "completed"
    (second,) = backend.parked(task_id)
    assert second.wake_event != first.wake_event
    assert tree.calls == ["act"]

    backend.emit_event(task_id, second.wake_event, {"decision": "approve"})
    assert backend.run_until_result(task_id).state == "completed"
    assert tree.calls == ["act", "act"]


@pytest.mark.parametrize("scoping", SCOPINGS)
def test_each_round_spawns_its_own_child(backend, scoping):
    child_task, task = private("round-child"), private("rounds")
    backend.register_body(child_task, lambda params, ctx: "ok")
    args = SpawnArgs(task_name=child_task, params={}).model_dump(exclude_none=True)
    domain = MeteredInterpreter(
        llm=lambda _op: ("", Usage()),
        tools=make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(backend.spawner)}),
    )
    node = acting(SPAWN_TOOL, args, SpawnResult)
    backend.register(task, two_rounds(node, scoping), domain, Fault(), [])
    snapshot = backend.run_until_result(backend.spawn(task, str(uuid4())))

    assert snapshot.state == "completed", snapshot
    assert len(backend.enqueued(child_task)) == 2
    assert len({str(result["task_id"]) for result in snapshot.result}) == 2


class Counts:
    """A domain that records the tools it is asked for."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name=str(name)):
                self.calls.append(name)
                return len(self.calls)
        raise TypeError(op)


# --- the spawning crossing: a Branch's children as tasks of their own --------------------------


class SpawningTree(Tree):
    """The tree, with a spawn tool that enqueues on the backend."""

    def __init__(self, spawner) -> None:
        super().__init__()
        self._spawn = spawn_tool(spawner)

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name=str(name)) if name == SPAWN_TOOL:
                return self._spawn(op)
        return super().run(op)


type TaskProgram = Callable[[Mapping[str, Any]], Effect[Any]]


def across(task: str) -> AcrossTasks[str, str]:
    return AcrossTasks(task, TypeAdapter(str), TypeAdapter(str))


def halving_by_hand(run_id: str, task: str) -> TaskProgram:
    """Divide and conquer across tasks, written directly: the level under `d:{depth}`, then every
    child spawned under `rec:{i}` with one level less, then every child joined in order."""

    def body(params: Mapping[str, Any]) -> Effect[str]:
        path, depth, levels = (
            params[CONTEXT_PARAM],
            params[DEPTH_PARAM],
            params[BUDGET_DEPTH_PARAM],
        )

        def level() -> Effect[tuple[str, list[str]]]:
            if levels == 0:
                return (yield from leaf(run_id, path)), []
            children = yield from call_tool("split", {"path": path}, list)
            if not children:
                return (yield from leaf(run_id, path)), []
            return "", children

        value, children = yield from scoped(level_scope(depth), level)
        if not children:
            return value
        spawned = []
        for i, child in enumerate(children):
            child_params = {
                CONTEXT_PARAM: child,
                DEPTH_PARAM: depth + 1,
                BUDGET_DEPTH_PARAM: levels - 1,
            }
            spawn = partial(spawn_child, task, "child", child_params)
            spawned.append((yield from scoped(compose_key(t"rec:{Index(i)}"), spawn)))
        values = []
        for handle in spawned:
            values.append((yield from join_answer(handle)))
        return (yield from joined(values))

    return body


ACROSS_SPELLINGS: dict[str, Callable[[str, str], TaskProgram]] = {
    "unfold": lambda run_id, task: partial(
        unfold_task, node=halving(run_id), crossing=across(task)
    ),
    "by hand": halving_by_hand,
}

ACROSS_BUDGETS = {"exact": DEPTH, "short": DEPTH - 1}


@dataclass(frozen=True)
class Grown:
    snap: Any
    root: UUID
    task: str
    run_id: str
    tree: Tree


def settle(backend, task: UUID) -> Any:
    """Work the queue until `task` is terminal; a tree of tasks takes more batches than one."""
    return backend.engine.run_until_result(task, max_batches=400)


def grow(
    backend,
    program_for,
    budget: int,
    fault: Fault | None = None,
    fault_for=None,
    run_id: str | None = None,
) -> Grown:
    task, run_id = private("node"), run_id or str(uuid4())
    tree = SpawningTree(backend.spawner)
    backend.register_child(
        task, run_id, program_for(run_id, task), tree, fault or Fault(), fault_for
    )
    params = across(task).params(ROOT, depth=0, levels=budget)
    root = UUID(backend.spawner(task, params, str(uuid4()), "default"))
    snap = settle(backend, root)
    # A child that answered and then crashed is still retrying when its answer completes the root.
    for task_id, _params in backend.enqueued(task):
        settle(backend, task_id)
    return Grown(snap, root, task, run_id, tree)


def tasks(backend, grown: Grown) -> list[UUID]:
    return [task_id for task_id, _params in backend.enqueued(grown.task)]


def names_by_task(backend, grown: Grown) -> list[list[str]]:
    """Each task's checkpoint names, with the tree's task ids written `{task}` and its run id
    `{run}`: a done event and its emit name the task that spawned it, and a ledger row its run."""
    ids = [str(task_id) for task_id in tasks(backend, grown)]

    def anonymous(name: str) -> str:
        for task_id in ids:
            name = name.replace(task_id, "{task}")
        return name.replace(grown.run_id, "{run}")

    by_task = (sorted(map(anonymous, backend.checkpoint_keys(t))) for t in tasks(backend, grown))
    return sorted(by_task)


def states(backend, grown: Grown) -> list[str]:
    return sorted(backend.engine.fetch_task_result(t).state for t in tasks(backend, grown))


def returned(grown: Grown) -> Any:
    assert grown.snap is not None
    assert grown.snap.state == "completed", grown.snap
    match grown.snap.result:
        case {"answer": {"kind": "returned", "value": value}}:
            return value
    raise AssertionError(f"the root did not return a value: {grown.snap.result!r}")


def refusals(grown: Grown) -> list[list[str]]:
    assert grown.snap is not None
    assert grown.snap.state == "completed", grown.snap
    match grown.snap.result:
        case {"answer": {"kind": "refused", "refusals": list(causes)}}:
            return causes
    raise AssertionError(f"the root did not report a refusal: {grown.snap.result!r}")


@pytest.mark.parametrize("budget", list(ACROSS_BUDGETS.values()), ids=list(ACROSS_BUDGETS))
def test_unfold_across_tasks_reproduces_the_hand_spelling(backend, budget):
    by_hand = grow(backend, ACROSS_SPELLINGS["by hand"], budget)
    by_unfold = grow(backend, ACROSS_SPELLINGS["unfold"], budget)

    for grown in (by_hand, by_unfold):
        assert returned(grown) == answer("halving", budget)
        assert len(tasks(backend, grown)) == 2 ** (budget + 1) - 1
        assert states(backend, grown) == ["completed"] * (2 ** (budget + 1) - 1)
        assert ledger_ids(backend, grown.run_id) == expected_rows(grown.run_id, "halving", budget)
    assert names_by_task(backend, by_unfold) == names_by_task(backend, by_hand)
    assert by_unfold.tree.calls == by_hand.tree.calls
    # Every child answers its parent in a checkpointed step, so a retry answers at most once.
    emits = [
        sum(n.startswith("emit;") for n in names) for names in names_by_task(backend, by_unfold)
    ]
    assert sorted(emits) == [0] + [1] * (2 ** (budget + 1) - 2)


type Aim = Callable[[Mapping[str, Any], Any], Fault]


def in_the_task_at(path: str, aim: Aim) -> tuple[Callable[[Mapping[str, Any], Any], Fault], list]:
    """A fault for the task unfolding `path` alone, made once from that task's params and ctx, so
    its retry replays past the crash and no other task in the tree is touched."""
    made: list[Fault] = []

    def fault_for(params: Mapping[str, Any], ctx: Any) -> Fault:
        if params[CONTEXT_PARAM] != path:
            return Fault()
        if not made:
            made.append(aim(params, ctx))
        return made[0]

    return fault_for, made


def parent_of(params: Mapping[str, Any]) -> str:
    """The id of the task that spawned this one, the first coordinate of its done event."""
    spawned_by = Key.parse(params[DONE_EVENT_PARAM]).terms()[0].coordinates[0]
    return spawned_by.atoms[0].text


def after_the_step(anonymous: str, run_id: str) -> Aim:
    """A crash after the named step's effect, its ids filled in from the task that runs it: an
    emit names its parent, and a ledger row its run."""

    def aim(params: Mapping[str, Any], ctx: Any) -> Fault:
        task = parent_of(params) if anonymous.startswith("emit;") else str(ctx.task_id)
        named = anonymous.replace("{task}", task).replace("{run}", run_id)
        return Fault(position=FaultPosition.AFTER_THUNK, named=named)

    return aim


def before_the_join(spawn: str) -> Aim:
    """A crash before a task awaits the answer of the child its `spawn` step placed, on the name
    the handler gave that child."""

    def aim(_params: Mapping[str, Any], ctx: Any) -> Fault:
        placement = Key.parse(spawn)
        done = spawn_done_name(Writer(task=str(ctx.task_id), placement=placement))
        return Fault(position=FaultPosition.BEFORE_OP, named=done.stored())

    return aim


def test_a_crash_at_every_step_and_join_of_every_task_converges(backend):
    """Reddens if any task of the tree, crashed after one of its steps or before one of its joins,
    fails to converge to the uncrashed tree's answer, rows and names."""
    budget = ACROSS_BUDGETS["short"]
    base = grow(backend, ACROSS_SPELLINGS["unfold"], budget)
    by_path = {params[CONTEXT_PARAM]: task_id for task_id, params in backend.enqueued(base.task)}
    ids = [str(task_id) for task_id in by_path.values()]

    def anonymous(name: str) -> str:
        for task_id in ids:
            name = name.replace(task_id, "{task}")
        return name.replace(base.run_id, "{run}")

    cells: list[tuple[str, str]] = []
    for path, task_id in by_path.items():
        names = [anonymous(name) for name in backend.checkpoint_keys(task_id)]
        cells += [(path, name) for name in names]
        cells += [(path, f"join {name}") for name in names if name.startswith("rec:")]
    assert len(cells) == len(set(cells))
    assert sum(label.startswith("join") for _path, label in cells) == 2 * (2**budget - 1)

    for path, label in cells:
        run_id = str(uuid4())
        match label.split():
            case ["join", spawn]:
                aim = before_the_join(spawn)
            case _:
                aim = after_the_step(label, run_id)
        fault_for, made = in_the_task_at(path, aim)
        crashed = grow(
            backend, ACROSS_SPELLINGS["unfold"], budget, fault_for=fault_for, run_id=run_id
        )
        cell = (path, label)
        assert made, f"no task unfolded {path!r}"
        assert not made[0].armed, f"the crash at {cell} never fired"
        assert returned(crashed) == answer("halving", budget), cell
        assert set(states(backend, crashed)) == {"completed"}, cell
        assert ledger_ids(backend, crashed.run_id) == expected_rows(
            crashed.run_id, "halving", budget
        ), cell
        assert names_by_task(backend, crashed) == names_by_task(backend, base), cell


NEVER_RUN = private("never-run")


def spawning_at_its_final_level(run_id: str) -> Callable[[str, Level], Effect[Decision[str, str]]]:
    """A node that spawns a child of its own when it should answer, around `unfold`."""
    node = halving(run_id)

    def stubborn(path: str, level: Level) -> Effect[Decision[str, str]]:
        if level.final:
            args = SpawnArgs(task_name=NEVER_RUN, params={}).model_dump(exclude_none=True)
            yield from call_tool(SPAWN_TOOL, args, Spawned)
        return (yield from node(path, level))

    return stubborn


def descending_at_its_final_level(
    run_id: str,
) -> Callable[[str, Level], Effect[Decision[str, str]]]:
    """A node that descends when it should answer."""
    node = halving(run_id)

    def stubborn(path: str, level: Level) -> Effect[Decision[str, str]]:
        if level.final:
            return Deeper(path)
        return (yield from node(path, level))

    return stubborn


def test_a_refusal_at_a_leaf_task_climbs_to_the_root(backend):
    """Reddens if a leaf task that stops at a refusal leaves its parent waiting, or if the refusal
    loses its cause on the way up. The handler refuses a spawn from a task whose levels are spent,
    so a node's final level is exactly where its spawn is refused."""
    budget = ACROSS_BUDGETS["short"]

    def program(run_id: str, task: str) -> TaskProgram:
        return partial(
            unfold_task, node=spawning_at_its_final_level(run_id), crossing=across(task)
        )

    grown = grow(backend, program, budget)

    ((kind, cause),) = refusals(grown)
    assert kind == "ChildRefused"
    assert "Refused" in cause
    # The root answers at the first refusal it joins, so a sibling may still be running: settle
    # each, and none may be left waiting.
    assert [settle(backend, t).state for t in tasks(backend, grown)] == ["completed"] * (
        2 ** (budget + 1) - 1
    )
    assert backend.enqueued(NEVER_RUN) == []


@pytest.mark.parametrize("budget", list(ACROSS_BUDGETS.values()), ids=list(ACROSS_BUDGETS))
def test_a_root_with_no_depth_of_its_own_holds_its_tree_to_the_unfold_budget(backend, budget):
    """Reddens if a spawned child's levels come from anywhere but its parent's remainder: a root
    task enqueued without a depth leaves the handler nothing to hold its children to."""
    root_task, task, run_id = private("root"), private("node"), str(uuid4())
    tree = SpawningTree(backend.spawner)
    backend.register_child(task, run_id, ACROSS_SPELLINGS["unfold"](run_id, task), tree, Fault())
    backend.register_child(
        root_task,
        run_id,
        lambda _params: unfold(ROOT, halving(run_id), budget=budget, crossing=across(task)),
        tree,
        Fault(),
    )
    root = UUID(backend.spawner(root_task, {}, str(uuid4()), "default"))
    snap = settle(backend, root)
    grown = Grown(snap, root, task, run_id, tree)

    assert returned(grown) == answer("halving", budget)
    assert len(tasks(backend, grown)) == 2 ** (budget + 1) - 2
    assert ledger_ids(backend, grown.run_id) == expected_rows(grown.run_id, "halving", budget)


def test_a_programming_error_at_a_leaf_fails_every_task_on_its_path_once(backend):
    """Reddens if a leaf's programming error is answered as a refusal or retried: each leaf fails
    of it on its first attempt, and each parent fails once of the child failure it joined, naming
    that child, up to the root."""
    budget = ACROSS_BUDGETS["short"]

    def program(run_id: str, task: str) -> TaskProgram:
        return partial(
            unfold_task, node=descending_at_its_final_level(run_id), crossing=across(task)
        )

    grown = grow(backend, program, budget)
    ended = {task_id: settle(backend, task_id) for task_id in tasks(backend, grown)}

    assert grown.snap.state == "failed", grown.snap
    assert backend.failure_kind(grown.snap) == "ChildFailed"
    assert any(str(child) in str(grown.snap.failure) for child in ended if child != grown.root)
    assert sorted(backend.failure_kind(snap) for snap in ended.values()) == sorted(
        ["ChildFailed"] * (2**budget - 1) + ["DescendedPastBudget"] * 2**budget
    )
    assert {backend.task_attempts(task_id) for task_id in ended} == {1}
    assert backend.enqueued(NEVER_RUN) == []


def crashing_at_its_final_level(run_id: str) -> Callable[[str, Level], Effect[Decision[str, str]]]:
    node = halving(run_id)

    def crashing(path: str, level: Level) -> Effect[Decision[str, str]]:
        if level.final:
            raise ValueError("a leaf that crashes on every attempt")
        return (yield from node(path, level))

    return crashing


def test_a_child_that_crashes_to_death_answers_its_parent(backend):
    """Reddens if a parent waits forever on a child that failed: a leaf retried to its last
    attempt answers `Failed`, and each parent fails once of it, up to the root."""
    budget = ACROSS_BUDGETS["short"]

    def program(run_id: str, task: str) -> TaskProgram:
        return partial(
            unfold_task, node=crashing_at_its_final_level(run_id), crossing=across(task)
        )

    grown = grow(backend, program, budget)
    ended = {task_id: settle(backend, task_id) for task_id in tasks(backend, grown)}

    assert grown.snap is not None
    assert grown.snap.state == "failed", grown.snap
    assert backend.failure_kind(grown.snap) == "ChildFailed"
    assert sorted(backend.failure_kind(snap) for snap in ended.values()) == sorted(
        ["ChildFailed"] * (2**budget - 1) + ["ValueError"] * 2**budget
    )
