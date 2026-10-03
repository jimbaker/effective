"""`fixpoint` on both engines, spelled by the combinator and by open recursion closed with `fix`.

The `fix` spelling is the reference: each application of the step scoped under `d:{n}`, the
recursive call outside that scope, and its own catch of a governed refusal, so a defect in
`within_budget` reaches one spelling only. Each row is a `_shapes.Shape`, compared by `agree` and
crashed by `sweep`.

Halving converges in five steps: `8 4 2 1 0`, and `0` halves to itself. Each step appends a ledger
row for its input before it asks the tool, so a refused step leaves the row it wrote."""

import operator
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, assert_never

import pytest
from _conformance import FaultPosition
from _shapes import (
    ENGINE_DEPTH,
    RECORDER_DEPTH,
    Ones,
    Program,
    Shape,
    agree,
    governing,
    granted_by,
    granting_at,
    run,
    sweep,
)

from effective.api import Effect, append_ledger, ask_llm, call_tool, gather, scoped
from effective.budget import Grant, MeasuredBudget, depth_grant_name
from effective.budget import as_policy as budget_policy
from effective.combinators import Converged, Grantor, Unconverged, fix, fixpoint
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.domain import CallTool, DomainOp
from effective.govern import BudgetRefused, Exceeded, GateState, Policy, Proceed, Refuse, Verdict
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Index, Key, Name, Run, compose_key
from effective.ops import LedgerRow, Step, WorkflowOp

START = 8
HALVINGS = [8, 4, 2, 1, 0]
"""The inputs halving steps through from `START`: the last one converges."""


class Halver:
    """The domain: halves a number, and lists a node's edges."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="halve", args={"n": int(n)}):
                self.calls.append("halve")
                return n // 2
            case CallTool(name="edges", args={"node": str(node)}):
                self.calls.append("edges")
                return list(EDGES[node])
            case CallTool(name=str(name)) if name in PROBES:
                self.calls.append(name)
                return 0
        raise TypeError(f"the halver answers halve, edges and the probes, not {op!r}")


def halved_id(run_id: str, n: int) -> Key:
    return compose_key(t"halved:{Run(run_id)},{Index(n)}")


def halve(run_id: str, n: int) -> Effect[int]:
    yield from append_ledger(LedgerRow(event_id=halved_id(run_id, n), kind="halved", n=n))
    return (yield from call_tool("halve", {"n": n}, int))


def settled(result: Converged[Any] | Unconverged[Any]) -> list[Any]:
    """A result as the task answers it: which case, and what it holds."""
    match result:
        case Converged(value=value):
            return ["converged", value]
        case Unconverged(value=value, cause=cause):
            return ["exhausted", value, cause]
        case unreachable:
            assert_never(unreachable)


type Stepper = Callable[[Any], Effect[Any]]


def open_fixpoint(
    step: Stepper,
    budget: int,
    converged: Callable[[Any, Any], bool] = operator.eq,
    grantor: Grantor | None = None,
):
    """The reference spelling: a level per application, its own catch of a refusal, and its own
    ask for more levels, made outside the level's scope."""

    def close(again: Callable[..., Effect[Any]]) -> Callable[..., Effect[Any]]:
        def body(value: Any, depth: int, left: int) -> Effect[Converged[Any] | Unconverged[Any]]:
            if left == 0:
                granted: tuple[int] | None = None
                with suppress(BudgetRefused):
                    granted = ((yield from granted_by(grantor, depth)),)
                match granted:
                    case None:
                        return Unconverged(value, "governed")
                    case (more,):
                        left = more
            if left == 0:
                return Unconverged(value, "levels")
            level = partial(scoped, compose_key(t"d:{Index(depth)}"), partial(step, value))
            got: tuple[Any] | None = None
            with suppress(BudgetRefused):  # a group from one refused branch matches as well
                got = ((yield from level()),)
            match got:
                case None:
                    return Unconverged(value, "governed")
                case (stepped,) if converged(value, stepped):
                    return Converged(stepped)
                case (stepped,):
                    return (yield from again(stepped, depth + 1, left - 1))
                case unreachable:
                    assert_never(unreachable)

        return body

    return close


def spellings(
    step_for: Callable[[str], Stepper],
    initial: Any,
    budget: int,
    converged: Callable[[Any, Any], bool] = operator.eq,
    grantor: Grantor | None = None,
    answer: Callable[[Any], list[Any]] = lambda result: settled(result),
) -> dict[str, Program]:
    def by_fixpoint(run_id: str) -> Effect[list[Any]]:
        step = step_for(run_id)
        result = yield from fixpoint(
            initial, step, budget=budget, converged=converged, grantor=grantor
        )
        return answer(result)

    def by_fix(run_id: str) -> Effect[list[Any]]:
        closed = fix(open_fixpoint(step_for(run_id), budget, converged, grantor))
        return answer((yield from closed(initial, 0, budget)))

    return {"fixpoint": by_fixpoint, "fix": by_fix}


def halving(
    budget: int,
    answer: list[Any],
    inputs: Sequence[int],
    *,
    refused: bool = False,
    converged: Callable[[int, int], bool] = operator.eq,
    grantor: Grantor | None = None,
    **row: Any,
) -> Shape:
    """Halving from `START` with `budget` steps: its answer, and the inputs whose rows it writes.

    A step checkpoints its row and its tool. A `refused` last step wrote its row and never reached
    its tool."""
    return Shape(
        spellings=spellings(
            lambda run_id: partial(halve, run_id), START, budget, converged, grantor
        ),
        domain=Halver,
        answer=lambda: answer,
        ledger_ids=lambda run_id: sorted(halved_id(run_id, n).stored() for n in inputs),
        count=lambda _base: 2 * len(inputs) - int(refused),
        calls=lambda halver: halver.calls,
        **row,
    )


def stopping(_depth: int) -> Effect[Grant]:
    yield from ()
    return Grant(stop=True)


def refusing_to_halve(n: int) -> Callable[[str], Policy]:
    """A gate that refuses the halving of `n` for its spend."""

    def policy(op: WorkflowOp, state: GateState) -> Verdict:
        match op:
            case Step(op=CallTool(name="halve", args={"n": int(m)})) if m == n:
                return Refuse(("halve refused",), Exceeded(spent=1.0, ceiling=1.0))
            case _:
                return Proceed()

    return lambda _run_id: policy


def refusing(
    *, spend: frozenset[str] = frozenset(), deny: frozenset[str] = frozenset()
) -> Callable[[str], Policy]:
    """A gate that refuses the tools in `spend` for their spend and denies those in `deny`."""

    def policy(op: WorkflowOp, state: GateState) -> Verdict:
        match op:
            case Step(op=CallTool(name=str(name))) if name in spend:
                return Refuse((f"{name} refused",), Exceeded(spent=1.0, ceiling=1.0))
            case Step(op=CallTool(name=str(name))) if name in deny:
                return Refuse((f"{name} denied",))
            case _:
                return Proceed()

    return lambda _run_id: policy


def asking_for_more(depth: int) -> Effect[Grant]:
    """A grantor whose ask is itself an op a gate can refuse."""
    levels = yield from call_tool("more", {"depth": depth}, int)
    return Grant(add_depth=levels)


ROWS = {
    "converges before the budget": halving(9, ["converged", 0], HALVINGS),
    "converges on the last step": halving(5, ["converged", 0], HALVINGS),
    "exhausts its levels": halving(3, ["exhausted", 1, "levels"], HALVINGS[:3]),
    "a grant extends its levels": halving(
        2, ["converged", 0], HALVINGS, grantor=granting_at(2, levels=3)
    ),
    "a grant of one level": halving(
        2, ["exhausted", 1, "levels"], HALVINGS[:3], grantor=granting_at(2, levels=1)
    ),
    "a refused ask for more levels": halving(
        2,
        ["exhausted", 2, "governed"],
        HALVINGS[:2],
        grantor=asking_for_more,
        layers=governing(refusing(spend=frozenset({"more"}))),
    ),
    "a stop grant ends it": halving(2, ["exhausted", 2, "levels"], HALVINGS[:2], grantor=stopping),
    "converges within a tolerance": halving(
        9, ["converged", 1], HALVINGS[:3], converged=lambda before, after: before - after <= 1
    ),
    "a governed refusal mid-step": halving(
        9,
        ["exhausted", 2, "governed"],
        HALVINGS[:3],
        refused=True,
        layers=governing(refusing_to_halve(2)),
    ),
}
"""Each row's budget, its answer, and the inputs a run writes a row for."""


@pytest.mark.parametrize("row", ROWS)
def test_fixpoint_reproduces_its_fix_spelling(backend, row):
    agree(backend, ROWS[row])


@pytest.mark.parametrize("spelling", ["fixpoint", "fix"])
@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
@pytest.mark.parametrize(
    "row",
    ["converges before the budget", "a governed refusal mid-step", "a grant extends its levels"],
)
def test_a_crash_at_every_checkpoint_resumes_to_the_base_run(backend, row, position, spelling):
    sweep(backend, ROWS[row], spelling, position)


def test_a_budget_of_zero_runs_no_step(backend):
    agree(backend, halving(0, ["exhausted", START, "levels"], []))


def test_a_depth_grant_park_through_run_id_extends_the_steps(backend):
    """Halving from 8 on a budget of 2 parks after two steps, on the name `descend` parks on, and
    a grant of three more lets it converge."""

    def program(run_id: str) -> Effect[list[Any]]:
        step = partial(halve, run_id)
        return settled((yield from fixpoint(START, step, budget=2, run_id=run_id)))

    outcome = run(backend, program, Halver(), max_attempts=3)
    assert outcome.snap.state not in ("completed", "failed"), outcome.snap
    (parked,) = backend.parked(outcome.task)
    park = depth_grant_name(outcome.run_id, depth=2, generation=0)
    assert str(parked.wake_event) == park.stored()

    backend.emit_event(outcome.task, parked.wake_event, Grant(add_depth=3).model_dump())
    snap = backend.run_until_result(outcome.task)
    assert snap.state == "completed", snap
    assert snap.result == ["converged", 0]
    assert outcome.domain.calls == ["halve"] * len(HALVINGS)


# --- a step that gathers: a refusal group -----------------------------------------------------

PROBES = ("probe-a", "probe-b")
"""Two tools a gathered step asks beside its halving, for a gate to refuse."""


def gathered(run_id: str, n: int) -> Effect[int]:
    """Halves `n` beside the two probes, as three branches."""
    halved, *_ = yield from gather(
        [
            partial(halve, run_id, n),
            *(partial(call_tool, probe, {"n": n}, int) for probe in PROBES),
        ]
    )
    return halved


def gathering(budget: int, answer: list[Any], inputs: Sequence[int], **row: Any) -> Shape:
    """A gathered step from `START`; the branches run in any order, so their calls are sorted."""
    return Shape(
        spellings=spellings(lambda run_id: partial(gathered, run_id), START, budget),
        domain=Halver,
        answer=lambda: answer,
        ledger_ids=lambda run_id: sorted(halved_id(run_id, n).stored() for n in inputs),
        calls=lambda halver: sorted(halver.calls),
        **row,
    )


def test_a_budget_refusal_beside_a_branch_that_went_through_is_governed(backend):
    """The halving branch wrote its row and returned; the probe refused for its spend ends the
    loop with the value the step was handed."""
    row = gathering(
        9,
        ["exhausted", 8, "governed"],
        [8],
        layers=governing(refusing(spend=frozenset({"probe-a"}))),
    )
    agree(backend, row)
    sweep(backend, row, "fixpoint")


GROUPS = {
    "a budget refusal beside a denial": refusing(
        spend=frozenset({"probe-a"}), deny=frozenset({"probe-b"})
    ),
    "a denial alone": refusing(deny=frozenset({"probe-b"})),
}
"""A gathered step whose group holds a refusal that is not the budget's: it is raised."""


@pytest.mark.parametrize("spelling", ["fixpoint", "fix"])
@pytest.mark.parametrize("group", GROUPS)
def test_a_refusal_that_is_not_the_budgets_is_raised(backend, group, spelling):
    program = spellings(lambda run_id: partial(gathered, run_id), START, 9)[spelling]
    outcome = run(backend, program, Halver(), layers=governing(GROUPS[group]), max_attempts=3)

    assert outcome.snap.state == "failed", outcome.snap
    assert backend.failure_kind(outcome.snap) == "Refused"
    assert backend.task_attempts(outcome.task) == 1


# --- a spend ceiling reached on the last permitted step ----------------------------------------


def asked_halving(n: int) -> Effect[int]:
    return (yield from ask_llm("halve", str(n), int))


def test_a_ceiling_reached_on_the_last_permitted_step_is_governed(backend):
    """Three steps of a budget of 3 each spend one ask, reaching a ceiling of 3 asks: the park for
    more levels is refused for its spend, and the loop ends governed with the value it reached."""
    halving_asks = MeteredInterpreter(
        llm=lambda op: (int(op.messages) // 2, Usage(cost=1.0)), tools=lambda _op: 0
    )

    def program(run_id: str) -> Effect[list[Any]]:
        return settled((yield from fixpoint(64, asked_halving, budget=3, run_id=run_id)))

    def ceiling(run_id: str) -> Policy:
        return budget_policy(MeasuredBudget(overall=3.0, run_id=run_id, on_exhaust="fail"))

    outcome = run(
        backend,
        program,
        halving_asks,
        layers=governing(ceiling),
        contract=Contract.V1,
        max_attempts=3,
    )
    assert outcome.snap.state == "completed", outcome.snap
    assert outcome.snap.result == ["exhausted", 8, "governed"]


# --- a worklist is a fixpoint over what is pending and what is known -------------------------

EDGES: Mapping[str, Sequence[str]] = {"a": ("b", "c", "b"), "b": ("a", "c"), "c": ("d",), "d": ()}
"""A graph with a cycle, a repeated edge and a sink."""


@dataclass(frozen=True)
class Work:
    """What is left to visit, first in first out; every node ever enqueued; and the nodes
    visited, in the order they were."""

    pending: tuple[str, ...]
    known: frozenset[str]
    visited: tuple[str, ...] = ()


def visited_id(run_id: str, node: str) -> Key:
    return compose_key(t"visited:{Run(run_id)},{Name(node)}")


def worklist_pass(run_id: str, work: Work) -> Effect[Work]:
    """Visits every pending node in order and enqueues the successors nobody has enqueued.

    A node is known from the moment it is enqueued, so a pass enqueues only nodes no pass has
    enqueued and records each node it visits: its result equals its input exactly when it had
    nothing pending."""
    pending: list[str] = []
    known = set(work.known)
    visited = list(work.visited)
    for node in work.pending:
        visited.append(node)
        successors = yield from scoped(
            compose_key(t"item:{Name(node)}"), partial(visit, run_id, node)
        )
        for successor in successors:
            if successor not in known:
                known.add(successor)
                pending.append(successor)
    return Work(tuple(pending), frozenset(known), tuple(visited))


def visit(run_id: str, node: str) -> Effect[list[str]]:
    yield from append_ledger(LedgerRow(event_id=visited_id(run_id, node), kind="visited"))
    return (yield from call_tool("edges", {"node": node}, list[str]))


def worklist_answer(result: Converged[Work] | Unconverged[Work]) -> list[Any]:
    match result:
        case Converged(value=Work(pending=pending, visited=visited)):
            return ["converged", list(pending), list(visited)]
        case Unconverged(value=Work(pending=pending, visited=visited), cause=cause):
            return ["exhausted", list(pending), list(visited), cause]
        case unreachable:
            assert_never(unreachable)


def worklist_spellings(budget: int) -> dict[str, Program]:
    initial = Work(("a",), frozenset({"a"}))
    return spellings(
        lambda run_id: partial(worklist_pass, run_id), initial, budget, answer=worklist_answer
    )


WORKLIST = Shape(
    spellings=worklist_spellings(9),
    domain=Halver,
    answer=lambda: ["converged", [], ["a", "b", "c", "d"]],  # first in, first visited
    ledger_ids=lambda run_id: sorted(visited_id(run_id, node).stored() for node in "abcd"),
    count=lambda _base: 2 * len("abcd"),
    calls=lambda halver: halver.calls,
)
"""Passes visit `a`, then `b c`, then `d`, and a fourth pass over nothing converges: each node
once, first in first out, whatever the cycle and the repeated edge offer again."""


def test_a_worklist_visits_each_node_once_and_stops_when_nothing_is_pending(backend):
    agree(backend, WORKLIST)


@pytest.mark.parametrize("spelling", ["fixpoint", "fix"])
@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_crash_at_every_checkpoint_of_a_worklist_resumes_to_the_base_run(
    backend, position, spelling
):
    sweep(backend, WORKLIST, spelling, position)


# --- the step runs as a loop, so a fixpoint goes deeper than the recursion limit --------------


def count_down(n: int) -> Effect[int]:
    yield from call_tool("tick", {}, int)
    return max(n - 1, 0)


class OnesDomain:
    def run(self, op: DomainOp[int]) -> int:
        return 1


def deep(depth: int) -> Callable[[str], Effect[list[Any]]]:
    return spellings(lambda _run_id: count_down, depth, depth + 1)["fixpoint"]


def test_the_fix_spelling_exceeds_the_recursion_limit():
    with pytest.raises(RecursionError):
        RecordingHandler(Ones()).run(lambda: fix_spelling(ENGINE_DEPTH)("r"))


def test_fixpoint_past_the_recursion_limit_on_the_recorder_and_replay():
    handler = RecordingHandler(Ones())
    assert handler.run(lambda: deep(RECORDER_DEPTH)("r")) == ["converged", 0]
    assert ReplayHandler(handler.trace).run(lambda: deep(RECORDER_DEPTH)("r")) == ["converged", 0]


def test_fixpoint_past_the_recursion_limit_on_both_engines(backend):
    outcome = run(backend, deep(ENGINE_DEPTH), OnesDomain(), max_attempts=1)
    assert outcome.snap.state == "completed", outcome.snap
    assert outcome.snap.result == ["converged", 0]


def in_a_branch(program: Program) -> Program:
    """`program` as the one branch of a gather, which runs it on a worker thread."""

    def branched(run_id: str) -> Effect[Any]:
        (value,) = yield from gather([lambda: program(run_id)])
        return value

    return branched


def fix_spelling(depth: int) -> Program:
    return spellings(lambda _run_id: count_down, depth, depth + 1)["fix"]


def test_a_branch_keeps_the_depth_arm_on_the_recorder_and_replay():
    with pytest.raises(ExceptionGroup) as raised:
        RecordingHandler(Ones()).run(lambda: in_a_branch(fix_spelling(ENGINE_DEPTH))("r"))
    assert raised.value.subgroup(RecursionError) is not None
    handler, program = RecordingHandler(Ones()), in_a_branch(deep(RECORDER_DEPTH))
    assert handler.run(lambda: program("r")) == ["converged", 0]
    assert ReplayHandler(handler.trace).run(lambda: program("r")) == ["converged", 0]


@pytest.mark.parametrize(
    ("program", "ending"),
    [(deep, "completed"), (fix_spelling, "RecursionError")],
    ids=["fixpoint completes", "fix exceeds the limit"],
)
def test_a_branch_keeps_the_depth_arm_on_both_engines(backend, program, ending):
    outcome = run(backend, in_a_branch(program(ENGINE_DEPTH)), OnesDomain(), max_attempts=1)
    match ending:
        case "completed":
            assert outcome.snap.state == "completed", outcome.snap
            assert outcome.snap.result == ["converged", 0]
        case kind:
            assert outcome.snap.state == "failed", outcome.snap
            assert backend.failure_kind(outcome.snap) == kind, outcome.snap.failure
