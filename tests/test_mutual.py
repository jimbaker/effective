"""`mutual` on both engines, spelled by the combinator and by open recursion closed with `fix`.

The `fix` spelling is the reference: one body over the tagged hand, each role run under
`d:{n};state:{role}` and the recursive call outside those scopes. Each row is a
`_shapes.Shape`, compared by `agree` and crashed by `sweep`.

Every role asks the domain once and appends a ledger row for the number it was handed, before it
answers or hands off, so a run writes one row per hop."""

from collections.abc import Callable, Mapping
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
    ledger_ids,
    run,
    sweep,
)

from effective.api import Effect, append_ledger, call_tool, scoped
from effective.budget import Grant, depth_grant_name
from effective.combinators import (
    Answered,
    DescendedPastBudget,
    Grantor,
    Hand,
    Level,
    Role,
    fix,
    mutual,
)
from effective.domain import CallTool, DomainOp
from effective.govern import Exceeded, GateState, Policy, Proceed, Refuse, Verdict
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Index, Key, Name, Run, compose_key
from effective.ops import CompositionRefused, LedgerRow, Step, WorkflowOp

type Roles = Mapping[str, Role[str, int, Any]]


class Ticks:
    """The domain: records which role asked, and answers 1."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="tick", args={"role": str(role)}):
                self.calls.append(role)
                return 1
        raise TypeError(f"the ticks answer tick, not {op!r}")


def hop_id(run_id: str, n: int) -> Key:
    return compose_key(t"hop:{Run(run_id)},{Index(n)}")


def hop(run_id: str, role: str, n: int) -> Effect[None]:
    yield from call_tool("tick", {"role": role}, int)
    yield from append_ledger(LedgerRow(event_id=hop_id(run_id, n), kind="hop", role=role))


# --- the roles ---------------------------------------------------------------------------------


def parity(run_id: str) -> Roles:
    """`even` and `odd`, each answering whether the number it holds is its parity."""

    def even(n: int, level: Level) -> Effect[Answered[bool] | Hand[str, int]]:
        yield from hop(run_id, "even", n)
        return Answered(True) if n == 0 else Hand("odd", n - 1)

    def odd(n: int, level: Level) -> Effect[Answered[bool] | Hand[str, int]]:
        yield from hop(run_id, "odd", n)
        return Answered(False) if n == 0 else Hand("even", n - 1)

    return {"even": even, "odd": odd}


def rotation(run_id: str) -> Roles:
    """Three roles counting down in turn: the one handed 0 answers its own name."""
    order = ["a", "b", "c"]

    def role(name: str) -> Role[str, int, Any]:
        def body(n: int, level: Level) -> Effect[Answered[str] | Hand[str, int]]:
            yield from hop(run_id, name, n)
            after = order[(order.index(name) + 1) % len(order)]
            return Answered(name) if n == 0 else Hand(after, n - 1)

        return body

    return {name: role(name) for name in order}


# --- the reference spelling ------------------------------------------------------------------


def open_mutual(roles: Roles, budget: int, grantor: Grantor | None = None):
    """The reference spelling: a level per hop, and its own ask for more levels when the budget
    is spent, made outside the level's scope. A role is final when no more were granted."""

    def close(again: Callable[..., Effect[Any]]):
        def body(hand: Hand[str, int], depth: int, left: int) -> Effect[Any]:
            if hand.role not in roles:
                raise CompositionRefused(f"{hand.role!r} is not a role")
            if left == 0:
                left = yield from granted_by(grantor, depth)
            level = Level(depth=depth, model="", final=left == 0)
            role = partial(roles[hand.role], hand.value, level)
            state = partial(scoped, compose_key(t"state:{Name(hand.role)}"), role)
            match (yield from scoped(compose_key(t"d:{Index(depth)}"), state)):
                case Answered(value=value):
                    return value
                case Hand() as handed if handed.role not in roles:
                    raise CompositionRefused(f"{handed.role!r} is not a role")
                case Hand() as handed if level.final:
                    raise DescendedPastBudget(f"{handed.role!r} was handed the final level")
                case Hand() as handed:
                    return (yield from again(handed, depth + 1, left - 1))
                case unreachable:
                    assert_never(unreachable)

        return body

    return close


def spellings(
    roles_for: Callable[[str], Roles],
    start: Hand[str, int],
    budget: int,
    grantor: Grantor | None = None,
) -> dict[str, Program]:
    def by_mutual(run_id: str) -> Effect[Any]:
        return (yield from mutual(start, roles_for(run_id), budget=budget, grantor=grantor))

    def by_fix(run_id: str) -> Effect[Any]:
        return (yield from fix(open_mutual(roles_for(run_id), budget, grantor))(start, 0, budget))

    return {"mutual": by_mutual, "fix": by_fix}


def row(
    roles_for: Callable[[str], Roles],
    start: Hand[str, int],
    answer: Any,
    *,
    budget: int = 9,
    grantor: Grantor | None = None,
) -> Shape:
    """A run from `start` and its answer. Each hop writes the row for the number it held,
    counting down from `start.value`."""
    return Shape(
        spellings=spellings(roles_for, start, budget, grantor),
        domain=Ticks,
        answer=lambda: answer,
        ledger_ids=lambda run_id: sorted(
            hop_id(run_id, n).stored() for n in range(start.value + 1)
        ),
        count=lambda _base: 2 * (start.value + 1),
        calls=lambda ticks: ticks.calls,
    )


ROWS = {
    "two roles": row(parity, Hand("even", 5), False),
    "three roles": row(rotation, Hand("a", 7), "b"),
    "a grant extends the hops": row(
        parity, Hand("even", 5), False, budget=2, grantor=granting_at(2, levels=5)
    ),
}

CALLERS = {
    "two roles": ["even", "odd"] * 3,
    "three roles": ["a", "b", "c"] * 2 + ["a", "b"],
    "a grant extends the hops": ["even", "odd"] * 3,
}
"""The roles each row runs, in the order it runs them."""


@pytest.mark.parametrize("name", ROWS)
def test_mutual_reproduces_its_fix_spelling(backend, name):
    outcomes = agree(backend, ROWS[name])
    assert all(outcome.domain.calls == CALLERS[name] for outcome in outcomes.values())


@pytest.mark.parametrize("spelling", ["mutual", "fix"])
@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
@pytest.mark.parametrize("name", ROWS)
def test_a_crash_at_every_checkpoint_resumes_to_the_base_run(backend, name, position, spelling):
    sweep(backend, ROWS[name], spelling, position)


# --- the refusals ------------------------------------------------------------------------------


def handing_to_nobody(run_id: str) -> Roles:
    def first(n: int, level: Level) -> Effect[Answered[str] | Hand[str, int]]:
        yield from hop(run_id, "first", n)
        return Hand("nobody", n - 1)

    return {"first": first}


@pytest.mark.parametrize("spelling", ["mutual", "fix"])
def test_a_hand_to_a_role_that_is_not_one_is_refused_before_it_runs(backend, spelling):
    program = spellings(handing_to_nobody, Hand("first", 3), 9)[spelling]
    outcome = run(backend, program, Ticks(), max_attempts=3)

    assert outcome.snap.state == "failed", outcome.snap
    assert backend.failure_kind(outcome.snap) == "CompositionRefused"
    assert backend.task_attempts(outcome.task) == 1
    assert outcome.domain.calls == ["first"], "the role that handed off ran, and nothing after"


@pytest.mark.parametrize("spelling", ["mutual", "fix"])
def test_a_start_that_is_not_a_role_is_refused_before_anything_runs(backend, spelling):
    program = spellings(parity, Hand("nobody", 3), 9)[spelling]
    outcome = run(backend, program, Ticks(), max_attempts=3)

    assert backend.failure_kind(outcome.snap) == "CompositionRefused"
    assert outcome.domain.calls == []


SHORT = {
    "a budget of 2": (None, ["even", "odd", "even"]),
    "a grant one level short": (granting_at(2, levels=2), ["even", "odd"] * 2 + ["even"]),
}
"""Parity from 5 takes six hops. A budget of 2 makes the third hop final, and a grant of 2 more at
that hop makes the fifth final; the final role hands off either way."""


@pytest.mark.parametrize("spelling", ["mutual", "fix"])
@pytest.mark.parametrize("short", SHORT)
def test_a_hand_at_the_final_level_descends_past_the_budget(backend, short, spelling):
    grantor, callers = SHORT[short]
    program = spellings(parity, Hand("even", 5), 2, grantor)[spelling]
    outcome = run(backend, program, Ticks(), max_attempts=3)

    assert outcome.snap.state == "failed", outcome.snap
    assert backend.failure_kind(outcome.snap) == "DescendedPastBudget"
    assert backend.task_attempts(outcome.task) == 1
    assert outcome.domain.calls == callers


@pytest.mark.parametrize(
    "roles", [{"bad:role": None}, {"even": None, "bad role": None}], ids=["start", "another"]
)
def test_a_role_name_that_cannot_be_a_key_is_refused_before_any_role_runs(backend, roles):
    def program(run_id: str) -> Effect[Any]:
        table = {name: parity(run_id)["even"] for name in roles}
        return (yield from mutual(Hand(next(iter(roles)), 3), table, budget=9))

    outcome = run(backend, program, Ticks(), max_attempts=3)

    assert backend.failure_kind(outcome.snap) == "UnusableRoleName"
    assert backend.task_attempts(outcome.task) == 1
    assert outcome.domain.calls == []


# --- the hops run as a loop, so mutual recursion goes deeper than the recursion limit --------


def ping_pong(_run_id: str) -> Roles:
    def ping(n: int, level: Level) -> Effect[Answered[str] | Hand[str, int]]:
        yield from call_tool("tick", {"role": "ping"}, int)
        return Answered("ping") if n == 0 else Hand("pong", n - 1)

    def pong(n: int, level: Level) -> Effect[Answered[str] | Hand[str, int]]:
        yield from call_tool("tick", {"role": "pong"}, int)
        return Answered("pong") if n == 0 else Hand("ping", n - 1)

    return {"ping": ping, "pong": pong}


def deep(depth: int, spelling: str) -> Program:
    return spellings(ping_pong, Hand("ping", depth), depth)[spelling]


def test_the_fix_spelling_exceeds_the_recursion_limit():
    with pytest.raises(RecursionError):
        RecordingHandler(Ones()).run(lambda: deep(ENGINE_DEPTH, "fix")("r"))


def test_mutual_past_the_recursion_limit_on_the_recorder_and_replay():
    answer = "ping" if RECORDER_DEPTH % 2 == 0 else "pong"
    handler = RecordingHandler(Ones())
    assert handler.run(lambda: deep(RECORDER_DEPTH, "mutual")("r")) == answer
    assert ReplayHandler(handler.trace).run(lambda: deep(RECORDER_DEPTH, "mutual")("r")) == answer


def test_mutual_past_the_recursion_limit_on_both_engines(backend):
    outcome = run(backend, deep(ENGINE_DEPTH, "mutual"), Ticks(), max_attempts=1)
    assert outcome.snap.state == "completed", outcome.snap
    assert outcome.snap.result == ("ping" if ENGINE_DEPTH % 2 == 0 else "pong")


# --- the contract's other clauses ---------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["mutual", "fix"])
def test_a_hand_to_a_role_that_is_not_one_at_the_final_level_is_refused(backend, spelling):
    """The hand is checked before the level: it names no role, whatever level it came from."""
    program = spellings(handing_to_nobody, Hand("first", 3), 0)[spelling]
    outcome = run(backend, program, Ticks(), max_attempts=3)

    assert backend.failure_kind(outcome.snap) == "CompositionRefused"
    assert backend.task_attempts(outcome.task) == 1


def refusing_the_tick_of(role: str) -> Callable[[str], Policy]:
    """A gate that refuses `role`'s tick for its spend."""

    def policy(op: WorkflowOp, state: GateState) -> Verdict:
        match op:
            case Step(op=CallTool(name="tick", args={"role": str(asker)})) if asker == role:
                return Refuse((f"{role} refused",), Exceeded(spent=1.0, ceiling=1.0))
            case _:
                return Proceed()

    return lambda _run_id: policy


@pytest.mark.parametrize("spelling", ["mutual", "fix"])
def test_a_refusal_inside_a_role_propagates(backend, spelling):
    """Parity from 5 reaches `odd` second; its tick is refused, and the row `even` wrote stands."""
    program = spellings(parity, Hand("even", 5), 9)[spelling]
    outcome = run(
        backend, program, Ticks(), layers=governing(refusing_the_tick_of("odd")), max_attempts=3
    )

    assert outcome.snap.state == "failed", outcome.snap
    assert backend.failure_kind(outcome.snap) == "BudgetRefused"
    assert backend.task_attempts(outcome.task) == 1
    assert outcome.domain.calls == ["even"]
    assert ledger_ids(backend, outcome.run_id) == [hop_id(outcome.run_id, 5).stored()]


def test_a_depth_grant_park_through_run_id_extends_the_hops(backend):
    """Parity from 5 takes six hops. On a budget of 2 it parks at the third, on the name `descend`
    parks on, and a grant of the four hops left lets it answer."""

    def program(run_id: str) -> Effect[Any]:
        return (yield from mutual(Hand("even", 5), parity(run_id), budget=2, run_id=run_id))

    outcome = run(backend, program, Ticks(), max_attempts=3)
    assert outcome.snap.state not in ("completed", "failed"), outcome.snap
    (parked,) = backend.parked(outcome.task)
    assert (
        str(parked.wake_event) == depth_grant_name(outcome.run_id, depth=2, generation=0).stored()
    )

    backend.emit_event(outcome.task, parked.wake_event, Grant(add_depth=4).model_dump())
    snap = backend.run_until_result(outcome.task)
    assert snap.state == "completed", snap
    assert snap.result is False
    assert outcome.domain.calls == ["even", "odd"] * 3


def test_the_roles_are_the_ones_handed_in_when_it_starts(backend):
    """A role that rewrites the mapping it was handed changes nothing: the recursion keeps the
    roles it started with."""

    def program(run_id: str) -> Effect[Any]:
        roles = dict(parity(run_id))
        even = roles["even"]

        def rewriting(n: int, level: Level) -> Effect[Answered[bool] | Hand[str, int]]:
            roles["odd"] = even
            return (yield from even(n, level))

        roles["even"] = rewriting
        return (yield from mutual(Hand("even", 3), roles, budget=9))

    outcome = run(backend, program, Ticks())
    assert outcome.snap.result is False
    assert outcome.domain.calls == ["even", "odd", "even", "odd"]
