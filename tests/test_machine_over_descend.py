"""`run_machine` is a `descend` judge: a visit is a level, the final level mints `Exhausted`
without running the state, and a grantor adds visits before it.

The depth test is a guard: a `for` loop passed it too. It fails for a walk that keeps a frame per
visit, as a `yield from` recursion does (`tests/test_descend_depth.py`), and the recorder is its
instrument because every handler drives the same generator."""

import sys
from collections.abc import Iterator, Mapping
from enum import StrEnum
from typing import Any, assert_never
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op

from effective.api import Effect, call_tool, qualified_event_name
from effective.budget import Grant, depth_grant_name
from effective.combinators import (
    Answered,
    Deeper,
    DescendedPastBudget,
    Level,
    descend,
    grant_cascade,
    human_grant,
)
from effective.domain import CallTool, DomainOp
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Index, Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Advance, Exhausted, Outcome, Park, ParkReason
from effective.machine.spec import Ctx, Report, StateSpec
from effective.machine.trampoline import SUITE_TOOL, Session, run_machine, stop_record

DEPTH = 2 * sys.getrecursionlimit()
GREEN = CommandRun(exit_code=0)


class Loop(StrEnum):
    SPIN = "spin"


class Spun(StrEnum):
    AGAIN = "again"


def spin(ctx: Ctx[Loop]) -> Effect[Report[Spun]]:
    yield from call_tool("turn", {}, int)
    return Report(Spun.AGAIN)


SPECS = {Loop.SPIN: StateSpec(Loop.SPIN, spin)}


def park_when_exhausted(state: Loop, verdict: Spun | Exhausted[Loop]) -> Outcome[Loop]:
    match verdict:
        case Exhausted():
            return Park(state, ParkReason.EXHAUSTED)
        case Spun.AGAIN:
            return Advance(Loop.SPIN)
        case unreachable:
            assert_never(unreachable)


def advance_always(state: Loop, verdict: Spun | Exhausted[Loop]) -> Outcome[Loop]:
    return Advance(Loop.SPIN)


def machine(
    budget: int, transition=park_when_exhausted, grantor=None, run_id: str = "spin-1"
) -> Effect[Session]:
    return (
        yield from run_machine(
            Run(run_id), "spin", SPECS, transition, start=Loop.SPIN, budget=budget, grantor=grantor
        )
    )


class Answers(Mapping[str, Any]):
    """The suite is green and every other tool answers 1. A `Mapping` the recorder reads as
    given, where it copies a `dict`."""

    def __getitem__(self, key: str) -> Any:
        return GREEN if key.endswith(SUITE_TOOL) else 1

    def __contains__(self, key: object) -> bool:
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


class SpinDomain:
    def __init__(self) -> None:
        self.turns = 0

    def run(self, op: DomainOp[Any]) -> Any:
        assert isinstance(op, CallTool)
        if op.name == SUITE_TOOL:
            return GREEN
        self.turns += 1
        return 1


def walk(handler: RecordingHandler, budget: int, **kwargs: Any) -> Session:
    match handler.run(lambda: machine(budget, **kwargs)):
        case Suspended() as parked:
            raise AssertionError(f"no park was expected: {parked}")
        case session:
            return session


def turns_asked(handler: RecordingHandler) -> int:
    return sum(1 for entry in handler.trace if entry.key.stored().endswith("tool:turn"))


def test_run_machine_past_the_recursion_limit_on_the_recorder_and_replay():
    handler = RecordingHandler(Answers())
    session = walk(handler, DEPTH)
    assert len(session.turns) == DEPTH + 1
    assert stop_record(session.stopped).kind == "machine-parked"
    replayed = ReplayHandler(handler.trace).run(lambda: machine(DEPTH))
    assert replayed.turns == session.turns


def test_a_zero_budget_is_one_final_visit_that_runs_no_state():
    handler = RecordingHandler(Answers())
    session = walk(handler, 0)
    assert [turn.verdict for turn in session.turns] == [Exhausted(Loop.SPIN, level=0)]
    assert turns_asked(handler) == 0
    assert [row.kind for row in handler.ledger] == ["machine-committed", "machine-parked"]


def advancing(run_id: str) -> Effect[Session]:
    return (yield from machine(1, transition=advance_always, run_id=run_id))


def test_an_advance_at_the_final_visit_fails_once_before_the_record_on_both_engines(backend):
    name, run_id = compose_key(t"advancing:{Run(str(uuid4()))}").stored(), str(uuid4())
    domain = SpinDomain()
    backend.register(name, advancing, domain, Fault(), [])
    task_id = backend.spawn(name, run_id, max_attempts=3)
    snap = backend.run_until_result(task_id)
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == DescendedPastBudget.__name__
    assert backend.task_attempts(task_id) == 1
    assert backend.ledger_kinds(run_id) == []
    assert domain.turns == 1


def add_two_visits_once(depth: int) -> Effect[Grant]:
    return Grant(add_depth=2) if depth == 1 else Grant(stop=True)
    yield  # pragma: no cover  -- a `Grantor` is a generator


def granted(run_id: str) -> Effect[dict[str, Any]]:
    session = yield from machine(1, grantor=add_two_visits_once, run_id=run_id)
    return {"visits": len(session.turns), "stopped": stop_record(session.stopped).kind}


def test_a_grantor_adds_visits_on_both_engines(backend):
    name, run_id = compose_key(t"granted:{Run(str(uuid4()))}").stored(), str(uuid4())
    domain = SpinDomain()
    backend.register(name, granted, domain, Fault(), [])
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    assert snap.state == "completed", snap
    assert snap.result == {"visits": 4, "stopped": "machine-parked"}
    assert domain.turns == 3


def test_the_final_visit_reports_the_level_it_exhausted_at():
    handler = RecordingHandler(Answers())
    session = walk(handler, 1, grantor=add_two_visits_once)
    assert session.turns[-1].verdict == Exhausted(Loop.SPIN, level=3)


def test_an_exhausted_run_concludes_from_the_last_visit_that_ran():
    """The last TURN and the last VISIT are different turns once a run exhausts.

    `_visit` mints `Exhausted` without running the state, so the final turn carries no report,
    and a caller reading `turns[-1]` reads nothing on the stop small-model scale makes ordinary."""
    handler = RecordingHandler(Answers())
    session = walk(handler, 1, grantor=add_two_visits_once)
    assert session.turns[-1].report is None
    assert session.concluded is session.turns[-2].report


def granted_by_a_human(run_id: str) -> Effect[dict[str, Any]]:
    grantor = grant_cascade([human_grant(run_id)])
    session = yield from machine(1, grantor=grantor, run_id=run_id)
    return {"visits": len(session.turns), "stopped": stop_record(session.stopped).kind}


def answer_depth_grants(backend, task_id, run_id: str, grants: list[dict[str, Any]]):
    """Answer each park in turn; the machine's own parks sit outside every level, unframed."""
    depth = 1
    for grant in grants:
        snap = backend.run_until_result(task_id)
        assert snap.state not in ("completed", "failed"), (depth, snap)
        name = depth_grant_name(run_id, generation=0, depth=depth)
        backend.emit_event(task_id, name.stored(), grant)
        depth += grant.get("add_depth", 0)
    return backend.run_until_result(task_id)


GRANT_CYCLE_OPS = {FaultPosition.BEFORE_OP: 14, FaultPosition.AFTER_THUNK: 6}
"""Where a crash can land in a park, grant, park and stop run: before each of its ops, or after
each step's thunk, which the awaits and scope frames do not run."""


def run_granted(backend, fault: Fault):
    name, run_id = compose_key(t"human-granted:{Run(str(uuid4()))}").stored(), str(uuid4())
    domain = SpinDomain()
    backend.register(name, granted_by_a_human, domain, fault, [])
    grants = [{"add_depth": 1}, {"stop": True}]
    snap = answer_depth_grants(backend, backend.spawn(name, run_id), run_id, grants)
    return snap, domain, run_id


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_human_grant_adds_visits_across_a_crash_at_every_op_on_both_engines(backend, position):
    """A park for more visits, a grant, a second park, a stop: the refill path between levels. A
    crash lands at every position the run has, and it converges to the same visits and rows."""
    unarmed = Fault(position=position)
    snap, _, _ = run_granted(backend, unarmed)
    assert snap.state == "completed", snap
    assert unarmed.count == GRANT_CYCLE_OPS[position], (
        "the walk changed shape; re-derive the bound"
    )
    for k, fault in at_every_op(unarmed):
        snap, domain, run_id = run_granted(backend, fault)
        assert snap.state == "completed", (k, snap)
        assert snap.result == {"visits": 3, "stopped": "machine-parked"}, k
        assert backend.ledger_kinds(run_id) == ["machine-committed", "machine-parked"], k
        if position is FaultPosition.BEFORE_OP:
            assert domain.turns == 2, (k, domain.turns)


def machines_in_descend_levels(run_id: str, *, own_ids: bool) -> Effect[int]:
    """A `descend` whose every level runs a machine with a human grantor, under the descend's
    `run_id` or under the machine's own."""

    def judge(visits: int, level: Level) -> Effect[Answered[int] | Deeper[int]]:
        inner = f"{run_id}-{level.depth}"
        grantor = grant_cascade([human_grant(inner if own_ids else run_id)])
        session = yield from machine(0, grantor=grantor, run_id=inner)
        total = visits + len(session.turns)
        return Answered(total) if level.final else Deeper(total)

    return (yield from descend(0, judge, budget=1, run_id=run_id))


def own_grant_ids(run_id: str) -> Effect[int]:
    return (yield from machines_in_descend_levels(run_id, own_ids=True))


def shared_grant_ids(run_id: str) -> Effect[int]:
    return (yield from machines_in_descend_levels(run_id, own_ids=False))


@pytest.mark.parametrize("own_ids", [True, False], ids=["own-ids", "shared-id"])
def test_a_nested_machine_granting_under_its_own_id_is_woken_by_its_frames(backend, own_ids):
    """Machines at `d:0` and `d:1` of a `descend`, each parking for a grant, with the descend's own
    refill between them. Granting under the machine's own id, a park is named by its frames. Under
    the shared id, the second machine's park is the second ask of one name in the task, so its
    emitter needs the occurrence `#2`."""
    # A name atom, so the machines' own ids `{run_id}-{level}` are atoms too.
    name, run_id = compose_key(t"nested-grants:{Run(str(uuid4()))}").stored(), f"r{uuid4().hex}"
    workflow = own_grant_ids if own_ids else shared_grant_ids
    backend.register(name, workflow, SpinDomain(), Fault(), [])
    task_id = backend.spawn(name, run_id)

    def machine_park(level: int):
        grantee = f"{run_id}-{level}" if own_ids else run_id
        bare = depth_grant_name(grantee, generation=0, depth=0).stored()
        return qualified_event_name(compose_key(t"d:{Index(level)}"), name=bare)

    second = machine_park(1) if own_ids else machine_park(1).occurrence(2)
    parks = [machine_park(0), depth_grant_name(run_id, generation=0, depth=1), second]
    for park in parks:
        snap = backend.run_until_result(task_id)
        assert snap.state not in ("completed", "failed"), (park, snap)
        backend.emit_event(task_id, park.stored(), {"stop": True})
    snap = backend.run_until_result(task_id)
    assert snap.state == "completed", snap
    assert snap.result == 2
