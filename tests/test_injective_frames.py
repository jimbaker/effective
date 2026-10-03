"""A scope entered twice keeps its ops apart, on both engines.

A gather's ordinal, and the occurrence of a settlement await, count across every entry of a scope
in one handler. So the second entry of `scoped(K){…}` names its positional ops and settlement
awaits differently from the first, and nothing the first entry wrote or settled under a settlement
name can answer or hide the second's. An await whose name declares no settlement is one question
however often it is asked, so a scope entered twice shares it. Each case below is a composition
that shares a name if a scope restarts its handler's ordinals.
"""

from typing import Any
from uuid import uuid4

import pytest
from _conformance import Approval, Fault, private

from effective.api import append_ledger, call_tool, gather, scoped, step
from effective.budget import Grant, MeasuredBudget, depth_grant_name
from effective.combinators import Answered, Deeper, Level, descend
from effective.cost import Usage
from effective.domain import CallTool, DomainOp
from effective.fork import OpIndex, fork_at, measured_drive, replay_prefix
from effective.handlers.absurd import DurableHandler
from effective.handlers.base import TraceEntry, op_key
from effective.handlers.recording import RecordingHandler, Suspended
from effective.keys import Key, Run, Segment, compose_key
from effective.ops import LedgerRow, Step
from effective.permission import APPROVE, Allow, Escalate, cascade, human, rules

ROUND = compose_key(t"round")
OUTER = compose_key(t"outer")
FIRST_ASK = depth_grant_name("r", depth=1, generation=0)


class Counting:
    """Answers every tool with its call count."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name=str(name)):
                self.calls.append(name)
                return len(self.calls)
        raise TypeError(op)


def twice(body):
    def workflow(_run_id: str):
        first = yield from scoped(ROUND, body)
        second = yield from scoped(ROUND, body)
        return [first, second]

    return workflow


def test_two_appends_of_one_event_id_in_a_scope_entered_twice_are_refused(backend):
    """The second entry's append is a second writer of one canonical row, so it is refused; with
    the restart it was the first writer again, and the row it meant to write was lost."""
    run_id, task = str(uuid4()), private("rows")
    event_id = compose_key(t"once:{Run(run_id)}")

    def body():
        yield from gather([lambda: append_ledger(LedgerRow(event_id=event_id, kind="k"))])
        return "appended"

    backend.register(task, twice(body), Counting(), Fault(), [])
    snapshot = backend.run_until_result(backend.spawn(task, run_id, max_attempts=1))

    assert snapshot.state == "failed", snapshot
    assert backend.failure_kind(snapshot) == "PlacedWriterCollision"


def _gate_act(op):
    if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name == "act":
        return Escalate("needs a ruling")
    return Allow()


def test_one_approval_settles_one_scope_entry(backend):
    """Each entry's gated op parks for its own approval; one ruling runs one op."""
    run_id, task, domain = str(uuid4()), private("approve"), Counting()
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

    def body():
        (acted,) = yield from gather([lambda: call_tool("act", {}, int)])
        return acted

    backend.register(task, twice(body), domain, Fault(), (gate,))
    task_id = backend.spawn(task, run_id)
    backend.run_until_result(task_id)
    (first,) = backend.parked(task_id)

    backend.emit_event(task_id, first.wake_event, {"decision": "approve"})
    snapshot = backend.run_until_result(task_id)
    assert snapshot.state != "completed", snapshot
    assert domain.calls == ["act"]
    (second,) = backend.parked(task_id)
    assert second.wake_event != first.wake_event

    backend.emit_event(task_id, second.wake_event, {"decision": "approve"})
    snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    assert domain.calls == ["act", "act"]


def _judge(context: str, level: Level):
    """Looks once per level, and answers only when the level is final."""
    yield from call_tool("judge", {}, int)
    return Answered(context) if level.final else Deeper(context)


def _drill(run_id: str):
    return lambda: descend("ctx", _judge, budget=1, run_id=run_id)


def test_a_grant_park_in_a_scope_entered_twice_is_asked_twice(backend):
    """A settlement await counts its occurrences across scope entries, so the second descent's
    grant is its own ask rather than the first descent's, already answered."""
    run_id, task = str(uuid4()), private("grant")
    backend.register(task, twice(_drill(run_id)), Counting(), Fault(), [])
    task_id = backend.spawn(task, run_id)
    backend.run_until_result(task_id)
    (first,) = backend.parked(task_id)

    backend.emit_event(task_id, first.wake_event, {"add_depth": 0})
    snapshot = backend.run_until_result(task_id)
    assert snapshot.state != "completed", snapshot
    (second,) = backend.parked(task_id)
    assert second.wake_event != first.wake_event

    backend.emit_event(task_id, second.wake_event, {"add_depth": 0})
    snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    assert snapshot.result == ["ctx", "ctx"]


def test_the_recorder_asks_a_scope_entered_twice_for_two_grants_by_different_names():
    handler = RecordingHandler(responses={"tool:judge": 0})
    parked = handler.run(lambda: twice(_drill("r"))("r"))
    assert isinstance(parked, Suspended)
    first = parked.awaiting.stored()

    parked = parked.resume(Grant(add_depth=0))
    assert isinstance(parked, Suspended)
    assert parked.awaiting.stored() != first


class CrashOnce(Counting):
    """Raises before its fourth call runs, once: a crash inside the second entry's gather."""

    def run(self, op: DomainOp[Any]) -> Any:
        if len(self.calls) == 3 and "crashed" not in self.calls:
            self.calls.append("crashed")
            raise RuntimeError("crash in the second entry's gather")
        return super().run(op)


def test_a_crash_inside_a_scope_entered_twice_resumes_every_branch(backend):
    """Concurrent branches in both entries re-bind their own checkpoints after a crash."""
    run_id, task, domain = str(uuid4()), private("crash"), CrashOnce()

    def body():
        return (
            yield from gather([lambda: call_tool("a", {}, int), lambda: call_tool("b", {}, int)])
        )

    backend.register(task, twice(body), domain, Fault(), [])
    snapshot = backend.run_until_result(backend.spawn(task, run_id))

    assert snapshot.state == "completed", snapshot
    assert [name for name in domain.calls if name != "crashed"].count("a") == 2
    assert len([name for name in domain.calls if name != "crashed"]) == 4


# --- the fork walks count the same way --------------------------------------------------------


class Looks:
    """A metered domain for a fork tail: every look is free and answers 0."""

    def run_metered(self, op: object) -> tuple[int, Usage]:
        return 0, Usage()


def _asks_twice(run_id: str, *, nested: bool):
    """Two entries of one scope, each descending to a grant park, after a top-level op a fork can
    substitute; `nested` puts both entries inside an outer scope, a walk's recursion one level
    down."""

    def entries():
        return (yield from twice(_drill(run_id))(run_id))

    def program():
        yield from step("start", CallTool(name="op", result_schema=str))
        if nested:
            return (yield from scoped(OUTER, entries))
        return (yield from entries())

    return program


def _recorded(program) -> tuple[list[TraceEntry], list[Key]]:
    """The base run's trace, answering each grant with 0, and the names it parked on in order."""
    handler = RecordingHandler(responses={"start": "s", "tool:judge": 0})
    parked, asked = handler.run(program), []
    while isinstance(parked, Suspended):
        asked.append(parked.awaiting)
        parked = parked.resume(Grant(add_depth=0))
    return handler.trace, asked


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_a_fork_prefix_replays_a_scope_entered_twice(nested):
    program = _asks_twice("r", nested=nested)
    trace, asked = _recorded(program)
    assert len(asked) == len(set(asked)) == 2

    replay_prefix(program, trace, OpIndex(len(trace)))


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_a_fork_tail_asks_the_second_entry_for_its_own_grant(nested):
    """Answer only the first entry's grant: the tail must park on the second's. A fork's `grants`
    are keyed by the await's own name, which the scope does not prefix."""
    program = _asks_twice("r", nested=nested)
    trace, (_, second) = _recorded(program)

    tail = fork_at(
        program, trace, OpIndex(0), "s", Looks(), grants={FIRST_ASK: Grant(add_depth=0)}
    )
    assert tail.parked_at == second


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_a_measured_drive_asks_the_second_entry_for_its_own_grant(nested):
    program = _asks_twice("r", nested=nested)
    _, (_, second) = _recorded(program)

    tail = measured_drive(
        program,
        MeasuredBudget(overall=1e9, run_id="r"),
        Looks(),
        {FIRST_ASK: Grant(add_depth=0)},
    )
    assert tail.tripped_at == second


# --- one handler run per claim --------------------------------------------------------------


def test_a_second_handler_run_over_one_claim_is_kept_apart_only_by_the_engine(backend):
    """A handler run is one claim's walk. A second run over the same ctx restarts the handler's
    counts, so it places its ops as the first did; the engine's per-claim count is what keeps their
    checkpoints apart, and both runs' effects happen."""
    task, domain = private("two-runs"), Counting()

    def body(params, ctx):
        first = DurableHandler(ctx, domain).run(lambda: call_tool("a", {}, int))
        second = DurableHandler(ctx, domain).run(lambda: call_tool("a", {}, int))
        return [first, second]

    backend.register_body(task, body)
    task_id = backend.spawn(task, str(uuid4()))
    snapshot = backend.run_until_result(task_id)

    assert snapshot.state == "completed", snapshot
    assert snapshot.result == [1, 2]
    assert sorted(backend.checkpoint_keys(task_id)) == ["step;tool:a", "step;tool:a#2"]
