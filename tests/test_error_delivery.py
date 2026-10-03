"""An error out of a `scoped` or `gather` body reaches the workflow that yielded it when every leaf
is a refusal, on the recorder, replay and both engines. A crash stays task-level.

A refusal re-derives on a retry, so a workflow that catches it takes the same path on every
attempt and on replay. That is the rule a spawned child's join already follows (`stopped_at`)."""

from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault

from effective.api import Effect, await_event, call_tool, gather, scoped
from effective.domain import CallTool, DomainOp
from effective.govern import Refused
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Run, compose_key
from effective.ops import Step, leaves
from effective.permission import Allow, Deny, cascade, rules


def deny_refused_tools(op: Any) -> Any:
    if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name.startswith("no-"):
        return Deny(f"{op.op.name} is refused")
    return Allow()


def tool(name: str) -> Callable[[], Effect[int]]:
    def body() -> Effect[int]:
        return (yield from call_tool(name, {}, int))

    return body


def waits(run_id: str) -> Effect[str]:
    """Parks on a name its run owns, since an engine keeps an emitted event for the queue."""
    return (yield from await_event(compose_key(t"wake:{Run(run_id)}"), str))


def refused_beside_a_park(run_id: str) -> Effect[str]:
    outcome = "not refused"
    try:
        yield from gather([partial(waits, run_id), tool("no-branch")])
    except* Refused:
        outcome = "caught"
    return outcome


def refused_after_a_park(run_id: str) -> Effect[str]:
    def body() -> Effect[int]:
        yield from waits(run_id)
        return (yield from tool("no-after")())

    try:
        yield from scoped(compose_key(t"s"), body)
    except Refused:
        return "caught"
    return "not refused"


def crashes() -> Effect[int]:
    yield from call_tool("ok", {}, int)
    raise ValueError("a crash in the body")


def scoped_refusal(_run_id: str) -> Effect[str]:
    try:
        yield from scoped(compose_key(t"s"), tool("no-scoped"))
    except Refused:
        return "caught"
    return "not refused"


def gathered_refusal(_run_id: str) -> Effect[str]:
    outcome = "not refused"
    try:
        yield from gather([tool("ok"), tool("no-branch")])
    except* Refused:
        outcome = "caught"
    return outcome


def scoped_crash(_run_id: str) -> Effect[str]:
    try:
        yield from scoped(compose_key(t"s"), crashes)
    except Exception:
        return "caught a crash"
    return "no crash"


def gathered_refusal_beside_crash(_run_id: str) -> Effect[str]:
    outcome = "nothing raised"
    try:
        yield from gather([tool("no-branch"), crashes])
    except* Refused:
        outcome = "caught a refusal beside a crash"
    return outcome


TASK_LEVEL = {
    "scoped crash": scoped_crash,
    "refusal beside a crash": gathered_refusal_beside_crash,
}


class Ones(Mapping[str, object]):
    """Every tool answers 1."""

    def __getitem__(self, key: str) -> object:
        return 1

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: object) -> bool:
        return True


class OnesDomain:
    def run(self, op: DomainOp[Any]) -> Any:
        return 1


def record() -> RecordingHandler:
    return RecordingHandler(Ones(), op_layers=[cascade([rules(deny_refused_tools)])])


DELIVERED = {"scoped": scoped_refusal, "gather": gathered_refusal}


@pytest.mark.parametrize("program", DELIVERED.values(), ids=DELIVERED)
def test_a_refusal_reaches_the_workflow_on_the_recorder_and_on_replay(program):
    handler = record()
    assert handler.run(lambda: program("r")) == "caught"
    assert ReplayHandler(handler.trace).run(lambda: program("r")) == "caught"


@pytest.mark.parametrize("program", DELIVERED.values(), ids=DELIVERED)
def test_a_refusal_reaches_the_workflow_on_both_engines(backend, program):
    name, run_id = compose_key(t"deliver:{Run(str(uuid4()))}").stored(), str(uuid4())
    gate = cascade([rules(deny_refused_tools)])
    backend.register(name, program, OnesDomain(), Fault(), [gate])
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    assert snap.state == "completed", snap
    assert snap.result == "caught"


@pytest.mark.parametrize("program", TASK_LEVEL.values(), ids=TASK_LEVEL)
def test_a_crash_stays_task_level_on_the_recorder(program):
    with pytest.raises((ValueError, ExceptionGroup)) as raised:
        record().run(lambda: program("r"))
    assert any(isinstance(leaf, ValueError) for leaf in leaves(raised.value))


@pytest.mark.parametrize("program", TASK_LEVEL.values(), ids=TASK_LEVEL)
def test_a_crash_stays_task_level_on_both_engines(backend, program):
    name, run_id = compose_key(t"deliver:{Run(str(uuid4()))}").stored(), str(uuid4())
    gate = cascade([rules(deny_refused_tools)])
    backend.register(name, program, OnesDomain(), Fault(), [gate])
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    assert snap.state == "failed", snap


PARKED = {
    "refused beside a park": refused_beside_a_park,
    "refused after a park": refused_after_a_park,
}


class Unanswered(Ones):
    """Every tool answers 1, and nothing answers an await."""

    def __contains__(self, key: object) -> bool:
        return "wake" not in str(key)


@pytest.mark.parametrize("program", PARKED.values(), ids=PARKED)
def test_a_park_comes_before_the_refusal_on_the_recorder(program):
    handler = RecordingHandler(Unanswered(), op_layers=[cascade([rules(deny_refused_tools)])])
    parked = handler.run(lambda: program("r"))
    assert isinstance(parked, Suspended)
    assert parked.resume("now") == "caught"


@pytest.mark.parametrize("program", PARKED.values(), ids=PARKED)
def test_a_park_comes_before_the_refusal_on_both_engines(backend, program):
    name, run_id = compose_key(t"deliver:{Run(str(uuid4()))}").stored(), str(uuid4())
    gate = cascade([rules(deny_refused_tools)])
    backend.register(name, program, OnesDomain(), Fault(), [gate])
    task = backend.spawn(name, run_id, max_attempts=1)
    assert backend.run_until_result(task).state != "completed"
    (parked,) = backend.parked(task)
    backend.emit_event(task, parked.wake_event, "now")
    snap = backend.run_until_result(task)
    assert snap.state == "completed", snap
    assert snap.result == "caught"


def crashes_after_a_park(run_id: str) -> Effect[list[str]]:
    def wakes_then_crashes() -> Effect[str]:
        yield from waits(run_id)
        raise ValueError("a crash after the wake")

    def waits_longer() -> Effect[str]:
        return (yield from await_event(compose_key(t"later:{Run(run_id)}"), str))

    return (yield from gather([wakes_then_crashes, waits_longer]))


def test_a_crash_in_a_resumed_round_raises_on_that_resume_on_the_recorder():
    parked = RecordingHandler({}).run(lambda: crashes_after_a_park("r"))
    assert isinstance(parked, Suspended)
    with pytest.raises(ExceptionGroup) as raised:
        parked.resume("now")
    assert [type(leaf) for leaf in leaves(raised.value)] == [ValueError]


def test_a_crash_in_a_resumed_round_fails_the_task_on_that_wake_on_both_engines(backend):
    name, run_id = compose_key(t"deliver:{Run(str(uuid4()))}").stored(), str(uuid4())
    backend.register(name, crashes_after_a_park, OnesDomain(), Fault(), [])
    task = backend.spawn(name, run_id, max_attempts=1)
    assert backend.run_until_result(task).state != "completed"
    parked = backend.parked(task)
    first = next(p for p in parked if "wake:" in str(p.wake_event))
    backend.emit_event(task, first.wake_event, "now")
    snap = backend.run_until_result(task)
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "ValueError"
