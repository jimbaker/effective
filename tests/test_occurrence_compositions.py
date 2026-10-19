"""Every walk numbers a repeated key alike under composition: gathers, scopes, parks and layers."""

from collections.abc import Iterator, Mapping
from contextlib import suppress
from typing import Any
from uuid import uuid4

import pytest
from _conformance import CountingDomain, Fault
from pydantic_core import to_jsonable_python

from effective.api import await_event, call_tool, gather, qualified_event_name, scoped
from effective.budget import Grant, depth_grant_name
from effective.checkpoints import is_engine_internal
from effective.cost import Usage
from effective.fork import OpIndex, fork_at
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler, ReplayMismatch
from effective.keys import Key, Name, compose_key, scope_prefix
from effective.layers import TransientError, layer_run_state, op_layer, retry
from effective.ops import Addressing, FramedEventName, event_name
from effective.permission import Refused

LANE = compose_key(t"lane:{Name('p')}")
GRANT = depth_grant_name("r1", depth=1, generation=0)


def _a():
    return (yield from call_tool("a", {}, int))


def _aa():
    first = yield from call_tool("a", {}, int)
    second = yield from call_tool("a", {}, int)
    return first + second


def gather_in_a_scope_twice(run_id: str = ""):
    one = yield from scoped(LANE, lambda: gather([_a, _a]))
    two = yield from scoped(LANE, lambda: gather([_a, _a]))
    return [one, two]


def repeat_in_a_branch(run_id: str = ""):
    return (yield from gather([_aa, _a]))


def scope_twice_in_a_branch(run_id: str = ""):
    def branch():
        first = yield from scoped(LANE, _a)
        second = yield from scoped(LANE, _a)
        return first + second

    return (yield from gather([branch, branch]))


def repeat_in_a_nested_gather(run_id: str = ""):
    def outer():
        return (yield from gather([_aa, _aa]))

    return (yield from gather([outer, _aa]))


def gathers_then_a_repeat(run_id: str = ""):
    return [(yield from gather([_a])), (yield from gather([_a])), (yield from _aa())]


def _engine_steps(backend, task_id) -> list[str]:
    return [key for key in backend.checkpoint_keys(task_id) if not is_engine_internal(key)]


@pytest.mark.parametrize(
    "workflow",
    [
        gather_in_a_scope_twice,
        repeat_in_a_branch,
        scope_twice_in_a_branch,
        repeat_in_a_nested_gather,
        gathers_then_a_repeat,
    ],
    ids=lambda workflow: workflow.__name__,
)
def test_the_engines_checkpoint_each_composed_repeat_the_recorder_records(backend, workflow):
    """Sorted on the engine side: concurrent branches commit in schedule order."""
    recorder = RecordingHandler(responses={"tool:a": 10})
    result = recorder.run(workflow)
    keys = [entry.key.stored() for entry in recorder.trace]
    assert len(set(keys)) == len(keys), keys
    assert ReplayHandler(recorder.trace).run(workflow) == result

    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, workflow, CountingDomain(), Fault(None), ())
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == to_jsonable_python(result)
    assert sorted(_engine_steps(backend, task_id)) == sorted(keys)


@op_layer
def _refuse_the_first_b(op):
    state = layer_run_state(compose_key(t"probe:{Name('refuse')}"))
    if op_names_tool(op, "b") and not state.get("refused"):
        state["refused"] = True
        raise Refused(op, "the first b")
    return (yield op)


def op_names_tool(op: Any, tool: str) -> bool:
    return getattr(getattr(op, "op", None), "name", None) == tool


def refused_then_asked(run_id: str = ""):
    with suppress(Refused):
        yield from call_tool("b", {}, int)
    return [(yield from call_tool("b", {}, int)), (yield from scoped(LANE, _b))]


def _b():
    return (yield from call_tool("b", {}, int))


def asks_a_twice(run_id: str = ""):
    return (yield from _aa())


class _FirstAFails(Mapping[str, Any]):
    """Canned answers whose first `a` raises a transient error, as `CountingDomain(flaky=True)`."""

    def __init__(self, flaky: bool) -> None:
        self.flaky, self.asked = flaky, 0

    def __getitem__(self, name: str) -> Any:
        match name:
            case "tool:a":
                self.asked += 1
                if self.flaky and self.asked == 1:
                    raise TransientError("the first a")
                return 10
            case "tool:b":
                return 20
            case _:
                raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(["tool:a", "tool:b"])

    def __len__(self) -> int:
        return 2


@pytest.mark.parametrize(
    ("workflow", "layers", "flaky"),
    [(refused_then_asked, (_refuse_the_first_b,), False), (asks_a_twice, (retry(),), True)],
    ids=["a refused ask", "a retried ask"],
)
def test_a_layer_refusing_or_retrying_keeps_the_walks_agreeing(backend, workflow, layers, flaky):
    recorder = RecordingHandler(responses=_FirstAFails(flaky), op_layers=layers)
    result = recorder.run(workflow)
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, workflow, CountingDomain(flaky=flaky), Fault(None), layers)
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == result
    # a refused op leaves an entry on the recorder and no checkpoint on an engine
    answered = [entry.key.stored() for entry in recorder.trace if entry.error is None]
    assert len(answered) < len(recorder.trace) or flaky, "the layer refused nothing"
    assert _engine_steps(backend, task_id) == answered


def _go(run_id: str) -> Key:
    return compose_key(t"go:{Name(run_id)}")


def a_park_between_repeats_in_a_branch(run_id: str = ""):
    def branch():
        first = yield from call_tool("a", {}, int)
        answer = yield from await_event(_go(run_id), dict)
        return [first, answer, (yield from call_tool("a", {}, int))]

    return (yield from gather([branch, _a]))


def a_park_between_repeats_in_a_scope(run_id: str = ""):
    def body():
        return [(yield from call_tool("a", {}, int)), (yield from await_event(_go(run_id), dict))]

    return [
        (yield from scoped(LANE, body)),
        (yield from scoped(LANE, body)),
        (yield from call_tool("a", {}, int)),
    ]


@pytest.mark.parametrize(
    "workflow",
    [a_park_between_repeats_in_a_branch, a_park_between_repeats_in_a_scope],
    ids=lambda workflow: workflow.__name__,
)
def test_a_resumed_run_keeps_the_walks_agreeing(backend, workflow):
    run_id = f"r-{uuid4().hex[:8]}"
    answer = {"v": 1}
    recorder = RecordingHandler(responses={"tool:a": 10})
    out = recorder.run(lambda: workflow(run_id))
    while isinstance(out, Suspended):
        out = out.resume(answer)
    assert ReplayHandler(recorder.trace).run(lambda: workflow(run_id)) == out
    steps = [key for entry in recorder.trace if "event;" not in (key := entry.key.stored())]

    backend.register(run_id, workflow, CountingDomain(), Fault(None), ())
    task_id = backend.spawn(run_id, run_id)
    snapshot = backend.run_until_result(task_id)
    while snapshot is None or snapshot.state != "completed":
        for park in backend.parked(task_id):
            backend.emit_event(task_id, Key.parse(park.wake_event), answer)
        snapshot = backend.run_until_result(task_id)
    assert snapshot.result == to_jsonable_python(out)
    engine = [key for key in _engine_steps(backend, task_id) if "event;" not in key]
    assert sorted(engine) == sorted(steps)


@pytest.mark.parametrize(
    "framed",
    [
        qualified_event_name(LANE, name=GRANT),
        GRANT.prefixed(scope_prefix(LANE)),
        GRANT.occurrence(2).prefixed(scope_prefix(LANE)),
    ],
    ids=["qualified", "prefixed", "a prefixed occurrence"],
)
def test_a_relative_await_refuses_a_framed_settlement_key(framed):
    with pytest.raises(FramedEventName):
        event_name(framed)


def test_an_absolute_await_takes_a_framed_settlement_key():
    framed = qualified_event_name(LANE, name=GRANT)
    assert event_name(framed, Addressing.ABSOLUTE) == framed


def _grant_twice_then_a():
    first = yield from await_event(GRANT, Grant)
    second = yield from await_event(GRANT, Grant)
    return [first, second, (yield from call_tool("a", {}, int))]


class _Live:
    def run_metered(self, op: Any) -> tuple[Any, Usage]:
        return 20, Usage()


def test_a_pending_park_fork_takes_the_name_the_run_parked_on():
    answer = Grant(add_depth=0)
    recorder = RecordingHandler(responses={GRANT.stored(): answer})
    parked = recorder.run(_grant_twice_then_a)
    assert isinstance(parked, Suspended)
    at = OpIndex(len(recorder.trace))
    tail = fork_at(
        _grant_twice_then_a, recorder.trace, at, answer, _Live(), pending_name=parked.awaiting
    )
    assert tail.result == [answer, answer, 20]
    with pytest.raises(ReplayMismatch):
        fork_at(_grant_twice_then_a, recorder.trace, at, answer, _Live(), pending_name=GRANT)


SUB = compose_key(t"sub:{Name('x')}")


def _absolute_in_a_scope():
    def body():
        return (yield from await_event(SUB, dict, addressing=Addressing.ABSOLUTE))

    return (yield from scoped(LANE, body))


@pytest.mark.xfail(
    strict=True,
    reason="the recorder frames a canned absolute await and leaves a resumed one unframed",
)
def test_a_resumed_absolute_await_in_a_scope_replays():
    recorder = RecordingHandler()
    parked = recorder.run(_absolute_in_a_scope)
    assert isinstance(parked, Suspended)
    assert parked.resume({"v": 1}) == {"v": 1}
    assert ReplayHandler(recorder.trace).run(_absolute_in_a_scope) == {"v": 1}
