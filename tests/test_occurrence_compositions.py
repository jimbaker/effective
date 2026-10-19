"""Every walk numbers a repeated key alike under composition: gathers, scopes, parks and layers."""

import threading
from collections.abc import Iterator, Mapping
from contextlib import suppress
from typing import Any
from uuid import uuid4

import pytest
from _conformance import CountingDomain, Fault
from pydantic_core import to_jsonable_python

from effective.api import await_event, call_tool, gather, qualified_event_name, race, scoped
from effective.budget import Grant, depth_grant_name
from effective.checkpoints import is_engine_internal
from effective.cost import Usage
from effective.fork import ForkAtAnInjectedOp, OpIndex, fork_at
from effective.handlers import recording
from effective.handlers.admission import ChoiceBox
from effective.handlers.base import kinded_name
from effective.handlers.recording import RecordingHandler, Suspended
from effective.handlers.replay import ReplayHandler, ReplayMismatch
from effective.keys import Key, Name, compose_key, scope_prefix
from effective.layers import (
    TransientError,
    current_placement,
    layer_run_state,
    op_layer,
    retry,
)
from effective.ops import Addressing, FramedEventName, NumberedEventName, event_name
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


@op_layer
def _ask_b_before_a(op):
    if op_names_tool(op, "a"):
        yield from call_tool("b", {}, int)
    return (yield op)


def a_then_b(run_id: str = ""):
    return [(yield from call_tool("a", {}, int)), (yield from call_tool("b", {}, int))]


def a_then_b_in_a_branch(run_id: str = ""):
    return (yield from gather([a_then_b, _b]))


def a_then_b_in_a_scope(run_id: str = ""):
    return (yield from scoped(LANE, a_then_b))


def a_then_b_in_a_race(run_id: str = ""):
    """One branch, since a loser's work before it stops is the schedule's."""
    return (yield from race([a_then_b]))


@pytest.mark.parametrize(
    "workflow",
    [a_then_b, a_then_b_in_a_branch, a_then_b_in_a_scope, a_then_b_in_a_race],
    ids=lambda workflow: workflow.__name__,
)
def test_an_op_a_layer_yields_is_placed_alike_on_every_walk(backend, workflow):
    """The layer asks `b` before each `a`, so the workflow's own `b` is its second placement."""
    layers = (_ask_b_before_a,)
    recorder = RecordingHandler(responses={"tool:a": 10, "tool:b": 20}, op_layers=layers)
    result = recorder.run(workflow)
    keys = [entry.key.stored() for entry in recorder.trace]
    assert ReplayHandler(recorder.trace).run(workflow) == result

    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, workflow, CountingDomain(), Fault(None), layers)
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == to_jsonable_python(result)
    assert sorted(_engine_steps(backend, task_id)) == sorted(keys)


@op_layer
def _watch(op):
    _WATCHED.append(current_placement())
    return (yield op)


_WATCHED: list[Any] = []


def test_a_layer_sees_one_placement_on_every_walk(backend):
    _WATCHED.clear()
    responses = {"tool:a": 10, "tool:b": 20}
    RecordingHandler(responses=responses, op_layers=(_watch,)).run(gather_in_a_scope_twice)
    recorded = sorted(placed.stored() for placed in _WATCHED)
    _WATCHED.clear()

    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, gather_in_a_scope_twice, CountingDomain(), Fault(None), (_watch,))
    backend.run_until_result(backend.spawn(run_id, run_id))
    assert recorded == sorted(placed.stored() for placed in _WATCHED)


@op_layer
def _refuse_b(op):
    if op_names_tool(op, "b"):
        raise Refused(op, "no b")
    return (yield op)


@op_layer
def _try_b_before_a(op):
    if op_names_tool(op, "a"):
        with suppress(Refused):
            yield from call_tool("b", {}, int)
    return (yield op)


def test_an_op_a_layer_yields_and_a_lower_layer_refuses_leaves_no_entry(backend):
    layers = (_try_b_before_a, _refuse_b)
    recorder = RecordingHandler(responses={"tool:a": 10, "tool:b": 20}, op_layers=layers)
    result = recorder.run(asks_a_twice)
    keys = [entry.key.stored() for entry in recorder.trace]
    assert keys == ["step;tool:a", "step;tool:a#2"]
    assert ReplayHandler(recorder.trace).run(asks_a_twice) == result

    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, asks_a_twice, CountingDomain(), Fault(None), layers)
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == result
    assert _engine_steps(backend, task_id) == keys


def test_a_retry_over_a_layers_op_places_alike_on_every_walk(backend):
    """The first `a` fails, so `retry` forwards it again and the layer asks `b` again: the engine
    serves that `b` its first checkpoint, so the workflow's own `b` is the second."""
    layers = (retry(), _ask_b_before_a)
    recorder = RecordingHandler(responses=_FirstAFails(flaky=True), op_layers=layers)
    result = recorder.run(a_then_b)
    keys = [entry.key.stored() for entry in recorder.trace]
    assert ReplayHandler(recorder.trace).run(a_then_b) == result

    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, a_then_b, CountingDomain(flaky=True), Fault(None), layers)
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == result
    assert sorted(_engine_steps(backend, task_id)) == sorted(keys)


class _Fails(Mapping[str, Any]):
    """Canned answers whose n-th ask of a tool raises a transient error, for each n in `fail`."""

    def __init__(self, fail: dict[str, set[int]]) -> None:
        self.fail, self.asked = fail, dict[str, int]()

    def __getitem__(self, name: str) -> Any:
        self.asked[name] = n = self.asked.get(name, 0) + 1
        if n in self.fail.get(name, set()):
            raise TransientError(f"{name}, ask {n}")
        return _ANSWERS[name]

    def __contains__(self, name: object) -> bool:
        """A lookup asks nothing, so `asked` counts the answers alone."""
        return name in _ANSWERS

    def __iter__(self) -> Iterator[str]:
        return iter(_ANSWERS)

    def __len__(self) -> int:
        return len(_ANSWERS)

    def run(self, op: Any) -> Any:
        """The engine's domain, failing as the canned answers do."""
        return self[kinded_name(op.name, "tool").stored()]


_ANSWERS = {"tool:a": 10, "tool:b": 20, "tool:c": 30}


@op_layer
def _ask_b_then_c_before_a(op):
    """A layer whose op differs on a re-forward: `b` the first time, `c` after."""
    state = layer_run_state(compose_key(t"probe:{Name('alternate')}"))
    if op_names_tool(op, "a"):
        state["n"] = n = state.get("n", 0) + 1
        yield from call_tool("b" if n == 1 else "c", {}, int)
    return (yield op)


def a_then_b_then_c(run_id: str = ""):
    return [(yield from a_then_b()), (yield from call_tool("c", {}, int))]


@pytest.mark.parametrize(
    ("workflow", "layers", "fail"),
    [
        (a_then_b, (retry(), _ask_b_before_a), {"tool:b": {1}}),
        (a_then_b, (retry(), _ask_b_before_a), {"tool:b": {1}, "tool:a": {1}}),
        (a_then_b, (retry(), _try_b_before_a, _refuse_the_first_b), {"tool:a": {1}}),
        (a_then_b_then_c, (retry(), _ask_b_then_c_before_a), {"tool:a": {1}}),
    ],
    ids=[
        "the layer's op failed first",
        "the layer's op, then the walk op, failed first",
        "the layer's op refused first",
        "another layer's op on the re-forward",
    ],
)
def test_a_reforward_places_alike_on_every_walk(backend, workflow, layers, fail):
    """A re-forwarded layer op the engine checkpointed is served, so each walk asks each tool
    alike."""
    answers, domain = _Fails(fail), _Fails(fail)
    recorder = RecordingHandler(responses=answers, op_layers=layers)
    result = recorder.run(workflow)
    keys = sorted(entry.key.stored() for entry in recorder.trace if entry.error is None)
    assert ReplayHandler(recorder.trace).run(workflow) == result

    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, workflow, domain, Fault(None), layers)
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == to_jsonable_python(result)
    assert sorted(_engine_steps(backend, task_id)) == keys
    assert answers.asked == domain.asked


@op_layer
def _gather_then_c_before_a(op):
    if op_names_tool(op, "a"):
        yield from gather([_b, _b])
        yield from call_tool("c", {}, int)
    return (yield op)


def test_a_branch_a_layer_gathers_is_placed_apart_from_the_layers_forwards(backend):
    """The layer's own `c` is served on the re-forward however many branches its gather ran.
    The recorder refuses a gather a layer yields, so the engines alone run this."""
    domain = _Fails({"tool:a": {1}})
    run_id = f"r-{uuid4().hex[:8]}"
    layers = (retry(), _gather_then_c_before_a)
    backend.register(run_id, a_then_b, domain, Fault(None), layers)
    task_id = backend.spawn(run_id, run_id)
    assert backend.run_until_result(task_id).result == [10, 20]
    assert domain.asked["tool:c"] == 1
    assert _engine_steps(backend, task_id).count("step;tool:c") == 1


class _LoserAsksFirst(Mapping[str, Any]):
    """The winner's `b` waits for the loser's, so the loser's layer has asked before the race is
    decided."""

    def __init__(self) -> None:
        self.loser_asked = threading.Event()

    def __getitem__(self, name: str) -> Any:
        match name:
            case "race:0,0;tool:b":
                assert self.loser_asked.wait(5)
                return 20
            case "race:0,1;tool:b":
                self.loser_asked.set()
                return 20
            case "tool:a":
                return 10
            case _:
                raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(["race:0,0;tool:b", "race:0,1;tool:b", "tool:a"])

    def __len__(self) -> int:
        return 3


def _b_against_a(run_id: str = ""):
    return (yield from race([_b, _a]))


def test_a_loser_stopped_after_its_layers_op_replays(monkeypatch):
    decided = threading.Event()

    class _Witnessed(ChoiceBox):
        def put(self, choice):
            super().put(choice)
            decided.set()

    @op_layer
    def ask_b_then_wait(op):
        if op_names_tool(op, "a"):
            yield from call_tool("b", {}, int)
            assert decided.wait(5)
        return (yield op)

    monkeypatch.setattr(recording, "ChoiceBox", _Witnessed)
    recorder = RecordingHandler(responses=_LoserAsksFirst(), op_layers=(ask_b_then_wait,))
    result = recorder.run(_b_against_a)
    assert [entry.injected for entry in recorder.trace].count(True) == 1
    assert ReplayHandler(recorder.trace).run(_b_against_a) == result


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


def test_a_relative_await_refuses_an_authors_numbered_name():
    with pytest.raises(NumberedEventName):
        event_name(Key.parse("go#2"))


@pytest.mark.parametrize(
    ("name", "addressing"),
    [
        (GRANT.occurrence(2), Addressing.RELATIVE),
        (Key.parse("lane:p;sub:x#2"), Addressing.ABSOLUTE),
    ],
    ids=["a substrate park's occurrence", "an absolute address"],
)
def test_an_await_keeps_an_occurrence_the_walk_placed(name, addressing):
    assert event_name(name, addressing) == name


def _grant_twice_then_a():
    first = yield from await_event(GRANT, Grant)
    second = yield from await_event(GRANT, Grant)
    return [first, second, (yield from call_tool("a", {}, int))]


class _Live:
    def run_metered(self, op: Any) -> tuple[Any, Usage]:
        return 20, Usage()


@pytest.mark.parametrize(
    ("at", "result"), [(1, [99, 20]), (2, [10, 99])], ids=["the op it ran for", "a later op"]
)
def test_a_fork_places_the_ops_a_layer_yielded_before_its_point(at, result):
    recorder = RecordingHandler(
        responses={"tool:a": 10, "tool:b": 20}, op_layers=(_ask_b_before_a,)
    )
    recorder.run(a_then_b)
    assert [entry.injected for entry in recorder.trace] == [True, False, False]
    assert fork_at(a_then_b, recorder.trace, OpIndex(at), 99, _Live()).result == result


def test_a_fork_at_an_op_a_layer_yielded_is_refused():
    recorder = RecordingHandler(
        responses={"tool:a": 10, "tool:b": 20}, op_layers=(_ask_b_before_a,)
    )
    recorder.run(a_then_b)
    with pytest.raises(ForkAtAnInjectedOp):
        fork_at(a_then_b, recorder.trace, OpIndex(0), 99, _Live())


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


def test_a_resumed_absolute_await_in_a_scope_replays():
    recorder = RecordingHandler()
    parked = recorder.run(_absolute_in_a_scope)
    assert isinstance(parked, Suspended)
    assert parked.resume({"v": 1}) == {"v": 1}
    assert ReplayHandler(recorder.trace).run(_absolute_in_a_scope) == {"v": 1}
