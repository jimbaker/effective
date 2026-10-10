"""The schedule instrument's own pins. An instrument is a claim, so the claims are here.

The suites that use it (`test_reader_quotient.py`, `test_gate_meter.py`) assert over the schedules
a `Turnstile` forces. What nothing else says is what the turnstile does when a run cannot supply
the order, when two ops share a label, and when an op in the order does not commit. Most run on
the recorder, which needs no engine; a park and the branch check need an engine's placements.
"""

import threading
from functools import partial
from uuid import uuid4

import pytest
from _conformance import private
from _schedules import Turnstile, step_name

from effective.api import ask_llm, await_event, call_tool, gather
from effective.govern import Refused
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.keys import Key
from effective.layers import op_layer
from effective.ops import AwaitEvent, WorkflowOp

RESPONSES = {"a": "x", "b": "y", "tool:a": 1, "tool:b": 2, "tool:same": 1}


def one_then_other():
    yield from ask_llm("a", "p", str)
    yield from ask_llm("b", "p", str)


def run(program, order, *below, timeout=5.0):
    turnstile = Turnstile(order, step_name, timeout=timeout)
    RecordingHandler(responses=RESPONSES, op_layers=(turnstile.layer(), *below)).run(program)
    return turnstile


def two_tools(first: str, second: str):
    tools = [partial(call_tool, first, {}, int), partial(call_tool, second, {}, int)]
    return lambda: gather(tools)


def test_an_order_names_each_label_once():
    with pytest.raises(ValueError, match=r"'a' appears twice"):
        Turnstile(["a", "a", "b"], step_name)


def test_a_second_op_under_a_label_is_refused():
    """Two branches reaching one label would share its turn and run together, so the second to
    arrive fails. A `Barrier` holds both inside the layer stack, where an overlap shows."""
    inside = threading.Barrier(2, timeout=1)

    @op_layer
    def rendezvous(op):
        inside.wait()
        return (yield op)

    with pytest.raises(ExceptionGroup) as raised:
        run(two_tools("same", "same"), ["tool:same"], rendezvous, timeout=0.5)
    refusals = raised.value.subgroup(AssertionError)
    assert refusals is not None
    (refusal,) = refusals.exceptions
    assert "'tool:same' names a second op" in str(refusal)


def test_a_refused_op_ends_its_turn():
    @op_layer
    def refuse_a(op):
        if step_name(op) == "tool:a":
            raise Refused(op, "no")
        return (yield op)

    turnstile = Turnstile(["tool:a", "tool:b"], step_name, timeout=0.5)
    handler = RecordingHandler(responses=RESPONSES, op_layers=(turnstile.layer(), refuse_a))
    with pytest.raises(ExceptionGroup):
        handler.run(two_tools("a", "b"))
    assert turnstile.outcomes == [("tool:a", "Refused"), ("tool:b", "committed")]
    assert turnstile.kept_its_schedule()


def test_an_order_the_run_cannot_take_waits_out_its_timeout():
    """A deadlock is the correct answer, and the message carries what did commit.

    This run is sequential, so nothing can reach `b` before `a`: a turnstile asked for an
    unreachable order has no move but to say so."""
    with pytest.raises(AssertionError, match=r"'a' waited out the gather; ended \[\]"):
        run(one_then_other, ["b", "a"], timeout=0.2)


def awaited(op: WorkflowOp) -> str | None:
    return "await" if isinstance(op, AwaitEvent) else step_name(op)


def durable(backend, program, turnstile):
    name, run_id = private("turnstile"), str(uuid4())
    backend.register_body(
        name,
        lambda params, ctx: DurableHandler(ctx, Ones(), op_layers=(turnstile.layer(),)).run(
            program
        ),
    )
    task = backend.spawn(name, run_id, max_attempts=1)
    return task, backend.run_until_result(task)


class Ones:
    def run(self, op):
        return 1


def test_a_parked_op_ends_its_turn(backend):
    """The park passes the layer by as a `BaseException`, so the layer never resumes. Its turn
    ends when CPython finalizes the abandoned generator, once `DurableHandler.run` drops the park's
    traceback."""
    event = Key.parse(f"ready:{uuid4()}")
    turnstile = Turnstile(["await", "tool:b"], awaited)

    def program():
        return (
            yield from gather([lambda: await_event(event, int), partial(call_tool, "b", {}, int)])
        )

    task, snap = durable(backend, program, turnstile)
    assert snap.state not in ("completed", "failed"), snap
    assert turnstile.outcomes == [("await", "closed"), ("tool:b", "committed")]
    assert "gather:0,1;step;tool:b" in backend.checkpoint_keys(task)


def test_the_check_wants_two_branches(backend):
    """A sequential run keeps any order it can reach, and orders nothing between branches."""
    sequential = Turnstile(["tool:a", "tool:b"], step_name)

    def program():
        yield from call_tool("a", {}, int)
        return (yield from call_tool("b", {}, int))

    durable(backend, program, sequential)
    assert sequential.kept_its_schedule()
    with pytest.raises(AssertionError, match="no two sibling branches"):
        sequential.check()

    nested = Turnstile(["tool:a", "tool:b"], step_name)

    def parent_then_child():
        yield from call_tool("a", {}, int)
        return (yield from gather([partial(call_tool, "b", {}, int)]))

    durable(backend, lambda: gather([parent_then_child]), nested)
    assert nested.kept_its_schedule()
    with pytest.raises(AssertionError, match="no two sibling branches"):
        nested.check()

    gathered = Turnstile(["tool:b", "tool:a"], step_name)
    durable(backend, two_tools("a", "b"), gathered)
    gathered.check()
