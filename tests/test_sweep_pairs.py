"""`_shapes.sweep_pairs` on rows small enough to say which pairs it must crash twice.

| row                           | pairs crashed in two attempts |
|-------------------------------|-------------------------------|
| three steps in sequence       | all three                     |
| the same, listed out of order | all three                     |
| two sibling branches          | none, and that is no failure  |
"""

import threading
from functools import partial
from typing import Any

import pytest
from _conformance import Fault, FaultPosition
from _shapes import Shape, run, sweep_pairs
from test_race_shapes import until

from effective.api import call_tool, gather
from effective.domain import CallTool, DomainOp


class Echo:
    """Answers a call with the value it was handed."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(args={"value": value}):
                return value
        raise TypeError(f"echo answers a call, not {op!r}")


def in_sequence(_run_id: str):
    answers = []
    for name in ("z", "a", "m"):
        answers.append((yield from call_tool(name, {"value": name}, str)))
    return answers


def siblings(_run_id: str):
    return (
        yield from gather([partial(call_tool, name, {"value": name}, str) for name in ("a", "b")])
    )


class _AFirst:
    """Holds `b`'s call until `a`'s has ended, raised or not, so `b` runs in the attempt that
    `a`'s crash ends."""

    def __init__(self, ctx: Any) -> None:
        self._ctx, self._ended = ctx, threading.Event()

    def step(self, name, thunk):
        if name.stored().endswith("tool:b"):
            until(self._ended)
        try:
            return self._ctx.step(name, thunk)
        finally:
            if name.stored().endswith("tool:a"):
                self._ended.set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)


SEQUENCE = Shape({"seq": in_sequence}, Echo, answer=lambda: ["z", "a", "m"])


def test_every_pair_in_sequence_crashes_in_two_attempts(backend):
    assert sweep_pairs(backend, SEQUENCE, "seq") == 3


def test_a_listing_out_of_commit_order_loses_no_pair(backend, monkeypatch):
    """A store whose listing is not commit order, as a clock stepping back makes Absurd's."""
    listed = backend.checkpoint_keys
    monkeypatch.setattr(backend, "checkpoint_keys", lambda task: listed(task)[::-1])
    assert sweep_pairs(backend, SEQUENCE, "seq") == 3


def test_a_sibling_in_the_crashed_attempt_cannot_take_the_second_aim(backend):
    """`b` runs after `a` crashed and before its attempt ends, so a second aim armed at once
    would crash that attempt twice and the run would never reach a third."""
    fault = Fault(
        named="gather:0,0;step;tool:a",
        position=FaultPosition.AFTER_THUNK,
        then=Fault(named="gather:0,1;step;tool:b", position=FaultPosition.AFTER_THUNK),
    )
    outcome = run(backend, siblings, Echo(), fault=fault, wrap=_AFirst)
    assert outcome.snap.result == ["a", "b"]
    assert (fault.fired, backend.task_attempts(outcome.task)) == (1, 2)


def test_siblings_committed_in_one_attempt_are_no_pair(backend, monkeypatch):
    registered = backend.register

    def register(name, program, domain, fault, layers, wrap=None, **kw):
        return registered(name, program, domain, fault, layers, _AFirst, **kw)

    monkeypatch.setattr(backend, "register", register)
    shape = Shape({"gather": siblings}, Echo, answer=lambda: ["a", "b"])
    assert sweep_pairs(backend, shape, "gather") == 0


def test_the_base_run_is_held_to_the_rows_count(backend):
    shape = Shape({"seq": in_sequence}, Echo, answer=lambda: ["z", "a", "m"], count=lambda _: 99)
    with pytest.raises(AssertionError):
        sweep_pairs(backend, shape, "seq")
