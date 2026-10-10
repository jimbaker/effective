"""A crash at every op, over the shapes whose occurrences a layer or a race could misnumber.

The walk mints every occurrence and the engine stores at the name it is handed, so each shape
below resumes from any crash to the same result, the same domain calls, and the same checkpoints.
"""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from _conformance import CountingDomain, Fault, at_every_op

from effective.api import append_ledger, call_tool, gather, race, scoped
from effective.choice import Chosen, Won
from effective.keys import Key, compose_key
from effective.layers import current_placement, op_layer, retry
from effective.ops import LedgerRow
from effective.permission import Allow, Deny, Refused, cascade, rules


def _refused_then_asked(run_id: str):
    with contextlib.suppress(Refused):
        yield from call_tool("a", {}, int)
    return [(yield from call_tool("a", {}, int)), (yield from call_tool("b", {}, int))]


def _deny_the_first_ask():
    """Refuses `step;tool:a` and allows `#2`: decided from the placement, so every attempt
    decides the same way."""

    def policy(op):
        placement = current_placement()
        return Deny("first ask") if placement == Key.parse("step;tool:a") else Allow()

    return cascade([rules(policy)])


def _retried(run_id: str):
    a = yield from call_tool("a", {}, int)
    yield from append_ledger(LedgerRow(event_id=Key.parse(f"{run_id}:e1"), kind="k1"))
    return [a, (yield from call_tool("b", {}, int))]


@op_layer
def _forward_twice(op):
    """Forwards each op a second time once it succeeded, and answers with the second value."""
    yield op
    return (yield op)


def _retried_in_branches(run_id: str):
    return (yield from gather([lambda: call_tool("a", {}, int), lambda: call_tool("b", {}, int)]))


def _repeated_in_a_scoped_race(run_id: str):
    def branch():
        return [(yield from call_tool("a", {}, int)), (yield from call_tool("a", {}, int))]

    match (yield from scoped(compose_key(t"rec:{0}"), lambda: race([branch]))):
        case Chosen(endings=[Won(value=value)]):
            return value
        case unexpected:
            raise AssertionError(unexpected)


@dataclass(frozen=True)
class Shape:
    factory: Callable[[str], Any]
    layers: Callable[[], tuple[Any, ...]]
    domain: Callable[[], CountingDomain]
    result: Any
    calls: list[str]
    steps: list[str]


SHAPES = {
    "refused-then-asked": Shape(
        _refused_then_asked,
        lambda: (_deny_the_first_ask(),),
        CountingDomain,
        [10, 20],
        ["a", "b"],
        ["step;tool:a#2", "step;tool:b"],
    ),
    "retried": Shape(
        _retried,
        lambda: (retry(2),),
        lambda: CountingDomain(flaky=True),
        [10, 20],
        ["a", "a", "b"],
        ["step;tool:a", "step;tool:b"],
    ),
    "forwarded-twice": Shape(
        _retried,
        lambda: (_forward_twice,),
        lambda: CountingDomain(incrementing=True),
        [11, 21],
        ["a", "b"],
        ["step;tool:a", "step;tool:b"],
    ),
    "retried-in-branches": Shape(
        _retried_in_branches,
        lambda: (retry(2),),
        lambda: CountingDomain(flaky=True),
        [10, 20],
        ["a", "a", "b"],
        ["gather:0,0;step;tool:a", "gather:0,1;step;tool:b"],
    ),
    "repeated-in-a-scoped-race": Shape(
        _repeated_in_a_scoped_race,
        tuple,
        lambda: CountingDomain(incrementing=True),
        [11, 12],
        ["a", "a"],
        ["rec:0;race:0,0;step;tool:a", "rec:0;race:0,0;step;tool:a#2"],
    ),
}


def _run(backend, shape: Shape, fault: Fault) -> tuple[Any, CountingDomain, list[str]]:
    domain, run_id = shape.domain(), f"r-{uuid4().hex[:8]}"
    name = f"occurrences-{run_id}"
    backend.register(name, shape.factory, domain, fault, shape.layers())
    task_id = backend.spawn(name, run_id)
    snapshot = backend.run_until_result(task_id)
    steps = sorted(k for k in backend.checkpoint_keys(task_id) if "step;tool:" in k)
    return snapshot, domain, steps


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_a_crash_at_every_op_resumes_to_the_same_occurrences(backend, shape):
    unarmed = Fault(None)
    snapshot, domain, steps = _run(backend, shape, unarmed)
    assert snapshot.state == "completed", snapshot
    assert (snapshot.result, sorted(domain.calls), steps) == (
        shape.result,
        shape.calls,
        shape.steps,
    )

    for k, fault in at_every_op(unarmed):
        snapshot, domain, steps = _run(backend, shape, fault)
        assert snapshot.state == "completed", f"k={k}: {snapshot}"
        assert (snapshot.result, sorted(domain.calls), steps) == (
            shape.result,
            shape.calls,
            shape.steps,
        ), f"k={k}"
