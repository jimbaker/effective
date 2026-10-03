"""`descend` runs deeper than Python's recursion limit, because `Deeper` is a tail call the driver
performs as a loop. The same recursion through `yield from` keeps a frame per level and does not.

The recorder runs twice the interpreter's recursion limit. The engines checkpoint every level, so
their cost is linear in depth, and they run a tenth past the limit. The `yield from` spelling is
asserted to raise at the engines' depth, so every arm runs past where plain recursion fails."""

import sys
from collections.abc import Iterator, Mapping
from uuid import uuid4

import pytest
from _conformance import Fault

from effective.api import Effect, call_tool
from effective.combinators import Answered, Deeper, Level, descend
from effective.domain import DomainOp
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Run, compose_key

RECORDER_DEPTH = 2 * sys.getrecursionlimit()
ENGINE_DEPTH = sys.getrecursionlimit() + sys.getrecursionlimit() // 10


def turn(n: int, level: Level) -> Effect[Answered[int] | Deeper[int]]:
    yield from call_tool("turn", {}, int)
    return Answered(level.depth) if n == 0 else Deeper(n - 1)


def trampolined(depth: int) -> Effect[int]:
    return (yield from descend(depth, turn, budget=depth))


def recursive(n: int) -> Effect[int]:
    yield from call_tool("turn", {}, int)
    return 0 if n == 0 else 1 + (yield from recursive(n - 1))


class Ones(Mapping[str, int]):
    """Every tool answers 1. A `Mapping` the recorder reads as given, where it copies a `dict`."""

    def __getitem__(self, key: str) -> int:
        return 1

    def __contains__(self, key: object) -> bool:
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


class OnesDomain:
    def run(self, op: DomainOp[int]) -> int:
        return 1


def test_the_same_recursion_through_yield_from_exceeds_the_limit():
    with pytest.raises(RecursionError):
        RecordingHandler(Ones()).run(lambda: recursive(ENGINE_DEPTH))


def test_descend_past_the_recursion_limit_on_the_recorder_and_replay():
    handler = RecordingHandler(Ones())
    assert handler.run(lambda: trampolined(RECORDER_DEPTH)) == RECORDER_DEPTH
    assert ReplayHandler(handler.trace).run(lambda: trampolined(RECORDER_DEPTH)) == RECORDER_DEPTH


def test_descend_past_the_recursion_limit_on_both_engines(backend):
    name, run_id = compose_key(t"deep:{Run(str(uuid4()))}").stored(), str(uuid4())
    backend.register(name, lambda _run_id: trampolined(ENGINE_DEPTH), OnesDomain(), Fault(), [])
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    assert snap.state == "completed", snap
    assert snap.result == ENGINE_DEPTH
