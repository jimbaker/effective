"""A value no store holds alike is refused where it enters, on both engines alike.

Postgres's `jsonb` holds no string with a NUL or without UTF-8 bytes; SQLite stores both escaped.

| where the value enters      | refused by                          |
|-----------------------------|-------------------------------------|
| a machine's seed            | `run_machine`, before the first op  |
| a tree a worker returns     | `run_machine`, before it is carried |
| a direct `store_artifact`   | `StoreArtifact` at construction     |
"""

from enum import StrEnum
from functools import partial
from typing import Any

import pytest
from _shapes import run

from effective.api import call_tool, store_artifact
from effective.keys import Run
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Finish
from effective.machine.spec import Evidence, Fused, StateSpec
from effective.machine.trampoline import run_machine

UNSTORABLE = {
    "nul-content": {"a.txt": "\x00"},
    "nul-key": {"a\x00.txt": "text"},
    "surrogate-content": {"a.txt": "\udc80"},
    "surrogate-key": {"\udc80.txt": "text"},
}
STORABLE = {"escaped-nul": {"a.txt": r"\u0000"}, "controls": {"a.txt": "\x01\x1f￿\U0010ffff"}}


class _State(StrEnum):
    WORK = "work"


class _Verdict(StrEnum):
    DONE = "done"


class _Suite:
    def run(self, op: Any) -> Any:
        return CommandRun(exit_code=0) if op.name == "run_suite" else "input"


def _machine(run_id: str, tree: dict[str, str], enters: str):
    def worker(_ctx):
        yield from call_tool("input", {}, str)
        return Evidence(summary="done", tree=tree if enters == "worker" else None)

    def judge(_ctx, _evidence):
        yield from ()
        return _Verdict.DONE

    yield from run_machine(
        Run(run_id),
        "store it",
        {_State.WORK: StateSpec(_State.WORK, Fused(worker, judge))},
        lambda *_: Finish(),
        start=_State.WORK,
        budget=2,
        tree=tree if enters == "seed" else {},
    )


def _direct(_run_id: str, tree: dict[str, str]):
    return (yield from store_artifact(tree, "application/json"))


def _program(enters: str, tree: dict[str, str]):
    if enters == "artifact":
        return partial(_direct, tree=tree)
    return partial(_machine, tree=tree, enters=enters)


@pytest.mark.parametrize("enters", ["seed", "worker", "artifact"])
@pytest.mark.parametrize("tree", UNSTORABLE.values(), ids=UNSTORABLE.keys())
def test_an_unstorable_value_fails_once_on_either_engine(backend, enters, tree):
    outcome = run(backend, _program(enters, tree), _Suite())
    failed = backend.failure_kind(outcome.snap) if outcome.snap.state == "failed" else None
    assert (outcome.snap.state, backend.task_attempts(outcome.task), failed) == (
        "failed",
        1,
        "CompositionRefused",
    )


@pytest.mark.parametrize("enters", ["seed", "worker", "artifact"])
@pytest.mark.parametrize("tree", STORABLE.values(), ids=STORABLE.keys())
def test_a_storable_value_completes(backend, enters, tree):
    assert run(backend, _program(enters, tree), _Suite()).snap.state == "completed"


def _unserializable(_run_id: str):
    return (yield from store_artifact(object(), "application/json"))


def test_an_artifact_no_serializer_takes_fails_once(backend):
    outcome = run(backend, _unserializable, _Suite())
    failed = backend.failure_kind(outcome.snap) if outcome.snap.state == "failed" else None
    assert (backend.task_attempts(outcome.task), failed) == (1, "CompositionRefused")
