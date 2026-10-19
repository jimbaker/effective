"""A measured fork refuses a step prefix whose earlier occurrences a layer refused."""

from contextlib import suppress
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from _conformance import PG_DSN, CountingDomain, Fault
from _steps import placed_step

from effective.api import call_tool
from effective.checkpoints import SparsePrefix, dense_key
from effective.keys import FramePosition, Key
from effective.layers import current_placement, op_layer
from effective.permission import Refused

A, A2, A3 = (placed_step("a", "tool", occurrence=n) for n in (1, 2, 3))
B, B2 = (placed_step("b", "tool", occurrence=n) for n in (1, 2))


@op_layer
def _deny_the_first_ask(op):
    if current_placement() == Key.parse("step;tool:a"):
        raise Refused(op, "first ask")
    return (yield op)


def _refused_then_asked(run_id: str):
    first: Any = "refused"
    with suppress(Refused):
        first = yield from call_tool("a", {}, int)
    return [first, (yield from call_tool("a", {}, int))]


def _export(backend, task_id, run_id: str):
    if backend.name == "sqlite":
        from effective.bridge_sqlite import export_measured_prefix

        return export_measured_prefix(backend.app.conn, str(task_id), run_id)
    from effective.bridge_absurd import export_measured_prefix

    with psycopg.connect(PG_DSN) as conn:
        return export_measured_prefix(conn, task_id, run_id)


def test_a_prefix_missing_a_refused_ask_is_refused_on_export(backend):
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(
        run_id, _refused_then_asked, CountingDomain(), Fault(None), (_deny_the_first_ask,)
    )
    task_id = backend.spawn(run_id, run_id)
    snapshot = backend.run_until_result(task_id)
    assert snapshot.state == "completed", snapshot
    assert backend.checkpoint_keys(task_id) == ["step;tool:a#2"]

    with pytest.raises(SparsePrefix, match="occurrence 2"):
        _export(backend, task_id, run_id)


@pytest.mark.parametrize(
    "names",
    [[A2], [A, A3], [A, B2]],
    ids=["first-missing", "second-missing", "another-name-counted"],
)
def test_a_prefix_with_a_gap_in_any_name_is_refused(names):
    seen = FramePosition()
    for name in names[:-1]:
        dense_key(name, seen)
    with pytest.raises(SparsePrefix):
        dense_key(names[-1], seen)


def test_a_dense_prefix_keeps_each_occurrence():
    seen = FramePosition()
    names = [A, B, A2, A3]
    assert [dense_key(n, seen).stored() for n in names] == names
