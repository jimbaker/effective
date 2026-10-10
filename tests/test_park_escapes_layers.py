"""A park unwinds the walk past every layer: a layer catching every exception never sees one,
and a layer whose cleanup raises or yields cannot replace it."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from _conformance import Fault, review_name

from effective.api import await_event, sleep_until
from effective.layers import op_layer, retry


def _catching(caught: list[str]):
    @op_layer
    def catches_everything(op):
        try:
            return (yield op)
        except Exception as exc:
            caught.append(type(exc).__name__)
            raise

    return catches_everything


def _awaits(run_id: str):
    return (yield from await_event(review_name(run_id), dict))


def _sleeps(run_id: str):
    yield from sleep_until(datetime.now(UTC) + timedelta(hours=1))
    return "woke"


PARKS = {"await": _awaits, "sleep": _sleeps}


@op_layer
def _cleanup_raises(op):
    try:
        return (yield op)
    finally:
        raise ValueError("cleanup")


@op_layer
def _cleanup_yields(op):
    try:
        return (yield op)
    finally:
        yield op


CLEANUPS = {"none": (), "raises": (_cleanup_raises,), "yields": (_cleanup_yields,)}

pytestmark = pytest.mark.conformance


@pytest.mark.parametrize("cleanup", CLEANUPS.values(), ids=CLEANUPS.keys())
@pytest.mark.parametrize("parks", PARKS.values(), ids=PARKS.keys())
def test_a_park_reaches_no_layer_and_survives_its_cleanup(backend, parks, cleanup):
    caught: list[str] = []
    run_id = f"r-{uuid4().hex[:8]}"
    layers = (retry(2, on=Exception), _catching(caught), *cleanup)
    backend.register(run_id, parks, None, Fault(None), layers)
    task_id = backend.spawn(run_id, run_id)
    snapshot = backend.run_until_result(task_id)
    assert snapshot.state in {"waiting", "sleeping"}, snapshot
    assert caught == []
