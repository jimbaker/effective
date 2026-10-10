"""A layer closed under a park is finished, and its cleanup's exception becomes a note on it."""

from typing import Any

import pytest

from effective.handlers.admission import drive_racing
from effective.handlers.base import EngineSignal
from effective.layers import drive_through, op_layer

pytestmark = pytest.mark.conformance


class _CleanupAbort(BaseException):
    pass


@op_layer
def _cleanup_aborts(op):
    try:
        return (yield op)
    finally:
        raise _CleanupAbort("cleanup")


@op_layer
def _cleanup_yields_then_raises(op):
    try:
        return (yield op)
    finally:
        try:
            yield op
        finally:
            raise RuntimeError("second close")


def _parks(op):
    raise EngineSignal("park")


DRIVERS = {
    "through": lambda layers: drive_through(layers, object(), _parks, escapes=(EngineSignal,)),
    "racing": lambda layers: drive_racing(
        layers, object(), _parks, new_work=lambda: None, stopped=(EngineSignal,)
    ),
}
FAILING_CLEANUPS = {"aborts": _cleanup_aborts, "yields-then-raises": _cleanup_yields_then_raises}


@pytest.mark.parametrize("cleanup", FAILING_CLEANUPS.values(), ids=FAILING_CLEANUPS.keys())
@pytest.mark.parametrize("drive", DRIVERS.values(), ids=DRIVERS.keys())
def test_a_cleanup_that_fails_leaves_the_park_in_flight_with_a_note(drive, cleanup):
    started: list[Any] = []

    def layer(op):
        started.append(gen := cleanup(op))
        return gen

    with pytest.raises(EngineSignal) as parked:
        drive([layer])
    assert started[0].gi_frame is None
    assert any("cleanup" in note or "second close" in note for note in parked.value.__notes__)
