"""Known hazards pinned as `xfail(strict)`: each pin turns red the moment its fix lands.

`strict` fires on XPASS and treats any *exception* as an ordinary xfail. A fix that changes the API
a pin CALLS makes the pin raise, report a quiet XFAIL, and stay here with the suite green, so each
pin calls only stable surface (`RecordingHandler.run`). Delete the file when the last pin passes.

The term-head refusals (only a `Tag` may stand as a whole term; only a `Tag` may name a term)
live in `tests/test_op_key_injectivity.py`, beside the composer they guard.
"""

import pytest

from effective.api import append_ledger, gather
from effective.handlers.recording import RecordingHandler
from effective.keys import Segment, compose_key
from effective.ops import LedgerRow

# --- the recorder is a third interpreter -----------------------------------------------------

EV = compose_key(t"done:{Segment('m1')}")


def _colliding_branch():
    yield from append_ledger(LedgerRow(event_id=EV, kind="done"))
    return None


def _colliding_gather():
    yield from gather([_colliding_branch, _colliding_branch])
    return None


@pytest.mark.xfail(
    strict=True,
    reason="RecordingHandler appends to a plain list with no writer and no UNIQUE(event_id), so "
    "a placed-writer collision that FAILS both durable engines completes silently in memory with "
    "two rows. Fix: give the recorder a per-run task token, key its ledger by event_id, and call "
    "the shared refuse_placed_writer_collision, so one decision function serves all three.",
)
def test_the_recorder_refuses_a_placed_writer_collision_like_the_engines_do():
    """Two gather branches appending the SAME authored `event_id`.

    Both durable engines fail the task with `PlacedWriterCollision` and keep exactly one row.
    The recorder, the author's infra-free harness, completes with **two**: it reports green on a
    composition production refuses. The engine-parametrized conformance suite cannot see it,
    because the recorder is no engine."""
    handler = RecordingHandler({})
    with pytest.raises(Exception, match="ollision"):
        handler.run(_colliding_gather)


def test_the_recorder_divergence_the_pin_above_describes():
    """The before-picture, for the same reason as its sibling above."""
    handler = RecordingHandler({})
    handler.run(_colliding_gather)
    rows = [r for r in handler.ledger if r.event_id == EV]
    assert len(rows) == 2, "two rows for one event_id, where both engines keep one"
