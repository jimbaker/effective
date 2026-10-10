"""The sweep's outcome classes, closed under composition, show no futile retry and no race lost
to a branch's error, on either engine.

`tests/_sweep.py` carries the grammar and the oracles; `scripts/futile_sweep.py` runs the same
closure from the command line and reports what each round cost.
"""

import pytest
from _durable import DSN, pg_ready
from _sweep import Rounds, closure, differ_on_absurd, run_alone, seeded

from effective.combinators import Converged
from effective.engines import open
from effective.handlers.durable import DurableHandler

ROUNDS = 4


@pytest.fixture(scope="module")
def closed():
    """The closure, run once for the module as a durable task whose rounds are domain ops."""
    first = seeded()
    outer = open("sqlite://")
    outer.register_task("sweep")(
        lambda params, ctx: DurableHandler(ctx, Rounds()).run(lambda: closure(first, ROUNDS))
    )
    snapshot = outer.run_until_result(outer.spawn("sweep", {}))
    outer.close()
    assert snapshot is not None
    return snapshot.result


def test_the_classes_reach_a_fixpoint_within_the_round_budget(closed):
    """`Converged` holds its value alone; `Unconverged` adds why the budget stopped it."""
    assert set(closed) == set(Converged.__dataclass_fields__)
    assert len(closed["value"]["rounds"]) < ROUNDS


def test_no_composition_retries_what_a_replay_would_raise_again(closed):
    assert closed["value"]["findings"] == []


def test_each_leaf_a_retry_can_recover_completes_alone():
    """A layer's fallback and a schema default hand an attempt a value it never recorded, so a
    retry can change what the workflow raises on."""
    recoverable = ("ok", "flaky", "fallback", "default")
    classes = {kind: run_alone(["leaf", kind], kind).cls for kind in recoverable}
    assert {kind: cls.split(":")[0] for kind, cls in classes.items()} == dict.fromkeys(
        recoverable, "completed"
    )


@pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")
def test_each_class_lands_in_the_same_class_on_absurd(closed):
    assert differ_on_absurd(closed["value"]["classes"], DSN) == []
