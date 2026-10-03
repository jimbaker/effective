"""A respawned generation lands on its PARENT's queue, on the real engine.

`_respawn` builds the successor's spawn itself, and must pass the caller's queue through as
`fork.py` and `agent/compose.py` do. A literal `"default"` would aim generation *n+1* of a chain
started on any other queue at a queue nobody is working.

**How that presents depends on a precondition, and the loud case is the common one.** If the
spawning app has REGISTERED the task for its own queue, as this test's does, the SDK refuses the
cross-queue spawn outright (`Absurd._prepare_spawn`: *"registered for queue X but spawn requested
queue default"*), the run fails every attempt, and nothing lands anywhere. The quiet version, where
the parent completes and an unclaimed successor sits in `default`, needs an app that has not
registered it.

Every other respawn test runs on the embedded SQLite engine, which has no queues at all: the one
engine where the defect cannot exist.

Absurd creates a table set per queue (`t_`/`r_`/`e_`/`c_`/`i_`/`w_` + name), so "which queue"
is answerable by asking which tables hold the rows. That is what makes this checkable rather
than a matter of reading the spawn args back.
"""

import uuid
from typing import Any

import psycopg
import pytest
from _durable import DSN, pg_ready
from pydantic import BaseModel

from effective.api import call_tool
from effective.budget import Budget
from effective.combinators import Again, Chain, Done, Turn, respawn
from effective.handlers.absurd import DurableHandler

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")


class Carry(BaseModel):
    seen: int = 0


def _absurd_spawner(app: Any):
    """A spawner that HONOURS the queue it is handed.

    Worth spelling out, because it is the other half of the defect: `_respawn` putting the
    parent's queue in the tool args only matters if the domain's spawner passes it on, and the
    only spawner in the tree before this one is `test_respawn_durable.py`'s, which drops the
    argument because SQLite has no queues to route between. A substrate that sends the right
    queue to a spawner that ignores it is still a chain landing in the wrong place."""

    def spawn(
        task_name: str,
        params: dict,
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> Any:
        spawned = app.spawn(
            task_name,
            params,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            queue=queue,
        )
        return spawned["task_id"] if isinstance(spawned, dict) else spawned

    return spawn


class _SpawningDomain:
    """Answers the substrate's spawn tool, and one observational call per generation.

    The observational call exists because `respawn`'s step must be a GENERATOR — a plain
    function returning `Again`/`Done` is not an `Effect` and never drives the engine."""

    def __init__(self, app: Any) -> None:
        from agent.runtime import spawn_tool

        self._spawn = spawn_tool(_absurd_spawner(app))

    def run(self, op: Any) -> Any:
        from effective.domain import SPAWN_TOOL

        return self._spawn(op) if op.name == SPAWN_TOOL else "tick"


def _count(queue: str, table: str, task_name: str) -> int:
    """Rows for `task_name` in one queue's task table. Absurd namespaces per queue, so this
    is the whole question: a successor on the wrong queue is a row in the wrong table."""
    with psycopg.connect(DSN) as conn:
        row = conn.execute(
            t"SELECT count(*) FROM absurd.{f'{table}_{queue}':i} WHERE task_name = {task_name}"
        ).fetchone()
        return row[0] if row else 0


@pytest.fixture
def isolated_queue():
    """A queue of this test's own, dropped afterwards — `drop_queue` removes its whole table
    set, so the fixture leaves no rows behind for the shared lane to trip over."""
    name = f"qroute{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("SELECT absurd.create_queue(%s)", (name,))
    yield name
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("SELECT absurd.drop_queue(%s)", (name,))


def test_a_respawned_generation_stays_on_its_parents_queue(isolated_queue):
    """Two generations on a non-default queue: the successor must be enqueued there too.

    Under the defect this queue holds exactly one task where two belong, which is why the
    assertion is on the SUCCESSOR's location rather than on the chain merely finishing."""
    from effective.absurd_worker import absurd_worker

    app: Any = absurd_worker(DSN, queue_name=isolated_queue)
    task_name = f"chain{uuid.uuid4().hex[:8]}"

    def body(carry: Carry, turn: Turn):
        yield from call_tool(f"tick{turn.generation}", {}, str)
        if turn.final:
            return Done({"seen": carry.seen + 1})
        return Again(Carry(seen=carry.seen + 1))

    @app.register_task(task_name)
    def task(params, ctx):
        chain = Chain.from_params(params, task=task_name, schema=Carry, initial=Carry())
        return DurableHandler(ctx, _SpawningDomain(app), ledger=None, params=params).run(
            lambda: respawn(body, chain, budget=Budget(generations=2))
        )

    app.spawn(task_name, {"run_id": f"r-{task_name}"})
    for _ in range(12):
        app.work_batch()

    assert _count(isolated_queue, "t", task_name) == 2, (
        "the successor generation is missing from this queue — `_respawn` sent it elsewhere, "
        "which is the defect this test exists for"
    )
    # NOT an anti-vacuity check: under the defect the SDK refuses the spawn before any row
    # reaches `default`, so this can never fail here and never runs at all in the red case (the
    # assertion above fires first). It guards against a FUTURE change that makes the cross-queue
    # spawn succeed, and a zero here proves nothing on its own.
    assert _count("default", "t", task_name) == 0, (
        "a generation leaked onto the `default` queue: `_respawn` is not reading the parent's "
        "queue name"
    )
