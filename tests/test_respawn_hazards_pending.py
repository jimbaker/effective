"""The engine semantics a respawn chain has to account for.

This is not a test of respawn. It pins the **substrate behavior respawn has to account for**,
Absurd's own event semantics, so it passes whether or not the substrate composes
generation-distinct names, and it never flips. The substrate's half, that every settlement
namespace separates two generations of one run, is
`test_parked_reader.py::test_a_settlement_namespace_separates_two_GENERATIONS_of_one_run`.
"""

import uuid

import pytest
from _durable import absurd, pg_ready, run_until_result

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")


def test_absurd_events_alias_across_tasks_sharing_a_composed_name():
    """Absurd events alias across tasks that compose the same name.

    Absurd events are queue-global and immutable: `emit_event` allows exactly one
    NULL->payload transition per name, ever (`infra/absurd/absurd.sql:1832-1841`), and a
    *fresh* await is answered instantly from the cached payload. The substrate's own
    authority names embed the run id plus a counter that restarts whenever a new handler is
    constructed (`DurableHandler._trips = 0`, `handlers/absurd.py:758`; composed at
    `budget.py:220`).

    A respawn chain keeps `workflow_run_id` STABLE across generations by design, so a name
    built from the run id and a restarting counter is the same in generation n+1 as in
    generation n, and generation n+1 is granted without parking, without a human, and without
    any record that two generations shared one authorization. Run-scoping the name does not
    separate them, because the run id is constant across the chain.

    **SQLite cannot show this**: its events are addressed by `(task_id, name)`
    (`sqlite.py:153-157`), so a green 0<->1 pass proves nothing here.

    **This test pins ENGINE semantics and never flips.** It imports no `effective` module and
    composes the park name by hand, so what it asserts is Absurd's own behavior: a
    queue-global immutable event answering a fresh await. Generation-distinct names change
    what the *substrate* composes, not this, so inverting it would produce a spurious red. This
    one is the standing proof of *why* those names are required.
    """
    app = absurd()
    suffix = uuid.uuid4().hex[:8]
    # A stable run id + a counter that restarts per generation: a respawn chain's shape.
    name = f"budget-grant:r-{suffix},0"

    @app.register_task(f"gen-{suffix}", default_max_attempts=1)
    def gen(params, ctx):
        return {"generation": params["g"], "granted": ctx.await_event(name)}

    def spawn(generation: int):
        task = app.spawn(f"gen-{suffix}", {"g": generation})
        return task["task_id"] if isinstance(task, dict) else task

    generation_0 = spawn(0)
    app.work_batch()  # runs, then parks on `name`
    app.emit_event(name, {"add_dollars": 25.0})  # the human answers GENERATION 0
    result_0 = run_until_result(app, generation_0)
    assert result_0.state == "completed"
    assert result_0.result["granted"] == {"add_dollars": 25.0}

    # Generation 1: a DIFFERENT task, composing the same name — nobody grants it anything.
    result_1 = run_until_result(app, spawn(1), max_batches=3)

    assert result_1 is not None, "generation 1 produced no result at all"
    assert result_1.state == "completed", (
        "expected generation 1 to be answered instantly by generation 0's grant. This pins "
        "Absurd's own event semantics: if it now parks, the ENGINE changed, not the "
        "substrate. Do not invert; see the docstring."
    )
    assert result_1.result["granted"] == {"add_dollars": 25.0}, (
        "generation 1 received generation 0's payload — one authorization, two generations, "
        "no record that they shared it"
    )
