"""The carrier audit: what survives a task boundary, and what silently does not.

A `respawn` generation IS a fresh task that keeps the same `workflow_run_id`. So the chain
question is answerable without respawn: spawn two tasks with one run id and look at what
carried. That is what this file does.

Everything a `DurableHandler` bookkeeps is per-task and re-derived from *this task's*
checkpoints: `self._meter`, `self._grants`, `self._trips` (`handlers/absurd.py:790-800`), the
per-attempt `run_scope()` dict (`layers.py:142`), the checkpoint history itself. Each therefore
needs a chain-level answer, and one of them has no owner anywhere:

    `Budget(dollars=100)` on a chain means $100 *per generation*, with no chain total,
    for a run the ledger presents as one run.

and *nothing red fires*: a 50-generation chain spending $5,000 under `Budget(dollars=100)`
passes every other test. This file is the red.

**These tests do not say the behavior is wrong.** Per-generation is a legal answer. They say
it must be CHOSEN at `Budget` and written down, rather than inherited from the fact that a
handler happens to be per-task. `Budget` declares TWO ceilings, `overall` and `per_generation`;
`enforce_generation` is called by no handler, and once it is wired the row's assertion changes to
match and the test stops being an audit and becomes the pin.

Park names aliasing across generations, the other half of the same question, is pinned by
`test_respawn_hazards_pending.py`, on raw Absurd, and is deliberately not duplicated here.
"""

from uuid import uuid4

import pytest
from _conformance import Contract, Fault, MeteringDomain, metered_trip_wf

from effective.keys import Segment, compose_key
from effective.layers import layer_run_state, run_scope

# A ceiling between 2x and 3x the per-ask cost, so the third ask trips it — the same
# arithmetic `test_measured_trip_fail_fast_aborts_within_the_ceiling` uses.
PER_ASK = 0.001
CEILING = 0.0015
# Enough generations that LINEAR growth is visible as growth, not as a doubling.
GENERATIONS = 4


def _run_one_generation(backend, run_id: str, generation: int):
    """One generation of a chain: a fresh task, the SAME run id, the SAME declared ceiling."""
    domain = MeteringDomain(cost=PER_ASK)
    name = f"gen{generation}-{run_id}"
    backend.register(
        name,
        metered_trip_wf,
        domain,
        Fault(),
        (),
        budget_limit=CEILING,
        on_exhaust="fail",
    )
    task_id = backend.spawn(name, run_id, max_attempts=1, contract=Contract.V1)
    return backend.run_until_result(task_id), domain


def test_the_measured_ceiling_re_arms_every_generation(backend):
    """A declared `$CEILING` buys `CEILING` per TASK, not per run.

    N tasks, one `run_id`: the shape of a respawn chain, which keeps the run id stable so the
    ledger sees one run. Each gets its own `DurableHandler`, hence its own
    `self._meter = Usage()` and `self._grants = 0.0`, re-derived from *its own* checkpoints.
    No generation can see another's spend.

    Run over N generations rather than two, because the shape is what matters: spend grows
    LINEARLY in N against a fixed declared ceiling, so it is unbounded.
    Measured here: ceiling $0.0015, four generations, $0.0080 spent = 5.3x. At one ceiling per
    generation, a 50-generation chain spends $5,000 under `Budget(dollars=100)`. A budget pool
    shared across tasks is deferred and undecided.

    **This is an audit.** Per-generation may well be the right semantics: it is bounded per
    unit of work and needs no new carrier. What is wrong is that nobody chose it; it fell out
    of a handler being per-task. When `enforce_generation` is wired this assertion changes to
    match, and the test becomes the pin."""
    run_id = f"chain-{uuid4().hex[:8]}"

    spend_per_generation = []
    for generation in range(GENERATIONS):
        snap, domain = _run_one_generation(backend, run_id, generation)
        # Each generation enforces the ceiling correctly *for itself* — that is the point.
        # Nothing here is broken locally, which is why nothing red fires today.
        assert snap is not None, generation
        assert snap.state == "failed", (generation, snap)
        assert len(domain.calls) == 2, (generation, domain.calls)
        spend_per_generation.append(len(domain.calls) * domain.cost)

    assert len(set(spend_per_generation)) == 1, "every generation buys the identical allowance"
    spent_by_the_run = sum(spend_per_generation)

    assert spent_by_the_run == pytest.approx(GENERATIONS * spend_per_generation[0]), (
        "spend is LINEAR in the number of generations — the ceiling is per task, so the run "
        "has no bound at all"
    )
    assert spent_by_the_run > CEILING, (
        f"one run id, one declared ceiling of {CEILING}, and the run spent {spent_by_the_run} "
        f"over {GENERATIONS} generations ({spent_by_the_run / CEILING:.1f}x). Nothing in the "
        "substrate holds a chain total."
    )


def test_a_fresh_task_starts_with_an_empty_checkpoint_history(backend):
    """The row that is working as intended — and the whole point of respawn.

    Recorded beside the others because an audit that only lists problems does not tell a
    reader which carriers are settled. Bounding replay history is exactly what a generation
    boundary buys; here it is, as a fact rather than a design claim."""
    run_id = f"hist-{uuid4().hex[:8]}"
    snap_0, _ = _run_one_generation(backend, run_id, 0)
    assert snap_0 is not None

    domain = MeteringDomain(cost=PER_ASK)
    name = f"fresh-{run_id}"
    backend.register(name, metered_trip_wf, domain, Fault(), (), budget_limit=1.0)
    task_id = backend.spawn(name, run_id, max_attempts=1, contract=Contract.V1)
    snap = backend.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "completed", snap
    assert backend.checkpoint_keys(task_id) != []
    # The generation's own history only — it re-executed every ask rather than re-binding
    # anything the earlier task committed under the same run id.
    assert len(domain.calls) == 3, "a fresh task re-runs its own work; nothing carried"


def test_the_layer_run_scope_does_not_cross_a_task_boundary():
    """`run_scope()` is per `Handler.run()` — per attempt, therefore per generation.

    Infra-free, because the lifetime is a `ContextVar`'s and needs no engine. This is the
    settled row: a gate's occurrence counters and accrued answers are re-derived from the
    durable event store on every replay rather than carried in process memory, which is what
    stopped a `govern()` closure letting task B proceed on task A's grant (`layers.py:119-136`).

    Its chain answer follows for free — a generation is an attempt, so nothing leaks — and
    saying so is the audit's job even when the answer is 'already correct'."""
    key = compose_key(t"gate-state:{Segment('g')},{Segment('r1')}")

    with run_scope():
        first = layer_run_state(key)
        first["occurrences"] = {"op": 3}

    with run_scope():
        second = layer_run_state(key)

    assert second == {}, "a second run (a second generation) must start clean"
    assert first is not second
