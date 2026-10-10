"""`govern`'s merged park on the REAL durable engine: the Quint model's claims, executed.

`govern_park.qnt` model-checks the protocol (fail-closed under crash, exactly-once absorption,
one-park-one-name, liveness). This file runs the same protocol against Absurd + Postgres, because
the house rule is that a park is verified on the real engine, not by design or by an in-memory
handler. The recording core cannot even park mid-layer (`RecordingHandler._interpret` says so in
as many words), so the in-memory tests exercise the ruling, never the suspend.

The central case is the one merged-park exists for: TWO heterogeneous policies (a dollars
accumulator and an approval classifier) blocking the same op, parking ONCE, and both being
settled by ONE resolution delivered to ONE event name, across a worker death. Two sequential
parks would ask a human twice for one decision; this proves the gate asks once.

Mechanism: the injected `AwaitEvent` reaches
`ctx.await_event`, Absurd raises `SuspendTask`, it propagates out through `drive_through`, and the
task parks durably. On delivery Absurd replays from the top, the gate re-runs, re-derives the SAME
park name from `|absorbed answers|` (the durable-state-only claim the Quint tooth attacks), and
re-binds by name. No captured layer stack.

Gated on the Podman test Postgres (`just pgt-up`).
"""

from uuid import uuid4

import pytest
from _approval_domain import CannedDomain, process_refund
from _durable import (
    DSN,
    absurd,
    ledger_kinds,
    pg_ready,
)
from _gate import at_spend

from effective import permission
from effective.budget import Grant, MeasuredBudget
from effective.budget import as_policy as budget_policy
from effective.govern import GateState, Resolution, govern
from effective.handlers.durable import DurableHandler
from effective.keys import Key
from effective.ledger import PostgresLedger
from effective.ops import Step, WorkflowOp
from effective.permission import Allow, Escalate

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")

GATE = "spend"
LIMIT = 0.005
OVER = 0.010  # a meter reading over the ceiling, so the budget policy parks


def _park_name(mid: str, before: str = "step:assess_request", pass_n: int = 0) -> Key:
    """The gate's park name, DERIVED from `GateState`. It is op-scoped, so a hard-coded
    `govern:{gate}:{run}:{pass}` would silently fail to match."""
    return GateState(run_id=mid, gate=GATE, pass_n=pass_n, op_key=before).park_name


def _gated(op: WorkflowOp, before: str = "assess_request") -> bool:
    """Gate exactly one Step, so the park's position in the run is unambiguous."""
    return isinstance(op, Step) and op.name == before


def _only_before(policy, before: str = "assess_request"):
    """Apply `policy` to the gated op only; every other op proceeds untouched."""
    from effective.govern import Proceed

    def scoped(op, state):
        return policy(op, state) if _gated(op, before) else Proceed()

    return scoped


def _register(app, name: str, run_id: str, domain, policies, *, max_attempts: int = 3):
    @app.register_task(name, default_max_attempts=max_attempts)
    def task(params, ctx, _domain=domain, _policies=tuple(policies)):
        ledger = PostgresLedger(DSN, workflow_run_id=params["request_id"])
        gate = govern(*_policies, gate=GATE, run_id=run_id)
        try:
            return DurableHandler(ctx, _domain, ledger=ledger, op_layers=(gate,)).run(
                lambda: process_refund(params["request_id"])
            )
        finally:
            ledger.close()

    return task


def _policies(run_id: str, *, spent: float = OVER, tier=None):
    """The two real policies, scoped to the gated op."""
    budget = MeasuredBudget(overall=LIMIT, run_id=run_id, on_exhaust="park")
    return [
        _only_before(at_spend(spent, budget_policy(budget))),
        _only_before(permission.as_policy([tier or (lambda op: Escalate())])),
    ]


def test_two_policies_park_ONCE_and_one_resolution_settles_both():
    """Merged-park on the durable path: one `AwaitEvent`, one answer, both policies cleared."""
    app = absurd()
    try:
        mid = f"gov-{uuid4().hex[:8]}"
        domain = CannedDomain(amount="42.00")
        _register(app, f"t-{mid}", mid, domain, _policies(mid))

        spawned = app.spawn(f"t-{mid}", {"request_id": mid})
        app.work_batch()  # runs to the gate and suspends

        snap = app.fetch_task_result(spawned)
        assert snap is not None
        assert snap.state != "completed"  # parked
        assert domain.calls == ["fetch_request"], domain.calls  # the gated op did NOT run
        assert ledger_kinds(mid) == []

        # ONE event name, carrying BOTH answers — the whole point of merged-park.
        app.emit_event(
            _park_name(mid).stored(),
            Resolution(
                answers={
                    "budget": Grant(add_dollars=0.010).model_dump(),
                    "permission": {"decision": "approve"},
                }
            ).model_dump(),
        )
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert domain.calls == ["fetch_request", "assess_request"], domain.calls
        assert ledger_kinds(mid) == ["assessment", "commitment"]
    finally:
        app.close()


def test_a_merged_park_survives_worker_death_and_resumes_on_a_fresh_worker():
    """`failClosed` + `awaitingMatchesPass`, executed: worker 1 parks and dies; worker 2 — a new
    process, registry, and domain — replays, re-derives the SAME park name from durable state,
    and binds the delivered resolution by name. The op runs exactly once, on worker 2."""
    mid = f"govdie-{uuid4().hex[:8]}"
    name = f"t-{mid}"

    app1 = absurd()
    domain1 = CannedDomain(amount="42.00")
    _register(app1, name, mid, domain1, _policies(mid))
    spawned = app1.spawn(name, {"request_id": mid})
    task_id = spawned
    app1.work_batch()  # parks at the merged gate
    assert domain1.calls == ["fetch_request"]
    app1.close()  # worker 1 is gone; the park lives in Postgres

    app2 = absurd()
    domain2 = CannedDomain(amount="42.00")
    _register(app2, name, mid, domain2, _policies(mid))
    app2.emit_event(
        _park_name(mid).stored(),
        Resolution(
            answers={
                "budget": Grant(add_dollars=0.010).model_dump(),
                "permission": {"decision": "approve"},
            }
        ).model_dump(),
    )
    snap = app2.run_until_result(task_id)
    try:
        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        # fetch_request was committed by worker 1 and replays from its checkpoint; only the
        # gated op runs live on worker 2.
        assert domain2.calls == ["assess_request"], domain2.calls
        assert ledger_kinds(mid) == ["assessment", "commitment"]
    finally:
        app2.close()


def test_a_refusing_policy_blocks_even_though_the_other_would_have_parked():
    """`refuse_dominates` on the durable path: an unattended (fail-mode) budget refuses, and the
    human is never asked — no park, no event, nothing past the gate."""
    app = absurd()
    try:
        mid = f"govref-{uuid4().hex[:8]}"
        domain = CannedDomain(amount="42.00")
        budget = MeasuredBudget(overall=LIMIT, run_id=mid, on_exhaust="fail")
        policies = [
            _only_before(at_spend(OVER, budget_policy(budget))),
            _only_before(permission.as_policy([lambda op: Escalate()])),
        ]
        _register(app, f"t-{mid}", mid, domain, policies, max_attempts=1)

        spawned = app.spawn(f"t-{mid}", {"request_id": mid}, max_attempts=1)
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "failed"  # the gate refused; the op never forwarded
        assert domain.calls == ["fetch_request"], domain.calls
        assert ledger_kinds(mid) == []
    finally:
        app.close()


def test_an_unobjecting_gate_is_transparent():
    """A gate whose policies all proceed changes nothing — no park, no extra ops, same result.
    (The colorless-workflow payoff: `govern` is handler assembly, not a workflow change.)"""
    app = absurd()
    try:
        mid = f"govok-{uuid4().hex[:8]}"
        domain = CannedDomain(amount="42.00")
        budget = MeasuredBudget(overall=LIMIT, run_id=mid, on_exhaust="park")
        policies = [
            _only_before(at_spend(0.0, budget_policy(budget))),  # well under the ceiling
            _only_before(permission.as_policy([lambda op: Allow()])),
        ]
        _register(app, f"t-{mid}", mid, domain, policies)

        spawned = app.spawn(f"t-{mid}", {"request_id": mid})
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert domain.calls == ["fetch_request", "assess_request"], domain.calls
        assert ledger_kinds(mid) == ["assessment", "commitment"]
    finally:
        app.close()
