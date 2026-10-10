"""A layer-INJECTED, suspending op survives suspend/resume.

An op-layer that injects an `AwaitEvent` the workflow never yielded (a permission gate
parking for a human) must survive the suspend and resume deterministically on replay.

It works on the durable path, and here is *why*: when the injected `AwaitEvent`
reaches `ctx.await_event` and the event isn't ready, Absurd raises `SuspendTask`,
which propagates out through `drive_through` (the trampoline throws it into the
layer's `yield`, the layer doesn't catch it, it bubbles to Absurd's suspend
handler) and the task parks durably. On delivery, Absurd **replays the task from
the top**, re-running the op-layer, which re-yields the *same deterministically
named* `AwaitEvent`; `ctx.await_event` now returns the committed event from
Absurd's durable log, and the layer forwards the original op. The injected
suspend is re-bound by name on replay, and no layer-stack continuation is captured.

This is a *mechanism* proof using a minimal inline permission gate; the real tiers are
`effective.permission.cascade`. Gated on the Podman test Postgres.
"""

from uuid import uuid4

import pytest
from _approval_domain import ApprovalEvent, CannedDomain, committed_id, process_refund
from _durable import (
    DSN,
    absurd,
    ledger_kinds,
    pg_ready,
)

from effective.handlers.durable import DurableHandler
from effective.keys import Key, Segment, compose_key
from effective.layers import op_layer
from effective.ledger import PostgresLedger
from effective.ops import AppendLedgerRow, AwaitEvent, Step, WorkflowOp
from effective.permission import Allow, Deny, Escalate, cascade, human, rules

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")


class Refused(Exception):
    """A denial from a permission gate — the op is not forwarded."""


def _gate(event_name: Key, before: str = "assess_request"):
    """A minimal permission op-layer: inject an approval `AwaitEvent` before the
    `before` Step, then forward (or refuse on a reject)."""

    @op_layer
    def permission(op: WorkflowOp):
        if isinstance(op, Step) and op.name == before:
            approval = yield AwaitEvent(name=event_name, schema=ApprovalEvent)
            if approval.decision == "reject":
                raise Refused(approval.rationale or "denied")
        return (yield op)

    return permission


def _register(app, name: str, mid: str, domain, layers, max_attempts: int = 3):
    @app.register_task(name, default_max_attempts=max_attempts)
    def task(params, ctx, _domain=domain, _layers=tuple(layers)):
        ledger = PostgresLedger(DSN, workflow_run_id=params["request_id"])
        try:
            return DurableHandler(ctx, _domain, ledger=ledger, op_layers=_layers).run(
                lambda: process_refund(params["request_id"])
            )
        finally:
            ledger.close()

    return task


def test_injected_await_suspends_then_resumes_on_approve():
    app = absurd()
    try:
        mid = f"perm-{uuid4().hex[:8]}"
        ev = compose_key(t"approve:{Segment(mid)}")
        domain = CannedDomain(amount="42.00")
        name = f"t-{mid}"
        _register(app, name, mid, domain, [_gate(ev)])

        spawned = app.spawn(name, {"request_id": mid})
        app.work_batch()  # runs to the INJECTED await and suspends

        snap = app.fetch_task_result(spawned)
        assert snap is not None
        assert snap.state != "completed"  # parked at the injected gate
        # The gated op (assess) has NOT run: the layer blocked it before forwarding:
        assert domain.calls == ["fetch_request"], domain.calls
        assert ledger_kinds(mid) == []  # nothing past the gate committed

        app.emit_event(
            ev.stored(), {"decision": "approve", "actor": "approver", "rationale": "ok"}
        )
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["status"] == "committed"
        # After approval the layer forwarded the op; assess ran exactly once:
        assert domain.calls == ["fetch_request", "assess_request"], domain.calls
        assert ledger_kinds(mid) == ["assessment", "commitment"]
    finally:
        app.close()


def test_injected_await_refuses_on_deny():
    app = absurd()
    try:
        mid = f"deny-{uuid4().hex[:8]}"
        ev = compose_key(t"approve:{Segment(mid)}")
        domain = CannedDomain(amount="42.00")
        name = f"t-{mid}"
        # max_attempts=1: a deliberate denial is terminal, not a retryable failure.
        _register(app, name, mid, domain, [_gate(ev)], max_attempts=1)

        spawned = app.spawn(name, {"request_id": mid}, max_attempts=1)
        app.work_batch()  # suspends at the injected await
        app.emit_event(
            ev.stored(), {"decision": "reject", "actor": "approver", "rationale": "nope"}
        )
        snap = app.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "failed"  # the gate refused; the op never forwarded
        assert domain.calls == ["fetch_request"], domain.calls  # assess never ran
        assert ledger_kinds(mid) == []  # nothing committed past the gate
    finally:
        app.close()


def test_parked_permission_survives_worker_death_resumes_on_fresh_worker():
    """The headline, on the durable path: a permission prompt parked by an
    injected layer survives the worker dying — a *fresh* worker (new process, new
    registry, new domain) resumes it by replay. fetch_request was committed by worker
    1 and replays from checkpoint on worker 2; only assess runs on worker 2."""
    mid = f"hand-{uuid4().hex[:8]}"
    ev = compose_key(t"approve:{Segment(mid)}")
    name = f"t-{mid}"

    # Worker 1: run to the injected gate and suspend, then DIE (close the app/conn).
    app1 = absurd()
    domain1 = CannedDomain(amount="42.00")
    _register(app1, name, mid, domain1, [_gate(ev)])
    spawned = app1.spawn(name, {"request_id": mid})
    task_id = spawned
    app1.work_batch()  # suspends at the injected await
    assert domain1.calls == ["fetch_request"]
    app1.close()  # worker 1 is gone — the task is durably parked in Postgres

    # Worker 2: a fresh process/registry/domain. Deliver the event and resume.
    app2 = absurd()
    domain2 = CannedDomain(amount="42.00")
    _register(app2, name, mid, domain2, [_gate(ev)])
    app2.emit_event(ev.stored(), {"decision": "approve", "actor": "approver", "rationale": "ok"})
    snap = app2.run_until_result(task_id)

    try:
        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["status"] == "committed"
        # fetch_request replayed from worker 1's checkpoint (NOT re-run); assess ran on worker 2:
        assert domain2.calls == ["assess_request"], domain2.calls
        assert ledger_kinds(mid) == ["assessment", "commitment"]
    finally:
        app2.close()


# --- the real cascade (effective.permission) over an UNMODIFIED process_refund -----------


def _gate_commit(op):
    """Policy: escalate the final commitment to a human; allow everything else."""
    if isinstance(op, AppendLedgerRow) and op.row.get("kind") == "commitment":
        return Escalate("commit needs sign-off")
    return Allow()


def test_cascade_escalates_commit_to_human_then_commits():
    """The DX thesis: a `cascade([rules, human])` op-layer adds HITL sign-off to the
    *unmodified* auto-path workflow. `rules` allows every op but the commitment, which
    it escalates to `human`, injecting an approval await that parks durably; on approve
    the op forwards and commits."""
    app = absurd()
    try:
        mid = f"casc-{uuid4().hex[:8]}"
        # The human tier parks on `approve;{op_key of the gated op}`, and the gated op is the
        # commitment — so ASK for the id rather than spelling it. A hand-written durable name
        # here fails as a HANG (the emit never matches the park), not as an assertion.
        # lint: terminal-hole — `committed_id` returns a `Key`, which the rule cannot see.
        ev = compose_key(t"approve;ledger;{committed_id(mid):domain=address}")
        domain = CannedDomain(amount="42.00")  # auto path: no built-in review
        name = f"t-{mid}"
        layers = [cascade([rules(_gate_commit), human(ApprovalEvent)])]
        _register(app, name, mid, domain, layers)

        spawned = app.spawn(name, {"request_id": mid})
        app.work_batch()  # runs to the commit; cascade escalates -> human await -> suspends

        snap = app.fetch_task_result(spawned)
        assert snap is not None
        assert snap.state != "completed"  # parked at the cascade's human tier
        assert domain.calls == ["fetch_request", "assess_request"]  # pre-commit ops allowed + ran
        assert ledger_kinds(mid) == ["assessment"]  # assessment allowed; commitment gated

        app.emit_event(
            ev.stored(), {"decision": "approve", "actor": "approver", "rationale": "ok"}
        )
        snap = app.run_until_result(spawned)
        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["status"] == "committed"
        assert ledger_kinds(mid) == ["assessment", "commitment"]
    finally:
        app.close()


def test_cascade_rules_deny_blocks_the_commit_without_a_human():
    """A `rules` tier that Denies decides immediately — no human tier consulted. The
    commit is Refused (terminal); the prior assessment was allowed and stays committed."""
    app = absurd()
    try:
        mid = f"cdeny-{uuid4().hex[:8]}"
        domain = CannedDomain(amount="42.00")
        name = f"t-{mid}"

        def deny_commit(op):
            if isinstance(op, AppendLedgerRow) and op.row.get("kind") == "commitment":
                return Deny("policy blocks commits")
            return Allow()

        layers = [cascade([rules(deny_commit)])]
        _register(app, name, mid, domain, layers, max_attempts=1)

        spawned = app.spawn(name, {"request_id": mid}, max_attempts=1)
        snap = app.run_until_result(spawned)
        assert snap is not None
        assert snap.state == "failed"  # Refused at the commit
        assert ledger_kinds(mid) == ["assessment"]  # assessment allowed; commitment blocked
    finally:
        app.close()
