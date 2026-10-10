"""Crash-at-every-op durable replay: the no-layer baseline.

The property a stop-anywhere durable substrate must hold: *if the worker dies at
any op boundary, a fresh worker reconstructs the identical committed outcome by
replaying committed checkpoints* (the Temporal/Absurd model: no first-class
continuations). The layer stack builds on this property; here it is established for
the *bare* workflow, with no op-layer in the stack.

How we crash deterministically, without OS-level kills: a `FaultCtx` proxy raises
once, *before* the k-th `ctx.step`, so ops 1..k-1 have committed their checkpoints
and op k has not. Absurd's own retry then re-runs the task; `_create_task_context`
reloads the checkpoint cache and `ctx.step` returns the cached value for the
already-committed ops (their thunks are NOT re-run), so the model/tool side-effects
fire **exactly once** no matter where the crash lands. We spawn with an immediate
(`base_seconds=0`) retry strategy so the retry is claimable by the next
`work_batch()` with no backoff stall.

Isolation is by unique `workflow_run_id` + committed state (NOT transaction
rollback): the whole point is that a *fresh connection* must see the prior worker's
committed checkpoints. Gated on the Podman test Postgres
(`just pgt-up`); skips otherwise so the infra-free suite stays green.

Run:  just pgt-up  &&  DATABASE_URL=...:5432/effective uv run pytest tests/test_replay_crash.py -v
"""

from typing import Any
from uuid import uuid4

import pytest
from _approval_domain import CannedDomain, process_refund
from _durable import (
    DSN,
    Fault,
    FaultCtx,
    absurd,
    ledger_kinds,
    pg_ready,
)

from effective.handlers.durable import DurableHandler
from effective.layers import retry
from effective.ledger import PostgresLedger

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")


def _run_one(
    app,
    mid: str,
    crash_at: int | None,
    op_layers=(),
    domain=None,
) -> tuple[Any, CannedDomain, Fault]:
    """Spawn one process_refund run, crashing before op `crash_at` (None = clean).

    `op_layers` is the DurableHandler op-layer stack (e.g. `[retry(...)]`); `domain`
    defaults to a fresh `CannedDomain` (auto path -> commit).
    """
    if domain is None:
        domain = CannedDomain(amount="42.00")  # under threshold -> auto -> commit
    fault = Fault(crash_at)
    name = f"crash-{crash_at}-{mid}"

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx, _domain=domain, _fault=fault, _layers=tuple(op_layers)):
        ledger = PostgresLedger(DSN, workflow_run_id=params["request_id"])
        try:
            return DurableHandler(
                FaultCtx(ctx, _fault), _domain, ledger=ledger, op_layers=_layers
            ).run(lambda: process_refund(params["request_id"]))
        finally:
            ledger.close()

    spawned = app.spawn(name, {"request_id": mid})
    snap = app.run_until_result(spawned)
    return snap, domain, fault


def test_crash_at_every_op_replays_to_identical_outcome():
    app = absurd()
    try:
        # Baseline: a clean run establishes the reference outcome AND the op count N.
        base_mid = f"base-{uuid4().hex[:8]}"
        snap, domain, fault = _run_one(app, base_mid, crash_at=None)
        assert snap is not None
        assert snap.state == "completed"
        assert snap.result["status"] == "committed"
        assert domain.calls == ["fetch_request", "assess_request"]
        n_ops = fault.count
        # the 5 ops: fetch_request, artifact, assess, ledger:assessed, ledger:committed
        assert n_ops == 5, n_ops
        ref_ledger = ledger_kinds(base_mid)
        assert ref_ledger == ["assessment", "commitment"]

        # Crash before each op boundary 1..N; every one must replay to the same result.
        for k in range(1, n_ops + 1):
            mid = f"crash{k}-{uuid4().hex[:8]}"
            snap, domain, fault = _run_one(app, mid, crash_at=k)

            assert fault.armed is False, f"k={k}: fault never fired — no crash exercised"
            assert snap is not None, f"k={k}: no result"
            assert snap.state == "completed", f"k={k}: state={snap.state} failure={snap.failure}"
            assert snap.result["status"] == "committed", f"k={k}: {snap.result}"
            # Exactly-once side effects across the crash+replay, at EVERY boundary:
            assert domain.calls == ["fetch_request", "assess_request"], f"k={k}: {domain.calls}"
            # The append-only ledger is identical to the no-crash run (idempotent by event_id):
            assert ledger_kinds(mid) == ref_ledger, f"k={k}: {ledger_kinds(mid)}"
    finally:
        app.close()


def test_retry_op_layer_recovers_transient_failure_durably():
    """The first op-layer proven through the REAL durable path (not _LocalCtx).

    A `retry` op-layer recovers a transient assessment failure *within one Absurd
    attempt*: the re-forward re-calls `ctx.step(Key.parse("assess_request"))`, which works
    because Absurd checkpoints only *successful* steps — a failed thunk leaves
    nothing to collide with, so the C2 `op_key`-attempt-disambiguation worry does
    not arise on the durable path. The committed outcome and append-only ledger
    are correct; the assessment thunk ran twice (fail, then success) but committed once.
    """
    app = absurd()
    try:
        mid = f"retry-{uuid4().hex[:8]}"
        flaky = CannedDomain(amount="42.00", flaky_assess=True)
        snap, domain, _ = _run_one(
            app, mid, crash_at=None, op_layers=[retry(attempts=2)], domain=flaky
        )

        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result["status"] == "committed"
        # retry re-ran the assessment thunk once (transient), then it succeeded:
        assert domain.calls == ["fetch_request", "assess_request", "assess_request"], domain.calls
        # ...but the durable record committed exactly once:
        assert ledger_kinds(mid) == ["assessment", "commitment"]
    finally:
        app.close()


def test_retry_op_layer_in_stack_survives_crash_at_every_op():
    """An op-layer present in the stack must not perturb durable crash-replay.

    With `retry` composed into the handler and NOT triggered (the domain is not flaky),
    crashing before every op boundary must still replay to the identical committed outcome.
    Proves an op-layer is replay-safe on the durable path even across worker death, the
    property the cascade needs.
    """
    app = absurd()
    layers = [retry(attempts=2)]
    try:
        base_mid = f"rbase-{uuid4().hex[:8]}"
        snap, domain, fault = _run_one(app, base_mid, crash_at=None, op_layers=layers)
        assert snap is not None
        assert snap.state == "completed"
        n_ops = fault.count
        assert n_ops == 5, n_ops  # retry not triggered -> same 5 boundaries as the bare run
        ref_ledger = ledger_kinds(base_mid)
        assert ref_ledger == ["assessment", "commitment"]

        for k in range(1, n_ops + 1):
            mid = f"rcrash{k}-{uuid4().hex[:8]}"
            snap, domain, fault = _run_one(app, mid, crash_at=k, op_layers=layers)
            assert fault.armed is False, f"k={k}: fault never fired"
            assert snap is not None, f"k={k}: no result"
            assert snap.state == "completed", f"k={k}: {snap.state}"
            assert snap.result["status"] == "committed", f"k={k}: {snap.result}"
            assert domain.calls == ["fetch_request", "assess_request"], f"k={k}: {domain.calls}"
            assert ledger_kinds(mid) == ref_ledger, f"k={k}: {ledger_kinds(mid)}"
    finally:
        app.close()
