"""`improve` replays to an identical Pareto frontier across crash-at-every-op.

`improve` holds its candidate pool *in-memory* and re-derives it on replay from the recorded
`propose`/`score` ops, so a worker death at ANY op boundary must reconstruct the identical
frontier. The sharp case is a **self-refine** `improve` whose `propose` yields a
*candidate-producing* op, so the next candidate rides a committed checkpoint. One child per round
keeps it sequential, so crash-at-every-op is well-defined by op count; gather's concurrent
branches are name-targeted, and `test_conformance.py` proves those.

The mechanism (same as `test_replay_crash.py`): a `FaultCtx` raises once before the k-th
`ctx.step`, so ops 1..k-1 have committed and op k has not; Absurd's retry reloads the
checkpoint cache and returns committed op results without re-running their thunks, so the
frontier re-derives from the *recorded* candidates and measures. Gated on the Podman test
Postgres (`just pgt-up`); skips otherwise.

Run:  just pgt-up  &&  uv run pytest tests/test_replay_improve.py -v
"""

from uuid import uuid4

import pytest
from _durable import (
    DSN,
    Fault,
    FaultCtx,
    absurd,
    pg_ready,
)

from effective import step
from effective.cost import CostBudget, MeteredInterpreter
from effective.domain import CallTool
from effective.handlers.durable import DurableHandler
from effective.improve import Measurement, improve
from effective.ledger import PostgresLedger
from effective.pareto import Objective

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")

MAX_VAL = [Objective("val", "max")]


def _propose(parents, reflection):
    """Self-refine: mutate the best candidate up by one, via a recorded op (so the next
    candidate is a committed checkpoint, not a value the workflow recomputes)."""
    best = max(parents, key=lambda s: s.measures["val"])
    nxt = yield from step(
        "propose",
        CallTool(name="propose", result_schema=int, args={"parent": best.candidate}),
    )
    return [nxt]


def _score(c):
    m = yield from step("score", CallTool(name="score", result_schema=Measurement, args={"c": c}))
    return m


def _climb_wf():
    # seed 0 -> climbs 1, 2, 3 over three rounds; argmax singleton frontier = [3]
    return improve(0, _propose, _score, objectives=MAX_VAL, rounds=3)


def _no_llm(op):
    raise AssertionError("a climb makes no model calls")


def _climb_tools(op: CallTool):
    match op.name:
        case "propose":
            return op.args["parent"] + 1
        case "score":
            return Measurement(measures={"val": float(op.args["c"])})
    raise ValueError(f"unknown tool: {op.name!r}")


def _run(app, run_id: str, crash_at: int | None):
    fault = Fault(crash_at)
    name = f"climb-{crash_at}-{run_id}"

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx, _fault=fault):
        ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
        interp = MeteredInterpreter(llm=_no_llm, tools=_climb_tools, budget=CostBudget(1.0))
        try:
            return DurableHandler(FaultCtx(ctx, _fault), interp, ledger=ledger).run(_climb_wf)
        finally:
            ledger.close()

    spawned = app.spawn(name, {"run_id": run_id})
    return app.run_until_result(spawned), fault


def test_improve_replays_to_identical_frontier_across_crash_at_every_op():
    app = absurd()
    try:
        # Baseline: a clean run fixes the reference frontier AND the op count N.
        base = f"base-{uuid4().hex[:8]}"
        snap, fault = _run(app, base, crash_at=None)
        assert snap is not None
        assert snap.state == "completed", snap.failure
        ref = snap.result
        assert [p["candidate"] for p in ref] == [3]  # the climbed argmax
        assert ref[0]["measures"]["val"] == 3.0
        n_ops = fault.count
        assert n_ops == 7, n_ops  # seed/score + 3 x (propose, score)

        # Crash before every op boundary; each must re-derive the identical frontier.
        for k in range(1, n_ops + 1):
            rid = f"c{k}-{uuid4().hex[:8]}"
            snap, fault = _run(app, rid, crash_at=k)
            assert fault.armed is False, f"k={k}: fault never fired — no crash exercised"
            assert snap is not None, f"k={k}: no result"
            assert snap.state == "completed", f"k={k}: state={snap.state} failure={snap.failure}"
            assert snap.result == ref, f"k={k}: {snap.result} != {ref}"
    finally:
        app.close()
