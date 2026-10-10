"""Measured-spend accrual: the usage-in-checkpoint envelope + the handler-owned,
replay-derived meter. The core claim: a measured trip must be a function of *recorded*
state, so the meter re-derives on replay without re-running the domain. These run infra-free
over the embedded SQLite engine's durable ctx (checkpoints persist across handler instances, so a
second run IS a replay)."""

import json
from typing import Any
from uuid import UUID

import pytest

from effective.api import ask_llm, gather
from effective.budget import MeasuredBudget
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.handlers.durable import DurableHandler

# These tests build a ctx directly rather than spawning, so the task id is a FIXTURE.
# A fixed literal, not `uuid7()`: the id keys every checkpoint row these tests write, and a
# constant keeps those rows byte-stable run to run. The shape is a real v7 uuid so it parses
# the way an engine-minted one does.
_TASK = UUID("019fa000-0000-7000-8000-000000000001")

# Two AskLLM turns with distinct usage, so a mis-fold is visible in the totals.
USAGES: list[tuple[str, Usage]] = [
    ("ans-1", Usage(prompt_tokens=10, completion_tokens=5, cost=0.001)),
    ("ans-2", Usage(prompt_tokens=20, completion_tokens=8, cost=0.002)),
]
BOTH = Usage(prompt_tokens=30, completion_tokens=13, cost=0.003)


def _metering_llm(turns):
    """An LLMCall that hands back `(result, usage)` per call, in order."""
    it = iter(turns)

    def call(_op):
        return next(it)

    return call


def _no_tools(_op):
    raise AssertionError("this workflow yields no CallTool")


def _two_ask_wf():
    a = yield from ask_llm("ask1", "m1", str)
    b = yield from ask_llm("ask2", "m2", str)
    return {"a": a, "b": b}


def _gather_ask_wf():
    def branch(name, msg):
        def thunk():
            return (yield from ask_llm(name, msg, str))

        return thunk

    results = yield from gather([branch("ask0", "m0"), branch("ask1", "m1")])
    return {"results": results}


def _stored(app: SqliteApp, name: str) -> Any:
    row = app.conn.execute(
        "SELECT state FROM checkpoints WHERE task_id=? AND name=?", (_TASK, name)
    ).fetchone()
    assert row is not None, f"no checkpoint {name!r}"
    return json.loads(row[0])


@pytest.fixture
def app():
    """A fresh in-memory durable engine, closed on teardown so the sqlite3 connection
    isn't GC'd unclosed (a ResourceWarning) — test hygiene, not a durability requirement."""
    a = SqliteApp(":memory:")
    yield a
    a.close()


def test_v1_envelopes_askllm_and_meters_above_the_checkpoint(app):
    interp = MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools)
    handler = DurableHandler(SqliteTaskContext(app.conn, _TASK), interp, contract=Contract.V1)

    out = handler.run(_two_ask_wf)

    assert out == {"a": "ans-1", "b": "ans-2"}
    assert handler.meter == BOTH  # both usages folded into the handler-owned meter
    stored = _stored(app, "step:ask1")
    assert set(stored) == {"result", "usage"}  # enveloped, not a bare result
    assert stored["result"] == "ans-1"  # the workflow still sees the bare result on read
    assert stored["usage"]["cost"] == 0.001


def test_v1_meter_rederives_on_replay_without_rerunning_the_domain(app):
    """The F1 fix, as a poison-stub replay proof: the second run's domain RAISES if
    touched, yet the meter re-derives to the same total from the recorded envelopes."""
    h1 = DurableHandler(
        SqliteTaskContext(app.conn, _TASK),
        MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools),
        contract=Contract.V1,
    )
    h1.run(_two_ask_wf)
    assert h1.meter == BOTH

    def _poison(_op):
        raise AssertionError("replay must not re-run the domain (F1: usage is recorded state)")

    h2 = DurableHandler(
        SqliteTaskContext(app.conn, _TASK),
        MeteredInterpreter(llm=_poison, tools=_poison),
        contract=Contract.V1,
    )
    out = h2.run(_two_ask_wf)

    assert out == {"a": "ans-1", "b": "ans-2"}
    assert h2.meter == h1.meter  # re-derived from the record, never re-run


def test_v1_handler_on_a_v0_checkpoint_fails_loud_not_silent(app):
    """The migration guard's failure mode (the footgun `Contract.from_params` exists to
    prevent): if a v1 handler ever reads a v0 (bare) checkpoint — e.g. an operator bumps a
    task's contract ignoring its params — it must fail LOUD, never silently misread the
    usage as $0 and mis-derive the trip. The bare result `"ans-1"` has no `["result"]`."""
    DurableHandler(
        SqliteTaskContext(app.conn, _TASK),
        MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools),
        contract=Contract.V0,
    ).run(_two_ask_wf)  # commit bare v0 rows

    with pytest.raises(TypeError):  # decoding a bare string as an envelope raises, not $0
        DurableHandler(
            SqliteTaskContext(app.conn, _TASK),
            MeteredInterpreter(llm=_no_tools, tools=_no_tools),  # poison — never reached
            contract=Contract.V1,
        ).run(_two_ask_wf)


def test_v0_is_byte_identical_bare_result_and_no_handler_accrual(app):
    interp = MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools)
    handler = DurableHandler(SqliteTaskContext(app.conn, _TASK), interp, contract=Contract.V0)

    out = handler.run(_two_ask_wf)

    assert out == {"a": "ans-1", "b": "ans-2"}
    assert _stored(app, "step:ask1") == "ans-1"  # bare result — identical to the pre-ADR format
    assert handler.meter == Usage()  # the handler meter is dormant on v0
    assert interp.meter == BOTH  # the telemetry meter (metered layer) still accrues live


def test_v1_gather_folds_branch_subtotals_into_parent_in_index_order(app):
    """§4 / Lean Model B: each branch accrues into its own subtotal; the barrier folds
    them into the parent in branch-index order. Sequential ctx (no write_lock)."""
    interp = MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools)
    handler = DurableHandler(SqliteTaskContext(app.conn, _TASK), interp, contract=Contract.V1)

    out = handler.run(_gather_ask_wf)

    assert out == {"results": ["ans-1", "ans-2"]}
    assert handler.meter == BOTH  # both branches' subtotals reached the parent meter
    assert isinstance(_stored(app, "gather:0,0;step:ask0"), dict)  # per-branch envelope, prefixed


@pytest.mark.parametrize("contract", [Contract.V0, Contract.V1])
def test_contract_from_params_round_trips(contract):
    from effective.cost import CONTRACT_PARAM

    assert Contract.from_params({CONTRACT_PARAM: contract.value}) is contract
    assert Contract.from_params({}) is Contract.V0  # absent → v0 (in-flight tasks)


# ── the measured trip: park-and-ask on the durable engine ───────────────
def _flat_llm(cost: float = 0.001):
    """A stateless LLMCall — a fixed `(result, usage)` per call, so replay's re-run order
    (committed steps re-bind without touching the domain) can't skew the totals."""

    def call(_op):
        return "ans", Usage(prompt_tokens=10, completion_tokens=5, cost=cost)

    return call


def _three_ask_wf():
    a = yield from ask_llm("ask1", "m1", str)
    b = yield from ask_llm("ask2", "m2", str)
    c = yield from ask_llm("ask3", "m3", str)
    return {"a": a, "b": b, "c": c}


def _spawn_measured(app, run_id, *, limit, on_exhaust, max_attempts=3, cost=0.001):
    """A v1 task whose 3 asks (0.001 each) trip a `limit`-dollar ceiling on the 3rd."""

    @app.register_task(run_id)
    def task(params, ctx):
        interp = MeteredInterpreter(llm=_flat_llm(cost), tools=_no_tools)
        budget = MeasuredBudget(overall=limit, run_id=params["run_id"], on_exhaust=on_exhaust)
        return DurableHandler(ctx, interp, contract=Contract.V1, budget=budget).run(_three_ask_wf)

    return app.spawn(run_id, {"run_id": run_id}, max_attempts=max_attempts)


def test_measured_trip_parks_then_a_grant_resumes(app):
    tid = _spawn_measured(app, "r1", limit=0.0015, on_exhaust="park")

    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "waiting"  # parked on the 3rd ask's trip, not complete

    app.emit_event("budget-grant:r1,0", {"add_dollars": 0.01})  # trip_n=0
    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "completed"
    assert snap.result == {"a": "ans", "b": "ans", "c": "ans"}  # the 3rd ask ran after the grant


def test_measured_trip_fail_fast_raises_without_parking(app):
    tid = _spawn_measured(app, "r1", limit=0.0015, on_exhaust="fail", max_attempts=1)

    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "failed"
    assert "budget exceeded" in str(snap.failure or "").lower()


def test_measured_trip_stop_grant_refuses_the_over_budget_ask(app):
    tid = _spawn_measured(app, "r1", limit=0.0015, on_exhaust="park", max_attempts=1)

    app.run_until_result(tid)  # parks on budget-grant:r1,0
    app.emit_event("budget-grant:r1,0", {"stop": True})  # answer with what you have
    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "failed"
    assert "budget exceeded" in str(snap.failure or "").lower()
