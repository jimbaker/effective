"""Measured-spend accrual: the usage-in-checkpoint envelope + the handler-owned,
replay-derived meter. The core claim: a measured trip must be a function of *recorded*
state, so the meter re-derives on replay without re-running the domain. These run infra-free
over the embedded SQLite engine's durable ctx (checkpoints persist across handler instances, so a
second run IS a replay)."""

import json
from typing import Any
from uuid import UUID

import pytest
from _authority import ruled

from effective.api import ask_llm, gather
from effective.budget import MeasuredBudget
from effective.cost import (
    FORMAT_PARAM,
    NEWEST_FORMAT,
    MeteredInterpreter,
    RecordedFormat,
    Usage,
    marked,
)
from effective.engines.sqlite import SqliteApp, SqliteTaskContext
from effective.handlers.durable import DurableHandler
from effective.layers import compose_domain
from effective.ops import CompositionRefused, RecordedValueRejected, UnreadFormat

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


def _stored(app: SqliteApp, task: UUID, name: str) -> Any:
    row = app.conn.execute(
        "SELECT state FROM checkpoints WHERE task_id=? AND name=?", (task, name)
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


@pytest.fixture
def task(app):
    return app.spawn("accrual", {})


def test_v1_envelopes_askllm_and_meters_above_the_checkpoint(app, task):
    interp = MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools)
    handler = DurableHandler(SqliteTaskContext(app.conn, task, clock=app.clock), interp)

    out = handler.run(_two_ask_wf)

    assert out == {"a": "ans-1", "b": "ans-2"}
    assert handler.meter == BOTH  # both usages folded into the handler-owned meter
    stored = _stored(app, task, "step:ask1")
    assert set(stored) == {"result", "usage"}  # enveloped, not a bare result
    assert stored["result"] == "ans-1"  # the workflow still sees the bare result on read
    assert stored["usage"]["cost"] == 0.001


def test_v1_meter_rederives_on_replay_without_rerunning_the_domain(app, task):
    """The F1 fix, as a poison-stub replay proof: the second run's domain RAISES if
    touched, yet the meter re-derives to the same total from the recorded envelopes."""
    h1 = DurableHandler(
        SqliteTaskContext(app.conn, task, clock=app.clock),
        MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools),
    )
    h1.run(_two_ask_wf)
    assert h1.meter == BOTH

    def _poison(_op):
        raise AssertionError("replay must not re-run the domain (F1: usage is recorded state)")

    h2 = DurableHandler(
        SqliteTaskContext(app.conn, task, clock=app.clock),
        MeteredInterpreter(llm=_poison, tools=_poison),
    )
    out = h2.run(_two_ask_wf)

    assert out == {"a": "ans-1", "b": "ans-2"}
    assert h2.meter == h1.meter  # re-derived from the record, never re-run


def test_a_bare_model_checkpoint_fails_loud_not_silent(app, task):
    """A model call recorded bare, as a release before the envelope wrote it, fails by name:
    read as `$0` it would mis-derive the trip."""
    DurableHandler(
        SqliteTaskContext(app.conn, task, clock=app.clock),
        MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools),
    ).run(_two_ask_wf)
    app.conn.execute(
        "UPDATE checkpoints SET state = ? WHERE task_id = ? AND name = ?",
        (json.dumps("ans-1"), task, "step:ask1"),
    )

    with pytest.raises(RecordedValueRejected, match=r"\{result, usage\}"):
        DurableHandler(
            SqliteTaskContext(app.conn, task, clock=app.clock),
            MeteredInterpreter(llm=_no_tools, tools=_no_tools),  # poison — never reached
        ).run(_two_ask_wf)


def test_v1_gather_folds_branch_subtotals_into_parent_in_index_order(app, task):
    """§4 / Lean Model B: each branch accrues into its own subtotal; the barrier folds
    them into the parent in branch-index order. Sequential ctx (no write_lock)."""
    interp = MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools)
    handler = DurableHandler(SqliteTaskContext(app.conn, task, clock=app.clock), interp)

    out = handler.run(_gather_ask_wf)

    assert out == {"results": ["ans-1", "ans-2"]}
    assert handler.meter == BOTH  # both branches' subtotals reached the parent meter
    assert isinstance(
        _stored(app, task, "gather:0,0;step:ask0"), dict
    )  # per-branch envelope, prefixed


@pytest.mark.parametrize("recorded_format", list(RecordedFormat), ids=str)
def test_contract_from_params_round_trips(recorded_format):
    from effective.cost import FORMAT_PARAM

    assert RecordedFormat.from_params({FORMAT_PARAM: recorded_format.value}) is recorded_format


@pytest.mark.parametrize(
    "params", [{}, {"__accrual_contract__": "v1"}], ids=["unmarked", "old-key"]
)
def test_params_without_the_formats_key_are_refused(params):
    with pytest.raises(UnreadFormat):
        RecordedFormat.from_params(params)


def test_params_are_marked_once_with_the_newest_format():
    params = marked({"run_id": "r"})
    assert (params, RecordedFormat.from_params(params), marked(params)) == (
        {"run_id": "r", FORMAT_PARAM: NEWEST_FORMAT.value},
        NEWEST_FORMAT,
        params,
    )


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
        return DurableHandler(ctx, interp, budget=budget).run(_three_ask_wf)

    return app.spawn(run_id, {"run_id": run_id}, max_attempts=max_attempts)


def test_measured_trip_parks_then_a_grant_resumes(app):
    tid = _spawn_measured(app, "r1", limit=0.0015, on_exhaust="park")

    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "waiting"  # parked on the 3rd ask's trip, not complete

    app.emit_event("budget-grant:r1,0", ruled({"add_dollars": 0.01}))  # trip_n=0
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
    app.emit_event("budget-grant:r1,0", ruled({"stop": True}))  # answer with what you have
    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "failed"
    assert "budget exceeded" in str(snap.failure or "").lower()


def test_a_measured_budget_over_a_domain_that_reports_no_usage_is_refused(app, task):
    hidden = compose_domain(
        [], base=MeteredInterpreter(llm=_metering_llm(USAGES), tools=_no_tools)
    )
    with pytest.raises(CompositionRefused, match="MeasuredBudget"):
        DurableHandler(
            SqliteTaskContext(app.conn, task, clock=app.clock),
            hidden,
            budget=MeasuredBudget(overall=1.0, run_id="r"),
        )
