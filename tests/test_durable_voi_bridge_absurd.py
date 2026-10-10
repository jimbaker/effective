"""The durable VOI probe on ABSURD/Postgres, the deployed engine.

The Absurd sibling of `test_durable_voi_bridge.py`: drive the production `DurableHandler` on a real
`ConcurrentAbsurdCtx` to a measured park, bridge its Postgres checkpoints + delivered grants via
`effective.bridge_absurd`, replay through `measured_drive`, and assert cross-implementation trip
parity (P3a/P3b/P3c). Needs Postgres/Absurd (`just pgt-up`); auto-skips otherwise. Unique run ids
per test (the Absurd events table is global per queue and persists across runs).
"""

import uuid

import psycopg
import pytest
from _durable import DSN, absurd, pg_ready

from effective.api import append_ledger, scoped, step
from effective.bridge_absurd import export_measured_prefix, park_name
from effective.budget import Grant, MeasuredBudget
from effective.cost import CONTRACT_PARAM, Contract, MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool
from effective.engines.absurd import ConcurrentAbsurdCtx
from effective.fork import measured_drive
from effective.handlers.absurd import DurableHandler
from effective.keys import Key, Segment, compose_key
from effective.ledger import PostgresLedger
from effective.ops import LedgerRow

pytestmark = pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")

COST = 0.001


def _flat_llm(_op):
    return "ans", Usage(prompt_tokens=10, completion_tokens=5, cost=COST)


def _domain():
    return MeteredInterpreter(llm=_flat_llm, tools=lambda _op: "tool-done")


def _ask(name):
    return step(name, AskLLM(messages="m", response_schema=str))


def _tool(name):
    return step(name, CallTool(name=name, result_schema=str))


def _five_asks():
    out = []
    for i in range(5):
        out.append((yield from _ask(f"ask{i}")))
    return out


def _ask_tool_ask():
    a = yield from _ask("ask:a")
    yield from _tool("tool:t")
    b = yield from _ask("ask:b")
    return [a, b]


def _two_asks():
    a = yield from _ask("ask0")
    b = yield from _ask("ask1")
    return [a, b]


# The L1 confusable on the REAL engine (jsonb): a tool result shaped exactly like a usage envelope.
ENVELOPE_TOOL_RESULT = {
    "result": "report.pdf",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.005},
}


def _env_domain():
    return MeteredInterpreter(llm=_flat_llm, tools=lambda _op: dict(ENVELOPE_TOOL_RESULT))


def _ask_envtool_asks():
    a = yield from _ask("ask:a")
    yield from step("tool:t", CallTool(name="t", result_schema=dict))
    b = yield from _ask("ask:b")
    c = yield from _ask("ask:c")
    yield from _ask("ask:d")
    return [a, b, c]


@pytest.fixture
def app():
    return absurd()


@pytest.fixture
def conn():
    c = psycopg.connect(DSN, autocommit=True)
    yield c
    c.close()


def _spawn(app, program, run_id, limit, *, domain=_domain):
    name = f"voi-{run_id}"

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx):
        rid = params["run_id"]
        ledger = PostgresLedger(DSN, workflow_run_id=rid)
        try:
            budget = MeasuredBudget(overall=limit, run_id=rid, on_exhaust="park")
            return DurableHandler(
                ConcurrentAbsurdCtx(ctx),
                domain(),
                ledger=ledger,
                contract=Contract.V1,
                budget=budget,
            ).run(program)
        finally:
            ledger.close()

    return app.spawn(name, {"run_id": run_id, CONTRACT_PARAM: Contract.V1.value})["task_id"]


def _run_until_park(app, conn, tid, expected):
    for _ in range(24):
        if park_name(conn, tid) == expected:
            return
        app.work_batch()
    assert park_name(conn, tid) == expected, park_name(conn, tid)


def _fork(program, run_id, limit, entries, grants, *, domain=_domain):
    return measured_drive(
        program,
        MeasuredBudget(overall=limit, run_id=run_id, on_exhaust="park"),
        domain(),
        grants,
        recorded=entries,
    )


def test_P3a_askllm_only_trip0_parity(app, conn):
    rid = f"a{uuid.uuid4().hex[:8]}"
    tid = _spawn(app, _five_asks, rid, limit=0.0015)  # crosses on ask2
    _run_until_park(app, conn, tid, f"budget-grant:{rid},0")

    entries, grants = export_measured_prefix(conn, tid, rid)
    assert [e.key.stored() for e in entries] == ["step:ask0", "step:ask1"]
    assert grants == {}

    tail = _fork(_five_asks, rid, 0.0015, entries, grants)
    assert tail.tripped_at == Key.parse(f"budget-grant:{rid},0")
    assert [e.key.stored() for e in tail.trace] == ["step:ask0", "step:ask1"]
    assert tail.usage.cost == pytest.approx(0.002)


def test_P3b_calltool_interleave_parity(app, conn):
    rid = f"b{uuid.uuid4().hex[:8]}"
    tid = _spawn(app, _ask_tool_ask, rid, limit=0.0005)  # crosses at ask:b, after the tool
    _run_until_park(app, conn, tid, f"budget-grant:{rid},0")

    entries, _ = export_measured_prefix(conn, tid, rid)
    assert [e.key.stored() for e in entries] == [
        "step;ask:a",
        "step;tool:t",
    ]  # the tool ran before the park

    tail = _fork(_ask_tool_ask, rid, 0.0005, entries, {})
    assert [e.key.stored() for e in tail.trace] == ["step;ask:a", "step;tool:t"]
    assert tail.tripped_at == Key.parse(f"budget-grant:{rid},0")


def test_P3c_trip_n_ge_1_needs_the_grant_export(app, conn):
    rid = f"c{uuid.uuid4().hex[:8]}"
    tid = _spawn(app, _five_asks, rid, limit=0.0015)
    _run_until_park(app, conn, tid, f"budget-grant:{rid},0")
    app.emit_event(f"budget-grant:{rid},0", {"add_dollars": 0.001})  # grant 0 (by name)
    _run_until_park(app, conn, tid, f"budget-grant:{rid},1")  # runs ask2, re-parks at :1

    entries, grants = export_measured_prefix(conn, tid, rid)
    assert [e.key.stored() for e in entries] == ["step:ask0", "step:ask1", "step:ask2"]
    assert grants == {compose_key(t"budget-grant:{Segment(rid)},0"): Grant(add_dollars=0.001)}

    # WITH the delivered grant, the fork re-derives the SAME park (:1, spend 0.003)
    tail = _fork(_five_asks, rid, 0.0015, entries, grants)
    assert tail.tripped_at == Key.parse(f"budget-grant:{rid},1")
    assert tail.usage.cost == pytest.approx(0.003)

    # WITHOUT it (F-3), the fork re-parks one trip early with the wrong name/spend
    wrong = _fork(_five_asks, rid, 0.0015, entries, {})
    assert wrong.tripped_at == Key.parse(f"budget-grant:{rid},0")
    assert wrong.usage.cost == pytest.approx(0.002)


def test_L1_envelope_shaped_tool_result_is_not_mis_decoded(app, conn):
    # ADVERSARIAL row on the REAL engine: a port is verified on the real engine, not by SQLite
    # alone. A CallTool returns a payload shaped like a usage envelope, stored as jsonb. The
    # bridge carries it raw; the driver decodes at the op (`metered_call` → CallTool is not
    # metered), so no phantom usage / truncation. Sniffing the payload's shape trips early.
    rid = f"e{uuid.uuid4().hex[:8]}"
    tid = _spawn(app, _ask_envtool_asks, rid, limit=0.003, domain=_env_domain)
    _run_until_park(app, conn, tid, f"budget-grant:{rid},0")

    entries, grants = export_measured_prefix(conn, tid, rid)
    assert [e.key.stored() for e in entries] == [
        "step;ask:a",
        "step;tool:t",
        "step;ask:b",
        "step;ask:c",
    ]

    tail = _fork(_ask_envtool_asks, rid, 0.003, entries, grants, domain=_env_domain)
    assert [e.key.stored() for e in tail.trace] == [
        "step;ask:a",
        "step;tool:t",
        "step;ask:b",
        "step;ask:c",
    ]
    assert tail.tripped_at == Key.parse(f"budget-grant:{rid},0")
    assert tail.usage.cost == pytest.approx(0.003)  # NOT 0.006

    tool_entry = next(e for e in tail.trace if e.key.stored() == "step;tool:t")
    assert tool_entry.result == ENVELOPE_TOOL_RESULT  # full dict, not truncated
    assert tool_entry.usage.cost == 0.0


def test_multirefill_within_one_ask_survives_real_suspend_resume(app, conn):
    # The durable trip is a DRIVER of the shared transition, parking on each `Parked` via a
    # fresh per-call local grants dict. A single ask
    # whose pre-check needs N>=2 grants must park at budget-grant:rid,0..:N-1 across REAL worker
    # suspend/resume, with THAT ask's checkpoint UNCOMMITTED the whole time — proving the local
    # dict rebuilds each resume (not a hidden bookkeeper). test_P3c only reaches :1 (one refill).
    rid = f"m{uuid.uuid4().hex[:8]}"
    tid = _spawn(app, _two_asks, rid, limit=0.0005)  # ask0 runs (meter 0.001); ask1 trips
    _run_until_park(app, conn, tid, f"budget-grant:{rid},0")
    assert [e.key.stored() for e in export_measured_prefix(conn, tid, rid)[0]] == [
        "step:ask0"
    ]  # ask1 parked

    # two insufficient refills (0.0002 each): the park advances :0 -> :1 -> :2 across real
    # suspend/resume, and ask1 STAYS uncommitted through every refill (the whole trip is one park).
    for k in (0, 1):
        app.emit_event(f"budget-grant:{rid},{k}", {"add_dollars": 0.0002})
        _run_until_park(app, conn, tid, f"budget-grant:{rid},{k + 1}")
        assert [e.key.stored() for e in export_measured_prefix(conn, tid, rid)[0]] == ["step:ask0"]

    # the 3rd grant lifts the ceiling above the meter; ask1 finally clears and commits.
    app.emit_event(f"budget-grant:{rid},2", {"add_dollars": 0.0002})
    for _ in range(24):
        app.work_batch()
        if [e.key.stored() for e in export_measured_prefix(conn, tid, rid)[0]] == [
            "step:ask0",
            "step:ask1",
        ]:
            break
    assert [e.key.stored() for e in export_measured_prefix(conn, tid, rid)[0]] == [
        "step:ask0",
        "step:ask1",
    ]


def _scoped_ledger_then_asks():
    """A ledger append INSIDE a `scoped(...)`, so its checkpoint reads `rec:0;ledger;…`."""
    yield from _ask("ask0")

    def body():
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"noted:{Segment('x')}"), kind="noted")
        )
        return (yield from _ask("ask:inner"))

    yield from scoped(compose_key(t"rec:{0}"), body)
    yield from _ask("ask1")
    yield from _ask("ask2")


def test_a_scoped_ledger_checkpoint_is_not_exported_as_a_step(app, conn):
    """A scoped checkpoint reads `rec:0;ledger;…`, so it does not START with its tag.

    The stakes are positional: `measured_drive` advances `state.steps` only on a `Step`, so one
    phantom entry shifts every later index and the fork replays the wrong value and usage."""
    rid = f"f{uuid.uuid4().hex[:8]}"
    tid = _spawn(app, _scoped_ledger_then_asks, rid, limit=0.0025)
    _run_until_park(app, conn, tid, f"budget-grant:{rid},0")

    entries, _ = export_measured_prefix(conn, tid, rid)
    assert [e.key.stored() for e in entries] == ["step:ask0", "rec:0;step;ask:inner", "step:ask1"]
