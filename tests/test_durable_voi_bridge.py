"""The durable VOI probe: fork a REAL durable (SQLite) measured park.

Drives the production `DurableHandler` on a `SqliteTaskContext` to a measured park, bridges its
checkpoints + delivered grants to `(entries, grants)`, replays through `measured_drive`, and
asserts the two interpreters park at the SAME step / grant name / spend: cross-implementation
trip parity, where `test_measured_fork.py` drives `measured_drive` against itself. Infra-free
(in-memory SQLite).
"""

import pytest

from effective.api import step
from effective.bridge_sqlite import export_measured_prefix, park_name
from effective.budget import Grant, MeasuredBudget
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool
from effective.engines.sqlite import SqliteApp
from effective.fork import measured_drive
from effective.handlers.durable import DurableHandler
from effective.keys import Key

COST = 0.001


def _flat_llm(_op):
    return "ans", Usage(prompt_tokens=10, completion_tokens=5, cost=COST)


def _tools(_op):
    return "tool-done"


def _domain():
    return MeteredInterpreter(llm=_flat_llm, tools=_tools)


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


# The envelope confusable: a tool whose result is a perfectly plausible payload shaped EXACTLY
# like a usage envelope `{result, usage}`. A bridge that sniffs that shape mis-decodes the row
# (phantom usage 0.005 + a truncated result) and diverges from the durable handler, which
# decides envelope-ness by op class (`metered_call`), never content.
ENVELOPE_TOOL_RESULT = {
    "result": "report.pdf",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.005},
}


def _envelope_tool(_op):
    return dict(ENVELOPE_TOOL_RESULT)


def _env_domain():
    return MeteredInterpreter(llm=_flat_llm, tools=_envelope_tool)


def _ask_envtool_asks():
    a = yield from _ask("ask:a")
    yield from step("tool:t", CallTool(name="t", result_schema=dict))  # returns the envelope shape
    b = yield from _ask("ask:b")
    c = yield from _ask("ask:c")
    yield from _ask("ask:d")
    return [a, b, c]


def _spawn(app, run_id, program, *, limit, max_attempts=3, domain=_domain):
    @app.register_task(run_id)
    def task(params, ctx):
        budget = MeasuredBudget(overall=limit, run_id=params["run_id"], on_exhaust="park")
        return DurableHandler(ctx, domain(), contract=Contract.V1, budget=budget).run(program)

    return app.spawn(run_id, {"run_id": run_id}, max_attempts=max_attempts)


def _fork(program, run_id, limit, entries, grants, *, domain=_domain):
    budget = MeasuredBudget(overall=limit, run_id=run_id, on_exhaust="park")
    return measured_drive(program, budget, domain(), grants, recorded=entries)


def _run_state(app, tid) -> str:
    snap = app.run_until_result(tid)
    assert snap is not None
    return snap.state


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()  # close the connection so it isn't GC'd unclosed (a ResourceWarning)


def test_P3a_askllm_only_trip0_parity(app):
    tid = _spawn(app, "r", _five_asks, limit=0.0015)  # crosses on ask2 (0.002 >= 0.0015)
    assert _run_state(app, tid) == "waiting"

    entries, grants = export_measured_prefix(app.conn, tid, "r")
    assert [e.key.stored() for e in entries] == [
        "step:ask0",
        "step:ask1",
    ]  # 2 committed before the trip
    assert grants == {}  # no grant delivered yet

    tail = _fork(_five_asks, "r", 0.0015, entries, grants)
    assert tail.tripped_at.stored() == park_name(app.conn, tid) == "budget-grant:r,0"
    assert [e.key.stored() for e in tail.trace] == ["step:ask0", "step:ask1"]
    assert tail.usage.cost == pytest.approx(0.002)


def test_P3b_calltool_interleave_parity_the_F2_regression(app):
    tid = _spawn(app, "r", _ask_tool_ask, limit=0.0005)  # crosses at ask:b (after the tool)
    assert _run_state(app, tid) == "waiting"

    entries, _ = export_measured_prefix(app.conn, tid, "r")
    # the durable run ran the TOOL after the ceiling crossing, then parked before ask:b
    assert [e.key.stored() for e in entries] == ["step;ask:a", "step;tool:t"]

    tail = _fork(_ask_tool_ask, "r", 0.0005, entries, {})
    # the fork matches: it too runs (replays) the tool, parks before ask:b
    assert [e.key.stored() for e in tail.trace] == ["step;ask:a", "step;tool:t"]
    assert tail.tripped_at.stored() == park_name(app.conn, tid) == "budget-grant:r,0"


def test_P3c_trip_n_ge_1_needs_the_grant_export(app):
    tid = _spawn(app, "r", _five_asks, limit=0.0015)
    assert _run_state(app, tid) == "waiting"  # parks at :0
    app.emit_event("budget-grant:r,0", {"add_dollars": 0.001})  # grant 0
    assert _run_state(app, tid) == "waiting"  # runs ask2, re-parks at :1

    entries, grants = export_measured_prefix(app.conn, tid, "r")
    assert [e.key.stored() for e in entries] == [
        "step:ask0",
        "step:ask1",
        "step:ask2",
    ]  # ask2 committed after the grant
    assert grants == {Key.parse("budget-grant:r,0"): Grant(add_dollars=0.001)}
    assert park_name(app.conn, tid) == "budget-grant:r,1"

    # WITH the delivered grant, the fork re-derives the SAME park (:1, spend 0.003)
    tail = _fork(_five_asks, "r", 0.0015, entries, grants)
    assert tail.tripped_at == Key.parse("budget-grant:r,1")
    assert tail.usage.cost == pytest.approx(0.003)

    # WITHOUT the grant export, the fork re-parks one trip early with the wrong name/spend
    wrong = _fork(_five_asks, "r", 0.0015, entries, {})
    assert wrong.tripped_at == Key.parse("budget-grant:r,0")
    assert wrong.usage.cost == pytest.approx(0.002)


def test_L1_envelope_shaped_tool_result_is_not_mis_decoded(app):
    # ADVERSARIAL row: a CallTool returns a payload shaped exactly like a usage envelope. The
    # durable handler stores it as a bare CallTool value (zero usage); the bridge must NOT sniff
    # `{result, usage}` and mis-read it, or it decodes phantom usage (0.005) + a truncated result
    # and the fork trips two steps early.
    tid = _spawn(app, "r", _ask_envtool_asks, limit=0.003, domain=_env_domain)
    assert _run_state(app, tid) == "waiting"

    entries, grants = export_measured_prefix(app.conn, tid, "r")
    # durable ran the tool AFTER ask:a (folding 0 usage), then asks b, c under the ceiling, and
    # parked before ask:d at 0.003 — the tool never advanced the meter.
    assert [e.key.stored() for e in entries] == [
        "step;ask:a",
        "step;tool:t",
        "step;ask:b",
        "step;ask:c",
    ]

    tail = _fork(_ask_envtool_asks, "r", 0.003, entries, grants, domain=_env_domain)
    # NOT truncated to 2 steps (phantom usage would have tripped early):
    assert [e.key.stored() for e in tail.trace] == [
        "step;ask:a",
        "step;tool:t",
        "step;ask:b",
        "step;ask:c",
    ]
    assert tail.tripped_at.stored() == park_name(app.conn, tid) == "budget-grant:r,0"
    assert tail.usage.cost == pytest.approx(0.003)  # NOT 0.006 — no phantom tool usage

    tool_entry = next(e for e in tail.trace if e.key.stored() == "step;tool:t")
    assert tool_entry.result == ENVELOPE_TOOL_RESULT  # the FULL dict, not truncated
    assert tool_entry.usage.cost == 0.0  # the tool folds zero — envelope-ness is by op class
