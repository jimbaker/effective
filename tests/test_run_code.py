"""``run_code``: the segment machine + the Monty engine.

Infra-free, three layers (the testing-layers discipline):

1. **Engine unit tests**: real Monty in-process (no I/O): live host calls,
   re-bind-by-index (replay-by-re-execution), action pause, denial injection,
   undeclared-name policy, resource limits, clock-as-effect.
2. **Combinator on RecordingHandler**: canned ``CodeOutcome`` segments pin the
   op-stream *shape*: deterministic step keys, explicit state threading
   (``fn_log`` / ``action_results`` ride op args), and the ``Refused`` →
   sandbox-``PermissionError`` route.
3. **End-to-end on DurableHandler + LocalCtx**: a real workflow whose code
   calls an observational function and a world-mutating action; the action
   lands as its own op and runs exactly once.

Durable crash-at-every-op / suspend-resume coverage is the pgt suite's job.
"""

import pytest

from effective.code import EXECUTE_TOOL, ActionCall, CodeOutcome, run_code
from effective.domain import CallTool
from effective.govern import BudgetRefused, Exceeded
from effective.handlers.absurd import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.layers import op_layer
from effective.monty import CodeEngineError, MontyEngine, execute_tool
from effective.ops import Minted, Step
from effective.permission import Refused

# ---------------------------------------------------------------- engine unit


def test_engine_runs_declared_functions_live_and_logs_them():
    calls: list[tuple] = []

    def fetch(prefix):
        calls.append(("fetch", prefix))
        return [{"name": f"{prefix}1", "credits": 3}, {"name": f"{prefix}2", "credits": 4}]

    engine = MontyEngine(functions={"fetch": fetch})
    outcome = engine.execute(
        {
            "code": (
                'rows = fetch(prefix)\n{"n": len(rows), "total": sum(r["credits"] for r in rows)}'
            ),
            "inputs": {"prefix": "COMS"},
            "functions": ["fetch"],
            "actions": [],
            "fn_log": [],
            "action_results": [],
        }
    )
    assert outcome.status == "complete"
    assert outcome.output == {"n": 2, "total": 7}
    assert calls == [("fetch", "COMS")]
    (entry,) = outcome.fn_log  # the ADR §3 shape: name + args-digest + value
    assert entry["name"] == "fetch"
    assert entry["value"] == [{"name": "COMS1", "credits": 3}, {"name": "COMS2", "credits": 4}]
    assert isinstance(entry["args"], str)
    assert entry["args"]


def test_engine_rebinds_from_fn_log_without_calling_live():
    """Replay-by-re-execution: with the recorded log threaded in, the registry
    is never hit — and the recorded (name, digest) is verified at the call site."""
    args = {
        "code": "rows = fetch(1)\nlen(rows)",
        "inputs": {},
        "functions": ["fetch"],
        "actions": [],
        "fn_log": [],
        "action_results": [],
    }
    live = MontyEngine(functions={"fetch": lambda n: [1, 2, 3]}).execute(args)

    def explode(*a, **k):
        raise AssertionError("live function called during re-bind")

    engine = MontyEngine(functions={"fetch": explode})
    outcome = engine.execute({**args, "fn_log": live.fn_log})
    assert outcome.status == "complete"
    assert outcome.output == 3
    assert outcome.fn_log == live.fn_log  # unchanged: nothing ran live


def test_engine_rebind_mismatch_is_loud_not_silent():
    """The args-digest detector: a recorded value that would land on the WRONG
    call site raises, never silently misbinds."""
    args = {
        "code": "x = fetch(1)\nx",
        "inputs": {},
        "functions": ["fetch"],
        "actions": [],
        "fn_log": [],
        "action_results": [],
    }
    live = MontyEngine(functions={"fetch": lambda n: n}).execute(args)
    tampered = [{**live.fn_log[0], "args": "0000000000000000"}]
    with pytest.raises(CodeEngineError, match="re-bind mismatch"):
        MontyEngine().execute({**args, "fn_log": tampered})


def test_engine_canonicalizes_at_the_boundary():
    """One canonical form (nan -> None, int keys -> str), live and recorded
    alike — the sandbox never sees a value its checkpoint round-trip changes."""
    engine = MontyEngine(functions={"probe": lambda: {"x": float("nan"), 1: "a"}})
    outcome = engine.execute(
        {
            "code": "v = probe()\n{'x_is_none': v['x'] is None, 'key': v['1']}",
            "inputs": {},
            "functions": ["probe"],
            "actions": [],
            "fn_log": [],
            "action_results": [],
        }
    )
    assert outcome.output == {"x_is_none": True, "key": "a"}
    assert outcome.fn_log[0]["value"] == {"x": None, "1": "a"}


def test_engine_pauses_at_action_and_resumes_from_threaded_result():
    engine = MontyEngine(functions={"llm_query": lambda p: f"summary of {p}"})
    args = {
        "code": (
            "s = llm_query('report')\n"
            "ack = send_email('approver@example.com', s)\n"
            "{'summary': s, 'ack': ack}"
        ),
        "inputs": {},
        "functions": ["llm_query"],
        "actions": ["send_email"],
        "fn_log": [],
        "action_results": [],
    }
    paused = engine.execute(args)
    assert paused.status == "action"
    assert paused.action == ActionCall(
        name="send_email", args=["approver@example.com", "summary of report"]
    )
    assert paused.fn_log[0]["value"] == "summary of report"  # the fn ran pre-pause

    # segment 2: fn re-binds from the log; the action re-binds from the result
    resumed = engine.execute(
        {**args, "fn_log": paused.fn_log, "action_results": [{"return_value": {"id": "msg-1"}}]}
    )
    assert resumed.status == "complete"
    assert resumed.output == {"summary": "summary of report", "ack": {"id": "msg-1"}}


def test_engine_injects_denial_as_permissionerror_the_code_can_catch():
    engine = MontyEngine()
    outcome = engine.execute(
        {
            "code": (
                "try:\n"
                "    r = send_email('x')\n"
                "except PermissionError as e:\n"
                "    r = f'routed around: {e}'\n"
                "r"
            ),
            "inputs": {},
            "functions": [],
            "actions": ["send_email"],
            "fn_log": [],
            "action_results": [
                {"exception": {"type": "PermissionError", "message": "denied by cascade"}}
            ],
        }
    )
    assert outcome.status == "complete"
    assert outcome.output == "routed around: denied by cascade"


def test_engine_undeclared_name_policy():
    engine = MontyEngine()
    base = {"inputs": {}, "functions": [], "actions": [], "fn_log": [], "action_results": []}
    # undeclared call -> NameError inside the sandbox (catchable)
    out = engine.execute(
        {**base, "code": "try:\n    x = mystery()\nexcept NameError:\n    x = 'caught'\nx"}
    )
    assert out.output == "caught"
    # undeclared OS access -> PermissionError inside the sandbox (denial as control flow)
    out = engine.execute(
        {
            **base,
            "code": "try:\n    open('/etc/hostname')\n    x = 'read'\n"
            "except PermissionError:\n    x = 'denied'\nx",
        }
    )
    assert out.output == "denied"
    # a bare undefined name is a code bug -> the step fails loudly, located
    with pytest.raises(CodeEngineError, match="undefined name 'ghost'"):
        engine.execute({**base, "code": "y = ghost\ny"})
    # a declared function missing from the registry is a deployment bug
    with pytest.raises(CodeEngineError, match="missing from the engine registry"):
        engine.execute({**base, "code": "f()", "functions": ["f"]})


def test_engine_unsupported_sandbox_event_is_a_loud_engine_error(monkeypatch):
    # the dispatch's terminal arm: a sandbox event the engine does not model
    # (a future Monty snapshot type) fails loudly, never a silent guess
    import pydantic_monty as monty

    class AlienSnapshot:
        pass

    monkeypatch.setattr(monty.Monty, "start", lambda self, **kwargs: AlienSnapshot())
    engine = MontyEngine()
    with pytest.raises(CodeEngineError, match="unsupported sandbox event: AlienSnapshot"):
        engine.execute({"code": "1", "inputs": {}})


def test_engine_resource_limits_kill_runaway_code():
    engine = MontyEngine(limits={"max_duration_secs": 0.2})
    with pytest.raises(Exception, match="time limit exceeded"):
        engine.execute(
            {
                "code": "while True:\n    pass",
                "inputs": {},
                "functions": [],
                "actions": [],
                "fn_log": [],
                "action_results": [],
            }
        )


def test_engine_clock_as_effect():
    """Registering an ambient-looking name makes the clock a recorded effect."""
    # the sandbox surfaces the call with its tz argument -> the stub accepts it
    engine = MontyEngine(functions={"datetime.now": lambda tz=None: "2026-07-03T12:00:00"})
    outcome = engine.execute(
        {
            "code": "import datetime\nstr(datetime.datetime.now())",
            "inputs": {},
            "functions": ["datetime.now"],
            "actions": [],
            "fn_log": [],
            "action_results": [],
        }
    )
    assert outcome.output == "2026-07-03T12:00:00"
    # recorded -> replays without a clock
    assert outcome.fn_log[0]["value"] == "2026-07-03T12:00:00"


# ------------------------------------------------- combinator (RecordingHandler)


def _wf():
    result = yield from run_code(
        "audit",
        "irrelevant to the recorder",
        schema=dict,
        functions=("llm_query",),
        actions={"send_email": dict},
    )
    return result


def test_run_code_threads_state_through_deterministic_op_keys():
    paused = CodeOutcome(
        status="action",
        action=ActionCall(name="send_email", args=["approver@example.com"], kwargs={}),
        fn_log=["summary"],
    )
    done = CodeOutcome(status="complete", output={"sent": True}, fn_log=["summary"])
    handler = RecordingHandler(
        responses={
            "code:seg,0,audit": paused,
            "code:action,0,audit;tool:send_email": {"id": "msg-1"},
            "code:seg,1,audit": done,
        }
    )
    assert handler.run(_wf) == {"sent": True}

    keys = [e.key.stored() for e in handler.trace]
    assert keys == [
        "step;code:seg,0,audit",
        "step;code:action,0,audit;tool:send_email",
        "step;code:seg,1,audit",
    ]
    # the action op is an ordinary CallTool on the real tool name (cascade-visible)
    action_step = handler.trace[1].op
    assert isinstance(action_step, Step)
    assert isinstance(action_step.op, CallTool)
    assert action_step.op.name == "send_email"
    # the action asks the handler for its idempotency key rather than carrying one
    assert action_step.op.args == {"args": ["approver@example.com"], "kwargs": {}}
    assert action_step.idempotency_key == Minted()
    # segment 2's args carry the threaded state — no handler pin-state anywhere
    seg1 = handler.trace[2].op
    assert isinstance(seg1, Step)
    assert isinstance(seg1.op, CallTool)
    assert seg1.op.name == EXECUTE_TOOL
    assert seg1.op.args["fn_log"] == ["summary"]
    assert seg1.op.args["action_results"] == [{"return_value": {"id": "msg-1"}}]


def test_run_code_threads_refusal_into_the_next_segment():
    @op_layer
    def deny_send(op):
        if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name == "send_email":
            raise Refused(op, "cascade: outbound email blocked")
        result = yield op
        return result

    paused = CodeOutcome(
        status="action", action=ActionCall(name="send_email", args=["x"]), fn_log=[]
    )
    done = CodeOutcome(status="complete", output={"routed": True}, fn_log=[])
    handler = RecordingHandler(
        responses={"code:seg,0,audit": paused, "code:seg,1,audit": done},
        op_layers=[deny_send],
    )
    assert handler.run(_wf) == {"routed": True}
    seg1 = handler.trace[-1].op
    assert isinstance(seg1, Step)
    assert isinstance(seg1.op, CallTool)
    assert seg1.op.args["action_results"] == [
        {"exception": {"type": "PermissionError", "message": "cascade: outbound email blocked"}}
    ]


def test_run_code_stops_at_a_budget_refusal():
    @op_layer
    def over_budget(op):
        if isinstance(op, Step) and isinstance(op.op, CallTool) and op.op.name == "send_email":
            raise BudgetRefused(op, Exceeded(spent=0.01, ceiling=0.005))
        result = yield op
        return result

    paused = CodeOutcome(
        status="action", action=ActionCall(name="send_email", args=["x"]), fn_log=[]
    )
    handler = RecordingHandler(responses={"code:seg,0,audit": paused}, op_layers=[over_budget])
    with pytest.raises(BudgetRefused):
        handler.run(_wf)
    assert [e.key.stored() for e in handler.trace][
        -1
    ] == "step;code:action,0,audit;tool:send_email"


def test_a_pinned_run_is_SCOPED_and_an_improvised_one_is_not():
    """The pinned-vs-improvised split is STRUCTURAL: a pinned run's segments carry the pin's
    key as a frame, so a projection selects on it and nothing records it as a payload.
    `run_code` applies the scope itself rather than leaving it to the caller, so an author
    cannot pin a script and forget to say so.

    MUTATION: return `(yield from segments())` unconditionally in `run_code`, dropping the
    `scoped(frame, ...)` arm: the pinned key loses its frame and this reddens.
    """
    from effective.keys import Segment, compose_key
    from effective.skills import Script

    done = CodeOutcome(status="complete", output={"ok": True}, fn_log=[])
    pin_key = compose_key(t"skill:{Segment('math')},activate")

    def wf():
        pinned = yield from run_code("p", Script(source="1 + 1", key=pin_key), schema=dict)
        improvised = yield from run_code("i", "2 + 2", schema=dict)
        return (pinned, improvised)

    handler = RecordingHandler(responses={"code:seg,0,p": done, "code:seg,0,i": done})
    handler.run(wf)
    assert [e.key.stored() for e in handler.trace] == [
        "skill:math,activate;step;code:seg,0,p",  # the pack is IN the key
        "step;code:seg,0,i",  # improvised: no frame, nothing to join to
    ]
    # ...and no op carries provenance as an argument any more
    for entry in handler.trace:
        assert isinstance(entry.op, Step)
        assert isinstance(entry.op.op, CallTool)
        assert "origin" not in entry.op.op.args


def test_run_code_rejects_non_json_inputs_before_any_op():
    """A set input's iteration order leaks the host hash seed into the sandbox,
    so it is rejected loudly at entry, before any op yields."""

    def wf():
        return (yield from run_code("x", "topic", schema=str, inputs={"topic": {"a", "b"}}))

    handler = RecordingHandler(responses={})
    with pytest.raises(TypeError, match=r"inputs\['topic'\].*not a JSON value"):
        handler.run(wf)
    assert handler.trace == []  # rejected before anything durable happened

    def wf_nan():
        return (yield from run_code("x", "v", schema=str, inputs={"v": float("nan")}))

    with pytest.raises(TypeError, match="non-finite"):
        RecordingHandler(responses={}).run(wf_nan)


def test_run_code_schema_rejects_none_output():
    """`schema=int` means int, not `int | None`."""
    from pydantic import ValidationError

    done = CodeOutcome(status="complete", output=None, fn_log=[])

    def wf():
        return (yield from run_code("n", "None", schema=int))

    with pytest.raises(ValidationError):
        RecordingHandler(responses={"code:seg,0,n": done}).run(wf)


# --------------------------------------------------------- end-to-end (LocalCtx)


class _CodeInterp:
    """A minimal DomainInterpreter: the reserved execute tool + one action tool."""

    def __init__(self, engine: MontyEngine, actions: dict) -> None:
        self._execute = execute_tool(engine)
        self._actions = actions

    def run(self, op):
        assert isinstance(op, CallTool)
        if op.name == EXECUTE_TOOL:
            return self._execute(op)
        return self._actions[op.name](op.args)


def test_run_code_end_to_end_action_runs_exactly_once():
    from effective.contexts import LocalCtx

    class TaskCtx(LocalCtx):
        """An in-process run with a task, which an action's idempotency key is minted from."""

        task_id = "report-task"

    sent: list[dict] = []

    def send_email(args: dict) -> dict:
        sent.append(args)
        return {"id": f"msg-{len(sent)}"}

    engine = MontyEngine(functions={"llm_query": lambda p: f"summary of {p}"})
    handler = DurableHandler(ctx=TaskCtx(), domain=_CodeInterp(engine, {"send_email": send_email}))

    def wf():
        out = yield from run_code(
            "report",
            (
                "s = llm_query(topic)\n"
                "ack = send_email('approver@example.com', s)\n"
                "{'summary': s, 'ack': ack}"
            ),
            schema=dict,
            inputs={"topic": "sales"},
            functions=("llm_query",),
            actions={"send_email": dict},
        )
        return out

    result = handler.run(wf)
    assert result == {
        "summary": "summary of sales",
        "ack": {"id": "msg-1"},
    }
    # the world changed exactly once, via its own op — never inside a segment
    assert sent == [
        {
            "args": ["approver@example.com", "summary of sales"],
            "kwargs": {},
            "idempotency_key": "idempotency:report-task;step;code:action,0,report;tool:send_email",
        }
    ]


def test_skill_script_activates_pins_and_executes(tmp_path):
    """The skills x code join, end to end: activate a pack, thread the pin, run
    its script in the sandbox — the source comes off the activation checkpoint
    (a registry swap after activation changes nothing), and the segment records
    the hash-keyed origin."""
    from effective.contexts import LocalCtx
    from effective.skills import DISCLOSE_TOOL, SkillRegistry, activate_skill

    pack = tmp_path / "summarize"
    pack.mkdir()
    (pack / "SKILL.md").write_text(
        "---\nname: summarize\ndescription: summarize a topic\n---\nUse required.py.\n"
    )
    (pack / "scripts").mkdir()
    (pack / "scripts" / "required.py").write_text("s = llm_query(topic)\n{'summary': s}")
    registry = SkillRegistry.load(tmp_path)

    class Interp:
        def run(self, op):
            assert isinstance(op, CallTool)
            if op.name == DISCLOSE_TOOL:
                return registry.disclose(op.args["skill"])
            assert op.name == EXECUTE_TOOL
            return execute_tool(engine)(op)

    engine = MontyEngine(functions={"llm_query": lambda p: f"summary of {p}"})
    handler = DurableHandler(ctx=LocalCtx(), domain=Interp())

    def wf():
        pin = yield from activate_skill("summarize")
        out = yield from run_code(
            "r",
            pin.script("required.py"),
            schema=dict,
            inputs={"topic": "sales"},
            functions=("llm_query",),
        )
        return {"out": out, "pin_key": pin.script("required.py").key.stored()}

    result = handler.run(wf)
    assert result["out"] == {"summary": "summary of sales"}
    # the script is scoped to the ACTIVATION; the pack's content hash stays one join away,
    # on the activation checkpoint
    assert result["pin_key"] == "skill:summarize,activate"


# ----------------------------------------------------------- cross-process replay


_PHASED = '''
"""Phased runner for the cross-process replay test."""
import json
import sys

sys.path.insert(0, "tests")
from _conformance import Fault, FaultCtx
from pydantic import BaseModel

from effective.code import EXECUTE_TOOL, run_code
from effective.domain import CallTool
from effective.handlers.absurd import DurableHandler
from effective.monty import MontyEngine, execute_tool
from effective.sqlite import SqliteApp, SqliteLedger


class Tags(BaseModel):
    id: str
    tags: set[str]


BODY = (
    "r = get_tags()\\n"
    "first = r['tags'][0]\\n"
    "ack = send_email(first)\\n"
    "{'first': first, 'ack': ack}"
)


class Domain:
    def __init__(self):
        self.sent = []
        self._execute = execute_tool(MontyEngine())

    def run(self, op):
        assert isinstance(op, CallTool)
        if op.name == EXECUTE_TOOL:
            return self._execute(op)
        if op.name == "get_tags":
            return {"id": "t", "tags": ["alpha", "beta", "gamma", "delta", "epsilon"]}
        assert op.name == "send_email"
        self.sent.append(op.args)
        return {"id": "msg-1"}


def wf(run_id):
    out = yield from run_code(
        "c", BODY, schema=dict, actions={"get_tags": Tags, "send_email": dict}
    )
    return out


phase, db_path = sys.argv[1], sys.argv[2]
fault = Fault(on_name="code:seg,2,c") if phase == "phase1" else Fault(None)
app = SqliteApp(db_path)
domain = Domain()


@app.register_task("t")
def task(params, ctx):
    ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock)
    return DurableHandler(FaultCtx(ctx, fault), domain, ledger=ledger).run(
        lambda: wf(params["run_id"])
    )


if phase == "phase1":
    tid = app.spawn("t", {"run_id": "r1"})
    app.work_batch()  # runs to the injected crash at seg:2; task requeued
    print(json.dumps({"sent": [a["args"][0] for a in domain.sent]}))
else:
    row = app.conn.execute("SELECT task_id FROM tasks").fetchone()
    snap = app.run_until_result(row[0])
    print(json.dumps({"result": snap.result, "sent": [a["args"][0] for a in domain.sent]}))
'''


def test_run_code_replay_agrees_across_processes_and_hash_seeds(tmp_path):
    """Pinned at the family level: a crash + FRESH-PROCESS resume under a
    different PYTHONHASHSEED must agree with the pre-crash world. The action
    result (a set-bearing schema) threads in its raw checkpoint form; re-dumped
    from the validated object per process, the resumed run would report a
    recipient the world never saw."""
    import json as _json
    import subprocess
    import sys

    script = tmp_path / "phased.py"
    script.write_text(_PHASED)
    db = tmp_path / "wf.db"

    def run(phase: str, seed: str) -> dict:
        proc = subprocess.run(
            [sys.executable, str(script), phase, str(db)],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": "src:tests"},
            cwd=".",
            check=True,
        )
        return _json.loads(proc.stdout.strip().splitlines()[-1])

    phase1 = run("phase1", "1")
    assert len(phase1["sent"]) == 1  # the action fired pre-crash and committed
    world_saw = phase1["sent"][0]

    phase2 = run("phase2", "2")  # fresh process, different hash seed
    assert phase2["sent"] == []  # nothing re-fired on resume
    assert phase2["result"]["first"] == world_saw  # the record agrees with the world


# ----------------------- cross-process re-bind of a declared-function set ARG --

_PHASED_SETARG = '''
"""Durable twin of the _args_digest re-bind test.

The prior test threads a set through an ACTION *schema*; this one drives the other,
sharper path: the sandbox builds a set and passes it to a DECLARED FUNCTION (`tally`),
so `tally`'s call-site fingerprint (`monty._args_digest`, built on `canonical()`) is
recorded in the segment's `fn_log`. It then calls an ACTION (`save`), a segment
boundary where phase1 injects a crash. On resume in a FRESH process under a different
PYTHONHASHSEED, the resumed segment re-executes and RE-BINDS `tally`: it recomputes the
fingerprint over the set arg and compares it to the recorded one. `canonical()` routes
through the set-sorting `canonical_form`, so the two agree across the seed change; a set
re-serialized in hash order would give recomputed != recorded, a false `CodeEngineError`
on a legitimate resume."""
import json
import sys

sys.path.insert(0, "tests")
from _conformance import Fault, FaultCtx

from effective.code import EXECUTE_TOOL, run_code
from effective.domain import CallTool
from effective.handlers.absurd import DurableHandler
from effective.monty import MontyEngine, execute_tool
from effective.sqlite import SqliteApp, SqliteLedger


BODY = (
    "s = {'zebra', 'apple', 'mango'}\\n"
    "n = tally(s)\\n"
    "ack = save(n)\\n"
    "{'n': n, 'ack': ack}"
)


def tally(s):
    return len(s)


class Domain:
    def __init__(self):
        self.saved = []
        self._execute = execute_tool(MontyEngine(functions={"tally": tally}))

    def run(self, op):
        assert isinstance(op, CallTool)
        if op.name == EXECUTE_TOOL:
            return self._execute(op)
        assert op.name == "save"
        self.saved.append(op.args)
        return {"id": "rcpt-1"}


def wf(run_id):
    out = yield from run_code(
        "c", BODY, schema=dict, functions=("tally",), actions={"save": dict}
    )
    return out


phase, db_path = sys.argv[1], sys.argv[2]
fault = Fault(on_name="code:seg,2,c") if phase == "phase1" else Fault(None)
app = SqliteApp(db_path)
domain = Domain()


@app.register_task("t")
def task(params, ctx):
    ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock)
    return DurableHandler(FaultCtx(ctx, fault), domain, ledger=ledger).run(
        lambda: wf(params["run_id"])
    )


if phase == "phase1":
    app.spawn("t", {"run_id": "r1"})
    app.work_batch()  # runs to the injected crash at seg:2; task requeued
    print(json.dumps({"saved": domain.saved}))
else:
    row = app.conn.execute("SELECT task_id FROM tasks").fetchone()
    try:
        snap = app.run_until_result(row[0])
        print(json.dumps({"result": snap.result, "saved": domain.saved}))
    except Exception as exc:  # a false re-bind mismatch surfaces here
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
'''


def test_run_code_rebind_survives_fresh_process_resume_with_a_set_arg(tmp_path):
    """Pinned at the family level on the DURABLE path: a `run_code` segment that
    called a declared host function with a set-bearing argument must re-bind cleanly on
    a fresh-worker resume under a different PYTHONHASHSEED. The re-bind recomputes the
    call-site fingerprint; a hash-order serialization of the set would make the
    recomputed digest differ from the recorded one, raising a false `CodeEngineError`.
    This is the durable twin of the engine-level pin in `test_canonical_determinism`."""
    import json as _json
    import subprocess
    import sys

    script = tmp_path / "phased_setarg.py"
    script.write_text(_PHASED_SETARG)
    db = tmp_path / "wf.db"

    def run(phase: str, seed: str) -> dict:
        proc = subprocess.run(
            [sys.executable, str(script), phase, str(db)],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": "src:tests"},
            cwd=".",
            check=True,
        )
        return _json.loads(proc.stdout.strip().splitlines()[-1])

    phase1 = run("phase1", "1")
    assert len(phase1["saved"]) == 1  # the action fired pre-crash and committed

    phase2 = run("phase2", "2")  # fresh process, different hash seed
    assert "error" not in phase2, (
        f"legitimate cross-process resume raised a false re-bind mismatch: {phase2}"
    )
    assert phase2["saved"] == []  # the action did not re-fire on resume
    assert phase2["result"]["n"] == 3  # |{'zebra','apple','mango'}| — re-bind agreed
