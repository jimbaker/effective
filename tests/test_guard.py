"""A guard judges each action before it runs: it proceeds, refuses, or asks the user and parks.

The door guard of `examples/smol_door.py` is the worked guard, run under the recorder and on both
engines, and crashed at every op."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultCtx, FaultPosition, at_every_op

from effective import Effect
from effective.api import qualified_event_name
from effective.combinators import Chain, Level
from effective.domain import ASK_TOOL, SPAWN_TOOL, Answers, AskLLM, CallTool, DomainOp, Judge
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.interpreters.tools import spawn_tool
from effective.keys import Index, Run, compose_key
from effective.react import (
    Allowed,
    Ask,
    AssistantTurn,
    Proceed,
    Refuse,
    Step,
    ToolRequest,
    ToolResult,
    Trajectory,
    Verdict,
    run_agent,
)
from effective.smol import Conversation, smol

EXAMPLES = Path(__file__).parent.parent / "examples"


def _door() -> Any:
    spec = importlib.util.spec_from_file_location("smol_door", EXAMPLES / "smol_door.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smol_door = _door()

SEND = AssistantTurn(
    thought="let Sam in",
    tool=ToolRequest(name="send_code", args={"visitor": "Sam", "minutes": 30}),
)
DONE = AssistantTurn(thought="done", answer="ok")
LEVEL = Level(depth=0, model="", final=False)


def _canned(**extra: Any) -> dict[str, Any]:
    return {
        "d:0;react:turn": SEND,
        "d:0;tool:send_code": ToolResult(content="sent"),
        "d:1;react:turn": DONE,
        **extra,
    }


def _always(verdict: Verdict) -> Callable[..., Effect[Verdict]]:
    def guard(request: ToolRequest, messages: list[dict[str, Any]], level: Level):
        yield from ()
        return verdict

    return guard


def _ran(recorder: RecordingHandler) -> list[str]:
    return [entry.key.stored() for entry in recorder.trace]


def _observed(recorder: RecordingHandler, verdict: Verdict) -> str:
    """The first turn's observation, when the run is guarded by `verdict`."""
    match recorder.run(lambda: run_agent("let Sam in", guard=_always(verdict))):
        case Trajectory(steps=[Step(observation=str(observation)), *_]):
            return observation
        case other:
            raise AssertionError(f"no first observation in {other!r}")


def test_a_proceed_runs_the_action():
    recorder = RecordingHandler(_canned())
    assert _observed(recorder, Proceed()) == "sent"
    assert "d:0;step;tool:send_code" in _ran(recorder)


def test_a_refusal_is_the_observation_and_the_action_never_runs():
    recorder = RecordingHandler(_canned())
    observation = _observed(recorder, Refuse("not asked"))
    assert observation.startswith("[refused] send_code")
    assert "not asked" in observation
    assert not any("tool:send_code" in name for name in _ran(recorder))


@pytest.mark.parametrize("allow", [True, False])
def test_an_ask_rings_the_user_then_parks_and_their_answer_decides(allow: bool):
    recorder = RecordingHandler(
        _canned(
            **{
                "d:0;guard:asked": ToolResult(content="asked"),
                "d:0;guard:ask": Allowed(allow=allow),
            }
        )
    )
    observation = _observed(recorder, Ask("Sam?"))
    ran = _ran(recorder)
    asked = ran.index("d:0;step;guard:asked")
    assert ran[asked + 1] == "d:0;event;guard:ask"
    assert ("d:0;step;tool:send_code" in ran) is allow
    assert observation.startswith("sent" if allow else "[refused] send_code")


def test_the_guard_reads_the_transcript_that_chose_the_action():
    seen: list[list[dict[str, Any]]] = []

    def guard(request: ToolRequest, messages: list[dict[str, Any]], level: Level):
        seen.append(messages)
        yield from ()
        return Proceed()

    RecordingHandler(_canned()).run(lambda: run_agent("let Sam in", guard=guard))
    [messages] = seen
    assert messages[0] == {"role": "user", "content": "let Sam in"}
    assert messages[-1]["role"] == "assistant"
    assert "send_code" in messages[-1]["content"]


def _judged(named: float, window: float) -> dict[str, Any]:
    return {"NAMED": {"p": named}, "SAME_WINDOW": {"p": window}}


@pytest.mark.parametrize(
    ("named", "window", "verdict"),
    [
        (0.95, 0.95, Proceed),
        (0.05, 0.95, Refuse),
        (0.95, 0.10, Refuse),
        (0.95, 0.50, Ask),
        (0.50, 0.95, Ask),
    ],
)
def test_the_door_proceeds_refuses_or_asks_by_the_two_facts(named, window, verdict):
    messages = [{"role": "user", "content": "let Sam in for half an hour"}]
    recorder = RecordingHandler({"judge:door": _judged(named, window)})
    got = recorder.run(lambda: smol_door.door(SEND.tool, messages, LEVEL))
    assert type(got) is verdict


def test_the_door_judges_only_a_door_code():
    heat = ToolRequest(name="thermostat", args={"celsius": 22})
    recorder = RecordingHandler({})
    assert recorder.run(lambda: smol_door.door(heat, [], LEVEL)) == Proceed()
    assert recorder.trace == []


LINES = [
    "let Sam in for 30 minutes",
    "Alex is at the door",
    "let Pat in for a bit",
    "let Quinn in for a bit",
    "/quit",
]
FACTS = {"Sam": (0.95, 0.95), "Alex": (0.05, 0.95), "Pat": (0.95, 0.5), "Quinn": (0.95, 0.5)}
VISITORS = ["Sam", "Alex", "Pat", "Quinn"]
ALLOWED = {2: True, 3: False}
"""The resident's answer to each turn's ask, by turn: both ask at the same depth."""


class House:
    """The model sends a code to whoever the line names, then answers with what it saw; the judge
    answers each visitor's facts from `FACTS`. Every answer is a function of the op alone."""

    def __init__(self) -> None:
        self.spawn: Callable[[CallTool[Any]], Any] | None = None
        self.sent: list[str] = []
        self.judged: list[str] = []
        self.asked: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=[*_, {"role": "tool", "content": str(seen)}]):
                return AssistantTurn(thought="done", answer=seen)
            case AskLLM(messages=[*_, {"role": "user", "content": str(line)}]):
                [visitor] = [v for v in VISITORS if v in line]
                args = {"visitor": visitor, "minutes": 30}
                return AssistantTurn(thought="door", tool=ToolRequest(name="send_code", args=args))
            case Judge(state={"visitor": str(visitor)}):
                self.judged.append(visitor)
                return Answers.model_validate(_judged(*FACTS[visitor]))
            case CallTool(name=name, args={"question": str(question)}) if name == ASK_TOOL:
                self.asked.append(question)
                return ToolResult(content="asked")
            case CallTool(name="send_code", args={"visitor": str(visitor)}):
                self.sent.append(visitor)
                return ToolResult(content=f"sent {visitor} a code")
            case CallTool(name=name) if name == SPAWN_TOOL and self.spawn is not None:
                return self.spawn(op)
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def converse(backend, domain: House, fault: Fault) -> Any:
    """Run the house conversation to its end, answering each turn's ask from `ALLOWED`."""
    run_id = f"h{uuid4().hex}"
    name = compose_key(t"smol-house:{Run(run_id)}").stored()
    scope = compose_key(t"smol:{Run(run_id)}")

    def body(params: dict[str, Any], ctx: Any) -> Any:
        chain = Chain.from_params(params, task=name, schema=Conversation, initial=Conversation())
        handler = DurableHandler(FaultCtx(ctx, fault), domain, ledger=None, params=params)
        return handler.run(lambda: smol(chain, guard=smol_door.door))

    domain.spawn = spawn_tool(backend.spawner)
    backend.register_body(name, body)
    for n, line in enumerate(LINES):
        park = qualified_event_name(scope, name=compose_key(t"user:{Index(n)}").stored())
        backend.emit_event(uuid4(), park.stored(), {"text": line})
    for n, allow in ALLOWED.items():
        ask = qualified_event_name(
            scope, compose_key(t"turn:{Index(n)}"), compose_key(t"d:{Index(0)}"), name="guard:ask"
        )
        backend.emit_event(uuid4(), ask.stored(), {"allow": allow})
    backend.spawn(name, run_id)
    ran, snap = 0, None
    while ran < len(tasks := [task_id for task_id, _ in backend.enqueued(name)]):
        for task_id in tasks[ran:]:
            snap = backend.run_until_result(task_id)
        ran = len(tasks)
    return snap


def _tools(snap: Any) -> list[str]:
    return [m["content"] for m in snap.result["messages"] if m["role"] == "tool"]


def test_the_door_sends_refuses_and_asks_in_one_conversation(backend):
    domain = House()
    snap = converse(backend, domain, Fault())
    assert snap.state == "completed", snap
    assert domain.sent == ["Sam", "Pat"]
    assert domain.judged == VISITORS
    assert domain.asked == [
        "Send Pat a door code for 30 minutes?",
        "Send Quinn a door code for 30 minutes?",
    ]
    sam, alex, pat, quinn = _tools(snap)
    assert (sam, pat) == ("sent Sam a code", "sent Pat a code")
    assert "never asked" in alex
    assert "declined" in quinn


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_judged_conversation_survives_a_crash_at_every_op(backend, position):
    """A crash before an op re-runs nothing recorded: each code is sent, each visitor judged and
    the resident asked once, however many incarnations the conversation takes."""
    unarmed = Fault(position=position)
    golden = converse(backend, House(), unarmed)
    assert unarmed.count > 0
    for k, fault in at_every_op(unarmed):
        domain = House()
        snap = converse(backend, domain, fault)
        assert snap.state == "completed", (k, snap)
        assert _tools(snap) == _tools(golden), k
        if position is FaultPosition.BEFORE_OP:
            assert domain.sent == ["Sam", "Pat"], k
            assert domain.judged == VISITORS, k
            assert len(domain.asked) == 2, k
