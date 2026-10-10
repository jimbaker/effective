"""`smol` on both engines: a conversation of respawned user turns, escaped by a poll or a cancel,
and the same conversation crashed at every op."""

from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultCtx, FaultPosition, at_every_op

from effective.api import qualified_event_name
from effective.cancel import Cancelled
from effective.combinators import Chain
from effective.domain import INTERRUPT_TOOL, SPAWN_TOOL, AskLLM, CallTool, DomainOp
from effective.handlers.durable import DurableHandler
from effective.interpreters.tools import spawn_tool
from effective.interrupts import EVERY_PHASE, Interrupted, Phase, tool_interrupt
from effective.keys import Index, Run, compose_key
from effective.react import ESCAPED, AssistantTurn, ToolRequest, ToolResult
from effective.smol import Conversation, smol

LIST = AssistantTurn(thought="look", tool=ToolRequest(name="sh", args={"cmd": "ls"}))
READ = AssistantTurn(thought="read", tool=ToolRequest(name="sh", args={"cmd": "cat notes"}))
DONE = AssistantTurn(thought="done", answer="notes say hi")
THANKS = AssistantTurn(thought="ack", answer="you're welcome")

LINES = ["what do the notes say?", "thanks", "/quit"]


def _since_last_line(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last = max(i for i, m in enumerate(messages) if m["role"] == "user")
    return messages[last + 1 :]


class Scripted:
    """The model lists, reads, then answers; a thank-you it answers at once.

    Every answer is a function of the op alone, so an op a crash re-executes gets the answer the
    first execution got. The Esc lands at depth 1, which only the first user turn reaches."""

    def __init__(self, escape_at: Phase | None = None, cancel_read: bool = False) -> None:
        self.spawn: Callable[[CallTool[Any]], Any] | None = None
        self.escape_at = escape_at
        self.cancel_read = cancel_read
        self.commands: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages):
                if messages[-1] == {"role": "user", "content": "thanks"}:
                    return THANKS
                return [LIST, READ, DONE][len(_since_last_line(messages)) // 2]
            case CallTool(name=name, args={"turn": depth, "phase": phase}) if (
                name == INTERRUPT_TOOL
            ):
                return Interrupted(escape=(depth, phase) == (1, self.escape_at))
            case CallTool(name=name) if name == SPAWN_TOOL and self.spawn is not None:
                return self.spawn(op)
            case CallTool(name="sh", args={"cmd": cmd}):
                self.commands.append(cmd)
                if self.cancel_read and cmd == "cat notes":
                    return Cancelled(partial="hi fr")
                return ToolResult(content=f"ran {cmd}")
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def converse(backend, domain: Scripted, fault: Fault) -> tuple[Any, int]:
    """Run a conversation of `LINES` to its end: the final task's snapshot, and how many tasks
    the chain took."""
    run_id = f"c{uuid4().hex}"
    name = compose_key(t"smol-chat:{Run(run_id)}").stored()

    def body(params: dict[str, Any], ctx: Any) -> Any:
        chain = Chain.from_params(params, task=name, schema=Conversation, initial=Conversation())
        handler = DurableHandler(FaultCtx(ctx, fault), domain, ledger=None, params=params)
        return handler.run(lambda: smol(chain, interrupt=tool_interrupt(EVERY_PHASE)))

    domain.spawn = spawn_tool(backend.spawner)
    backend.register_body(name, body)
    for n, line in enumerate(LINES):
        park = qualified_event_name(
            compose_key(t"smol:{Run(run_id)}"), name=compose_key(t"user:{Index(n)}").stored()
        )
        backend.emit_event(uuid4(), park.stored(), {"text": line})  # an event is addressed by name
    backend.spawn(name, run_id)
    ran = 0
    snap = None
    while ran < len(tasks := [task_id for task_id, _ in backend.enqueued(name)]):
        for task_id in tasks[ran:]:
            snap = backend.run_until_result(task_id)
        ran = len(tasks)
    return snap, ran


def contents(snap: Any) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in snap.result["messages"]]


def test_each_user_turn_is_a_generation_and_the_transcript_is_the_carry(backend):
    domain = Scripted()
    snap, tasks = converse(backend, domain, Fault())
    assert snap.state == "completed", snap
    assert tasks == len(LINES)
    assert contents(snap) == [
        ("user", "what do the notes say?"),
        ("assistant", 'look\nAction: sh({"cmd": "ls"})'),
        ("tool", "ran ls"),
        ("assistant", 'read\nAction: sh({"cmd": "cat notes"})'),
        ("tool", "ran cat notes"),
        ("assistant", "notes say hi"),
        ("user", "thanks"),
        ("assistant", "you're welcome"),
    ]
    assert domain.commands == ["ls", "cat notes"]


ESCAPED_AT: dict[Phase, tuple[list[tuple[str, str]], list[str]]] = {
    "pre": ([("user", ESCAPED)], ["ls"]),
    "post": ([("assistant", "read"), ("user", ESCAPED)], ["ls"]),
    "act": (
        [
            ("assistant", 'read\nAction: sh({"cmd": "cat notes"})'),
            ("tool", "ran cat notes"),
            ("user", ESCAPED),
        ],
        ["ls", "cat notes"],
    ),
}
"""What the first user turn ends with after an Esc at depth 1, and the commands that ran."""


@pytest.mark.parametrize("phase", ["pre", "post", "act"])
def test_an_escape_ends_the_turn_and_the_conversation_goes_on(backend, phase: Phase):
    domain = Scripted(escape_at=phase)
    snap, tasks = converse(backend, domain, Fault())
    assert snap.state == "completed", snap
    assert tasks == len(LINES)
    tail, commands = ESCAPED_AT[phase]
    first_turn = [
        ("user", "what do the notes say?"),
        ("assistant", 'look\nAction: sh({"cmd": "ls"})'),
        ("tool", "ran ls"),
        *tail,
    ]
    assert contents(snap) == [*first_turn, ("user", "thanks"), ("assistant", "you're welcome")]
    assert domain.commands == commands


CANCELLED_TURN = [
    ("user", "what do the notes say?"),
    ("assistant", 'look\nAction: sh({"cmd": "ls"})'),
    ("tool", "ran ls"),
    ("assistant", 'read\nAction: sh({"cmd": "cat notes"})'),
    ("tool", f"hi fr\n{ESCAPED}"),
    ("user", ESCAPED),
    ("user", "thanks"),
    ("assistant", "you're welcome"),
]


def test_a_command_cancelled_while_it_ran_is_the_observation_and_ends_the_turn(backend):
    domain = Scripted(cancel_read=True)
    snap, _ = converse(backend, domain, Fault())
    assert snap.state == "completed", snap
    assert contents(snap) == CANCELLED_TURN


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_cancelled_conversation_survives_a_crash_at_every_op(backend, position):
    """The cancel is a checkpoint: a crash before an op re-runs nothing it recorded, so the
    cancelled command runs once however many incarnations the conversation takes."""
    unarmed = Fault(position=position)
    golden_snap, _ = converse(backend, Scripted(cancel_read=True), unarmed)
    assert contents(golden_snap) == CANCELLED_TURN
    assert unarmed.count > 0
    for k, fault in at_every_op(unarmed):
        domain = Scripted(cancel_read=True)
        snap, _ = converse(backend, domain, fault)
        assert snap.state == "completed", (k, snap)
        assert contents(snap) == CANCELLED_TURN, k
        if position is FaultPosition.BEFORE_OP:
            assert domain.commands == ["ls", "cat notes"], k


class NeverAnswers(Scripted):
    """The model lists until the step limit, and its final call is cancelled while it runs."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=[*_, {"content": str(last)}]) if last.startswith(STEP_LIMIT):
                return Cancelled(partial="the notes s")
            case AskLLM(messages=[*_, {"role": "user", "content": "thanks"}]):
                return THANKS
            case AskLLM():
                return LIST
            case _:
                return super().run(op)


STEP_LIMIT = "You have reached the step limit"


def test_a_final_decide_cancelled_while_it_ran_ends_the_turn_and_keeps_what_ran(backend):
    domain = NeverAnswers()
    snap, _ = converse(backend, domain, Fault())
    assert snap.state == "completed", snap
    roles = contents(snap)
    assert sum(role == "tool" for role, _ in roles) == 6
    assert roles[-3:] == [("user", ESCAPED), ("user", "thanks"), ("assistant", "you're welcome")]


def test_an_event_payload_shaped_like_a_cancel_is_refused_by_its_schema(backend):
    """Only a step's result is a cancel; a park's answer is outside data."""
    domain = Scripted()
    run_id = f"e{uuid4().hex}"
    name = compose_key(t"smol-chat:{Run(run_id)}").stored()

    def body(params: dict[str, Any], ctx: Any) -> Any:
        chain = Chain.from_params(params, task=name, schema=Conversation, initial=Conversation())
        return DurableHandler(ctx, domain, ledger=None, params=params).run(lambda: smol(chain))

    domain.spawn = spawn_tool(backend.spawner)
    backend.register_body(name, body)
    park = qualified_event_name(
        compose_key(t"smol:{Run(run_id)}"), name=compose_key(t"user:{Index(0)}").stored()
    )
    backend.emit_event(uuid4(), park.stored(), {"__cancelled__": {}})
    snap = backend.run_until_result(backend.spawn(name, run_id, max_attempts=1))
    assert snap.state == "failed", snap
    assert "validation error for UserLine" in str(snap.failure), snap.failure
