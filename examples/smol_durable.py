"""smol_durable: `examples/smol_agent.py` as a house agent, durable, with Esc and a judged door.

Run it:  uv run python examples/smol_durable.py [--db build/smol.db] [--async]

`--async` answers the model and the judge with asyncio interpreters; the workflow and the handler
are the same either way.

Type a line and the agent works on it; press Esc while it works and the turn ends, the running
command or model call stopped where it was. An empty line or `/quit` ends the conversation, and
Ctrl-D leaves it. Kill the process at any point and run it again: the conversation resumes at the
line it was waiting for, and nothing it recorded runs twice.

The house has a thermostat and a door. Before a door code is sent, a judge reads what the
resident typed and answers two facts: the visitor is named, and the window is what was asked.

| the judge finds                  | the code                                        |
|----------------------------------|-------------------------------------------------|
| both facts hold                  | is sent                                         |
| either fact fails                | is refused, and the model reads why             |
| a fact it cannot call either way | waits: the phone asks the resident, who decides |

| set                | serves                                                                |
|--------------------|-----------------------------------------------------------------------|
| `OPENAI_API_KEY`   | the model, `gpt-5.6-luna`; without it an offline model parses the line |
| `JEV_API_KEY`      | the judge, Jev, capped at 200k tokens; without it an offline judge     |

The offline model understands `warm it to 22`, `let Sam in for half an hour`, `Alex is at the
door` (which it answers by sending Alex a code, for the judge to refuse), and `!cmd`, a shell
command, so `!sleep 30` then Esc shows a cancel.
"""

import argparse
import os
import re
import secrets
import select
import sys
import termios
import threading
import tty
from pathlib import Path
from typing import Any

from smol_door import door

from agent.runtime import spawn_tool
from effective.cancel import Cancelled, CancelToken
from effective.combinators import Chain
from effective.cost import MeteredInterpreter, Usage
from effective.domain import (
    ASK_TOOL,
    INTERRUPT_TOOL,
    SPAWN_TOOL,
    Answers,
    AskLLM,
    CallTool,
    Judge,
)
from effective.handlers.absurd import DurableHandler
from effective.interpreters.shell import Shelled, run_shell
from effective.interrupts import EVERY_PHASE, Interrupted, tool_interrupt
from effective.keys.frame import split_frames
from effective.ops import CARRY_PARAM
from effective.parked import read_sqlite_parked_conn
from effective.react import (
    AssistantTurn,
    ToolRequest,
    ToolResult,
)
from effective.smol import Conversation, smol
from effective.sqlite import SqliteApp

TASK = "smol-chat"
ESC = b"\x1b"
JEV_CAP = 200_000


def _function(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


TOOLS = [
    _function("thermostat", "Set the house's temperature.", {"celsius": {"type": "number"}}),
    _function(
        "send_code",
        "Text a visitor a door code valid for some minutes.",
        {"minutes": {"type": "integer"}, "visitor": {"type": "string"}},
    ),
    _function("sh", "Run a shell command and read its output.", {"cmd": {"type": "string"}}),
]

_WARM = re.compile(r"warm.*?(\d+)")
_LET_IN = re.compile(r"let (\w+)\b.*? in(?: for (\d+) minutes| for (half an hour))?")
_AT_DOOR = re.compile(r"(\w+) is at the door")


def offline(op: AskLLM[Any]) -> tuple[AssistantTurn, Usage]:
    """Parses the resident's line into one action, and answers with the observation after it."""
    last = op.messages[-1]
    if last["role"] == "tool":
        return AssistantTurn(thought="done", answer=last["content"]), Usage()
    line = last["content"]
    if line.startswith("!"):
        return _acting("sh", cmd=line[1:])
    if warm := _WARM.search(line):
        return _acting("thermostat", celsius=float(warm[1]))
    if let_in := _LET_IN.search(line):
        minutes = 30 if let_in[3] else int(let_in[2] or 60)
        return _acting("send_code", visitor=let_in[1], minutes=minutes)
    if at_door := _AT_DOOR.search(line):  # an eager model lets in whoever turns up
        return _acting("send_code", visitor=at_door[1], minutes=60)
    return AssistantTurn(thought="echo", answer=line), Usage()


def _acting(tool: str, **args: Any) -> tuple[AssistantTurn, Usage]:
    return AssistantTurn(thought=tool, tool=ToolRequest(name=tool, args=args)), Usage()


def offline_judge(op: Judge[Any]) -> tuple[Answers, Usage]:
    """Reads the words: "let <visitor> in", and the minutes spelled out. "a bit" is a tie."""
    words = " ".join(op.state["resident"]).lower()
    named = re.search(r"let " + re.escape(str(op.state["visitor"]).lower()) + r"\b.*? in", words)
    minutes = str(op.state["minutes"])
    spelled = minutes in words or (minutes == "30" and "half an hour" in words)
    window = 0.5 if "a bit" in words else 0.95 if spelled else 0.1
    answers = {"NAMED": {"p": 0.95 if named else 0.05}, "SAME_WINDOW": {"p": window}}
    return Answers.model_validate(answers), Usage()


def model(token: CancelToken, colored: bool) -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        return offline
    from openai import AsyncOpenAI, OpenAI

    from effective.interpreters.openai import (
        GPT56_LUNA,
        AsyncResponsesTurnCaller,
        ResponsesTurnCaller,
    )

    kind, client = (
        (AsyncResponsesTurnCaller, AsyncOpenAI) if colored else (ResponsesTurnCaller, OpenAI)
    )
    return kind(
        client=client(),
        system_prompt=(
            "You run a house for its resident. Act with one tool at a time; answer tersely "
            "when done. A refused action is final: say so rather than retry it."
        ),
        model="gpt-5.6-luna",
        price=GPT56_LUNA,
        tools=TOOLS,
        cancel=token,
    )


def judge_caller(db: Path, colored: bool) -> Any:
    key = os.environ.get("JEV_API_KEY")
    if not key:
        return offline_judge
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

    from effective.interpreters.jev import AsyncJev, Jev
    from effective.spend import TokenBudget

    spend, kept = TokenBudget(db.with_suffix(".jev-spend"), JEV_CAP), db.with_suffix(".jev")
    if colored:
        return AsyncJev(AsyncTypeSafeClient(api_key=key), spend, cache=kept)
    return Jev(TypeSafeClient(api_key=key), spend, cache=kept)


HOUSE = frozenset({ASK_TOOL, "thermostat", "send_code"})


def house(op: CallTool[Any]) -> ToolResult:
    """The devices: each prints what it did, which is how a replay shows it did nothing."""
    match op.name, op.args:
        case "thermostat", {"celsius": celsius}:
            print("[thermostat]", celsius, "C", flush=True)
            return ToolResult(content="set to " + str(celsius) + " C")
        case "send_code", {"visitor": visitor, "minutes": minutes}:
            code = str(secrets.randbelow(10_000)).zfill(4)
            print("[sms]", visitor, "code", code, "for", minutes, "minutes", flush=True)
            return ToolResult(content="sent " + str(visitor) + " a door code")
        case _, {"question": question}:
            print("[phone]", question, flush=True)
            return ToolResult(content="asked")
        case name, _:
            return ToolResult(content="bad arguments for " + name)


def serve(app: SqliteApp, token: CancelToken) -> Any:
    """The house, the shell, the interrupt poll read off the token, and the spawn a new generation
    of the conversation is enqueued through."""

    def enqueue(
        task_name: str,
        params: dict[str, Any],
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        return str(
            app.spawn(
                task_name, params, idempotency_key=idempotency_key, max_attempts=max_attempts
            )
        )

    spawn = spawn_tool(enqueue)

    def tools(op: CallTool[Any]) -> Any:
        match op.name:
            case name if name == INTERRUPT_TOOL:
                return Interrupted(escape=token.cancelled)
            case name if name == SPAWN_TOOL:
                return spawn(op)
            case name if name in HOUSE:
                return house(op)
            case "sh":
                match run_shell(op.args["cmd"], cancel=token, timeout_s=120):
                    case Shelled(exit_code=code, output=output):
                        return ToolResult(content=f"exit {code}\n{output}")
                    case Cancelled() as cancelled:
                        return cancelled
            case other:
                return ToolResult(content=f"no tool named {other}")

    return tools


def _keypress() -> bytes:
    """One key's bytes, read from the descriptor: an arrow key sends its escape sequence at once,
    and a buffered text read would take the whole sequence and leave `select` nothing to see."""
    chunk = os.read(sys.stdin.fileno(), 32)
    while select.select([sys.stdin], [], [], 0.03)[0]:  # a sequence that arrived split
        chunk += os.read(sys.stdin.fileno(), 32)
    return chunk


def drain_watching_esc(app: SqliteApp, token: CancelToken) -> None:
    """Run the engine until nothing is claimable, cancelling on Esc meanwhile."""

    def drain() -> None:
        while app.work_batch():
            pass

    worker = threading.Thread(target=drain)
    worker.start()
    if not sys.stdin.isatty():
        worker.join()
        return
    saved = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin)
        while worker.is_alive():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready and _keypress() == ESC and not token.cancelled:
                token.cancel()
                print("[esc]", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)
        worker.join()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="build/smol.db")
    parser.add_argument(
        "--async", dest="colored", action="store_true", help="asyncio interpreters"
    )
    args = parser.parse_args()
    db, colored = Path(args.db), args.colored
    db.parent.mkdir(parents=True, exist_ok=True)
    app, token = SqliteApp(str(db)), CancelToken()
    domain = MeteredInterpreter(
        llm=model(token, colored), tools=serve(app, token), judge=judge_caller(db, colored)
    )

    @app.register_task(TASK)
    def conversation(params: dict[str, Any], ctx: Any) -> Any:
        chain = Chain.from_params(params, task=TASK, schema=Conversation, initial=Conversation())
        handler = DurableHandler(ctx, domain, ledger=None, params=params)
        return handler.run(lambda: smol(chain, guard=door, interrupt=tool_interrupt(EVERY_PHASE)))

    if not app.conn.execute("SELECT 1 FROM tasks WHERE name = ?", (TASK,)).fetchone():
        app.spawn(TASK, {"run_id": "smol"})
    shown = 0
    while True:
        drain_watching_esc(app, token)
        parked = [p for p in read_sqlite_parked_conn(app.conn) if p.task_name == TASK]
        if not parked:
            print("conversation over")
            return
        messages = parked[-1].params.get(CARRY_PARAM, {}).get("messages", [])
        for message in messages[shown:]:
            if message["role"] == "assistant":
                print(message["content"])
        shown = len(messages)
        wake = parked[-1].wake_event
        asking = split_frames(wake)[1] == "guard:ask"
        try:
            line = input("allow? [y/N] " if asking else "> ")
        except EOFError:  # leave; the conversation stays parked for the next run
            print()
            return
        token.reset()
        app.emit_event(wake, {"allow": line.strip() == "y"} if asking else {"text": line})


if __name__ == "__main__":
    main()
