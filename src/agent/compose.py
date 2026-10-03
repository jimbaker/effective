"""Composing an agent loop — workflow-side `act` builders (HITL, …).

The loop (`effective.react`) is a durable driver; *what an action does* is the `act`
seam. This module builds `act`s that compose with algebraic effects:

- **HITL is a tool that suspends.** `make_act` routes a designated "ask" tool to an
  `await_event` instead of a `call_tool` — so the loop *parks durably* until a human
  answers, then threads the answer back as the next observation. This is the same
  suspend/resume mechanism the permission `human` tier uses (proven to survive
  crash/replay on the durable path), lifted to the agent boundary: no new machinery,
  just a different op yielded for one tool name.

The handler-side compositions (a subagent as an opaque tool, the local ctxs) live
in `agent.runtime`; the two sides meet at the `CallTool` / `AwaitEvent` ops.
"""

import json
from collections.abc import Mapping, Sequence, Set
from typing import Any

from pydantic import BaseModel, ValidationError

from effective import Effect, await_event, call_tool, compose_key, scoped
from effective.code import canonical, run_code
from effective.combinators import Level
from effective.interrupts import Interrupted as Interrupted
from effective.interrupts import tool_interrupt as tool_interrupt
from effective.keys import Index, Name
from effective.react import (
    Act,
    Decide,
    Interrupt,
    ToolRequest,
    ToolResult,
    bad_arguments,
    default_act,
    run_agent,
)
from effective.spawning import join_child, spawn_child

ASK_HUMAN = "ask_human"


class HumanAnswer(BaseModel):
    """The reviewer's reply to an Ask — delivered as the awaited event's value."""

    text: str


def make_act(ask_tools: Set[str] = frozenset({ASK_HUMAN})) -> Act:
    """An `act` that suspends on an Ask tool and dispatches everything else normally.

    A request whose name is in `ask_tools` yields `await_event("ask:<turn>", …)` — the
    loop parks (the handler returns `Suspended`; on the durable path the task suspends)
    until the named answer is delivered, then the answer's text becomes the observation.
    Every other request is an opaque `call_tool` (a plain tool *or* a subagent).
    """

    def act(request: ToolRequest, level: Level) -> Effect[ToolResult]:
        if request.name in ask_tools:
            turn = level.depth
            answer = yield from await_event(compose_key(t"ask:{Index(turn)}"), HumanAnswer)
            return ToolResult(content=answer.text)
        result = yield from call_tool(request.name, request.args, ToolResult)
        return result

    return act


CODE_TOOL = "run_code"
"""The tool name a code-emitting ``decide`` uses to request a sandboxed run."""


class CodeRequest(BaseModel):
    """The arguments a code run needs; `inputs` rides beside it untyped."""

    code: str


def code_act(
    *,
    inputs: Mapping[str, Any] | None = None,
    functions: Sequence[str] = (),
    actions: Mapping[str, type[Any]] | None = None,
    code_tool: str = CODE_TOOL,
    fallback: Act | None = None,
) -> Act:
    """An ``act`` that runs a code-emitting turn's code via ``run_code``, as in
    ``run_agent(decide=code_emitting, act=code_act(...))``.

    A request named ``code_tool`` carries a ``code`` string arg, or becomes a bad-arguments
    observation, and may carry a JSON ``inputs`` dict. It runs as ``run_code("act", ...)`` inside
    the turn's ``d:{i}`` frame, which keeps each turn's segment keys apart, so no caller threads a
    scope by hand. ``inputs`` are the *harness-supplied* typed values every code run receives
    (whole values, never stringly previews), merged over any model-supplied request ``inputs``;
    the harness wins. Any other request falls through to ``fallback`` (default: ``default_act``,
    the plain tool dispatch). The code's validated output returns to the loop as canonical JSON
    text, the observation the next turn reads.

    Namespacing is the caller's, applied structurally by wrapping:
    ``scoped(compose_key(t"sub:{Name(name)}"), lambda: run_agent(..., act=code_act(...)))``.
    """
    dispatch = fallback if fallback is not None else default_act
    fixed = dict(inputs or {})

    def act(request: ToolRequest, level: Level) -> Effect[ToolResult]:
        if request.name != code_tool:
            result = yield from dispatch(request, level)
            return result
        try:
            code = CodeRequest.model_validate(request.args).code
        except ValidationError as malformed:
            return bad_arguments(code_tool, malformed)
        merged = {**(request.args.get("inputs") or {}), **fixed}
        out = yield from run_code(
            compose_key(t"act").stored(),
            code,
            schema=object,
            inputs=merged,
            functions=functions,
            actions=actions,
        )
        return ToolResult(content=json.dumps(canonical(out)))

    return act


def spawn_subagent(
    name: str,
    task: str,
    *,
    decide: Decide | None = None,  # a child Decide; None -> the scoped default (ask_llm)
    act: Act | None = None,
    interrupt: Interrupt | None = None,
    max_iters: int = 4,
) -> Effect[ToolResult]:
    """Run a nested `run_agent` **inline** under the parent's handler, namespaced by
    `name`: the *durable* subagent, complementing `runtime.subagent_runner`.

    Because the child's ops flow through the parent's handler/ctx (inside
    `scoped(compose_key(t"sub:{Name(name)}"))`),
    its steps are checkpointed there: a parent crash *resumes* the child from its last
    checkpoint rather than restarting it (proven in test_durable_subagent). The trade-off
    vs the opaque `subagent_runner`: **context isolation is preserved** (the child reasons
    over its own messages, never the parent's), but **trace-opacity is given up** (the
    child's ops appear in the parent stream) — so the child's cost lands naturally on the
    parent meter, no telemetry-peek needed. The fully-opaque-*and*-durable variant is a
    spawned child Absurd task (a true fork); that needs Absurd spawn primitives.
    """
    traj = yield from scoped(
        compose_key(t"sub:{Name(name)}"),
        lambda: run_agent(task, max_iters=max_iters, decide=decide, act=act, interrupt=interrupt),
    )
    return ToolResult(content=traj.answer)


# --- the spawned subagent: a child Absurd *task* (opaque AND durable) ------------------
# `SPAWN_TOOL`/`SpawnResult` (the tool's wire protocol, shared with `effective.fork`) live
# in `effective.domain` beside the `CallTool` they type; `ChildResult` is this caller's own
# done-event payload (a fork child answers on `effective.spawning.ChildAnswer`).


class ChildResult(BaseModel):
    """The child's completion event payload (its final answer), awaited by the parent."""

    answer: str


def spawn_subagent_task(
    child_task: str,
    task: str,
    *,
    correlation: str,
    queue: str = "default",
) -> Effect[ToolResult]:
    """Spawn `child_task` as its **own durable Absurd task** and await its completion.

    The third corner of the subagent design (`runtime.subagent_runner` is opaque-but-
    restart; `spawn_subagent` is durable-but-visible): this is **opaque AND durable**. The
    parent's trace is just a `spawn` `CallTool` (checkpointed → idempotent on replay,
    so a parent crash never re-spawns) followed by an `AwaitEvent` on the child's
    completion (which *suspends*, releasing the worker — deadlock-free on one queue). The
    child runs as a separate task with its **own** checkpoints, so a parent crash never
    restarts it and a child crash never restarts the parent. The true fork point.

    `correlation` names the spawn op, and the handler names the done event from where that op is
    placed, so replay re-binds both. The handler-side `runtime.spawn_tool` performs the spawn.

    The handler holds the spawn to this task's depth, refuses one past it with `Refused`, and
    gives the child one level less.
    """
    spawned = yield from spawn_child(child_task, correlation, {"task": task}, queue=queue)
    child = yield from join_child(spawned, ChildResult)
    return ToolResult(content=child.answer)
