"""Domain operations — the inner payload of a durable Step.

These name the business-relevant actions whose *meaning* varies by handler:
production calls the real model/tool; recording returns canned values; replay
returns recorded values. They are inert data; they perform no I/O.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel

from effective.keys import Key


class AsksModel:
    """The class pattern for `ModelCall`: its subclasses are exactly that union's arms, each
    final, so `case AsksModel():` is the union."""


@final
@dataclass(frozen=True)
class AskLLM[T](AsksModel):
    messages: Any  # phase 0: opaque; phase 1+: list[Message] / PEP 750 Template
    response_schema: type[T]


@final
@dataclass(frozen=True)
class CallTool[T]:
    name: str
    result_schema: type[T]  # required, like AskLLM.response_schema (no `object` sentinel)
    args: dict[str, Any] = field(default_factory=dict)


class WireChoice(BaseModel):
    """One of a set: each label maps to a description, or to `None` when the label says it all."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]


class WireNoul(BaseModel):
    """The probability that a condition holds, with optional descriptions of either side."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[Literal["true", "false"], str]


class WireScore(BaseModel):
    """A position on ordered levels, lowest first."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


type WireQuestion = Annotated[WireChoice | WireNoul | WireScore, Field(discriminator="type")]


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p: float


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    confidence: float
    probabilities: dict[str, float]


class Answers(RootModel[dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]]):
    """A judgment's answers, keyed by question name. Each shape has a required field the others
    lack and admits no other, so an answer validates as exactly one of them."""


@final
@dataclass(frozen=True)
class Judge[T](AsksModel):
    """Bounded, typed questions over one state, asked in one call.

    The state and the questions travel apart, so no state value is read as an instruction. Any
    interpreter that answers this wire can serve it: Jev, a scripted table in a test, a recorded
    tape. It returns one `ChoiceAnswer`, `NoulAnswer` or `ScoreAnswer` per question."""

    state: dict[str, Any]
    questions: dict[str, WireQuestion]
    response_schema: type[T]


type ModelCall[T] = AskLLM[T] | Judge[T]
type DomainOp[T] = ModelCall[T] | CallTool[T]


# --- the spawn tool's wire protocol: a well-known `CallTool` name + its result schema ----
# Two callers yield this same `CallTool` (`agent.compose.spawn_subagent_task`, the N=1
# subagent; `effective.fork.spawn_fork`, the counterfactual child) and one handler answers
# it (`agent.runtime.spawn_tool`), so the pair is shared protocol, not any one of their
# implementations. It lives with `CallTool` because that is exactly what it types.

SPAWN_TOOL = "spawn"

ASK_TOOL = "ask"
"""The reserved tool name a guard's question reaches the user through, before the run parks."""

INTERRUPT_TOOL = "interrupt"
"""The reserved tool name of `effective.interrupts.tool_interrupt`'s non-blocking poll.

What keeps it from colliding with an author's tool named `interrupt` is arity: the poll keys
as `tool:interrupt,{phase}` and an author's tool as `tool:{name}`, as `SPAWN_TOOL` does."""


class ToolRefused(BaseModel):
    """A tool ran and said NO — as a recorded VALUE, not an exception.

    **The forcing case is the edit ladder's designed mainline.** If `coding.runners`' static gate
    refused a bad edit by raising, a handler-side exception would never cross the yield boundary
    into the workflow, so the op would never complete: measured on SQLite, a syntactically-invalid
    edit raising `ToolError('lint [syntax] invalid syntax')` produced **three identical attempts**
    as the engine retried a deterministic refusal, and **zero ledger rows**. The run died on
    exactly the answer the ladder exists to produce.

    A value fixes that where widening the exception-delivery set would not. The op COMPLETES, the
    refusal is checkpointed, and a replay re-serves it with nothing to re-derive, so the set of
    exceptions a child answers with stays `fork.REFUSALS`.

    **`refused: Literal[True]` is a discriminator, not decoration.** A tool's result schema
    becomes `ToolRefused | R`, and pydantic resolves a union left-to-right: without a
    discriminator this model's own dump is a perfectly good `dict[str, str]`, so a refusal
    round-tripped back as a plain dict and a workspace tree was indistinguishable from a refusal.
    Measured. The literal fixes it in both directions — a tree whose FILES are named `refused`,
    `tool` and `diagnostic` still reads as a tree, because its values are strings and this field
    demands the boolean. It has no default, so a record that does not say it refused, such as a
    predicate's `{tool, diagnostic}`, is never read as a refusal.

    Not a `CompositionRefused`: composing this call was fine. The tool was asked for something it
    cannot do, which is a runtime answer about the world and is what a judge should get to see."""

    refused: Literal[True]
    tool: str
    diagnostic: str


class SpawnResult(BaseModel):
    """What the `spawn` tool returns to the parent: the child task's id.

    **A `UUID`: this is the one place a task-identity type is declared.** The `TaskContext`
    Protocol names `step`/`await_event`/`sleep_until` and has no `spawn`, so each engine's `spawn`
    is an independent API and this model is where the two meet. Both produce a `UUID`: Absurd's
    SDK returns `row["task_id"]` off a `uuid` column (its own `SpawnResult` TypedDict annotates
    `str` and validates nothing), and `SqliteApp.spawn` mints `uuid7`.

    As a Pydantic model the declaration is enforced: passing a `str` raises here. It also gives
    the checkpoint round-trip: `to_jsonable_python` writes the canonical text and the op's
    `TypeAdapter` parses it back, so the parent replays a `UUID`."""

    task_id: UUID


class Spawned(BaseModel):
    """What a workflow's spawn returns: the child's task id, and the event its answer arrives on.

    The handler names the event from the spawn's own task and placement, so the name is fixed by
    where the workflow placed the spawn, and a replay re-binds it from the spawn's checkpoint."""

    task_id: UUID
    done_event: Key


class SpawnArgs(BaseModel):
    """What a spawn asks for: the task to enqueue, the params it starts with, and the queue.

    `idempotency_key` names the spawn to the engine, so a crash between the enqueue and its
    checkpoint enqueues one child. The handler sets it from the task that spawns and where the
    spawn was placed, and refuses a spawn that arrives carrying one."""

    model_config = ConfigDict(extra="forbid")

    task_name: str
    params: dict[str, Any]
    queue: str = "default"
    idempotency_key: str | None = None
    max_attempts: int | None = Field(default=None, ge=1)
    """How many executions the child may have; `None` leaves it to the engine."""

    def call(self) -> CallTool[Spawned]:
        return CallTool(
            name=SPAWN_TOOL, result_schema=Spawned, args=self.model_dump(exclude_none=True)
        )
