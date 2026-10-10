"""Spawn and join: a child task as its own durable run, answered on the event its spawn names.

A parent spawns and joins in two steps, so a fan-out spawns every child before it waits on any.
The children run concurrently in their own tasks, and the joins are a loop: a join inside a gather
branch is refused, since the branch would rescope an await its child cannot see.

    handles = []
    for i, params in enumerate(children):
        handles.append((yield from spawn_child(task, f"c{i}", params)))
    answers = []
    for handle in handles:
        answers.append((yield from join_child(handle, Answer)))

The child's side is `answer_parent`, called by its task body once it has an answer, or
`run_child`, which also reports a refusal the child stopped at or an error it failed of, so its
parent never waits on a child that cannot answer. `join_answer` is the parent's side of that pair:
it returns the value, or raises `ChildRefused` or `ChildFailed` for the parent to catch or let
climb."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, assert_never

from pydantic import BaseModel, Field

from effective.api import Effect, await_event, step
from effective.domain import SpawnArgs, Spawned
from effective.govern import REFUSALS, ChildRefused
from effective.handlers.base import Continued, Finished
from effective.keys import Key, Run, compose_key
from effective.ops import (
    DONE_EVENT_PARAM,
    Addressing,
    Unretryable,
    leaves,
    unretryable,
)


def spawn_child(
    task_name: str,
    name: str,
    params: Mapping[str, Any],
    *,
    queue: str = "default",
    max_attempts: int | None = None,
) -> Effect[Spawned]:
    """Enqueue `task_name` as its own durable task, once, with up to `max_attempts` executions
    (`None` leaves the limit to the engine).

    The spawn is a checkpointed step placed by `name`, so a replay returns the recorded child. The
    handler names the child's done event from that placement, and refuses the spawn with `Refused`
    once this task's depth is spent."""
    return (
        yield from step(
            # `spawn` is a static discriminator: a key-registry scan reads the template, and an
            # interpolated one would be a hole that no other arity-2 `tool:` variant is disjoint
            # from. `SPAWN_TOOL` is what a domain dispatches on.
            compose_key(t"tool:spawn,{Run(name)}").stored(),
            SpawnArgs(
                task_name=task_name, params=dict(params), queue=queue, max_attempts=max_attempts
            ).call(),
        )
    )


def join_child[T](spawned: Spawned, schema: type[T]) -> Effect[T]:
    """Wait for a spawned child's answer; the parent suspends and releases its worker meanwhile.

    The await is absolute: the child emits the name from its own params and has never seen a frame
    of this task's."""
    return (yield from await_event(spawned.done_event, schema, addressing=Addressing.ABSOLUTE))


def deliver(ctx: Any, event: Key, payload: Any) -> None:
    """Emit `payload` on `event`, the one call every engine answers.

    `emit_event` is an optional capability rather than part of the `TaskContext` protocol, as
    `repark` is, so the `Any` declares that looseness here once."""
    ctx.emit_event(event, payload)


def answer_parent(ctx: Any, params: Mapping[str, Any], payload: Any) -> None:
    """Answer the parent that spawned this task, on the done event its spawn named.

    A checkpointed step, so a retried child answers at most once. `payload` is plain JSON: the
    parent's `join_child` validates it back into its schema."""
    done_event = Key.parse(str(params[DONE_EVENT_PARAM]))
    ctx.step(
        compose_key(t"emit;{done_event:domain=address}"),
        lambda: deliver(ctx, done_event, payload) or {"emitted": True},
    )


class Returned(BaseModel):
    """A child's value, as plain JSON."""

    kind: Literal["returned"] = "returned"
    value: Any


class Refusal(BaseModel):
    """The refusals a child stopped at, each as its exception's type name and message."""

    kind: Literal["refused"] = "refused"
    refusals: list[tuple[str, str]]


class Failed(BaseModel):
    """The error a child failed of, and the errors raised beside it, each as its exception's type
    name and message."""

    kind: Literal["failed"] = "failed"
    error: tuple[str, str]
    notes: list[tuple[str, str]] = Field(default_factory=list)


class ChildAnswer(BaseModel):
    """What a spawned child answers its parent with, its arm naming how the child's task ended.
    One model, since an await takes one schema."""

    answer: Annotated[Returned | Refusal | Failed, Field(discriminator="kind")]


class ChildFailed(Unretryable):
    """A joined child failed. Raised in its parent, where a retry reads the same answer, carrying
    the errors raised beside the child's as notes."""

    def __init__(self, message: str, beside: Sequence[str] = ()) -> None:
        super().__init__(message)
        for error in beside:
            self.add_note(error)


def described(exc: BaseException) -> tuple[str, str]:
    """An exception as the wire carries it: its type name and message."""
    return type(exc).__name__, str(exc)


def failure_answer(leaf: BaseException, raised: BaseException) -> Any:
    """The answer a failed task's done event carries, as plain JSON: `leaf` as its error, every
    other leaf of `raised` as a note. The worker that fails the task sends it."""
    notes = [described(other) for other in leaves(raised) if other is not leaf]
    return ChildAnswer(answer=Failed(error=described(leaf), notes=notes)).model_dump(mode="json")


@dataclass(frozen=True)
class Crashed:
    """A child's crash raised beside refusals, as the group its worker sees: the crash first."""

    errors: BaseExceptionGroup[BaseException]


def stopped_at(
    raised: Exception, refusals: tuple[type[Exception], ...]
) -> Refusal | Crashed | None:
    """Where a child that raised `raised` stopped, for its task body to answer or re-raise.

    A `Refusal` when every leaf is one of `refusals`. When refusals were raised beside a crash, and
    no leaf would be raised again on a retry, every leaf with the first crash leading: a
    refusal alone would have completed the child, so the crash is what its worker retries and
    reports. `None` when `raised` goes on to the worker as it is."""
    stopped = list(leaves(raised))
    answers = [isinstance(leaf, refusals) for leaf in stopped]
    match all(answers), any(answers), unretryable(raised):
        case (True, _, _):
            return Refusal(refusals=[described(leaf) for leaf in stopped])
        case (False, True, None):
            cause = stopped[answers.index(False)]
            others = [leaf for leaf in stopped if leaf is not cause]
            return Crashed(BaseExceptionGroup("a crash raised beside refusals", [cause, *others]))
        case _:
            return None


def answer_with(ctx: Any, params: Mapping[str, Any], answer: Returned | Refusal | Failed) -> Any:
    """Answer the parent that spawned this task, if one did. Returns the answer as plain JSON."""
    payload = ChildAnswer(answer=answer).model_dump(mode="json")
    if DONE_EVENT_PARAM in params:
        answer_parent(ctx, params, payload)
    return payload


class RunsTasks(Protocol):
    """A handler that walks a task body and says whether the walk finished."""

    def run_task(self, program: Callable[[], Effect[Any]]) -> Finished | Continued: ...


def run_child(
    ctx: Any, params: Mapping[str, Any], handler: RunsTasks, program: Callable[[], Effect[Any]]
) -> Any:
    """Walk ``program`` as a spawned child and answer its parent, if it has one, with its value or
    the refusals it stopped at.

    Anything else propagates unchanged: the worker answers `Failed` when it fails the task for
    good, and a crash beside a refusal is retried. A refusal raised bare or from a gather
    classifies alike. A walk that ends at a generation boundary answers nobody: the successor
    carries the same done event and answers when the chain finishes. Returns the task's own
    result, so a root with no parent reports what a child would."""
    try:
        match handler.run_task(program):
            case Continued(result=result):
                return result
            case Finished(value=value):
                answer: Returned | Refusal = Returned(value=value)
            case unreachable:
                assert_never(unreachable)
    except Exception as raised:
        match stopped_at(raised, REFUSALS):
            case Refusal() as refusal:
                answer = refusal
            case Crashed(errors=errors):
                raise errors from None
            case None:
                raise
            case unreachable:
                assert_never(unreachable)
    return answer_with(ctx, params, answer)


def join_answer(spawned: Spawned) -> Effect[Any]:
    """Wait for a spawned child's answer: its value as plain JSON, `ChildRefused`, or
    `ChildFailed`."""
    reply = yield from join_child(spawned, ChildAnswer)
    match reply.answer:
        case Returned(value=value):
            return value
        case Refusal(refusals=refusals):
            raise ChildRefused("; ".join(f"{kind}: {message}" for kind, message in refusals))
        case Failed(error=(kind, message), notes=notes):
            raise ChildFailed(
                f"child task {spawned.task_id} failed of {kind}: {message}",
                [f"{noted_kind}: {noted_message}" for noted_kind, noted_message in notes],
            )
        case unreachable:
            assert_never(unreachable)
