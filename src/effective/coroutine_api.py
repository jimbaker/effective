"""An alternative authoring surface: awaitable coroutine effects.

Same op algebra, different surface. Each op is an ``Awaitable[T]`` whose
``__await__`` yields the op to the driver; workflows are ``async def`` using
``await``. The generator surface in ``effective.api`` is the primary one; this
one stays runnable so the comparison between them rests on code.

The same handler drives both surfaces (a sync generator and a coroutine both
speak ``.send()`` / yield over the same ops), so a parity test can assert that
the two surfaces produce an identical op trace.
"""

from collections.abc import Generator
from typing import Any, cast

from effective.cancel import served
from effective.domain import AskLLM, CallTool
from effective.keys import Key, Name, compose_key
from effective.ops import (
    AppendLedgerRow,
    AwaitEvent,
    LedgerRow,
    Step,
    StoreArtifact,
    WorkflowOp,
    event_name,
)


class _Effect[T]:
    """An awaitable wrapping one op. ``await`` it to perform the op."""

    def __init__(self, op: WorkflowOp) -> None:
        self.op = op

    def __await__(self) -> Generator[WorkflowOp, Any, T]:
        raw = yield self.op  # yields to the driver, not the asyncio loop
        match self.op:
            case Step() as step:
                return cast(T, served(step.name, raw))
            case _:
                return cast(T, raw)


def ask_llm[T](name: str, messages: Any, schema: type[T]) -> _Effect[T]:
    return _Effect(Step(name=name, op=AskLLM(messages=messages, response_schema=schema)))


def call_tool[T](name: str, args: dict[str, Any], schema: type[T]) -> _Effect[T]:
    return _Effect(
        # Composed, not f-stringed, and inlined rather than delegated to `api.direct_tool_key`:
        # `build_key_registry` scans `compose_key` call sites and nothing else, so a surface that
        # called the minter would mint the same bytes and be invisible to the source map.
        # The name rides INSIDE the hole, as author path text.
        Step(
            name=compose_key(t"tool:{Name(name)}").stored(),
            op=CallTool(name=name, args=args, result_schema=schema),
        )
    )


def await_event[T](name: str | Key, schema: type[T]) -> _Effect[T]:
    return _Effect(AwaitEvent(name=event_name(name), schema=schema))


def append_ledger(row: LedgerRow) -> _Effect[None]:
    return _Effect(AppendLedgerRow(row=row))


def store_artifact[T](value: T, content_type: str) -> _Effect[str]:
    return _Effect(StoreArtifact(value=value, content_type=content_type))
