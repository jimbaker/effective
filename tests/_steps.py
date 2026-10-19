"""A step's stored key as a test expects it, asked of the minter rather than spelled."""

from effective.api import call_tool
from effective.domain import OpKind
from effective.handlers.base import op_key, step_key
from effective.keys import Key
from effective.ops import Step, step_name_text


def placed_step(name: str | Key, kind: OpKind, frames: str = "", occurrence: int = 1) -> str:
    """The stored key of a step named `name` whose op is of `kind`, under `frames`.

    ========================================================  =======================
    call                                                      stored
    ========================================================  =======================
    `placed_step("extract", "llm")`                           `step;llm:extract`
    `placed_step(react_key("turn"), "llm", "d:0;")`           `d:0;step;llm;react:turn`
    `placed_step(react_key("turn"), "llm", "sub:a;", 2)`      `sub:a;step;llm;react:turn#2`
    ========================================================  ======================="""
    return step_key(name, kind).prefixed(frames).occurrence(occurrence).stored()


def called_step(tool: str, frames: str = "") -> str:
    """The stored key of the step `call_tool(tool, ...)` yields, under `frames`, asked of the op
    it yields: a tool call names its step from the tool, so no `placed_step` spells it."""
    called = call_tool(tool, {}, object)
    yielded = next(called)
    called.close()
    return op_key(yielded).prefixed(frames).stored()


def called_name(tool: str) -> str:
    """The name, as text, of the step `call_tool(tool, ...)` yields."""
    called = call_tool(tool, {}, object)
    yielded = next(called)
    called.close()
    assert isinstance(yielded, Step)
    return step_name_text(yielded.name)
