"""Precise-edit channel — a diff is a `Gated` output channel (the data axis on the
code-edit boundary).

The key move: the model emits a structured `Edit` (path / old / new); a `Gated` channel
checks that `old` matches the target file **exactly once** in a recorded snapshot, and
a breach returns a `Repair` that drives a bounded re-emit — malformed patches never
reach the world. Two precisions make it sound, not cute:

1. **The gate is pure because the snapshot is recorded.** File content threads into the
   predicate as a closure, so the gate is a pure function of recorded data — the whole
   resolve/Repair loop is replay-exact and a reader can compute the gate's answer from
   the trace (strictly more legible than a `git apply --check` gate).
2. **The Gate checks; the Action applies.** This module is *only* the emission guardrail
   (data axis: `Gated` → `Repair` → bounded re-emit). Applying an accepted edit is a
   separate control-axis concern — an `actions=` cascade-gated op, where a stale snapshot
   that passed the gate but fails the apply returns a typed observation and the loop
   continues.

This is the typed-extractor pattern (`TypedField` + `FormGate`) pointed at code editing, and the
repo's first real `Gated` channel. Agent-layer coding support: imports effective + agent only.
"""

import json
from collections.abc import Callable, Mapping
from string.templatelib import Template
from typing import Any, assert_never

from pydantic import BaseModel

from effective.channels import Gated, Prompt, Repair, render
from effective.cost import Usage
from effective.domain import AskLLM
from effective.interpreters.openai import Price, json_complete, messages_to_openai, resolve_price


class Edit(BaseModel):
    """One minimal, anchored edit — `old` must appear exactly once in the target file."""

    path: str
    old: str
    new: str


class EditResponse(BaseModel):
    """The channel output: a single gated `edit` (the wrapper the one `Gated` channel
    resolves into — one channel named `edit`, one field `edit`)."""

    edit: Edit


_ANCHOR_HINT = (
    "old must match the target file exactly once — copy more surrounding lines to anchor it"
)


def edit_template(snapshot: Mapping[str, str], failure: str) -> Template:
    """A precise-edit prompt whose `edit` slot is a `Gated` channel. The gate closes over
    the recorded `snapshot` (path -> file content), so it is a pure predicate over recorded
    data — no I/O in the channel."""

    def applies(e: Edit) -> bool:
        return snapshot.get(e.path, "").count(e.old) == 1

    edit = Gated(Edit, applies, _ANCHOR_HINT)
    return t"""Fix the failing test with ONE minimal, anchored edit.

Test failure:
{failure}

Respond with a JSON object with exactly this key:
  "edit" (object with "path", "old", "new"): {edit}"""


def _resolve_edit_with_repair(
    client: Any,
    *,
    model: str,
    price: Price,
    extra: dict[str, Any],
    prompt: Prompt[EditResponse],
    messages: list[dict[str, Any]],
    max_repairs: int,
) -> tuple[Edit | None, Usage]:
    """JSON-mode call -> ``prompt.resolve`` through the `edit` channel -> bounded repair
    re-prompt on a gate breach (old not exactly once). Returns the accepted ``Edit`` or
    ``None`` when the guardrail can't be satisfied within budget — the caller treats
    ``None`` as "no safe edit" (a denied/failed observation), never a silent bad patch."""
    total = Usage()
    outcome: EditResponse | Repair = Repair("not attempted")
    for _ in range(max_repairs + 1):
        data, usage = json_complete(
            client, model=model, messages=messages, price=price, extra=extra
        )
        total = total + usage
        match prompt.resolve(data):
            case EditResponse() as resp:
                return resp.edit, total
            case Repair() as outcome:
                fix = f"Rejected: {outcome.reason}. Re-emit the full corrected JSON object."
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
        messages = [
            *messages,
            {"role": "assistant", "content": json.dumps(data)},
            {"role": "user", "content": fix},
        ]
    return None, total  # guardrail unsatisfied within budget -> no safe edit


def make_precise_editor(
    client: Any,
    *,
    snapshot: Mapping[str, str],
    model: str = "gpt-5-nano",
    price: Price | None = None,
    extra: dict[str, Any] | None = None,
    max_repairs: int = 2,
) -> Callable[[AskLLM[Any]], tuple[Edit | None, Usage]]:
    """An LLMCall for an edit-emitting AskLLM: render the `Gated` edit template over a
    recorded ``snapshot``, call the model in JSON mode, resolve + repair. The op's last
    message is the test-failure text; the snapshot threads in as the gate's closure.

    ``snapshot`` is the recorded file content at edit time — in a live loop it arrives
    via a ``read_file`` op each turn; here it is passed in, which keeps this module the
    pure emission guardrail (the read-op threading is the loop-wiring follow-up)."""
    price = resolve_price(model, price)
    extra = extra if extra is not None else {"reasoning_effort": "low"}

    def edit(op: AskLLM[Any]) -> tuple[Edit | None, Usage]:
        failure = op.messages[-1]["content"] if op.messages else ""
        prompt = render(edit_template(snapshot, failure), output=EditResponse)
        return _resolve_edit_with_repair(
            client,
            model=model,
            price=price,
            extra=extra,
            prompt=prompt,
            messages=messages_to_openai(prompt.messages),
            max_repairs=max_repairs,
        )

    return edit
