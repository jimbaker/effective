"""Tool bracketing strategies + a local GBNF-masked turn caller.

The cache claim: *mask the selection and keep the prefix fixed.* Three strategies over
the **same** per-turn allowlist isolate the one variable that matters for the KV
cache, whether the tool catalog in the prompt prefix is stable:

| strategy | prefix and mask                                                                   |
|----------|-----------------------------------------------------------------------------------|
| `FULL`   | full catalog in the prefix, no decode mask: the no-bracketing baseline            |
| `MASK`   | full catalog in the prefix (stable); the per-turn allowlist enforced only as a    |
|          | **GBNF decode mask** (llama-server compiles the response schema's ``tool_name``   |
|          | enum to a grammar). The prefix is stable, so the cache is preserved               |
| `MUTATE` | the catalog in the prefix is **rewritten** to the per-turn allowlist each turn;   |
|          | the front of the context changes, so the KV cache busts                           |

``MASK`` and ``MUTATE`` apply the *identical* allowlist with the *identical* decode
mask; they differ only in prefix stability, so the measured ``cache_hit_ratio`` delta
between them is the cost of front-of-context mutation.

The turn itself is a **typed t-string channel** (`render` → `Prompt[_ChannelTurn]` →
`resolve`): the four output slots are channels, a `Gated` channel is the cloud-side
tool guardrail, and the per-turn bracketing decision is emitted as a t-string
telemetry `event` (the Sentry "point back at the code" surface). Data axis (channels)
and control axis (the masked op) meet on one turn.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from string.templatelib import Template
from typing import Any, assert_never

from pydantic import BaseModel

from effective.channels import Field, FormGate, Gated, Prompt, Repair, render
from effective.cost import Usage
from effective.domain import AskLLM
from effective.interpreters.openai import LOCAL_FREE, Price, to_openai_messages, usage_from_openai
from effective.react import AssistantTurn, ToolRequest
from effective.telemetry import Sink, event

# --- the over-provisioned toolbox (GSM8K needs only the arithmetic four) -------


def _num(x: Any) -> float | int:
    f = float(x)
    return int(f) if f.is_integer() else f


def _add(a: dict[str, Any]) -> Any:
    return _num(a["a"]) + _num(a["b"])


def _subtract(a: dict[str, Any]) -> Any:
    return _num(a["a"]) - _num(a["b"])


def _multiply(a: dict[str, Any]) -> Any:
    return _num(a["a"]) * _num(a["b"])


def _divide(a: dict[str, Any]) -> Any:
    return _num(a["a"]) / _num(a["b"])


def _na(_: dict[str, Any]) -> Any:
    """A distractor tool: present in the catalog so bracketing is non-trivial, but
    irrelevant to GSM8K — it should never be selected for a correct trajectory."""
    return "not applicable to this task"


# 4 real arithmetic tools + 8 plausible distractors = a 12-tool catalog. The distractors
# make the prefix fat enough that mutating it has a real cache cost, and model the
# realistic "over-provisioned toolbox" that motivates bracketing in the first place.
ARITHMETIC: tuple[str, ...] = ("add", "subtract", "multiply", "divide")

CATALOG_TOOLS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "add": _add,
    "subtract": _subtract,
    "multiply": _multiply,
    "divide": _divide,
    "search": _na,
    "unit_convert": _na,
    "lookup_constant": _na,
    "calendar": _na,
    "weather": _na,
    "translate": _na,
    "sql_query": _na,
    "http_get": _na,
}

CATALOG: list[str] = list(CATALOG_TOOLS)

TOOL_DOC: dict[str, str] = {
    "add": "add(a, b) — the sum of two numbers",
    "subtract": "subtract(a, b) — a minus b",
    "multiply": "multiply(a, b) — the product of two numbers",
    "divide": "divide(a, b) — a divided by b",
    "search": "search(query) — web search",
    "unit_convert": "unit_convert(value, from, to) — convert units",
    "lookup_constant": "lookup_constant(name) — a physical/math constant",
    "calendar": "calendar(date) — calendar facts for a date",
    "weather": "weather(city) — current weather",
    "translate": "translate(text, lang) — translate text",
    "sql_query": "sql_query(query) — run a SQL query",
    "http_get": "http_get(url) — fetch a URL",
}


# --- the per-turn selector: a realistic "tool retrieval" that varies per turn --

type Select = Callable[[int], set[str]]


def rotating_select(
    catalog: Sequence[str] = CATALOG, *, keep: Sequence[str] = ARITHMETIC, window: int = 2
) -> Select:
    """A per-turn allowlist that always keeps the task-relevant tools (`keep`) and
    rotates a `window` of distractors by turn index.

    This makes the allowlist genuinely *vary per turn* (so MUTATE busts the cache) while
    never dropping a tool the task needs (so quality is strategy-independent and only the
    cache differs). It stands in for a real top-k tool retriever."""
    distractors = [t for t in catalog if t not in set(keep)]

    def select(turn: int) -> set[str]:
        if not distractors:
            return set(keep)
        start = (turn * window) % len(distractors)
        extra = {distractors[(start + i) % len(distractors)] for i in range(window)}
        return set(keep) | extra

    return select


# --- the three strategies -----------------------------------------------------


@dataclass(frozen=True)
class Strategy:
    """A bracketing strategy = (is the prefix catalog stable?) x (mask the decode?)."""

    name: str
    stable_prefix: bool  # True: full catalog in the prompt; False: rewrite to the allowlist
    mask_decode: bool  # True: constrain tool_name enum to the allowlist; False: full enum


FULL = Strategy("full", stable_prefix=True, mask_decode=False)
MASK = Strategy("mask", stable_prefix=True, mask_decode=True)
MUTATE = Strategy("mutate", stable_prefix=False, mask_decode=True)
STRATEGIES: dict[str, Strategy] = {s.name: s for s in (FULL, MASK, MUTATE)}


# --- the turn as a typed t-string channel -------------------------------------


class _ChannelTurn(BaseModel):
    """The turn output schema; each field is an output channel in the template."""

    thought: str = ""
    tool_name: str = ""
    tool_args_json: str = ""
    answer: str = ""


def _json_object(s: str) -> bool:
    try:
        return isinstance(json.loads(s or "{}"), dict)
    except ValueError, TypeError:
        return False


# Cross-field guardrails (cf. Django Form.clean) — invariants no single field expresses.
_FORM_GATES: tuple[FormGate, ...] = (
    FormGate(
        lambda v: bool(v["tool_name"].strip() or v["answer"].strip()),
        "set either a tool or an answer, not neither",
    ),
    FormGate(
        lambda v: not v["tool_name"].strip() or _json_object(v["tool_args_json"]),
        "tool_args_json must be a JSON object when a tool is set",
    ),
)


def _catalog_block(tools: Sequence[str], doc: Mapping[str, str]) -> str:
    return "\n".join(f"  {doc[t]}" for t in tools)


# The benchmark-specific task framing (the rest of the turn prompt — response format,
# the act-or-finish protocol — is task-neutral). GSM8K's is the default so the existing
# bench is unchanged; HotpotQA passes its own (see agent.hotpotqa).
GSM8K_TURN_INSTRUCTIONS = (
    "You are a ReAct agent solving a grade-school math word problem. Each turn, call ONE "
    "arithmetic tool for the next calculation, using the observations from prior tool "
    "results; never redo a calculation you already have. The moment your last observation "
    "IS the final answer, FINISH and put that number (digits only) in answer."
)


def _turn_prompt(
    catalog_block: str, allowed: set[str], instructions: str = GSM8K_TURN_INSTRUCTIONS
) -> Prompt[_ChannelTurn]:
    """Render the turn as typed channels. `instructions` is the benchmark-specific framing;
    `catalog_block` rides the prompt prefix; the `Gated` channel rejects an out-of-allowlist
    tool (the cloud guardrail; locally the GBNF enum makes it un-decodable in the first
    place)."""
    thought = Field(str)
    tool_name = Gated(
        str, lambda n: n == "" or n in allowed, "tool_name must be empty or an allowed tool"
    )
    tool_args_json = Field(str)
    answer = Field(str)
    template: Template = t"""{instructions}

Each turn, do ONE of two things: call ONE tool, OR finish by setting tool_name to "" and
putting your final answer in answer. Do not repeat a tool call you already made.

Available tools:
{catalog_block}

Respond with a JSON object with EXACTLY these keys:
  "thought" (brief reasoning): {thought}
  "tool_name" (an allowed tool, or "" to finish): {tool_name}
  "tool_args_json" (a JSON object string of the tool's arguments): {tool_args_json}
  "answer" (your final answer when finishing, else ""): {answer}
"""
    return render(template, output=_ChannelTurn)


def _turn_schema(enum_tools: set[str]) -> dict[str, Any]:
    """A JSON schema whose `tool_name` enum is the decode mask. The native llama-server
    compiles this to a GBNF grammar, so a masked tool literally cannot be sampled — and
    the grammar constrains *decoding*, not the prompt tokens, so it never busts the cache."""
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool_name": {"type": "string", "enum": [*sorted(enum_tools), ""]},
            "tool_args_json": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["thought", "tool_name", "tool_args_json", "answer"],
    }


def _to_turn(ct: _ChannelTurn) -> AssistantTurn:
    name = ct.tool_name.strip()
    if name:
        args = json.loads(ct.tool_args_json or "{}")
        return AssistantTurn(thought=ct.thought, tool=ToolRequest(name=name, args=args))
    return AssistantTurn(thought=ct.thought, answer=ct.answer or None)


def _turn_index(messages: Sequence[Mapping[str, Any]]) -> int:
    """The current turn = how many assistant turns precede it (run_agent appends one
    assistant message per iteration). Determines the per-turn allowlist."""
    return sum(1 for m in messages if m.get("role") == "assistant")


def make_bracketed_caller(
    client: Any,
    model: str,
    *,
    strategy: Strategy,
    catalog: Sequence[str] = CATALOG,
    doc: Mapping[str, str] = TOOL_DOC,
    select: Select | None = None,
    instructions: str | None = None,
    price: Price = LOCAL_FREE,
    extra: dict[str, Any] | None = None,
    max_tokens: int = 512,
    max_repairs: int = 2,
    sink: Sink | None = None,
    session_id: str = "",
    cloud: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> Callable[[AskLLM[Any]], tuple[AssistantTurn, Usage]]:
    """A bracketing turn caller. Local (default) constrains decode with a **GBNF mask**
    (the response schema's `tool_name` enum); `cloud=True` (e.g. gpt-5-nano) cannot mask
    logits, so it sends plain `json_object` and the allowlist is enforced by the **`Gated`
    post-hoc repair** instead. The prefix-stability story is identical on
    both: both providers prefix-cache and report `prompt_tokens_details.cached_tokens`, so
    `Usage.cache_hit_ratio` is the measurement either way.

    Per turn it derives the allowlist (`select(turn)`) and renders the prompt (full catalog
    for a stable prefix, the allowlist for MUTATE). When a `sink` is given, each turn emits
    a t-string telemetry event with the bracketing decision + cache outcome."""
    select = select if select is not None else rotating_select(catalog)
    instructions = instructions or GSM8K_TURN_INSTRUCTIONS
    full = set(catalog)
    extra = extra if extra is not None else ({} if cloud else {"temperature": 0.0})

    def call(op: AskLLM[Any]) -> tuple[AssistantTurn, Usage]:
        turn = _turn_index(op.messages)
        allowed = full if strategy is FULL else select(turn)
        prompt_tools = sorted(catalog) if strategy.stable_prefix else sorted(allowed)
        enum_tools = full if not strategy.mask_decode else allowed

        prompt = _turn_prompt(_catalog_block(prompt_tools, doc), allowed, instructions)
        contract = "\n".join(m.content for m in prompt.messages)
        messages = [{"role": "system", "content": contract}, *to_openai_messages(op.messages)]
        schema = _turn_schema(enum_tools)

        total = Usage()
        outcome: _ChannelTurn | Repair = Repair("not attempted")
        for _ in range(max_repairs + 1):
            start = clock()
            if cloud:  # no logit mask available: json_object + the Gated repair
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=max_tokens,
                    **extra,
                )
            else:  # native llama-server: the schema's tool_name enum becomes a GBNF mask
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_body={"response_format": {"type": "json_object", "schema": schema}},
                    **extra,
                )
            elapsed = clock() - start
            total = total + replace(usage_from_openai(response, price), latency_s=elapsed)
            content = response.choices[0].message.content or "{}"
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {}
            outcome = prompt.resolve(data, form_gates=_FORM_GATES)
            # Both arms do work: one leaves the repair loop, one builds the next prompt. So
            # this is a decision over the outcome's arms.
            match outcome:
                case _ChannelTurn():
                    break
                case Repair(reason=reason):
                    pass
                case unreachable:
                    assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
            fix = f"Rejected: {reason}. Re-emit the full corrected JSON object."
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": fix},
            ]

        match outcome:
            case _ChannelTurn():
                turn_result = _to_turn(outcome)
                chose = outcome.tool_name.strip() or "(finish)"
            case Repair(reason=reason):
                # repairs exhausted -> finish on the thought rather than act on a bad turn
                turn_result = AssistantTurn(thought=reason, answer=f"[guardrail: {reason}]")
                chose = "(guardrail)"
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

        if sink is not None:
            # one self-contained line per turn: the bracketing *decision* (strategy,
            # allowlist, chosen tool) AND its cache *outcome* (prompt/cached tokens), so the
            # trajectory is legible turn-by-turn and the event's code.* points back here.
            strategy_name = strategy.name
            allowed_tools = sorted(allowed)
            prompt_tok = total.prompt_tokens
            cached_tok = total.cache_read_input_tokens
            event(
                t"bracket {strategy_name=} {turn=} {chose=} "
                t"{allowed_tools=} {prompt_tok=} {cached_tok=}",
                sink=sink,
                session_id=session_id,
                kind="CHAIN",
                iteration=turn,
            )
        return turn_result, total

    return call
