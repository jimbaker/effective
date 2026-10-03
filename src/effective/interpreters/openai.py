"""A thin OpenAI `LLMCall` for the ReAct agent (provider seam, not core).

This is the production half the cost layer was waiting for: given an ``AskLLM``
op (the conversation + the response schema), call OpenAI for one turn and return
``(AssistantTurn, Usage)`` — exactly the shape ``MeteredInterpreter`` folds.

Provider quirks are isolated here, away from the effective core:

- The model answers against a *wire* schema (``_WireTurn``) whose tool arguments
  are a JSON *string*, because OpenAI strict structured outputs reject the
  open-ended ``dict[str, Any]`` the domain ``ToolRequest`` carries. The caller
  re-parses the string into the domain ``AssistantTurn``.
- OpenAI returns token counts but no dollar cost, so we price the usage from a
  per-model ``Price`` table (cached input billed separately — the KV-cache the
  cost layer reports).

The OpenAI client is *injected* (duck-typed on ``client.chat.completions.parse``)
so unit tests pass a fake and this module imports no SDK. The real client is
built only in scripts/openai_agent_smoke.py.
"""

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, Field

from effective.cancel import Cancelled, CancelToken
from effective.channels import Message
from effective.cost import Usage
from effective.domain import AskLLM
from effective.interpreters.aio import LoopThread, Stopped, shared_loop
from effective.layers import RateLimited, TransientError
from effective.react import AssistantTurn, ToolRequest


def _transient_types() -> tuple[type[BaseException], ...]:
    """The OpenAI exception classes the substrate treats as transient — connection-class only
    (no response arrived, hence no usage). In-function import keeps this module SDK-free (the
    module docstring's contract); an absent SDK means a fake client that raises no OpenAI types,
    so `()` catches nothing. A `RateLimitError` maps to `RateLimited` instead (below): it IS
    transient, but `retry_domain` re-fires it only with a backoff configured."""
    try:
        import openai
    except ImportError:
        return ()
    return (openai.APIConnectionError, openai.InternalServerError)  # APITimeoutError ⊂ the first


def _rate_limit_types() -> tuple[type[BaseException], ...]:
    """The provider's 429 type, mapped separately so the substrate can distinguish "retry me"
    from "retry me, but not immediately"."""
    try:
        import openai
    except ImportError:
        return ()
    return (openai.RateLimitError,)


@contextmanager
def _as_transient() -> Iterator[None]:
    """Map provider connection/5xx/429 failures into the substrate's vocabulary.

    ONE definition wrapping ALL THREE call paths (`structured_extract`, `json_complete`, and
    `_BaseTurnCaller.__call__` via `_complete`), so a transient blip retries under `retry_domain`
    on every path. A provider→substrate boundary that covers one of three doors is not a boundary.

    Deliberately a context manager rather than three copies of the `try`: the mapping is one rule,
    and a rule copied three times drifts."""
    try:
        yield
    except _rate_limit_types() as exc:
        raise RateLimited(f"{type(exc).__name__}: {exc}") from exc
    except _transient_types() as exc:
        raise TransientError(f"{type(exc).__name__}: {exc}") from exc


@dataclass(frozen=True)
class Price:
    """Per-1M-token prices in USD."""

    input_per_1m: float
    cached_per_1m: float
    output_per_1m: float

    def cost(self, uncached: int, cached: int, completion: int) -> float:
        return (
            uncached * self.input_per_1m
            + cached * self.cached_per_1m
            + completion * self.output_per_1m
        ) / 1_000_000


# gpt-5-nano list pricing (USD / 1M tokens). Approximate; the meter is a guide
# and the smoke run is hard-capped by a CostBudget regardless.
GPT5_NANO = Price(input_per_1m=0.05, cached_per_1m=0.005, output_per_1m=0.40)

# gpt-5.6-luna list pricing (USD / 1M tokens), 2026-08-24.
#
# **SHORT-CONTEXT tier only, and the meter therefore UNDER-reports a long-context call.** Luna is
# priced in two tiers — $0.20/$0.02/$1.20 short, $0.40/$0.04/$1.80 long — and `Price` has three
# fields with no notion of a context threshold. Encoding the cheaper tier is the honest default
# (it is what a short agent turn actually pays) but a run that crosses into long context is
# charged more than this reports. Widening `Price` to a tiered shape is the fix if that ever
# matters; until then the meter is a guide and `CostBudget` is the hard cap.
GPT56_LUNA = Price(input_per_1m=0.20, cached_per_1m=0.02, output_per_1m=1.20)

# GPT-6 list pricing (USD / 1M tokens), released and read 2026-09-22 from
# developers.openai.com/api/docs/pricing. Priced only: neither has a measured `ModelProfile`.
GPT6_LUNA = Price(input_per_1m=0.10, cached_per_1m=0.01, output_per_1m=0.50)
GPT6_SOL = Price(input_per_1m=2.00, cached_per_1m=0.20, output_per_1m=10.00)

# A local model has no per-token price; latency_s carries the comparison signal.
LOCAL_FREE = Price(input_per_1m=0.0, cached_per_1m=0.0, output_per_1m=0.0)


type Wire = Literal["responses", "chat"]
"""Which OpenAI endpoint a caller drives — `/v1/responses` or `/v1/chat/completions`.

A `Literal`, not a `str`, so an unknown wire is a TYPE error at the call site rather than a
`ValueError` at the first live request. That matters more than usual here: the code that dispatches
on it lives in `examples/`, which `just lint --totality src` deliberately does not scan, so `ty` is
the only enforcer available."""


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """What a caller must know about a model beyond its id: what it costs, and how to ask it.

    **One record, because these knobs travel together and drifted apart twice in one sitting.**
    `price` was pinned to `GPT5_NANO` beside a parameterized `model`, so choosing a model repriced
    nothing. Fixing that left `reasoning_effort="minimal"` hardcoded one line below — and the very
    next model rejected it with a 400, because `minimal` is a gpt-5-nano value that gpt-5.6-luna
    does not accept. Two knobs, one shape: a model-specific setting spelled beside a model that
    varies. A profile keyed by model id is the structural answer to both.

    **`wire` is the third instance of that same shape**, and it arrived the same way: the endpoint
    was a caller-side default (`wire="responses"` for every model) while the two models disagree
    about which endpoint they work on. The asymmetry is measured and is a mirror image — on chat
    completions nano takes tools with reasoning and rejects `none`, while luna rejects tools with
    any reasoning and accepts them only at `none`. An endpoint is therefore a fact about the MODEL,
    not a preference of the caller."""

    price: Price
    reasoning_effort: str
    wire: Wire


MODELS: dict[str, ModelProfile] = {
    "gpt-5-nano": ModelProfile(price=GPT5_NANO, reasoning_effort="minimal", wire="chat"),
    "gpt-5.6-luna": ModelProfile(price=GPT56_LUNA, reasoning_effort="low", wire="responses"),
}
"""Model id -> everything a caller needs. `minimal` is nano's cheapest effort tier; luna refuses
it (400) and reports its supported values as `none`, `low`, `medium`, …, so `low` is the floor.

**Each `wire` is the endpoint that model is MEASURED good on, not a guess.** Luna on
`/v1/responses` with tools declared finished 6/6 cleanly against chat completions' 3/6, and acted
30/30 against 24/30. Nano's evidence is all on chat completions; it has never been run on
`/v1/responses` at all, so it is recorded where it has been proven rather than where its stablemate
happens to do well."""


def profile_for(model: str) -> ModelProfile:
    """The profile for `model`, or a refusal.

    **Refusing beats defaulting.** An unknown model priced at some other model's rate produces a
    cost that looks right and is not, which is worse than an error: the meter, the span's cost,
    and every projection folded from it would agree on a wrong number. The same
    argument covers the effort knob, where the failure is at least loud: a 400 on call one."""
    try:
        return MODELS[model]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise ValueError(
            f"no profile for model {model!r} — add it to MODELS (known: {known})"
        ) from None


def resolve_price(model: str, price: Price | None) -> Price:
    """The price a caller MEANT: their explicit one, or the model's own from the table.

    **One helper because this is the hard rule, and it was spelled eleven times as a default
    argument instead.** `price: Price = GPT5_NANO` beside a `model: str = "gpt-5-nano"` reads as a
    matched pair and stops being one the moment anybody passes a different model — which is how
    `gpt-5.6-luna` came to be metered at nano's rate, under-reporting 3x, at every call site that
    had not been individually remembered.

    A default argument cannot express "whichever model you named", so it expressed "nano" and was
    silently wrong for everything else. This can, and an unknown model refuses through
    `profile_for` rather than inheriting a stablemate's rate. An explicit `price=` still wins,
    which is what a local llama.cpp endpoint with no table entry passes."""
    return price if price is not None else profile_for(model).price


def price_for(model: str) -> Price:
    """`profile_for(model).price` — kept because pricing is what most callers want."""
    return profile_for(model).price


class _WireTool(BaseModel):
    """Provider-facing tool request: arguments as a JSON string (strict-mode safe)."""

    name: str = Field(description="The tool to call.")
    arguments_json: str = Field(description='Arguments as a JSON object string, e.g. {"a": 1}.')


class _WireTurn(BaseModel):
    """Provider-facing turn. Set `tool` to act (and `answer` null), or `answer` to finish."""

    thought: str = Field(description="Brief reasoning for this step.")
    tool: _WireTool | None = Field(description="The action to take, or null when answering.")
    answer: str | None = Field(description="The final answer, or null when acting.")


def _counter(usage: Any, name: str) -> int | None:
    """One token counter, or `None` when the reply did not carry a number for it.

    The whole discrimination rests on this distinction. `getattr(u, name, 0) or 0` collapses
    absent, `None` and a real `0` into the same answer, which is what let an unvalidated response
    price at zero. Present-but-`None` is the SDK's ordinary output for a field the wire omitted,
    so it must read as "not this shape", not as "this shape, costing nothing"."""
    value = getattr(usage, name, None)
    return int(value) if isinstance(value, int) else None


def usage_from_openai(response: Any, price: Price) -> Usage:
    """Read OpenAI token usage and price it (cached input billed at the cheap rate).

    **Two wire shapes, and an unknown one REFUSES rather than reporting zero.** Chat Completions
    says `prompt_tokens`/`completion_tokens`/`prompt_tokens_details.cached_tokens`; the Responses
    API says `input_tokens`/`output_tokens`/`input_tokens_details.cached_tokens`. Reading the
    first set through `getattr(..., 0)` would price a Responses reply at `Usage(cost=0.0)` with
    no exception.

    A silent zero is the worst available answer here. It is indistinguishable from a real
    measured-and-free call, so it would put a cost of `0.0` on every span, report `$0.00`
    from the meter, and let the two-bookkeepers check AGREE because both halves are zero. That is
    exactly the "free" vs "unmeasured" collapse `Node.cost` exists to prevent, arriving through
    the provider seam instead of the projection.

    **The refusal was written against attribute PRESENCE, and that is not where the hazard is.**
    `openai._models.construct_type` is the SDK's real response-parsing path and it does not
    validate, so a reply missing its counts comes back as a `CompletionUsage` whose every field is
    `None`. `hasattr` says yes, `int(x or 0)` says zero, and nothing raises — the exact silent zero
    the paragraph above says is the worst available answer, reachable through the door the refusal
    left open. The two shapes it did catch were the two nobody was going to send.

    So the discrimination is on VALUES, not on attribute names: a family counts as present only
    when its own prompt counter holds a number. Both families present is ambiguous rather than
    first-wins, and neither present refuses with what it actually saw.

    **Not dispatched on `ModelProfile.wire`, though the declared discriminator exists.** That was
    the tidier suggestion and it is the wrong shape here: this is called from eight places, several
    of which price a model with no table entry at all (`LlamaCppTurnCaller`, the bracket and
    skillsbench meters). A required `wire` would either thread a parameter through callers that
    cannot supply one, or push them through `profile_for` and make an unknown model fatal where it
    is fine today. The wire is a fact about the MODEL; this is a fact about the RESPONSE in hand,
    and reading the response is what stays correct when a caller is off the table."""
    u = response.usage
    if u is None:
        raise ValueError("response carries no usage — cannot price it")
    chat, responses = _counter(u, "prompt_tokens"), _counter(u, "input_tokens")
    match (chat, responses):
        case (int(), None):  # Chat Completions
            prompt = chat
            completion = _counter(u, "completion_tokens") or 0
            details = getattr(u, "prompt_tokens_details", None)
        case (None, int()):  # Responses
            prompt = responses
            completion = _counter(u, "output_tokens") or 0
            details = getattr(u, "input_tokens_details", None)
        case (int(), int()):
            raise ValueError(
                "ambiguous usage shape: it carries BOTH `prompt_tokens` (Chat Completions) and "
                "`input_tokens` (Responses), so which one is authoritative is a guess"
            )
        case _:
            raise ValueError(
                "unrecognized usage shape: expected a numeric `prompt_tokens` (Chat Completions) "
                "or `input_tokens` (Responses); the token-ish attributes present are "
                f"{ {k: getattr(u, k, None) for k in dir(u) if 'token' in k} }"
            )
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    uncached = max(prompt - cached, 0)
    cost = price.cost(uncached, cached, completion)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cache_read_input_tokens=cached,
        cost=cost,
    )


def structured_extract(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: Any,
    price: Price,
    max_completion_tokens: int = 4000,
    extra: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, Usage]:
    """Generic one-shot structured output: parse ``response_format`` from the model,
    returning the parsed instance plus a priced + timed ``Usage``.

    Schema-agnostic and domain-free: the caller passes the response model, and a domain's
    extraction schemas live with the domain.
    """
    start = clock()
    with _as_transient():  # provider → substrate vocabulary; retry_domain catches it
        response = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            **(extra or {}),
        )
    elapsed = clock() - start
    parsed = response.choices[0].message.parsed
    usage = replace(usage_from_openai(response, price), latency_s=elapsed)
    return parsed, usage


def json_complete(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    price: Price,
    max_completion_tokens: int = 4000,
    extra: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], Usage]:
    """One-shot JSON-mode completion: parse the response content as a dict, priced
    + timed. The schema lives in the prompt (e.g. a rendered t-string with typed
    channels), so this stays schema-agnostic and domain-free."""
    start = clock()
    with _as_transient():
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_completion_tokens,
            **(extra or {}),
        )
    elapsed = clock() - start
    data = json.loads(response.choices[0].message.content)
    usage = replace(usage_from_openai(response, price), latency_s=elapsed)
    return data, usage


def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Map the channel processor's provider-neutral ``Message``s to OpenAI dicts.

    This is the caller-side provider mapping the ADR reserves: ``render`` emits a
    neutral ``cache: bool`` boundary; OpenAI caches by prefix automatically, so cache
    is a no-op here (an Anthropic caller would instead place ``cache_control`` on the
    last ``cache=True`` message). Keeping the mapping in an interpreter is what lets
    ``effective.channels`` stay provider-neutral."""
    return [{"role": m.role, "content": m.content} for m in messages]


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt the loop's abstract messages to what OpenAI accepts.

    run_agent records observations as role:"tool", but OpenAI only allows a tool
    message in reply to an assistant `tool_calls` — which structured-output mode
    never emits. Present the observation as a user message instead.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({"role": "user", "content": f"Tool result: {m['content']}"})
        else:
            out.append(m)
    return out


def _to_turn(wire: _WireTurn) -> AssistantTurn:
    tool = None
    if wire.tool is not None:
        # strict=False: models embed literal newlines/tabs inside string values
        # when the argument is itself code or a multiline shell command — accept
        # the control characters rather than failing the turn.
        #
        # A small model can also emit MALFORMED json for a tool's arguments (an
        # unescaped quote splitting a value, a missing delimiter). That must not crash a
        # durable workflow: fall back to empty args so the tool surfaces an error
        # observation and the ReAct loop re-decides, rather than the task dying mid-run.
        try:
            args = json.loads(wire.tool.arguments_json or "{}", strict=False)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        tool = ToolRequest(name=wire.tool.name, args=args)
    return AssistantTurn(thought=wire.thought, tool=tool, answer=wire.answer)


class _BaseTurnCaller:
    """Shared ``LLMCall`` machinery for OpenAI-compatible endpoints.

    Carries the system prompt (protocol + tool catalog), since the prototype
    ``run_agent`` keeps only the conversation in the op — provider framing lives
    in the seam. Times each call with an injectable clock and records the wall
    time as ``Usage.latency_s`` (the local analog of dollar cost). Subclasses
    differ only in how they request structured output and read the turn back.
    """

    def __init__(
        self,
        client: Any,
        system_prompt: str,
        model: str,
        price: Price | None = None,
        max_completion_tokens: int = 2000,
        extra: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.model = model
        # THE ONE PLACE A PRICE IS CHOSEN. `price` defaulted to `GPT5_NANO` beside a `model` that
        # varied, so naming a model repriced nothing — `OpenAITurnCaller(model="gpt-5.6-luna")`
        # metered luna at nano's rate and under-reported by 3x. `ModelProfile` was introduced to
        # stop exactly that and bound the knobs in the TABLE only, which left every constructor
        # free to disagree with it.
        #
        # `None` means "ask the table", and an unknown model then REFUSES through `profile_for`
        # rather than inheriting some other model's rate — a cost that looks right and is not is
        # worse than an error, because the meter, the span's cost and every projection
        # folded from it would agree on the wrong number. An explicit `price=` still wins, which
        # is what a local llama.cpp endpoint with no table entry needs.
        self.price = resolve_price(model, price)
        self.max_completion_tokens = max_completion_tokens
        self.extra = extra or {}
        self.clock = clock

    def _complete(self, messages: list[dict[str, Any]]) -> tuple[_WireTurn, Any]:
        raise NotImplementedError

    def __call__(self, op: AskLLM[Any]) -> tuple[AssistantTurn | Cancelled, Usage]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *to_openai_messages(op.messages),
        ]
        start = self.clock()
        try:
            with _as_transient():  # the one door both subclasses' `_complete` go through
                wire, response = self._complete(messages)
        except _StreamStopped as stopped:
            # The provider reports no usage for a response it never finished.
            return Cancelled(partial=stopped.partial), Usage(latency_s=self.clock() - start)
        elapsed = self.clock() - start
        usage = replace(usage_from_openai(response, self.price), latency_s=elapsed)
        return _to_turn(wire), usage


class OpenAITurnCaller(_BaseTurnCaller):
    """Cloud OpenAI: strict structured output via the ``parse`` (json_schema) helper."""

    def __init__(
        self,
        client: Any,
        system_prompt: str,
        model: str = "gpt-5-nano",
        price: Price | None = None,
        max_completion_tokens: int = 2000,
        extra: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(client, system_prompt, model, price, max_completion_tokens, extra, clock)

    def _complete(self, messages: list[dict[str, Any]]) -> tuple[_WireTurn, Any]:
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=_WireTurn,
            max_completion_tokens=self.max_completion_tokens,
            **self.extra,
        )
        return response.choices[0].message.parsed, response


class ResponsesTurnCaller(_BaseTurnCaller):
    """OpenAI `/v1/responses`, with the action channel NATIVE rather than emulated.

    **Why this endpoint exists here, measured 2026-08-24.** In `/v1/chat/completions` the two
    models are mirror images: `gpt-5-nano` accepts function tools with `reasoning_effort` but
    rejects `none`; `gpt-5.6-luna` rejects function tools with ANY reasoning effort, including the
    default, and accepts them only at `none`. So on chat completions luna can have its reasoning
    or its tools, never both. `/v1/responses` lifts that.

    **What being native buys.** With tools declared only in system-prompt prose and no `tools`
    parameter on the request, luna refused to act on turn one in 6 of 30 trials — it said the
    tools were returning nothing, which from the API's point of view was true, since it had none.
    Declaring them natively: 30 of 30 acted (Fisher exact two-sided p = 0.024).

    **The union is native too, which is the pleasant part.** Setting `tools` AND `text_format`
    together means the reply is one or the other: a `function_call` output with `output_parsed`
    None, or a `message` with the parsed turn. That is exactly `AssistantTurn`'s tool-XOR-answer
    shape, so the envelope stops being emulated and becomes a fallback for the answer branch."""

    def __init__(
        self,
        client: Any,
        system_prompt: str,
        model: str,
        price: Price,
        max_completion_tokens: int = 2000,
        extra: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        tools: list[dict[str, Any]] | None = None,
        cancel: CancelToken | None = None,
    ) -> None:
        super().__init__(client, system_prompt, model, price, max_completion_tokens, extra, clock)
        self.tools = tools
        self.cancel = cancel

    def _complete(self, messages: list[dict[str, Any]]) -> tuple[_WireTurn, Any]:
        kwargs = self._request(messages)
        if self.cancel is not None:
            return _streamed(self.client, kwargs, self.cancel)
        response = self.client.responses.parse(**kwargs)
        return _wire_from_response(response), response

    def _request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        extra = dict(self.extra)
        # `reasoning_effort=` (chat completions) is spelled `reasoning={"effort": …}` here, and
        # the token cap is `max_output_tokens`. Translated rather than passed through, so a
        # `ModelProfile` written for either endpoint reaches this one unchanged.
        effort = extra.pop("reasoning_effort", None)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "text_format": _WireTurn,
            "max_output_tokens": self.max_completion_tokens,
            **extra,
        }
        if effort is not None:
            kwargs["reasoning"] = {"effort": effort}
        if self.tools:
            kwargs["tools"] = self.tools
            # ONE call per reply, because the envelope holds one. `AssistantTurn` has a single
            # `tool`, so a reply carrying two actions can only have one of them performed, and
            # the model would go on believing both ran. Asked for here and refused below, since
            # this is a request parameter rather than a guarantee.
            kwargs["parallel_tool_calls"] = False
        return kwargs


class AsyncResponsesTurnCaller(ResponsesTurnCaller):
    """`ResponsesTurnCaller` over `openai.AsyncOpenAI`, its coroutines run on `shared_loop()`.

    The same request and the same answer; a cancel is the task's cancellation, which lands at
    whatever the call awaits and closes its connection on the way out."""

    @property
    def loop(self) -> LoopThread:
        return shared_loop()

    def _complete(self, messages: list[dict[str, Any]]) -> tuple[_WireTurn, Any]:
        kwargs = self._request(messages)
        if self.cancel is None:
            response = self.loop.run(self.client.responses.parse(**kwargs))
            return _wire_from_response(response), response
        seen = _Seen()
        try:
            response = self.loop.run(_astreamed(self.client, kwargs, seen), self.cancel)
        except Stopped:
            with seen.lock:
                response, ended, text = seen.response, seen.ended, "".join(seen.arrived)
            if response is None and ended is None:
                raise _StreamStopped(text) from None
            if response is None:
                raise RuntimeError("the response ended without completing", ended) from None
        return _wire_from_response(response), response


class _Seen:
    """What an async stream delivered, read by the calling thread after a stop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.arrived: list[str] = []
        self.response: Any = None
        self.ended: str | None = None


async def _astreamed(client: Any, kwargs: dict[str, Any], seen: _Seen) -> Any:
    async with client.responses.stream(**kwargs) as stream:
        async for event in stream:
            with seen.lock:
                if event.type == "response.output_text.delta":
                    seen.arrived.append(event.delta)
                elif event.type == "response.completed":
                    seen.response = event.response
                elif event.type in ENDED:
                    seen.ended = event.type
        if seen.response is None and seen.ended is not None:  # as the threaded caller says it
            raise RuntimeError("the response ended without completing", seen.ended)
        return await stream.get_final_response()


def _sever(stream: Any) -> None:
    """Shut the stream's socket down, then close the stream.

    A close alone leaves a read blocked in another thread asleep, and on a pooled connection the
    provider never learns the client left, so it generates the whole reply. A shutdown wakes the
    read and sends the provider a FIN. The socket is reached through the httpx response the openai
    stream holds as `_response`; a stream without one is only closed."""
    response = getattr(stream, "_response", None)
    network = getattr(response, "extensions", {}).get("network_stream")
    sock = None if network is None else network.get_extra_info("socket")
    if sock is not None:
        with suppress(OSError):  # the connection is already gone
            sock.shutdown(socket.SHUT_RDWR)
    stream.close()


class _StreamStopped(Exception):
    """A cancel closed the stream; `partial` is the answer text that had arrived."""

    def __init__(self, partial: str) -> None:
        super().__init__("the response stream was closed by a cancel")
        self.partial = partial


ENDED = frozenset({"response.incomplete", "response.failed"})
"""The events that end a stream without an answer; a stop after one is not what ended it."""


class _Reader:
    """One streamed call's reader thread and its stop, which share the stream and what arrived."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.lock = threading.Lock()
        self.arrived: list[str] = []
        self.stream: Any = None
        self.response: Any = None
        self.ended: str | None = None
        self.error: Exception | None = None
        self.settled, self.stopped = threading.Event(), threading.Event()

    def read(self) -> None:
        with self.lock:
            if self.stopped.is_set():  # the Esc came before the request; send nothing
                self.settled.set()
                return
        try:
            stream = self.manager.__enter__()
            with self.lock:
                self.stream = stream
                late = self.stopped.is_set()
            if late:  # the stop came while the request was in flight
                _sever(stream)
                return
            for event in stream:
                with self.lock:
                    if event.type == "response.output_text.delta":
                        self.arrived.append(event.delta)
                    elif event.type == "response.completed":
                        self.response = event.response
                    elif event.type in ENDED:
                        self.ended = event.type
            final = stream.get_final_response()
            with self.lock:
                self.response = final
        except Exception as raised:
            with self.lock:
                self.error = raised
        finally:
            self.settled.set()

    def stop(self) -> None:
        with self.lock:
            self.stopped.set()
            stream = self.stream
        try:
            if stream is not None:
                _sever(stream)
        finally:
            self.settled.set()


def _streamed(client: Any, kwargs: dict[str, Any], cancel: CancelToken) -> tuple[_WireTurn, Any]:
    """The call streamed on a reader thread, so a cancel returns at once.

    The reader opens the stream too, so a stop reaches the call while it waits for the first
    byte. The stop severs the connection (`_sever`), which ends the reader and tells the provider
    the client has gone, and the op answers `Cancelled` without waiting for the reader. A reply
    whose `response.completed` arrived before the stop is the answer, and one that ended
    `incomplete` or `failed` raises as it would have without the stop."""
    if cancel.cancelled:  # the Esc came first; send nothing
        raise _StreamStopped("")
    reader = _Reader(client.responses.stream(**kwargs))
    with cancel.on_cancel(reader.stop):  # registered first, so no cancel falls between the two
        threading.Thread(target=reader.read, name="responses-stream", daemon=True).start()
        reader.settled.wait()
    with reader.lock:
        response, ended, error = reader.response, reader.ended, reader.error
        text = "".join(reader.arrived)
    if response is None and ended is None and reader.stopped.is_set():
        raise _StreamStopped(text)
    reader.manager.__exit__(None, None, None)
    match response, ended, error:
        case None, str(), _:  # it ended on its own, before or despite a stop
            raise RuntimeError("the response ended without completing", ended)
        case None, None, Exception():
            raise error
        case None, None, None:
            raise RuntimeError("the stream ended with no response")
        case _:
            return _wire_from_response(response), response


_NAMESPACE = "functions."
"""The prefix a model sometimes puts on a function it calls, and the catalog never does.

Measured on the first live `examples.coder` run (2026-09-15, `gpt-5.6-luna`): the reply named
`functions.edit` for a tool declared as `edit`, the loop found no such tool, and the call reached
a deployment that serves four names and refused. Stripped here, at the seam that knows this is one
provider's spelling of the name it was given, rather than in every table a loop dispatches on."""


def _wire_from_response(response: Any) -> _WireTurn:
    """Read one turn back: a native `function_call` if the model acted, else the parsed answer.

    A `function_call` carries no `thought` — the model's reasoning is either summarised in a
    `reasoning` item or not exposed at all — so the thought is left empty and the ACTION carries
    the information. `react._assistant` renders the action into the transcript, which is what a
    later turn actually needs to see."""
    calls = [o for o in response.output if o.type == "function_call"]
    if len(calls) > 1:
        raise ValueError(
            f"the reply carried {len(calls)} function calls and this envelope performs one: "
            f"{[o.name for o in calls]}. Taking the first would tell the model the rest ran."
        )
    call = calls[0] if calls else None
    if call is not None:
        tool = _WireTool(name=call.name.removeprefix(_NAMESPACE), arguments_json=call.arguments)
        return _WireTurn(thought="", tool=tool, answer=None)
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError(
            "responses reply carried neither a function_call nor a parsed turn; "
            f"output kinds were {[o.type for o in response.output]}"
        )
    return parsed


class LlamaCppTurnCaller(_BaseTurnCaller):
    """Local llama.cpp server: strict output via ``json_object`` + a JSON schema.

    llama-cpp-python rejects OpenAI's ``response_format: json_schema``; it instead
    takes ``{"type": "json_object", "schema": <json schema>}`` and compiles the
    schema to a GBNF grammar, constraining the decode to valid ``_WireTurn`` JSON.
    The schema rides in ``extra_body`` so the OpenAI SDK forwards it verbatim.
    """

    def __init__(
        self,
        client: Any,
        system_prompt: str,
        model: str,
        price: Price = LOCAL_FREE,
        max_completion_tokens: int = 2000,
        extra: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(client, system_prompt, model, price, max_completion_tokens, extra, clock)

    def _complete(self, messages: list[dict[str, Any]]) -> tuple[_WireTurn, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_completion_tokens,
            extra_body={
                "response_format": {
                    "type": "json_object",
                    "schema": _WireTurn.model_json_schema(),
                }
            },
            **self.extra,
        )
        wire = _WireTurn.model_validate_json(response.choices[0].message.content)
        return wire, response
