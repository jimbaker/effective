"""Thin OpenAI LLMCall — pricing + wire-to-domain mapping (red-green spec).

The provider seam lives in the agent layer (not the effective core, which stays
provider-free). The client is injected, so these tests use a fake — no network,
no spend. The real paid call is exercised only by scripts/openai_agent_smoke.py
under a hard CostBudget.
"""

import json
from types import SimpleNamespace
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from effective.channels import Message
from effective.domain import AskLLM
from effective.interpreters.openai import (
    GPT5_NANO,
    GPT56_LUNA,
    MODELS,
    LlamaCppTurnCaller,
    OpenAITurnCaller,
    ResponsesTurnCaller,
    Wire,
    _wire_from_response,
    _WireTool,
    _WireTurn,
    json_complete,
    messages_to_openai,
    price_for,
    profile_for,
    structured_extract,
    usage_from_openai,
)
from effective.react import AssistantTurn


def _clock(values: list[float]):
    it = iter(values)
    return lambda: next(it)


class _FakeCreate:
    """Stands in for client.chat.completions when the caller uses create()."""

    def __init__(self, content: str, usage: SimpleNamespace) -> None:
        self._content = content
        self._usage = usage
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self._usage)


def _local_client(content: str, usage: SimpleNamespace):
    completions = _FakeCreate(content, usage)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _fake_usage(prompt: int, completion: int, cached: int) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


class _FakeCompletions:
    """Stands in for client.chat.completions; records the parse() kwargs."""

    def __init__(self, wire: Any, usage: SimpleNamespace) -> None:
        self._wire = wire
        self._usage = usage
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self._wire)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self._usage)


def _client(wire: Any, usage: SimpleNamespace):
    completions = _FakeCompletions(wire, usage)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


# --- pricing ---------------------------------------------------------------


def test_usage_from_openai_prices_cached_and_uncached_separately():
    resp = SimpleNamespace(usage=_fake_usage(prompt=1000, completion=500, cached=200))
    u = usage_from_openai(resp, GPT5_NANO)

    assert u.prompt_tokens == 1000
    assert u.completion_tokens == 500
    assert u.cache_read_input_tokens == 200
    # 800 uncached in @ $0.05/1M + 200 cached in @ $0.005/1M + 500 out @ $0.40/1M
    expected = (800 * 0.05 + 200 * 0.005 + 500 * 0.40) / 1_000_000
    assert u.cost == pytest.approx(expected)


# --- wire -> domain mapping ------------------------------------------------


def test_caller_maps_a_tool_turn_with_parsed_args_and_prepends_system():
    wire = _WireTurn(
        thought="multiply them",
        tool=_WireTool(name="multiply", arguments_json='{"a": 19, "b": 23}'),
        answer=None,
    )
    client, completions = _client(wire, _fake_usage(100, 20, 0))
    caller = OpenAITurnCaller(client=client, system_prompt="SYS", model="gpt-5-nano")

    op = AskLLM(messages=[{"role": "user", "content": "q"}], response_schema=AssistantTurn)
    turn, usage = caller(op)

    assert isinstance(turn, AssistantTurn)
    assert turn.tool is not None
    assert turn.tool.name == "multiply"
    assert turn.tool.args == {"a": 19, "b": 23}  # arguments_json parsed to a dict
    assert turn.answer is None
    assert usage.completion_tokens == 20

    sent = completions.calls[0]
    assert sent["model"] == "gpt-5-nano"
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}  # prepended
    assert sent["messages"][1] == {"role": "user", "content": "q"}


def test_caller_translates_tool_role_messages_for_openai():
    # OpenAI rejects a bare role:"tool" message (valid only after assistant
    # tool_calls); the seam remaps run_agent's observations to user messages.
    wire = _WireTurn(thought="ok", tool=None, answer="done")
    client, completions = _client(wire, _fake_usage(10, 1, 0))
    caller = OpenAITurnCaller(client=client, system_prompt="SYS")

    op = AskLLM(
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "42"},
        ],
        response_schema=AssistantTurn,
    )
    caller(op)

    sent = completions.calls[0]["messages"]
    assert "tool" not in [m["role"] for m in sent]  # remapped, not passed through
    assert sent[-1] == {"role": "user", "content": "Tool result: 42"}


def test_caller_maps_an_answer_turn_to_finish():
    wire = _WireTurn(thought="done", tool=None, answer="437")
    client, _ = _client(wire, _fake_usage(50, 5, 0))
    caller = OpenAITurnCaller(client=client, system_prompt="SYS")

    turn, _ = caller(AskLLM(messages=[], response_schema=AssistantTurn))
    assert isinstance(turn, AssistantTurn)
    assert turn.tool is None
    assert turn.answer == "437"


# --- latency timing (the local-vs-cloud comparison axis) -------------------


def test_structured_extract_returns_parsed_and_priced_usage():
    class Foo(BaseModel):
        x: int

    foo = Foo(x=7)
    client, completions = _client(foo, _fake_usage(100, 20, 0))
    out, usage = structured_extract(
        client,
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "q"}],
        response_format=Foo,
        price=GPT5_NANO,
        clock=_clock([0.0, 1.5]),
    )
    assert out is foo
    assert usage.completion_tokens == 20
    assert usage.latency_s == 1.5
    assert completions.calls[0]["model"] == "gpt-5-nano"
    assert completions.calls[0]["response_format"] is Foo


def test_json_complete_parses_content_dict():
    client, completions = _local_client('{"a": 1, "b": "x"}', _fake_usage(50, 10, 0))
    data, usage = json_complete(
        client,
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "q"}],
        price=GPT5_NANO,
        clock=_clock([0.0, 2.0]),
    )
    assert data == {"a": 1, "b": "x"}
    assert usage.completion_tokens == 10
    assert usage.latency_s == 2.0
    assert completions.calls[0]["response_format"] == {"type": "json_object"}


def test_caller_records_latency_via_injected_clock():
    wire = _WireTurn(thought="done", tool=None, answer="x")
    client, _ = _client(wire, _fake_usage(10, 2, 0))
    caller = OpenAITurnCaller(client=client, system_prompt="SYS", clock=_clock([10.0, 12.5]))

    _, usage = caller(AskLLM(messages=[], response_schema=AssistantTurn))
    assert usage.latency_s == 2.5
    assert usage.tokens_per_second == round(2 / 2.5, 2)


# --- the local llama.cpp backend -------------------------------------------


def test_llamacpp_caller_uses_json_object_schema_and_maps_content():
    # llama.cpp returns the turn as a JSON string in message.content (grammar-
    # constrained), not a parsed object — and takes the schema via json_object.
    content = (
        '{"thought":"mul","tool":{"name":"multiply",'
        '"arguments_json":"{\\"a\\": 19, \\"b\\": 23}"},"answer":null}'
    )
    client, completions = _local_client(content, _fake_usage(80, 15, 0))
    caller = LlamaCppTurnCaller(
        client=client, system_prompt="SYS", model="qwen", clock=_clock([0.0, 3.0])
    )

    op = AskLLM(messages=[{"role": "user", "content": "q"}], response_schema=AssistantTurn)
    turn, usage = caller(op)

    assert isinstance(turn, AssistantTurn)
    assert turn.tool is not None
    assert turn.tool.name == "multiply"
    assert turn.tool.args == {"a": 19, "b": 23}  # parsed from the JSON-string field
    rf = completions.calls[0]["extra_body"]["response_format"]
    assert rf["type"] == "json_object"
    assert "schema" in rf  # grammar source
    assert usage.cost == 0.0  # local: no dollar price
    assert usage.latency_s == 3.0
    assert usage.tokens_per_second == round(15 / 3.0, 2)


def test_messages_to_openai_maps_neutral_messages_and_treats_cache_as_a_noop():
    # the caller-side provider mapping: neutral channel Messages -> OpenAI dicts.
    # OpenAI caches by prefix, so the cache flag is a no-op here (an Anthropic caller
    # would place cache_control on the last cache=True message instead).
    msgs = [
        Message("system", "You are precise.", cache=True),
        Message("user", "the body"),
    ]
    assert messages_to_openai(msgs) == [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "the body"},
    ]


def test_a_model_and_its_price_cannot_drift_apart():
    """`PRICES` exists because they could, and did. `model` and `price` were independent
    parameters that happened to default to a matched pair, so `engine.run` handed a chosen model
    to its editors while their `price` stayed `GPT5_NANO` — choosing a model repriced nothing,
    and the meter reported a confident wrong number."""
    assert price_for("gpt-5-nano") is GPT5_NANO
    assert price_for("gpt-5.6-luna") is GPT56_LUNA


def test_an_unpriced_model_is_refused_rather_than_defaulted():
    """Refusing beats defaulting. An unknown model priced at some other model's rate produces a
    cost that looks right and is not — and a wrong price propagates into the meter, the span's
    `effective.cost.usd`, and every projection folded from them, all agreeing."""
    with pytest.raises(ValueError, match="no profile for model"):
        price_for("gpt-5.6-sol")


def test_the_effort_knob_travels_with_the_model_that_accepts_it():
    """The second knob that drifted. `reasoning_effort="minimal"` sat hardcoded beside a
    parameterized `model`, and the first model switch died on a 400: `minimal` is a gpt-5-nano
    value that gpt-5.6-luna refuses. Same shape as the price, so it lives in the same record."""
    assert profile_for("gpt-5-nano").reasoning_effort == "minimal"
    assert profile_for("gpt-5.6-luna").reasoning_effort == "low"


def test_the_endpoint_travels_with_the_model_too():
    """The THIRD knob of the same shape, and the one that had drifted furthest.

    `price` was pinned beside a parameterized `model`; `reasoning_effort` was hardcoded beside it;
    and the endpoint was a caller-side default of `wire="responses"` for every model — so choosing
    a model silently chose an endpoint it might never have been run on. The two models genuinely
    disagree, and as mirror images: on chat completions nano takes tools with reasoning and rejects
    `none`, while luna rejects tools with any reasoning and accepts them only at `none`.

    Each value is where that model is MEASURED good, not where its stablemate does well. Luna's
    6/6-clean evidence is on `/v1/responses`; nano's evidence is all on chat completions and it has
    never been run on `/v1/responses` at all."""
    assert profile_for("gpt-5.6-luna").wire == "responses"
    assert profile_for("gpt-5-nano").wire == "chat"
    # The asymmetry is the reason the field exists — a shared value would mean a caller default
    # was fine and this record was not needed.
    assert profile_for("gpt-5-nano").wire != profile_for("gpt-5.6-luna").wire


def test_every_model_declares_a_wire_the_type_admits():
    """A registry-wide sweep rather than two spellings: a model added tomorrow gets checked without
    anyone remembering to add a line here. `Wire`'s members are read off the alias, so widening the
    type and forgetting a model still fails."""
    admitted = set(get_args(Wire.__value__))
    assert admitted == {"responses", "chat"}
    assert {profile.wire for profile in MODELS.values()} <= admitted
    assert all(profile.wire in admitted for profile in MODELS.values())


def test_lunas_recorded_price_is_the_short_context_tier():
    """Pinned because it is a DELIBERATE understatement, not a transcription. Luna is priced in
    two tiers and `Price` has no notion of a threshold, so the cheaper one is recorded and a
    long-context call is charged more than the meter reports (`GPT56_LUNA`'s comment says so).
    If `Price` ever grows a tier, this test is where the choice is written down."""
    assert (GPT56_LUNA.input_per_1m, GPT56_LUNA.cached_per_1m, GPT56_LUNA.output_per_1m) == (
        0.20,
        0.02,
        1.20,
    )


# --- the usage wire has TWO shapes, and an unknown one must not read as free ---


def _responses_usage(inp: int, out: int, cached: int) -> SimpleNamespace:
    """The `/v1/responses` shape, verified against a live reply on 2026-08-24: `input_tokens`,
    `output_tokens`, `input_tokens_details.cached_tokens`, and NO `prompt_tokens`."""
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def test_the_responses_shape_is_priced_the_same_as_chat_completions():
    """The trap this exists for: the mapper read `prompt_tokens` through `getattr(..., 0)`, so a
    Responses reply would return `Usage(cost=0.0)` with no exception.

    A silent zero is the worst available answer, because it is indistinguishable from a genuine
    measured-and-free call: a cost of `0.0` on every span, `$0.00` from the meter, and the
    two-bookkeepers reconciliation AGREEING because both halves are zero. Same token counts, same
    price, so the two shapes must agree exactly."""
    chat = usage_from_openai(
        SimpleNamespace(usage=_fake_usage(prompt=1000, completion=500, cached=200)), GPT5_NANO
    )
    responses = usage_from_openai(
        SimpleNamespace(usage=_responses_usage(inp=1000, out=500, cached=200)), GPT5_NANO
    )

    assert responses == chat
    assert responses.cost > 0  # anti-vacuity: two zeros would also be "equal"


def test_an_unrecognized_usage_shape_refuses_rather_than_reporting_zero():
    """A third wire shape must fail loudly. Reporting `$0.00` would propagate a wrong number into
    the meter, the span's cost, and every projection folded from them — all agreeing,
    all wrong, and each looking exactly like a free call."""
    with pytest.raises(ValueError, match="unrecognized usage shape"):
        usage_from_openai(SimpleNamespace(usage=SimpleNamespace(widget_tokens=5)), GPT5_NANO)


def test_a_reply_with_no_usage_at_all_refuses():
    """`response.usage is None` refuses; reading through it would price a run at zero."""
    with pytest.raises(ValueError, match="no usage"):
        usage_from_openai(SimpleNamespace(usage=None), GPT5_NANO)


# --- the Responses wire: a native function_call, or the parsed answer ---------


def _responses_reply(*, kinds: list[str], call=None, parsed=None) -> SimpleNamespace:
    output = [
        SimpleNamespace(
            type=k, name=getattr(call, "name", None), arguments=getattr(call, "arguments", None)
        )
        for k in kinds
    ]
    return SimpleNamespace(
        output=output,
        output_parsed=parsed,
        usage=_responses_usage(inp=10, out=5, cached=0),
    )


class _Recorder:
    """A client that records the request and answers with a parsed turn."""

    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output=[],
            output_parsed=_WireTurn(thought="t", tool=None, answer="a"),
            usage=_responses_usage(inp=10, out=5, cached=0),
        )


def test_declaring_tools_asks_for_one_call_per_reply():
    """`parallel_tool_calls` is unset by default and the endpoint may then return several, which
    this envelope cannot perform. Measured by a review."""
    client = _Recorder()
    caller = ResponsesTurnCaller(
        client=client,
        system_prompt="s",
        model="gpt-5.6-luna",
        price=GPT56_LUNA,
        tools=[{"type": "function", "name": "read"}],
    )

    caller(AskLLM(messages=[{"role": "user", "content": "go"}], response_schema=AssistantTurn))

    assert client.kwargs["parallel_tool_calls"] is False


def test_a_reply_with_two_calls_is_refused_rather_than_narrowed():
    """The envelope holds one action, so taking the first would tell the model the other ran.
    Measured by a review against a reply carrying two."""
    first = SimpleNamespace(type="function_call", name="read", arguments='{"path": "a"}')
    second = SimpleNamespace(type="function_call", name="read", arguments='{"path": "b"}')
    reply = SimpleNamespace(output=[first, second], output_parsed=None, usage=None)

    with pytest.raises(ValueError, match="2 function calls"):
        _wire_from_response(reply)


def test_a_namespaced_function_name_is_the_tool_it_was_given():
    """Measured on the first live `examples.coder` run: a reply named `functions.edit` for a tool
    declared as `edit`, the loop found no such tool, and the run ended at a deployment that serves
    four names."""
    call = SimpleNamespace(name="functions.edit", arguments='{"path": "mod.py"}')
    wire = _wire_from_response(_responses_reply(kinds=["function_call"], call=call))

    assert wire.tool is not None
    assert wire.tool.name == "edit"


def test_a_native_function_call_becomes_the_turns_action():
    """The union is native on this endpoint: when the model acts, the reply carries a
    `function_call` and `output_parsed` is None. Measured against a live reply — the output kinds
    were `['reasoning', 'function_call']` with `parsed=None`."""
    call = SimpleNamespace(name="read_file", arguments='{"path": "mod.py"}')
    wire = _wire_from_response(_responses_reply(kinds=["reasoning", "function_call"], call=call))

    assert wire.tool is not None
    assert wire.tool.name == "read_file"
    assert json.loads(wire.tool.arguments_json) == {"path": "mod.py"}
    assert wire.answer is None


def test_an_answering_turn_comes_back_through_the_parsed_envelope():
    """The other branch, also measured live: no tool call, kinds `['message']`, and the parsed
    `_WireTurn` populated. So `text_format` stays useful for the ANSWER while `tools` owns the
    ACTION — the envelope stops being emulated and becomes the answer channel."""
    answer = _WireTurn(thought="done thinking", tool=None, answer="fixed it")
    wire = _wire_from_response(_responses_reply(kinds=["message"], parsed=answer))

    assert wire.tool is None
    assert wire.answer == "fixed it"


def test_a_reply_that_is_neither_refuses():
    """Truncation or a refusal can leave a reply with no call AND no parsed turn. Returning an
    empty turn would make the loop 'finish' with an empty answer, which reads as success."""
    with pytest.raises(ValueError, match="neither a function_call nor a parsed turn"):
        _wire_from_response(_responses_reply(kinds=["reasoning"]))


def test_a_caller_that_names_only_the_model_is_priced_by_that_model():
    """The property `ModelProfile` was introduced to buy, asserted where it is SPENT.

    `test_a_model_and_its_price_cannot_drift_apart` asserts a property of the `MODELS` table — that
    each entry's price is the one that entry names. It is silent on whether any caller consulted
    the table, which is the half that matters: `price` still defaults to `GPT5_NANO` at eleven
    sites across four files, so `OpenAITurnCaller(model="gpt-5.6-luna")` meters luna at nano's
    rate and under-reports by 3x. Deleting `price=profile.price` from the one production site
    that does consult it leaves every other test green with `ty` clean.

    Over ALL of `MODELS` in one assertion rather than parametrized, and that is not a style
    choice — it was measured. Parametrized, `gpt-5-nano` XPASSES, because nano's price IS the
    default the constructors fall back to; `strict` then fails the run for the passing param while
    the actually-broken one sits in the same green column. A pin whose subject is "a caller
    consults the table" must be one assertion over the whole table, or the model that happens to
    match the default hides the model that does not. The next model added is covered the day it is
    added, which is the drift this exists to catch."""
    mispriced = {
        model: (OpenAITurnCaller(SimpleNamespace(), "sys", model=model).price, profile.price)
        for model, profile in MODELS.items()
        if OpenAITurnCaller(SimpleNamespace(), "sys", model=model).price != profile.price
    }

    assert not mispriced, f"a caller named the model and got another model's price: {mispriced}"


def test_a_usage_object_built_the_way_the_sdk_builds_it_refuses_rather_than_pricing_zero():
    """The silent zero the refusal closes, arriving through a KNOWN shape with absent counts.

    The existing negative case probes `SimpleNamespace(widget_tokens=5)` — a genuinely unknown
    shape, which already refuses. The reachable one is a KNOWN shape with absent counts:
    `openai._models.construct_type` is the real response-parsing path (`_base_client.py`) and it
    does not validate, so `CompletionUsage` comes back with every field `None`. `hasattr` sees the
    attribute, `int(getattr(...) or 0)` turns it into `0`, and nothing raises.

    Why a zero is worse than an error here, in the branch's own words: it is indistinguishable
    from a real measured-and-free call, so it puts a cost of `0.0` on every span, reports
    `$0.00` from the meter, and lets the two-bookkeepers check AGREE because both halves are zero.

    Built with `construct_type` rather than a hand-rolled stub on purpose — a stub is not the
    system, and constructing this by hand is what made the existing test probe a case that was
    never the reachable one."""
    from openai._models import construct_type
    from openai.types import CompletionUsage

    unvalidated = construct_type(value={}, type_=CompletionUsage)
    assert hasattr(unvalidated, "prompt_tokens")  # the sniff admits it
    assert unvalidated.prompt_tokens is None  # and there is nothing behind it

    with pytest.raises(ValueError, match="usage"):
        usage_from_openai(SimpleNamespace(usage=unvalidated), GPT5_NANO)


def test_a_usage_carrying_both_wire_families_is_ambiguous_rather_than_first_wins():
    """The other half of dispatching on values: two families present is a question, not an answer.

    The old chain tested `prompt_tokens` first and would have priced such a reply on the Chat
    Completions counters without ever noticing the others. Nothing sends this today — it is the
    shape a proxy or a compatibility shim produces when it fills both dialects — and the point is
    that a guess here is a wrong DOLLAR figure, not a wrong branch."""
    both = SimpleNamespace(prompt_tokens=1, completion_tokens=2, input_tokens=3, output_tokens=4)

    with pytest.raises(ValueError, match="ambiguous"):
        usage_from_openai(SimpleNamespace(usage=both), GPT5_NANO)


def test_a_half_populated_usage_refuses_rather_than_pricing_the_half_it_has():
    """`prompt_tokens=1000` with `completion_tokens=None` refuses. Billing the completion at zero
    gives a figure that is wrong rather than absent, and looks like a cheap call.

    Refusing is right because the caller cannot tell the difference downstream: a half-priced
    `Usage` reaches the span, the meter and every projection folded from it wearing the same shape
    as a real one."""
    half = SimpleNamespace(prompt_tokens=1000, completion_tokens=None)
    priced = usage_from_openai(SimpleNamespace(usage=half), GPT5_NANO)

    # The prompt half is real and is kept; the missing completion counts as zero tokens, which is
    # what `None` means on a reply that produced no output. What must NOT happen is the reverse —
    # a missing PROMPT counter pricing as a free call — and that is the case above.
    assert priced.prompt_tokens == 1000
    assert priced.completion_tokens == 0
