"""Tool bracketing strategies + the GBNF-masked local caller.

Offline (fake client): pins the one invariant that makes the cache measurement
meaningful: MASK keeps the *full* catalog in the prompt prefix while narrowing the
decode enum, MUTATE rewrites the prefix to the allowlist, and FULL does neither. Also
pins that cached tokens flow into `Usage.cache_hit_ratio` and that the per-turn
telemetry event carries the bracketing decision.
"""

import json
from types import SimpleNamespace

from agent.bracket import (
    CATALOG,
    FULL,
    MASK,
    MUTATE,
    make_bracketed_caller,
    rotating_select,
)
from effective.domain import AskLLM
from effective.react import AssistantTurn


class _FakeCreate:
    def __init__(self, contents, *, cached=0, prompt=100, completion=10):
        self.contents = list(contents)
        self.calls = []
        self.cached, self.prompt, self.completion = cached, prompt, completion

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=self.prompt,
            completion_tokens=self.completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached),
        )
        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _client(contents, **kw):
    fake = _FakeCreate(contents, **kw)
    return SimpleNamespace(chat=SimpleNamespace(completions=fake)), fake


def _turn_json(thought="t", tool_name="", tool_args_json="", answer=""):
    return json.dumps(
        {
            "thought": thought,
            "tool_name": tool_name,
            "tool_args_json": tool_args_json,
            "answer": answer,
        }
    )


def _op(n_assistant=0):
    messages = [{"role": "user", "content": "solve the problem"}]
    for _ in range(n_assistant):
        messages += [{"role": "assistant", "content": "..."}, {"role": "tool", "content": "9"}]
    return AskLLM(messages=messages, response_schema=AssistantTurn)


def _enum_of(call):
    return set(call["extra_body"]["response_format"]["schema"]["properties"]["tool_name"]["enum"])


def _system(call):
    return call["messages"][0]["content"]


def test_full_strategy_full_catalog_and_full_enum():
    client, fake = _client([_turn_json(tool_name="add", tool_args_json='{"a":1,"b":2}')])
    turn, _ = make_bracketed_caller(client, "m", strategy=FULL)(_op())
    assert turn.tool is not None
    assert turn.tool.name == "add"
    enum = _enum_of(fake.calls[0])
    assert set(CATALOG) <= enum  # every tool is decodable
    assert "search" in _system(fake.calls[0])  # full catalog in the prefix


def test_mask_keeps_full_prefix_but_narrows_enum():
    """The cache-preserving move: prefix advertises ALL tools (stable), enum is the subset."""
    client, fake = _client([_turn_json(tool_name="add", tool_args_json='{"a":1,"b":2}')])
    make_bracketed_caller(client, "m", strategy=MASK)(_op(n_assistant=0))
    allowed = rotating_select()(0)
    enum = _enum_of(fake.calls[0]) - {""}
    assert enum == allowed  # decode is masked to the allowlist
    # ...but the prompt still shows the FULL catalog (so the prefix never changes)
    for tool in CATALOG:
        assert tool in _system(fake.calls[0])


def test_mutate_rewrites_prefix_to_allowlist():
    """The naive trap: the prefix catalog shrinks to the allowlist, so it changes per turn."""
    client, fake = _client([_turn_json(tool_name="add", tool_args_json='{"a":1,"b":2}')])
    make_bracketed_caller(client, "m", strategy=MUTATE)(_op(n_assistant=0))
    allowed = rotating_select()(0)
    system = _system(fake.calls[0])
    assert "weather" not in allowed  # a dropped tool...
    assert "weather" not in system  # ...is gone from the prefix too
    assert _enum_of(fake.calls[0]) - {""} == allowed


def test_allowlist_varies_per_turn():
    """MUTATE's prefix differs across turns (that is what busts the cache)."""
    c0, f0 = _client([_turn_json(answer="done")])
    c1, f1 = _client([_turn_json(answer="done")])
    make_bracketed_caller(c0, "m", strategy=MUTATE)(_op(n_assistant=0))
    make_bracketed_caller(c1, "m", strategy=MUTATE)(_op(n_assistant=1))
    assert _system(f0.calls[0]) != _system(f1.calls[0])


def test_cached_tokens_flow_into_usage():
    client, _ = _client([_turn_json(answer="42")], cached=80, prompt=100)
    _, usage = make_bracketed_caller(client, "m", strategy=MASK)(_op())
    assert usage.cache_read_input_tokens == 80
    assert usage.cache_hit_ratio == 0.8


def test_gated_rejects_out_of_allowlist_then_repairs():
    # first turn picks a real tool that is NOT in this turn's allowlist -> Gated repair
    not_allowed = sorted(set(CATALOG) - rotating_select()(0))[0]
    client, fake = _client(
        [
            _turn_json(tool_name=not_allowed, tool_args_json="{}"),
            _turn_json(tool_name="add", tool_args_json='{"a":1,"b":2}'),
        ]
    )
    turn, _ = make_bracketed_caller(client, "m", strategy=MASK)(_op())
    assert turn.tool is not None
    assert turn.tool.name == "add"
    assert len(fake.calls) == 2  # one repair round-trip


def test_cloud_mode_uses_json_object_and_gated_not_gbnf():
    """cloud=True sends plain json_object (no GBNF schema) — the allowlist is the Gated
    post-hoc repair, and max_completion_tokens (gpt-5 shape), not max_tokens."""
    client, fake = _client([_turn_json(tool_name="add", tool_args_json='{"a":1,"b":2}')])
    make_bracketed_caller(client, "gpt-5-nano", strategy=MASK, cloud=True)(_op())
    call = fake.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert "extra_body" not in call  # no GBNF schema on the cloud path
    assert "max_completion_tokens" in call
    assert "max_tokens" not in call


def test_telemetry_event_carries_the_bracketing_decision():
    spans = []
    client, _ = _client([_turn_json(tool_name="add", tool_args_json='{"a":1,"b":2}')])
    make_bracketed_caller(client, "m", strategy=MASK, sink=spans.append, session_id="s1")(_op())
    assert len(spans) == 1
    fields = spans[0].fields
    assert fields["strategy_name"] == "mask"
    assert fields["turn"] == 0
    assert fields["chose"] == "add"
    assert set(fields["allowed_tools"]) == rotating_select()(0)
